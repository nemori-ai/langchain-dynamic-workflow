"""Unit tests for the Layer 2 codegen — the single ``exec`` point of the meta layer.

``compile_workflow_source`` is where an untrusted source string crosses into a
runnable orchestration callable: AST gate, then ``compile`` + a single ``exec``
into a namespace whose ``__builtins__`` is a curated safe whitelist (never the
real builtins), then the ``orchestrate`` coroutine is extracted. These tests pin
that the namespace is genuinely restricted, that structural defects (no
``orchestrate``, not a coroutine, wrong arity) and gate violations are rejected,
that ``extract_meta`` reads only a pure literal, and that
``run_workflow_from_source`` carries a compiled script end to end through the
engine with fake leaves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Roster
from langchain_dynamic_workflow._codegen import (
    compile_workflow_source,
    extract_meta,
    run_workflow_from_source,
)
from langchain_dynamic_workflow._errors import WorkflowScriptError

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]

_ARGS_ONLY_SCRIPT = """\
async def orchestrate(ctx, args):
    doubled = [n * 2 for n in sorted(args["nums"])]
    return sum(doubled)
"""

_AGENT_SCRIPT = """\
async def orchestrate(ctx, args):
    return await ctx.agent(f"Summarize {args['topic']}", agent_type="writer")
"""


async def test_compiles_and_runs_a_valid_script() -> None:
    fn = compile_workflow_source(_ARGS_ONLY_SCRIPT)
    unused_ctx: Any = object()  # this script ignores ctx
    result = await fn(unused_ctx, {"nums": [3, 1, 2]})
    assert result == 12  # (1 + 2 + 3) * 2


def test_restricted_namespace_excludes_real_builtins() -> None:
    # The compiled function's globals must expose only the safe whitelist as
    # __builtins__ — the real, escape-capable builtins must be absent.
    fn = compile_workflow_source(_ARGS_ONLY_SCRIPT)
    builtins_table = cast(Any, fn).__globals__["__builtins__"]
    for safe in ("len", "sorted", "sum", "range"):
        assert safe in builtins_table
    for dangerous in ("open", "__import__", "eval", "exec", "getattr", "globals"):
        assert dangerous not in builtins_table


def test_missing_orchestrate_is_rejected() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        compile_workflow_source("x = 1\n")
    assert "orchestrate" in str(exc.value)


def test_non_coroutine_orchestrate_is_rejected() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        compile_workflow_source("def orchestrate(ctx, args):\n    return 1\n")
    assert "orchestrate" in str(exc.value).lower()


def test_wrong_arity_orchestrate_is_rejected() -> None:
    with pytest.raises(WorkflowScriptError):
        compile_workflow_source("async def orchestrate(ctx):\n    return 1\n")


def test_gate_violation_propagates_through_compile() -> None:
    source = "import os\nasync def orchestrate(ctx, args):\n    return 1\n"
    with pytest.raises(WorkflowScriptError) as exc:
        compile_workflow_source(source)
    assert "import" in str(exc.value).lower()


def test_extract_meta_reads_a_pure_literal() -> None:
    source = 'meta = {"name": "deep_research", "phases": ["scope", "synth"]}\n' + _AGENT_SCRIPT
    meta = extract_meta(source)
    assert meta == {"name": "deep_research", "phases": ["scope", "synth"]}


def test_extract_meta_returns_none_when_absent() -> None:
    assert extract_meta(_AGENT_SCRIPT) is None


def test_extract_meta_rejects_a_non_literal() -> None:
    # A computed meta value cannot be statically extracted; it must be rejected
    # rather than silently executed.
    source = "meta = {'name': undefined_symbol}\n" + _AGENT_SCRIPT
    with pytest.raises(WorkflowScriptError):
        extract_meta(source)


async def test_run_workflow_from_source_runs_end_to_end(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("a summary")
    roster = Roster().register("writer", leaf)
    result = await run_workflow_from_source(
        _AGENT_SCRIPT, roster=roster, args={"topic": "batteries"}
    )
    assert result == "a summary"


async def test_run_script_can_call_injected_reduce_helpers() -> None:
    # A host-authored script reaches the reduce helpers by name (no import — the AST
    # gate forbids imports). Proves the meta-layer namespace injection delivers F to
    # the host's on-the-fly scripts, not only to imported developer workflows.
    from langchain_dynamic_workflow._codegen import compile_workflow_source

    source = (
        "async def orchestrate(ctx, args):\n"
        "    votes = [{'refuted': False}, {'refuted': True}, {'refuted': False}]\n"
        "    kept = survives(votes, against=lambda v: v['refuted'], kill_at=2)\n"
        "    groups = corroborate(['a', 'A', 'b'], key=lambda s: s.lower(), min_support=2)\n"
        "    review = [ReviewItem(item='x', verdicts=[{'k': True}, {'k': True}])]\n"
        "    bucket = reconcile(review, include=lambda v: v['k'])\n"
        "    uniq = dedup(['a', 'a', 'b'])\n"
        "    return (kept, len(groups), bucket.included, uniq)\n"
    )
    orchestrate = compile_workflow_source(source)
    # Actually RUN the compiled body (it touches only the injected reduce helpers +
    # literals, never ctx), and assert the computed result — proving the injected
    # names resolve AND execute correctly end to end, not merely that they exist:
    #   survives([F, T, F], kill_at=2)        -> True  (1 against < 2)
    #   corroborate(['a','A','b'], min=2)     -> 1 group ('a','A' share key 'a')
    #   reconcile([both-include 'x'])         -> included == ['x']
    #   dedup(['a','a','b'])                  -> ['a','b']
    unused_ctx: Any = object()  # this script ignores ctx
    result = await orchestrate(unused_ctx, {})
    assert result == (True, 1, ["x"], ["a", "b"])


async def test_run_script_can_construct_injected_race_types() -> None:
    # A host-authored script reaches the race value types by name (no import — the
    # AST gate forbids imports). Proves the meta-layer namespace injection delivers
    # the race primitive's surface to on-the-fly scripts, not only to imported
    # developer workflows.
    from langchain_dynamic_workflow._codegen import compile_workflow_source

    source = (
        "async def orchestrate(ctx, args):\n"
        "    candidates = [RaceCandidate(prompt=h, agent_type='inv') for h in args['hyps']]\n"
        "    probe = RaceResult(winner='x', winner_index=0)\n"
        "    return (len(candidates), candidates[0].agent_type, probe.won)\n"
    )
    orchestrate = compile_workflow_source(source)
    # Run the compiled body (it touches only the injected race types + literals,
    # never ctx) and assert the computed result — proving the injected names resolve
    # AND execute, not merely that they are present.
    unused_ctx: Any = object()
    result = await orchestrate(unused_ctx, {"hyps": ["a", "b", "c"]})
    assert result == (3, "inv", True)


async def test_run_script_can_call_ctx_batch_map() -> None:
    # batch_map is a Ctx method, so — like dag / loop_until / race — it needs NO
    # run_script global injection and is NOT on the AST gate's banned-name list.
    # This proves a host-authored script COMPILES with a ctx.batch_map(...) call
    # (the security gate + restricted exec accept it), the dual of the codegen tests
    # that prove the injected value types resolve. The compiled body is exercised
    # against a tiny in-process stub ctx whose batch_map applies fn to each item, so
    # the method-call wiring is proven end to end without a real engine.
    from langchain_dynamic_workflow._codegen import compile_workflow_source

    source = (
        "async def orchestrate(ctx, args):\n"
        "    out = await ctx.batch_map(args['xs'], lambda x: x, max_in_flight=2)\n"
        "    return out\n"
    )
    orchestrate = compile_workflow_source(source)

    class _StubCtx:
        async def batch_map(
            self, items: Any, fn: Any, *, max_in_flight: int | None = None
        ) -> list[Any]:
            return [fn(x) for x in items]

    stub_ctx: Any = _StubCtx()  # not a real Ctx; the call only needs .batch_map
    result = await orchestrate(stub_ctx, {"xs": [1, 2, 3]})
    assert result == [1, 2, 3]

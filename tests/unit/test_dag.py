"""Unit tests for the DAG topological-order fan-out scheduler and its value type."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_dynamic_workflow._codegen import compile_workflow_source
from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._dag import (
    DagNode,
    _validate_dag,  # pyright: ignore[reportPrivateUsage] - internal validator under test
    run_dag,
)
from langchain_dynamic_workflow._engine import run_workflow
from langchain_dynamic_workflow._errors import (
    WorkflowBudgetExceededError,
    WorkflowDagError,
    WorkflowNestingError,
)
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._observability import Span, SpanKind, SpanRecorder
from langchain_dynamic_workflow._roster import Roster


def _node(node_id: str, deps: list[str]) -> DagNode:
    async def _run(d: dict[str, object]) -> str:
        return node_id

    return DagNode(node_id, deps=deps, run=_run)


def test_validate_accepts_a_well_formed_dag() -> None:
    _validate_dag([_node("a", []), _node("b", ["a"]), _node("c", ["a", "b"])])


def test_validate_rejects_duplicate_ids() -> None:
    with pytest.raises(WorkflowDagError, match="duplicate"):
        _validate_dag([_node("a", []), _node("a", [])])


def test_validate_rejects_unknown_dependency() -> None:
    with pytest.raises(WorkflowDagError, match="unknown"):
        _validate_dag([_node("a", ["ghost"])])


def test_validate_rejects_self_dependency() -> None:
    with pytest.raises(WorkflowDagError, match="itself"):
        _validate_dag([_node("a", ["a"])])


def test_validate_rejects_a_cycle() -> None:
    with pytest.raises(WorkflowDagError, match="cycle"):
        _validate_dag([_node("a", ["b"]), _node("b", ["a"])])


def _leaf_node(node_id: str, deps: list[str], *, fail: bool = False) -> DagNode:
    # A node whose result records the deps it actually received, so a test can assert
    # topological data-flow. `fail=True` raises to exercise failure isolation.
    async def _run(d: dict[str, object]) -> dict[str, object]:
        if fail:
            raise RuntimeError(f"{node_id} boom")
        return {"id": node_id, "saw": dict(d)}

    return DagNode(node_id, deps=deps, run=_run)


async def test_run_dag_threads_predecessor_results_in_topo_order() -> None:
    results = await run_dag(
        [
            _leaf_node("pkg", []),
            _leaf_node("modA", ["pkg"]),
            _leaf_node("symA", ["modA"]),
        ]
    )
    # symA saw modA's result; modA saw pkg's result -> topological data-flow.
    sym_a = results["symA"]
    mod_a = results["modA"]
    assert sym_a is not None and mod_a is not None
    assert sym_a["saw"]["modA"]["id"] == "modA"
    assert mod_a["saw"]["pkg"]["id"] == "pkg"


async def test_run_dag_failure_skips_dependents_transitively() -> None:
    results = await run_dag(
        [
            _leaf_node("pkg", [], fail=True),
            _leaf_node("modA", ["pkg"]),  # depends on failed pkg -> skipped
            _leaf_node("symA", ["modA"]),  # depends on skipped modA -> skipped
            _leaf_node("indep", []),  # independent -> survives
        ]
    )
    assert results["pkg"] is None
    assert results["modA"] is None
    assert results["symA"] is None
    indep = results["indep"]
    assert indep is not None
    assert indep["id"] == "indep"


async def test_run_dag_legitimate_none_does_not_skip_dependents() -> None:
    async def _returns_none(_d: dict[str, object]) -> None:
        return None

    async def _child(d: dict[str, object]) -> dict[str, object]:
        return {"saw_keys": sorted(d)}

    results = await run_dag(
        [
            DagNode("root", deps=[], run=_returns_none),
            DagNode("child", deps=["root"], run=_child),
        ]
    )
    # root returned None *legitimately* (it did not raise) -> child still runs.
    assert results["root"] is None
    assert results["child"] == {"saw_keys": ["root"]}


async def test_run_dag_empty_returns_empty_dict() -> None:
    assert await run_dag([]) == {}


async def test_run_dag_control_flow_signal_fails_loud_not_masked() -> None:
    async def _budget_breach(_d: dict[str, object]) -> object:
        raise WorkflowBudgetExceededError("pool exhausted")

    async def _other(_d: dict[str, object]) -> str:
        return "ok"

    with pytest.raises(WorkflowBudgetExceededError):
        await run_dag(
            [DagNode("a", deps=[], run=_budget_breach), DagNode("b", deps=[], run=_other)]
        )


async def test_run_dag_cancellation_does_not_leak_tasks() -> None:
    # When run_dag is cancelled externally while a node is in-flight, the finally
    # block must cancel AND await the in-flight node tasks so none are left pending
    # (the "Task was destroyed but it is pending!" asyncio warning must not fire).
    started: list[str] = []

    async def _slow(_d: dict[str, object]) -> str:
        started.append("slow")
        await asyncio.sleep(10)  # long enough that the outer cancel hits mid-flight
        return "done"

    task = asyncio.create_task(run_dag([DagNode("a", deps=[], run=_slow)]))
    await asyncio.sleep(0.05)  # let the node start
    assert started == ["slow"], "node should have started before we cancel"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Give the event loop one turn to settle any residual task teardown.
    await asyncio.sleep(0)
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leaked == [], f"leaked {len(leaked)} pending task(s) after run_dag cancellation"


async def test_run_dag_isolates_a_synchronously_raising_run_callable() -> None:
    # A node whose `run` raises SYNCHRONOUSLY (a non-async callable, an async function
    # whose prologue raises, or one returning a non-coroutine) is constructed before
    # `create_task` receives it. The synchronous raise must be isolated exactly like an
    # async node-task failure — recorded as a None result with dependents skipped — not
    # allowed to escape `_start` and tear the whole graph down, destroying the healthy
    # independent `good` node.
    def sync_raiser(_d: dict[str, object]) -> Any:
        raise KeyError("missing")  # raises before returning a coroutine

    async def ok(_d: dict[str, object]) -> str:
        return "ok"

    results = await run_dag(
        [
            DagNode("bad", deps=[], run=sync_raiser),
            DagNode("good", deps=[], run=ok),
        ]
    )
    assert results["bad"] is None  # recorded as a failed node, not propagated raw
    assert results["good"] == "ok"  # the healthy sibling ran; the graph was not torn down


async def test_run_dag_synchronous_control_flow_signal_aborts_the_dag() -> None:
    # A control-flow signal raised SYNCHRONOUSLY during node construction must be
    # captured in `aborted` and re-raised after the graph drains — exactly like the
    # async path — never swallowed as a None hole. This pins the regular-exception vs.
    # control-flow-signal distinction across the synchronous-raise branch so a future
    # refactor cannot silently mask a budget/determinism/dag signal raised in `run`.
    #
    # The `sibling_ran` probe discriminates a clean capture-then-drain (the signal is
    # recorded, no further nodes launch, in-flight drains, then it re-raises) from a raw
    # escape: a sibling node already in flight must be awaited to completion under the
    # `finally` teardown rather than orphaned when the exception tears `run_dag` down.
    sibling_ran = asyncio.Event()

    def sync_budget_breach(_d: dict[str, object]) -> Any:
        raise WorkflowBudgetExceededError("pool exhausted")  # raises synchronously

    async def slow_sibling(_d: dict[str, object]) -> str:
        await asyncio.sleep(0.02)  # in flight when the signal is captured
        sibling_ran.set()
        return "ok"

    with pytest.raises(WorkflowBudgetExceededError):
        await run_dag(
            [
                # The ready queue is a LIFO stack popped from the end, so `slow_sibling`
                # (last) is launched FIRST and is genuinely in flight when the next pop
                # constructs `sync_budget_breach` and it raises synchronously.
                DagNode("breach", deps=[], run=sync_budget_breach),
                DagNode("sibling", deps=[], run=slow_sibling),
            ]
        )
    assert sibling_ran.is_set(), "the in-flight sibling must drain before the signal re-raises"


async def test_run_dag_synchronous_cancellation_propagates_not_swallowed() -> None:
    # A true process/control BaseException raised SYNCHRONOUSLY during node construction
    # — here asyncio.CancelledError (a race() loser / externally-cancelled run whose
    # cancellation lands during `node.run` construction) — must PROPAGATE out of run_dag,
    # never be recorded as a None hole. This matches the async settle path, where
    # Task.exception() re-raises a cancelled node task, and the repo-wide policy that
    # process/control signals (CancelledError / KeyboardInterrupt / SystemExit) surface
    # loud. CancelledError is NOT a member of WORKFLOW_CONTROL_FLOW_SIGNALS and is not an
    # `Exception` subclass, so it must be caught by neither isolation clause.
    def sync_cancel(_d: dict[str, object]) -> Any:
        raise asyncio.CancelledError  # process/control signal, not a leaf failure

    async def ok(_d: dict[str, object]) -> str:
        return "ok"

    with pytest.raises(asyncio.CancelledError):
        await run_dag(
            [
                DagNode("bad", deps=[], run=sync_cancel),
                DagNode("good", deps=[], run=ok),
            ]
        )


async def test_run_dag_synchronous_raise_skips_dependents() -> None:
    # The synchronous-raise isolation must cascade like an ordinary node failure: a
    # node depending on the sync-raising node is skipped and lands as None, while an
    # unrelated node survives.
    def sync_raiser(_d: dict[str, object]) -> Any:
        raise KeyError("missing")

    async def child(_d: dict[str, object]) -> str:
        return "child ran"

    async def ok(_d: dict[str, object]) -> str:
        return "ok"

    results = await run_dag(
        [
            DagNode("bad", deps=[], run=sync_raiser),
            DagNode("dependent", deps=["bad"], run=child),  # depends on failed bad -> skipped
            DagNode("indep", deps=[], run=ok),  # independent -> survives
        ]
    )
    assert results["bad"] is None
    assert results["dependent"] is None  # skipped because its predecessor failed
    assert results["indep"] == "ok"


async def test_run_dag_nesting_error_fails_loud() -> None:
    # A WorkflowNestingError (depth-cap breach) raised by a dag node must be
    # re-raised after drain, not masked as a None hole. This is the regression
    # for the M7 Codex-review BLOCKER fix: previously the error was absent from
    # WORKFLOW_CONTROL_FLOW_SIGNALS and would have been silently swallowed.
    async def _nest_breach(_d: dict[str, object]) -> object:
        raise WorkflowNestingError("nesting too deep")

    async def _other(_d: dict[str, object]) -> str:
        return "ok"

    with pytest.raises(WorkflowNestingError):
        await run_dag([DagNode("a", deps=[], run=_nest_breach), DagNode("b", deps=[], run=_other)])


# ---------------------------------------------------------------------------
# ctx.dag integration: fan-out frame + SpanKind.DAG
# ---------------------------------------------------------------------------


class _CountingLeaf:
    """A lightweight leaf runner that records prompts and counts invocations."""

    def __init__(self, *, prefix: str) -> None:
        self.calls = 0
        self.prefix = prefix

    async def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
    ) -> LeafOutcome:
        self.calls += 1
        return LeafOutcome(
            state={"messages": [AIMessage(content=f"{self.prefix}:{prompt}")]}, usage=0
        )


def _dag_ctx(leaf: _CountingLeaf, journal: InMemoryJournalStore, collected: list[Span]) -> Ctx:
    roster = Roster()
    roster.register("worker", object())  # type: ignore[arg-type]
    return Ctx(
        roster=roster,
        journal=journal,
        leaf_runner=leaf,
        gate=ConcurrencyGate(limit=8),
        spans=SpanRecorder(sink=collected.append),
    )


async def test_ctx_dag_runs_topologically_and_emits_a_dag_span() -> None:
    leaf = _CountingLeaf(prefix="R")
    collected: list[Span] = []
    ctx = _dag_ctx(leaf, InMemoryJournalStore(), collected)

    results = await ctx.dag(
        [
            DagNode("pkg", deps=[], run=lambda d: ctx.agent("doc package", agent_type="worker")),
            DagNode(
                "modA",
                deps=["pkg"],
                run=lambda d: ctx.agent(f"doc A | {d['pkg']}", agent_type="worker"),
            ),
        ]
    )
    assert results["pkg"] == "R:doc package"
    assert results["modA"] == "R:doc A | R:doc package"
    assert leaf.calls == 2
    # Verify a DAG span was emitted with the correct node/surviving counts.
    dag_spans = [s for s in collected if s.kind is SpanKind.DAG]
    assert len(dag_spans) == 1
    assert dag_spans[0].attributes["node_count"] == 2
    assert dag_spans[0].attributes["surviving_count"] == 2


async def test_authored_script_dagnode_resolves_at_runtime() -> None:
    # A genuine red→green test of the run_script namespace injection: the compiled
    # script names `DagNode` (no import) and is actually EXECUTED, so a missing
    # injection surfaces as a NameError through run_workflow rather than passing
    # silently.
    source = (
        "async def orchestrate(ctx, args):\n"
        "    nodes = [DagNode('a', deps=[], run=lambda d: ctx.agent('q', agent_type='w'))]\n"
        "    return await ctx.dag(nodes)\n"
    )
    orchestrate = compile_workflow_source(source)

    async def leaf(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return {"messages": [*inp["messages"], AIMessage(content="DOC")]}

    roster = Roster().register("w", RunnableLambda(leaf))

    async def top(ctx: Ctx) -> Any:
        return await orchestrate(ctx, {})

    result = await run_workflow(top, roster=roster, thread_id="t-inject")
    assert result == {"a": "DOC"}

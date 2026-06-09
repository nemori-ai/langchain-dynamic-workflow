"""Unit tests for the DAG topological-order fan-out scheduler and its value type."""

from __future__ import annotations

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
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError, WorkflowDagError
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

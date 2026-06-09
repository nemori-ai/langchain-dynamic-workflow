"""Unit tests for the DAG topological-order fan-out scheduler and its value type."""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow._dag import DagNode, _validate_dag, run_dag
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError, WorkflowDagError


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
    assert results["symA"]["saw"]["modA"]["id"] == "modA"
    assert results["modA"]["saw"]["pkg"]["id"] == "pkg"


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
    assert results["indep"]["id"] == "indep"


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

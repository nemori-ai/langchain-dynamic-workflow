"""Unit tests for the DAG topological-order fan-out scheduler and its value type."""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow._dag import DagNode, _validate_dag
from langchain_dynamic_workflow._errors import WorkflowDagError


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

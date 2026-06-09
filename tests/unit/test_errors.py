"""Unit tests for the engine fail-loud exception family."""

from __future__ import annotations

from langchain_dynamic_workflow._errors import (
    WORKFLOW_CONTROL_FLOW_SIGNALS,
    WorkflowCycleError,
    WorkflowDagError,
    WorkflowNestingError,
)


def test_dag_and_cycle_errors_are_runtime_errors() -> None:
    assert issubclass(WorkflowDagError, RuntimeError)
    assert issubclass(WorkflowCycleError, RuntimeError)


def test_structural_violations_fail_loud_inside_fanout() -> None:
    # A malformed dag / a nesting cycle / a depth-cap breach raised inside a
    # parallel/pipeline/race/dag frame must NOT be masked as a None hole — each is
    # an author bug (structural error), not a leaf failure.
    assert WorkflowDagError in WORKFLOW_CONTROL_FLOW_SIGNALS
    assert WorkflowCycleError in WORKFLOW_CONTROL_FLOW_SIGNALS
    assert WorkflowNestingError in WORKFLOW_CONTROL_FLOW_SIGNALS

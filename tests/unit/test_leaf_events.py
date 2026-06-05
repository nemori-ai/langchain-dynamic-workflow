"""Unit tests for the per-leaf callback tap that normalizes a leaf's run subtree.

LeafEventHandler is a LangChain BaseCallbackHandler appended to a leaf's callbacks
list. It closes over the owning leaf's span_id (correlation is by the handler
instance, NOT by inherited metadata, which deepagents drops at the subagent
boundary) and turns each on_*_start/end/error edge into a LeafEvent carrying the
run-tree node ids. These tests pin the normalization, the run-tree fields, the
shape-only default detail vs the payload opt-in, and that a sink failure is the
sink's own concern (the handler does not swallow on the engine's behalf).
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.outputs import LLMResult

from langchain_dynamic_workflow import LeafEvent
from langchain_dynamic_workflow._leaf_events import LeafEventHandler


def test_handler_normalizes_chat_model_start_end_into_leaf_events() -> None:
    events: list[LeafEvent] = []
    handler = LeafEventHandler(leaf_span_id="leaf-abc", sink=events.append)
    root = uuid4()
    child = uuid4()

    handler.on_chat_model_start(
        serialized={"name": "fake-model"},
        messages=[[]],
        run_id=child,
        parent_run_id=root,
    )
    handler.on_llm_end(response=_empty_llm_result(), run_id=child, parent_run_id=root)

    assert [e.phase for e in events] == ["start", "end"]
    start = events[0]
    assert start.leaf_span_id == "leaf-abc"
    assert start.kind == "chat_model"
    assert start.run_id == str(child)
    assert start.parent_run_id == str(root)
    assert start.name == "fake-model"
    assert start.ts > 0.0


def test_handler_tool_events_carry_name_and_run_tree() -> None:
    events: list[LeafEvent] = []
    handler = LeafEventHandler(leaf_span_id="leaf-1", sink=events.append)
    root = uuid4()
    tool_run = uuid4()

    handler.on_tool_start(
        serialized={"name": "search"},
        input_str="query",
        run_id=tool_run,
        parent_run_id=root,
    )
    handler.on_tool_end(output="result", run_id=tool_run, parent_run_id=root)

    assert [e.kind for e in events] == ["tool", "tool"]
    assert events[0].name == "search"
    assert events[0].parent_run_id == str(root)


def test_detail_is_shape_only_by_default_and_payloads_only_on_opt_in() -> None:
    # Default: detail never carries raw tool input / model text.
    shape_only: list[LeafEvent] = []
    handler = LeafEventHandler(leaf_span_id="leaf-x", sink=shape_only.append)
    run = uuid4()
    handler.on_tool_start(
        serialized={"name": "search"},
        input_str="SECRET-QUERY",
        run_id=run,
        parent_run_id=None,
    )
    assert "SECRET-QUERY" not in str(shape_only[0].detail)

    # Opt-in: the raw input appears under a bounded detail key.
    with_payload: list[LeafEvent] = []
    handler2 = LeafEventHandler(
        leaf_span_id="leaf-x", sink=with_payload.append, include_payloads=True
    )
    handler2.on_tool_start(
        serialized={"name": "search"},
        input_str="SECRET-QUERY",
        run_id=run,
        parent_run_id=None,
    )
    assert "SECRET-QUERY" in str(with_payload[0].detail)


def test_handler_does_not_swallow_a_raising_sink() -> None:
    # The handler is a thin normalizer; a raising sink is the consumer's concern
    # (the engine's documented inline-sink contract), so the handler propagates it.
    def boom(_event: LeafEvent) -> None:
        raise RuntimeError("sink boom")

    handler = LeafEventHandler(leaf_span_id="leaf-z", sink=boom)
    try:
        handler.on_tool_start(
            serialized={"name": "t"}, input_str="x", run_id=uuid4(), parent_run_id=None
        )
    except RuntimeError as exc:
        assert "sink boom" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("the handler must not swallow a raising sink")


def _empty_llm_result() -> LLMResult:
    return LLMResult(generations=[])

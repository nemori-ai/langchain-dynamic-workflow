"""Unit tests for the observability span layer (observability-by-default).

The orchestration primitives emit a structured :class:`Span` on completion so a
trace consumer sees what ran, whether a leaf was a journal hit, how many tokens it
burned, and how a fan-out fared — without the orchestration script doing anything.
These unit tests pin the span recorder mechanics in isolation; the integration
test drives spans through a real ``run_workflow``.
"""

from __future__ import annotations

from langchain_dynamic_workflow._observability import (
    Span,
    SpanKind,
    SpanRecorder,
)


def test_recorder_emits_a_completed_span_with_attributes() -> None:
    emitted: list[Span] = []
    recorder = SpanRecorder(sink=emitted.append)

    with recorder.span(SpanKind.AGENT, "researcher") as span:
        span.set("agent_type", "researcher")
        span.set("cached", False)
        span.set("usage_tokens", 42)

    assert len(emitted) == 1
    completed = emitted[0]
    assert completed.kind == SpanKind.AGENT
    assert completed.name == "researcher"
    assert completed.attributes == {"agent_type": "researcher", "cached": False, "usage_tokens": 42}
    # A completed span carries a non-negative wall-clock duration.
    assert completed.duration_s >= 0.0
    assert completed.error is None


def test_recorder_marks_a_span_that_raised_and_re_raises() -> None:
    # A span whose body raises is still emitted (failure is observable), carries
    # the error text, and the exception propagates — the recorder never swallows.
    emitted: list[Span] = []
    recorder = SpanRecorder(sink=emitted.append)

    try:
        with recorder.span(SpanKind.PARALLEL, "fanout"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:  # pragma: no cover - the body must raise
        raise AssertionError("the span body should have re-raised RuntimeError")

    assert len(emitted) == 1
    completed = emitted[0]
    assert completed.kind == SpanKind.PARALLEL
    assert completed.error is not None
    assert "boom" in completed.error


def test_default_recorder_is_a_silent_noop() -> None:
    # The default sink swallows spans so observability is opt-in at zero cost when
    # no sink is wired — a workflow with no trace consumer pays nothing.
    recorder = SpanRecorder()
    with recorder.span(SpanKind.PIPELINE, "refine") as span:
        span.set("item_count", 3)
    # No sink, no collection, no error — the recorder is a clean no-op.

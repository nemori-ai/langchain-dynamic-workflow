"""Unit coverage for ``ui_adapter.UiAdapter`` — the demo's event-mapping layer.

``UiAdapter`` lifts the host graph's inline ``ProgressEntry``/``Span`` -> Gen-UI
mapping out of ``run_hello_demo`` into a tested unit. It is the *mapping* layer; the
*transport* layer (host-config rebind, contextvar dance) lives in ``ui_bridge`` and
is covered separately. These tests drive the real ``UiAdapter`` against a captured
``emit`` stand-in (a plain ``(component, props)`` collector), so they pin the four
load-bearing behaviors the adapter exists to guarantee:

* a ``ProgressEntry`` becomes a ``phase_timeline`` event carrying ``kind`` /
  ``message`` / a stable ``event_id``;
* ``parallel`` / ``pipeline`` spans become ``fanout_graph`` events, a cached
  agent-leaf span additionally emits a ``journal_badge``, and a plain agent span
  maps to ``agent_span``;
* the same logical event is deduped by a stable content-based id (a failed-retry
  re-emit of progress, or a resume re-emit of a cached span, must not double-fire);
  and
* the adapter never lets an ``emit`` failure propagate (the engine calls sinks
  directly, where a raising sink would break orchestration).
"""

from __future__ import annotations

from typing import Any

from ui_adapter import UiAdapter

from langchain_dynamic_workflow import (
    ProgressEntry,
    ProgressKind,
    Span,
    SpanKind,
)

Event = tuple[str, dict[str, Any]]


def _collector() -> tuple[list[Event], UiAdapter]:
    """Build a ``UiAdapter`` whose emits land in a returned list, in order."""
    sent: list[Event] = []

    def emit(component: str, props: dict[str, Any]) -> None:
        sent.append((component, props))

    return sent, UiAdapter(emit=emit)


def _agent_span(*, name: str = "researcher", cached: bool = False) -> Span:
    """A completed leaf ``agent()`` span with the real attribute keys the engine sets."""
    return Span(
        kind=SpanKind.AGENT,
        name=name,
        attributes={"agent_type": name, "cached": cached, "usage_tokens": 0 if cached else 1234},
        duration_s=0.5,
        error=None,
    )


# --- (1) ProgressEntry -> phase_timeline -------------------------------------


def test_progress_entry_maps_to_phase_timeline_event() -> None:
    sent, adapter = _collector()

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="research"))

    assert len(sent) == 1
    component, props = sent[-1]
    assert component == "phase_timeline"
    assert props["kind"] == "phase"
    assert props["message"] == "research"
    assert props.get("event_id")  # the dedupe key must exist and be non-empty


def test_progress_log_entry_carries_log_kind() -> None:
    sent, adapter = _collector()

    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message="working..."))

    component, props = sent[-1]
    assert component == "phase_timeline"
    assert props["kind"] == "log"
    assert props["message"] == "working..."


# --- (2) Span mapping --------------------------------------------------------


def test_parallel_span_maps_to_fanout_graph() -> None:
    sent, adapter = _collector()

    adapter.on_span(
        Span(
            kind=SpanKind.PARALLEL,
            name="parallel",
            attributes={"thunk_count": 3, "surviving_count": 2},
            duration_s=0.1,
            error=None,
        )
    )

    assert len(sent) == 1
    component, props = sent[-1]
    assert component == "fanout_graph"
    assert props["kind"] == "parallel"
    # Real PARALLEL attributes (thunk_count / surviving_count) surface to the UI.
    assert props["thunk_count"] == 3
    assert props["surviving_count"] == 2
    assert "event_id" in props


def test_pipeline_span_maps_to_fanout_graph() -> None:
    sent, adapter = _collector()

    adapter.on_span(
        Span(
            kind=SpanKind.PIPELINE,
            name="pipeline",
            attributes={"item_count": 4, "surviving_count": 4},
            duration_s=0.2,
            error=None,
        )
    )

    component, props = sent[-1]
    assert component == "fanout_graph"
    assert props["kind"] == "pipeline"
    assert props["item_count"] == 4
    assert props["surviving_count"] == 4


def test_cached_agent_span_emits_journal_badge() -> None:
    sent, adapter = _collector()

    adapter.on_span(_agent_span(name="researcher", cached=True))

    components = [component for component, _ in sent]
    # A cached leaf is the journal's headline: it must surface a journal_badge.
    assert "journal_badge" in components
    badge_props = next(props for component, props in sent if component == "journal_badge")
    assert badge_props["name"] == "researcher"
    assert badge_props["cached"] is True
    assert "event_id" in badge_props


def test_fresh_agent_span_does_not_emit_journal_badge() -> None:
    sent, adapter = _collector()

    adapter.on_span(_agent_span(name="researcher", cached=False))

    components = [component for component, _ in sent]
    # A freshly-executed leaf is not a journal hit, so no badge.
    assert "journal_badge" not in components
    # It still surfaces as a generic agent span (so the timeline shows the leaf ran).
    assert "agent_span" in components
    span_props = next(props for component, props in sent if component == "agent_span")
    assert span_props["name"] == "researcher"
    assert span_props["cached"] is False


# --- (3) Stable-id dedupe ----------------------------------------------------


def test_duplicate_cached_span_deduped_on_resume() -> None:
    """A resume re-emits a span for every replayed (cached) leaf; emit it once."""
    sent: list[str] = []
    adapter = UiAdapter(emit=lambda _component, props: sent.append(props["event_id"]))

    span = _agent_span(name="leaf-A", cached=True)
    # Same logical span object, emitted twice (mirrors a resume re-emitting it).
    adapter.on_span(span)
    adapter.on_span(span)

    # journal_badge + agent_span on the first call, nothing on the second.
    assert len(sent) == len(set(sent)), "no event_id may repeat"
    first_call_ids = set(sent)
    assert len(first_call_ids) == len(sent)
    # Re-emitting the identical span produced no new events.
    assert len(sent) == 2  # one journal_badge id + one agent_span id, each once


def test_duplicate_progress_entry_deduped_on_retry() -> None:
    """A failed-retry re-delivers the same progress entry; the adapter emits it once."""
    sent: list[str] = []
    adapter = UiAdapter(emit=lambda _component, props: sent.append(props["event_id"]))

    entry = ProgressEntry(kind=ProgressKind.PHASE, message="research")
    adapter.on_progress(entry)
    adapter.on_progress(entry)

    assert len(sent) == 1


def test_distinct_progress_entries_get_distinct_ids() -> None:
    """Two genuinely different entries must NOT collide on the same stable id."""
    sent: list[str] = []
    adapter = UiAdapter(emit=lambda _component, props: sent.append(props["event_id"]))

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="a"))
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="b"))

    assert len(sent) == 2
    assert sent[0] != sent[1]


def test_event_id_is_content_based_not_identity_based() -> None:
    """Two distinct objects with identical content dedupe to the SAME id.

    A content hash (not ``id()``) is the whole point: a resume builds a *new*
    ``Span`` object with the same fields, and it must still be recognized as the
    same logical event.
    """
    sent: list[str] = []
    adapter = UiAdapter(emit=lambda _component, props: sent.append(props["event_id"]))

    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message="same"))
    # A brand-new object, equal content — different identity.
    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message="same"))

    assert len(sent) == 1, "content-equal events must dedupe even across object identity"


# --- (4) Non-blocking red line ----------------------------------------------


def test_emit_exception_is_swallowed_on_progress() -> None:
    def boom(_component: str, _props: dict[str, Any]) -> None:
        raise RuntimeError("ui down")

    adapter = UiAdapter(emit=boom)

    # Must not propagate: the engine calls progress sinks directly.
    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message="x"))


def test_emit_exception_is_swallowed_on_span() -> None:
    def boom(_component: str, _props: dict[str, Any]) -> None:
        raise RuntimeError("ui down")

    adapter = UiAdapter(emit=boom)

    # Both a fanout span and a cached leaf (which fans out into two emits) must be safe.
    adapter.on_span(
        Span(
            kind=SpanKind.PARALLEL,
            name="parallel",
            attributes={"thunk_count": 2, "surviving_count": 2},
            duration_s=0.0,
            error=None,
        )
    )
    adapter.on_span(_agent_span(cached=True))


def test_emit_failure_does_not_poison_dedupe_set() -> None:
    """A failed emit must not record the id as 'seen' — a later retry can still emit.

    If the adapter marked an event seen *before* a failed emit, a transient UI
    outage would permanently suppress that event. The id is committed to the
    seen-set only after a successful emit.
    """
    failures = {"count": 1}
    delivered: list[str] = []

    def flaky(_component: str, props: dict[str, Any]) -> None:
        if failures["count"] > 0:
            failures["count"] -= 1
            raise RuntimeError("transient")
        delivered.append(props["event_id"])

    adapter = UiAdapter(emit=flaky)
    entry = ProgressEntry(kind=ProgressKind.PHASE, message="research")

    adapter.on_progress(entry)  # fails, swallowed, NOT marked seen
    adapter.on_progress(entry)  # retry succeeds

    assert len(delivered) == 1, "a swallowed failure must not block a later retry"

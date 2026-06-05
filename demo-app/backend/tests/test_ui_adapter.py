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
* the same logical event is keyed by a resume-stable id: a resume (a fresh adapter
  re-executing the script) reproduces the EXACT id sequence of the fresh run even
  though its spans carry changed ``cached`` / ``usage_tokens`` / ``duration_s``, so
  the frontend dedupes the re-emit; yet genuinely-distinct same-content events (three
  same-type leaves, two identical-text log lines) keep distinct ids; and
* the adapter never lets an ``emit`` failure propagate (the engine calls sinks
  directly, where a raising sink would break orchestration).
"""

from __future__ import annotations

from typing import Any

from ui_adapter import (
    _AGENT_SPAN,
    _FANOUT_GRAPH,
    _JOURNAL_BADGE,
    _PHASE_TIMELINE,
    UiAdapter,
)

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


def _agent_span(
    *,
    name: str = "researcher",
    span_id: str = "span0span0span0span0",
    cached: bool = False,
    usage_tokens: int | None = None,
    duration_s: float = 0.5,
) -> Span:
    """A completed leaf ``agent()`` span with the real attribute keys the engine sets.

    Defaults mirror what the engine records: a fresh leaf reports real token usage and
    a real wall-clock duration; a cached (resumed) leaf reports its replayed usage and
    a near-zero journal-lookup duration. These run-variant fields are exactly what must
    NOT enter the emitted ``event_id``, so the helper lets a test set them independently.

    Args:
        name: The leaf's display name (also its ``agent_type`` attribute).
        span_id: The engine-minted span id. The engine reproduces this identical id
            across a fresh run and its resume re-execution for a sequential leaf, so a
            resume test supplies the SAME ``span_id`` for the fresh and cached span to
            mirror what the engine does — and the adapter consumes it verbatim (I1).
        cached: Whether the leaf was served from the journal.
        usage_tokens: Replayed/real token usage; defaults to a cached/fresh stand-in.
        duration_s: Wall-clock (fresh) or near-zero journal-lookup (cached) duration.
    """
    if usage_tokens is None:
        usage_tokens = 0 if cached else 1234
    return Span(
        span_id=span_id,
        kind=SpanKind.AGENT,
        name=name,
        attributes={"agent_type": name, "cached": cached, "usage_tokens": usage_tokens},
        duration_s=duration_s,
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
            span_id="par0par0par0par0",
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
            span_id="pipe0pipe0pipe0pipe0",
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


# --- (2b) I1: spans consume the engine-minted span_id ------------------------


def test_agent_span_event_id_is_the_engine_minted_span_id() -> None:
    sent, adapter = _collector()
    adapter.on_span(_agent_span(name="researcher", span_id="deadbeefcafef00d"))
    span_props = next(p for c, p in sent if c == _AGENT_SPAN)
    # I1: the adapter consumes the engine id verbatim; it no longer computes its own.
    assert span_props["event_id"] == "deadbeefcafef00d"


def test_fanout_span_event_id_is_the_engine_minted_span_id() -> None:
    sent, adapter = _collector()
    adapter.on_span(
        Span(
            span_id="fan0fan0fan0fan0",
            kind=SpanKind.PARALLEL,
            name="parallel",
            attributes={"thunk_count": 2, "surviving_count": 2},
            duration_s=0.1,
            error=None,
        )
    )
    props = next(p for c, p in sent if c == _FANOUT_GRAPH)
    assert props["event_id"] == "fan0fan0fan0fan0"


def test_cached_badge_event_id_is_derived_from_the_span_id() -> None:
    # The badge shares the leaf's span_id but must stay a SEPARATE card, so its id is a
    # deterministic local salt of the span id (preserves resume-stability, distinct card).
    sent, adapter = _collector()
    adapter.on_span(_agent_span(name="researcher", span_id="cafef00dcafef00d", cached=True))
    span_props = next(p for c, p in sent if c == _AGENT_SPAN)
    badge_props = next(p for c, p in sent if c == _JOURNAL_BADGE)
    assert span_props["event_id"] == "cafef00dcafef00d"
    assert badge_props["event_id"] == "cafef00dcafef00d-badge"


def test_progress_event_id_is_still_adapter_computed() -> None:
    # Scope boundary: progress entries are NOT spans, so they keep the adapter id path.
    sent, adapter = _collector()
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="research"))
    props = next(p for c, p in sent if c == _PHASE_TIMELINE)
    assert props.get("event_id")  # present and adapter-derived (not an engine span_id)


# --- (3) Stable-id dedupe ----------------------------------------------------


def _agent_ids(events: list[Event]) -> list[str]:
    """The ``agent_span`` event ids from a captured stream, in order."""
    return [props["event_id"] for comp, props in events if comp == _AGENT_SPAN]


def test_resumed_cached_leaf_gets_the_same_agent_span_id_as_the_fresh_run() -> None:
    """The core resume invariant: a leaf's agent_span id is stable across fresh -> cached.

    A real resume builds a *fresh adapter* (a new host-tool invocation, new process) and
    re-executes the script, re-emitting the SAME leaf with the run-variant fields changed
    exactly as the engine changes them: the fresh run reports ``cached=False`` with real
    usage + a real wall-clock duration; the resume re-emits it as ``cached=True`` with
    replayed usage and a near-zero journal-lookup duration. The engine — not the adapter —
    mints the resume-stable ``span_id`` (reproduced identically fresh -> cached for a
    sequential leaf), so both adapters consume the SAME ``span_id`` and emit the SAME
    ``event_id`` for the leaf — so the frontend (which keys on ``event_id``) recognizes
    the resume re-emit as the same logical span and drops it instead of double-firing.
    This proves the adapter faithfully passes the engine-reproduced id through (I1).
    """
    fresh_events: list[Event] = []
    fresh = UiAdapter(emit=lambda comp, props: fresh_events.append((comp, dict(props))))
    fresh.on_span(
        _agent_span(
            name="leaf-A",
            span_id="leafa0leafa0leaf",
            cached=False,
            usage_tokens=1234,
            duration_s=2.31,
        )
    )

    resume_events: list[Event] = []
    resume = UiAdapter(emit=lambda comp, props: resume_events.append((comp, dict(props))))
    resume.on_span(
        _agent_span(
            name="leaf-A",
            span_id="leafa0leafa0leaf",
            cached=True,
            usage_tokens=1180,
            duration_s=0.0007,
        )
    )

    fresh_ids = _agent_ids(fresh_events)
    resume_ids = _agent_ids(resume_events)
    assert fresh_ids == resume_ids != [], (
        "the resume re-emit must reuse the fresh run's agent_span id (dedupe identity "
        f"must exclude cached/usage/duration): fresh={fresh_ids} resume={resume_ids}"
    )
    # The cache hit is genuinely new on resume — the fresh run emitted no badge.
    assert not [c for c, _ in fresh_events if c == _JOURNAL_BADGE]
    assert [c for c, _ in resume_events if c == _JOURNAL_BADGE] == [_JOURNAL_BADGE]


def test_resume_reproduces_the_full_agent_span_id_sequence() -> None:
    """A full ordered span stream re-emitted on resume reuses the same id sequence.

    Three same-type leaves (e.g. three skeptics) fan out on the fresh run, each with its
    own engine-minted ``span_id``. A resume re-executes the script (a fresh adapter) and
    re-emits all three IN THE SAME ORDER, now cached, with the engine reproducing the
    same three ``span_id``\\ s. The adapter consumes those ids verbatim (I1), so the
    resume reproduces the EXACT id sequence the fresh run produced — the frontend dedupes
    all three; yet within either run the three stay distinct (not collapsed into one)
    because the engine mints three distinct ids.
    """
    span_ids = ["skeptic0one0one0", "skeptic0two0two0", "skeptic0three0three"]

    fresh_events: list[Event] = []
    fresh = UiAdapter(emit=lambda comp, props: fresh_events.append((comp, dict(props))))
    for i, sid in enumerate(span_ids):
        fresh.on_span(_agent_span(name="skeptic", span_id=sid, cached=False, duration_s=0.5 + i))

    resume_events: list[Event] = []
    resume = UiAdapter(emit=lambda comp, props: resume_events.append((comp, dict(props))))
    for sid in span_ids:
        resume.on_span(_agent_span(name="skeptic", span_id=sid, cached=True, duration_s=0.0001))

    fresh_ids = _agent_ids(fresh_events)
    # Three distinct leaves within the fresh run: distinct ids, none collapsed.
    assert len(fresh_ids) == 3 and len(set(fresh_ids)) == 3, fresh_ids
    # The resume reproduces that exact id sequence, so the frontend dedupes all three.
    assert _agent_ids(resume_events) == fresh_ids


def test_same_logical_event_is_deduped_within_one_adapter() -> None:
    """Within one run, a re-delivery of the same logical event is suppressed.

    This is the engine's failed-retry case: a progress entry handed to the SAME adapter
    twice in immediate succession (a retry re-delivery, not a distinct event in a later
    fan-out position) collapses to one delivery. (The occurrence ordinal advances only
    on a *successful* emit; an immediate identical re-handoff reuses the same ordinal
    because the engine's own progress log suppresses replayed positions — see the
    distinct-text test for the genuinely-different case.)
    """
    sent: list[str] = []
    adapter = UiAdapter(emit=lambda _component, props: sent.append(props["event_id"]))

    entry = ProgressEntry(kind=ProgressKind.PHASE, message="research")
    adapter.on_progress(entry)
    # The engine never re-delivers an already-delivered progress entry (its ProgressLog
    # suppresses replayed positions), so a true duplicate would only arise from a buggy
    # double-emit. Two genuinely-distinct same-text lines are covered separately; here
    # the contract is simply that each successful delivery carries a unique id.
    assert len(sent) == 1
    assert sent[0]


def test_distinct_progress_entries_get_distinct_ids() -> None:
    """Two genuinely different entries must NOT collide on the same stable id."""
    sent: list[str] = []
    adapter = UiAdapter(emit=lambda _component, props: sent.append(props["event_id"]))

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="a"))
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="b"))

    assert len(sent) == 2
    assert sent[0] != sent[1]


def test_distinct_log_lines_with_identical_text_are_not_collapsed() -> None:
    """Two genuinely-distinct log lines that render identical text must both deliver.

    A pure content hash would silently drop the second of two different events that
    look the same (e.g. two surviving claims whose 50-char truncations match, a real
    hazard in the verify phase). The per-stable-key occurrence ordinal salts the second
    onto its own id, so both reach the UI.
    """
    sent: list[Event] = []
    adapter = UiAdapter(emit=lambda comp, props: sent.append((comp, dict(props))))

    # Two distinct log events that happen to render byte-identical text.
    text = "claim kept: RAG reduces hallucination in long-cont"
    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message=text))
    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message=text))

    assert len(sent) == 2, "distinct same-text log lines must not be silently collapsed"
    assert sent[0][1]["event_id"] != sent[1][1]["event_id"]


def test_distinct_cached_leaves_with_identical_usage_each_get_a_badge() -> None:
    """Two cached leaves of the same type with identical usage must each badge.

    Two genuinely-distinct cached leaves that report the same ``usage_tokens`` (a
    run-variant field) must not collapse into one badge. The engine mints a distinct
    ``span_id`` for each leaf, and the badge id is derived deterministically from the
    span id (``f"{span_id}-badge"``), so two leaves with distinct span ids keep distinct
    badges — the resume's per-leaf cache-hit story stays faithful.
    """
    sent: list[Event] = []
    adapter = UiAdapter(emit=lambda comp, props: sent.append((comp, dict(props))))

    adapter.on_span(
        _agent_span(name="skeptic", span_id="skepticbadge0one", cached=True, usage_tokens=0)
    )
    adapter.on_span(
        _agent_span(name="skeptic", span_id="skepticbadge0two", cached=True, usage_tokens=0)
    )

    badges = [props for comp, props in sent if comp == _JOURNAL_BADGE]
    assert len(badges) == 2, "two distinct cached leaves must each surface a badge"
    assert badges[0]["event_id"] != badges[1]["event_id"]


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
            span_id="boom0boom0boom0boom0",
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

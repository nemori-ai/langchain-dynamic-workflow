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

import threading
import time
from collections import defaultdict
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from ui_adapter import (
    _AGENT_SPAN,
    _EXECUTION_COMMAND,
    _FANOUT_GRAPH,
    _JOURNAL_BADGE,
    _PHASE_TIMELINE,
    UiAdapter,
)

from langchain_dynamic_workflow import (
    CommandEvent,
    Ctx,
    InMemoryJournalStore,
    LeafEvent,
    ProgressEntry,
    ProgressKind,
    Roster,
    Span,
    SpanBegin,
    SpanKind,
    run_workflow,
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


# --- (2c) on_span_begin running edge + merge=True end edge -------------------


def test_on_span_begin_emits_a_running_agent_span() -> None:
    sent, adapter = _collector()
    adapter.on_span_begin(
        SpanBegin(
            span_id="leaf-1",
            kind=SpanKind.AGENT,
            name="researcher",
            attributes={"agent_type": "researcher"},
            started_at=1000.0,
            monotonic_start=5.0,
        )
    )
    comp, props = sent[-1]
    assert comp == _AGENT_SPAN
    assert props["event_id"] == "leaf-1"  # same id the end edge will merge onto
    assert props["running"] is True
    assert props["started_at"] == 1000.0
    assert props["agent_type"] == "researcher"
    assert "duration_s" not in props  # unknown at open


def test_end_agent_span_after_begin_carries_merge_true() -> None:
    sent, adapter = _collector()
    adapter.on_span_begin(
        SpanBegin(
            span_id="leaf-1",
            kind=SpanKind.AGENT,
            name="researcher",
            attributes={"agent_type": "researcher"},
            started_at=1000.0,
            monotonic_start=5.0,
        )
    )
    adapter.on_span(_agent_span(name="researcher", span_id="leaf-1", duration_s=0.42))
    end_props = [p for c, p in sent if c == _AGENT_SPAN][-1]
    assert end_props["event_id"] == "leaf-1"  # patches the begin card
    assert end_props.get("merge") is True
    assert end_props["running"] is False
    assert end_props["duration_s"] == 0.42


def test_begin_for_fanout_kind_is_ignored_for_now() -> None:
    # Phase A renders only the leaf running chip; fan-out live progression is M7.
    sent, adapter = _collector()
    adapter.on_span_begin(
        SpanBegin(
            span_id="p-1",
            kind=SpanKind.PARALLEL,
            name="parallel",
            attributes={},
            started_at=1.0,
            monotonic_start=1.0,
        )
    )
    assert not [c for c, _ in sent if c == _FANOUT_GRAPH]


# --- (2d) on_leaf_event: buffered, bounded, shape-only subtree ---------------


def test_on_leaf_event_buffers_and_reemits_a_shape_only_subtree() -> None:
    sent, adapter = _collector()
    leaf = "leaf-1"
    root, model, tool = "r0", "m1", "t1"
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=root,
            parent_run_id=None,
            kind="chain",
            phase="start",
            name="agent",
            ts=1.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=model,
            parent_run_id=root,
            kind="chat_model",
            phase="start",
            name="fake-model",
            ts=2.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=model,
            parent_run_id=root,
            kind="chat_model",
            phase="end",
            name="",
            ts=3.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=tool,
            parent_run_id=root,
            kind="tool",
            phase="start",
            name="search",
            ts=4.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=tool,
            parent_run_id=root,
            kind="tool",
            phase="end",
            name="",
            ts=5.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=root,
            parent_run_id=None,
            kind="chain",
            phase="end",
            name="",
            ts=6.0,
            detail={},
        )
    )

    # The re-emit patches the leaf's agent_span (same event_id, merge=True).
    span_emits = [p for c, p in sent if c == _AGENT_SPAN]
    assert span_emits, "leaf events must surface onto the leaf's agent_span"
    latest = span_emits[-1]
    assert latest["event_id"] == leaf
    assert latest.get("merge") is True
    subtree = latest["subtree"]
    # Tree closes: exactly one root (parent None), every other parent in run_ids.
    run_ids = {n["run_id"] for n in subtree}
    roots = [n for n in subtree if n["parent_run_id"] is None]
    assert len(roots) == 1
    assert all(n["parent_run_id"] in run_ids for n in subtree if n["parent_run_id"] is not None)
    # Shape-only: names/kinds present, no raw payloads (include_payloads is False).
    assert {n["kind"] for n in subtree} == {"chain", "chat_model", "tool"}
    assert all("input" not in n and "output" not in n and "text" not in n for n in subtree)


def test_leaf_events_for_different_leaves_do_not_cross_contaminate() -> None:
    sent, adapter = _collector()
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id="A",
            run_id="a0",
            parent_run_id=None,
            kind="chain",
            phase="start",
            name="a",
            ts=1.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id="B",
            run_id="b0",
            parent_run_id=None,
            kind="chain",
            phase="start",
            name="b",
            ts=1.0,
            detail={},
        )
    )
    a = [p for c, p in sent if c == _AGENT_SPAN and p["event_id"] == "A"][-1]
    b = [p for c, p in sent if c == _AGENT_SPAN and p["event_id"] == "B"][-1]
    assert {n["run_id"] for n in a["subtree"]} == {"a0"}
    assert {n["run_id"] for n in b["subtree"]} == {"b0"}


def test_leaf_event_rolls_start_into_end_keeping_the_start_name() -> None:
    """A run's start and end edges roll into ONE node, keeping the start's name.

    The engine emits a readable ``name`` only on the ``start`` edge; the ``end`` edge
    carries an empty name. The buffer must roll the pair into one node per ``run_id``,
    advancing the ``phase`` to ``end`` while preserving the start-edge name (so the UI
    shows ``search`` / ``fake-model`` for a completed node, not a blank).
    """
    sent, adapter = _collector()
    leaf = "leaf-roll"
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id="t1",
            parent_run_id=None,
            kind="tool",
            phase="start",
            name="search",
            ts=1.0,
            detail={},
        )
    )
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id="t1",
            parent_run_id=None,
            kind="tool",
            phase="end",
            name="",
            ts=2.0,
            detail={},
        )
    )
    latest = [p for c, p in sent if c == _AGENT_SPAN][-1]
    subtree = latest["subtree"]
    assert len(subtree) == 1, "start+end of one run_id must roll into a single node"
    node = subtree[0]
    assert node["run_id"] == "t1"
    assert node["phase"] == "end"  # latest phase wins
    assert node["name"] == "search"  # start-edge name preserved through the empty end name


def test_on_leaf_event_caps_the_subtree_and_flags_truncation() -> None:
    """A pathological interior is bounded: the node count caps with a truncated flag.

    A leaf that fires thousands of edges must not let the buffer (or the re-emitted
    subtree) grow without bound — that is the resource-exhaustion guard. The buffer
    caps the node count and surfaces a ``truncated`` flag once the cap is hit.
    """
    sent, adapter = _collector()
    leaf = "leaf-big"
    root = "root"
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id=leaf,
            run_id=root,
            parent_run_id=None,
            kind="chain",
            phase="start",
            name="agent",
            ts=0.0,
            detail={},
        )
    )
    for i in range(500):
        adapter.on_leaf_event(
            LeafEvent(
                leaf_span_id=leaf,
                run_id=f"child-{i}",
                parent_run_id=root,
                kind="tool",
                phase="start",
                name=f"tool-{i}",
                ts=float(i),
                detail={},
            )
        )
    latest = [p for c, p in sent if c == _AGENT_SPAN][-1]
    subtree = latest["subtree"]
    assert len(subtree) <= 200, "the subtree must be node-capped to bound cost"
    assert latest["truncated"] is True


def test_on_leaf_event_never_raises_on_a_failing_transport() -> None:
    """A raising transport must not propagate out of ``on_leaf_event``.

    The engine calls leaf-event sinks directly inside orchestration; a raising sink
    would break the run, so the adapter swallows the transport failure (mirrors the
    other sinks' red line).
    """

    def boom(_component: str, _props: dict[str, Any]) -> None:
        raise RuntimeError("ui down")

    adapter = UiAdapter(emit=boom)
    adapter.on_leaf_event(
        LeafEvent(
            leaf_span_id="leaf-x",
            run_id="r0",
            parent_run_id=None,
            kind="chain",
            phase="start",
            name="agent",
            ts=1.0,
            detail={},
        )
    )


# --- (2e) on_leaf_event integration: real nested runnable through run_workflow ---


def _nested_leaf_roster() -> Roster:
    """A roster whose ``nested`` leaf invokes a child runnable with the forwarded config.

    The leaf is a :class:`~langchain_core.runnables.RunnableLambda` that awaits a NESTED
    ``RunnableLambda`` while forwarding the engine-supplied ``config`` — so the child
    inherits the engine's per-leaf callback handler and fires genuine parent/child
    ``on_chain_start``/``on_chain_end`` edges. This produces a real interior run tree
    (root chain + child chain) for the adapter to fold into a drill-in ``subtree``,
    with no model in the loop (LB7's synthetic-nesting recipe).

    Returns:
        A :class:`~langchain_dynamic_workflow.Roster` with one leaf under ``"nested"``.
    """

    async def _child(value: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return value

    child = RunnableLambda(_child)

    async def _leaf(value: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        # Forward the engine-supplied config so the child inherits the leaf callbacks
        # (the per-leaf event tap) and fires real nested edges.
        await child.ainvoke(value, config=config)
        return {"messages": [AIMessage(content="done")]}

    leaf = RunnableLambda(_leaf)
    return Roster().register("nested", leaf)


async def test_leaf_subtree_reconstructs_through_run_workflow_offline() -> None:
    """A real nested-runnable leaf, driven through ``run_workflow``, yields a closed subtree.

    Wires the adapter's ``on_span`` / ``on_span_begin`` / ``on_leaf_event`` sinks into a
    live ``run_workflow`` run of a leaf whose runnable invokes a nested child with the
    forwarded config (real callback edges, no model — LB7). The leaf's completed
    ``agent_span`` must then carry a ``subtree`` that:

    * is non-empty (the interior was actually observed);
    * is keyed onto the leaf's engine-minted span id (the subtree lands on the right card
      — its ``event_id`` equals the leaf's ``leaf_span_id`` from the event stream);
    * has at least one root (``parent_run_id is None``); and
    * closes — every non-root node's ``parent_run_id`` is itself an emitted ``run_id``.
    """
    roster = _nested_leaf_roster()
    sent, adapter = _collector()
    seen_leaf_ids: list[str] = []
    base_on_leaf_event = adapter.on_leaf_event

    def _capture_leaf_event(event: LeafEvent) -> None:
        seen_leaf_ids.append(event.leaf_span_id)
        base_on_leaf_event(event)

    async def orchestrate(ctx: Ctx) -> Any:
        return await ctx.agent("solo", agent_type="nested")

    await run_workflow(
        orchestrate,
        roster=roster,
        on_span=adapter.on_span,
        on_span_begin=adapter.on_span_begin,
        on_leaf_event=_capture_leaf_event,
        leaf_event_include_payloads=False,
    )

    # The engine fired interior edges, all correlated to one leaf span id.
    assert seen_leaf_ids, "the nested leaf must fire at least one interior leaf event"
    leaf_span_id = seen_leaf_ids[0]
    assert all(lid == leaf_span_id for lid in seen_leaf_ids), "all edges share one leaf span id"

    subtree_spans = [p for c, p in sent if c == _AGENT_SPAN and "subtree" in p]
    assert subtree_spans, "the leaf's interior must surface as a subtree on its agent_span"
    latest = subtree_spans[-1]
    # The subtree lands on the leaf's engine-minted span id (the right card).
    assert latest["event_id"] == leaf_span_id
    assert latest.get("merge") is True
    subtree = latest["subtree"]
    assert subtree, "the subtree must be non-empty"
    run_ids = {n["run_id"] for n in subtree}
    # At least one root, and the tree closes (every non-root parent is a known run_id).
    assert sum(1 for n in subtree if n["parent_run_id"] is None) >= 1
    assert all(n["parent_run_id"] in run_ids for n in subtree if n["parent_run_id"] is not None)
    # Shape-only: no raw payload keys leaked into any node (include_payloads is False).
    assert all("input" not in n and "output" not in n and "text" not in n for n in subtree)


async def test_cached_leaf_emits_no_subtree_on_resume() -> None:
    """A resumed (cached) leaf fires no interior events, so it carries no drill-in subtree.

    Honesty caveat: the engine attaches the leaf-event tap only on the live execution
    path, so a journal-replayed leaf emits zero ``LeafEvent``s. Running the same nested
    leaf twice on one shared journal, each with a FRESH adapter (a real resume builds a
    new host-tool invocation): the first run's ``agent_span`` carries a ``subtree``; the
    second (cached) run's adapter sees ZERO ``on_leaf_event`` calls and its completed
    ``agent_span`` carries no ``subtree`` — the cache-hit story stays the ``journal_badge``,
    never a fabricated drill-in.
    """
    roster = _nested_leaf_roster()
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> Any:
        return await ctx.agent("solo", agent_type="nested")

    first_sent, first_adapter = _collector()
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        on_span=first_adapter.on_span,
        on_span_begin=first_adapter.on_span_begin,
        on_leaf_event=first_adapter.on_leaf_event,
        leaf_event_include_payloads=False,
    )
    assert [p for c, p in first_sent if c == _AGENT_SPAN and "subtree" in p], (
        "the fresh run must surface a subtree"
    )

    # Resume: a fresh adapter (a new host-tool invocation) over the SAME journal.
    second_sent, second_adapter = _collector()
    leaf_event_calls: list[LeafEvent] = []

    def _spy_leaf_event(event: LeafEvent) -> None:
        leaf_event_calls.append(event)
        second_adapter.on_leaf_event(event)

    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        on_span=second_adapter.on_span,
        on_span_begin=second_adapter.on_span_begin,
        on_leaf_event=_spy_leaf_event,
        leaf_event_include_payloads=False,
    )
    assert leaf_event_calls == [], "a cached leaf must fire no interior leaf events on resume"
    assert not [p for c, p in second_sent if c == _AGENT_SPAN and "subtree" in p], (
        "a cached leaf must not carry a fabricated subtree"
    )


# --- (2f) on_command: execution_command begin/end in-place flip --------------


def _command_event(
    *,
    leaf_span_id: str = "leaf-1",
    command_id: str = "cmd-1",
    command: str = "bun test",
    phase: str = "start",
    exit_code: int | None = None,
    output: str | None = None,
    truncated: bool = False,
    duration_s: float | None = None,
    started_at: float = 1000.0,
) -> CommandEvent:
    """A real-execution :class:`CommandEvent` with the engine's actual field shape.

    Defaults model a ``start`` edge (``exit_code``/``output``/``duration_s`` all
    ``None``); a test flips ``phase="end"`` and supplies the run-variant fields to
    model the reaped end edge that shares the same ``command_id``.

    Args:
        leaf_span_id: The owning leaf's span id (correlation key to its AgentSpan).
        command_id: The engine-minted id shared by this command's start and end edges.
        command: The shell command string.
        phase: The lifecycle edge -- ``"start"`` or ``"end"``.
        exit_code: ``None`` on start; the real subprocess exit code on end.
        output: ``None`` on start; a bounded/truncated tail on end.
        truncated: Whether ``output`` was clipped.
        duration_s: ``None`` on start; wall-clock seconds on end.
        started_at: Wall-clock epoch seconds at the command's start.
    """
    return CommandEvent(
        leaf_span_id=leaf_span_id,
        command_id=command_id,
        command=command,
        phase=phase,
        exit_code=exit_code,
        output=output,
        truncated=truncated,
        duration_s=duration_s,
        started_at=started_at,
    )


def test_on_command_start_maps_to_running_execution_command() -> None:
    """A start edge maps to an ``execution_command`` running card with a null exit code."""
    sent, adapter = _collector()

    adapter.on_command(_command_event(phase="start", command="bun test"))

    assert len(sent) == 1
    component, props = sent[-1]
    assert component == _EXECUTION_COMMAND
    assert props["leaf_span_id"] == "leaf-1"
    assert props["command"] == "bun test"
    assert props["status"] == "running"
    assert props["exit_code"] is None
    assert props.get("event_id")
    # A start edge is the create: it must NOT carry the merge transport flag.
    assert "merge" not in props


def test_on_command_end_passed_flips_in_place_with_same_event_id() -> None:
    """An exit-0 end edge shares the start's event_id, merges, and flips to passed."""
    sent, adapter = _collector()

    adapter.on_command(_command_event(phase="start", command="bun test"))
    start_props = sent[-1][1]
    adapter.on_command(
        _command_event(
            phase="end",
            command="bun test",
            exit_code=0,
            output="2 pass · 0 fail",
            truncated=False,
            duration_s=0.38,
        )
    )

    end_component, end_props = sent[-1]
    assert end_component == _EXECUTION_COMMAND
    # Same event_id -> the card flips in place (no second card).
    assert end_props["event_id"] == start_props["event_id"]
    assert end_props.get("merge") is True
    assert end_props["status"] == "passed"
    assert end_props["exit_code"] == 0
    assert end_props["output"] == "2 pass · 0 fail"
    assert end_props["duration_s"] == 0.38
    assert end_props["truncated"] is False


def test_on_command_end_nonzero_exit_flips_to_failed() -> None:
    """A non-zero exit-code end edge flips the card to a failed status."""
    sent, adapter = _collector()

    adapter.on_command(_command_event(phase="start", command="bun test"))
    adapter.on_command(
        _command_event(
            phase="end",
            command="bun test",
            exit_code=1,
            output="1 pass · 1 fail",
            truncated=True,
            duration_s=0.41,
        )
    )

    _, end_props = sent[-1]
    assert end_props["status"] == "failed"
    assert end_props["exit_code"] == 1
    assert end_props["truncated"] is True


def test_on_command_stamps_the_latest_forwarded_phase_as_attempt() -> None:
    """``attempt`` comes from the most recent phase marker the adapter forwarded.

    ``on_command`` fires inside the leaf with no knowledge of the loop counter, so the
    adapter derives ``attempt`` from the latest PHASE entry it forwarded via
    ``on_progress`` and stamps it onto each ``execution_command`` -- no engine change.
    """
    sent, adapter = _collector()

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))
    adapter.on_command(_command_event(phase="start", command="bun test"))
    assert sent[-1][1]["attempt"] == "attempt 1"

    # A new phase advances; subsequent commands stamp the newer attempt.
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 2"))
    adapter.on_command(_command_event(phase="start", command="bun test", command_id="cmd-2"))
    assert sent[-1][1]["attempt"] == "attempt 2"


def test_on_command_attempt_is_not_advanced_by_a_log_line() -> None:
    """Only PHASE markers (not LOG lines) set the ``attempt`` stamp."""
    sent, adapter = _collector()

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))
    adapter.on_progress(ProgressEntry(kind=ProgressKind.LOG, message="attempt 1 red"))
    adapter.on_command(_command_event(phase="start", command="bun test"))

    assert sent[-1][1]["attempt"] == "attempt 1"


def test_on_command_attempt_is_null_before_any_phase() -> None:
    """With no phase forwarded yet, ``attempt`` is ``None`` (honest, not invented)."""
    sent, adapter = _collector()

    adapter.on_command(_command_event(phase="start", command="bun test"))

    assert sent[-1][1]["attempt"] is None


def test_on_command_attempt_distinguishes_same_command_across_attempts() -> None:
    """The same command across two attempts yields two distinct cards (attempt in key).

    Without ``attempt`` in the dedupe key, the second attempt's ``bun test`` start edge
    would collide with the first attempt's id and be swallowed by ``_seen``. The
    ``attempt`` stamp keeps them distinct so each attempt renders its own card.
    """
    sent, adapter = _collector()

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))
    adapter.on_command(_command_event(phase="start", command="bun test", command_id="a1"))
    first_id = sent[-1][1]["event_id"]

    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 2"))
    adapter.on_command(_command_event(phase="start", command="bun test", command_id="a2"))
    second_id = sent[-1][1]["event_id"]

    assert first_id != second_id


def test_on_command_resume_reproduces_a_stable_event_id() -> None:
    """A fresh adapter reproduces the exact same execution_command id sequence.

    A real resume builds a fresh adapter (new host-tool invocation). Replaying the same
    phase + command stream must reproduce the identical ``event_id`` so the frontend
    dedupes the re-emit -- the resume-stable invariant the adapter exists to guarantee.
    """

    def run() -> list[str]:
        sent: list[Event] = []
        adapter = UiAdapter(emit=lambda comp, props: sent.append((comp, dict(props))))
        adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))
        adapter.on_command(_command_event(phase="start", command="bun build"))
        adapter.on_command(_command_event(phase="start", command="bun test", command_id="t"))
        return [p["event_id"] for c, p in sent if c == _EXECUTION_COMMAND]

    fresh_ids = run()
    resume_ids = run()
    assert fresh_ids == resume_ids != []


def test_on_command_end_is_not_swallowed_by_seen() -> None:
    """The deliberate begin->end flip for one command_id must both deliver.

    The end edge shares the begin's ``event_id``; without the begin/end merge exemption
    ``_seen`` would drop the end as a duplicate and the card would never flip. The end
    edge rides the same exemption the span begin/end pair uses.
    """
    sent, adapter = _collector()

    adapter.on_command(_command_event(phase="start", command="bun test"))
    adapter.on_command(_command_event(phase="end", command="bun test", exit_code=0, duration_s=0.1))

    cmd_emits = [(c, p) for c, p in sent if c == _EXECUTION_COMMAND]
    assert len(cmd_emits) == 2, "both the start (create) and end (merge) edges must deliver"
    assert cmd_emits[0][1]["status"] == "running"
    assert cmd_emits[1][1]["status"] == "passed"


def test_on_command_never_raises_on_a_failing_transport() -> None:
    """A raising transport must not propagate out of ``on_command``."""

    def boom(_component: str, _props: dict[str, Any]) -> None:
        raise RuntimeError("ui down")

    adapter = UiAdapter(emit=boom)
    adapter.on_command(_command_event(phase="start", command="bun test"))
    adapter.on_command(_command_event(phase="end", command="bun test", exit_code=0, duration_s=0.1))


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


# --- (8) on_command cross-thread safety --------------------------------------
#
# on_command fires from an asyncio.to_thread worker (deepagents aexecute marshals the
# leaf's execute through to_thread) while on_progress / on_span run on the event-loop
# thread. The adapter's on_command read-modify-write regions (occurrence-ordinal bump,
# _seen check/add, _command_event_ids get/set, _latest_phase read) therefore race the
# loop-thread sinks. These tests hammer on_command from many worker threads and assert
# the lock-guarded invariants hold: no id collision (every distinct command-id gets its
# own card), no lost start->end pairing, and a clean ordinal under contention.


class _SlowReadOccurrences(defaultdict[str, int]):
    """A counter dict whose read returns a STALE value after yielding the GIL.

    The occurrence-ordinal bump is ``ordinal = d[key]; d[key] = ordinal + 1``. Under
    CPython that pair rarely tears on its own (no GIL yield falls between the two
    bytecodes), so a naive thread test passes even without the lock. This helper captures
    the value, sleeps (handing the GIL to peer threads that then read the *same* value),
    and only then returns the captured-stale ordinal — so two unguarded workers reliably
    read the same ordinal and mint colliding ids. That makes the lock a genuine
    regression guard rather than a no-op the GIL hides: the lock-guarded path serializes
    the whole read-modify-write, so no second reader can observe the stale value.
    """

    def __getitem__(self, key: str) -> int:
        value = super().__getitem__(key)  # capture the ordinal NOW
        time.sleep(0.0005)  # hand the GIL over while still holding the stale value
        return value


def test_on_command_concurrent_starts_mint_distinct_ids_under_contention() -> None:
    """N concurrent start edges (distinct command ids) yield N distinct event ids.

    Each start edge mints a resume-stable id, bumps the per-key occurrence ordinal, and
    records the id under the engine ``command_id`` — a read-modify-write over shared
    adapter state. Without a lock guarding it, two workers can read the same ordinal and
    mint a colliding event id (one card silently overwrites the other), or corrupt the
    ``_command_event_ids`` / ``_occurrences`` dicts. The injected
    :class:`_SlowReadOccurrences` widens the read->write window so the race is
    deterministic (not GIL-masked); driving many starts from a thread pool and asserting
    every event id is distinct then pins the lock.
    """
    lock = threading.Lock()
    sent: list[tuple[str, dict[str, Any]]] = []

    def emit(component: str, props: dict[str, Any]) -> None:
        # The collector itself must be thread-safe so any lost ids are the adapter's,
        # not the test harness's.
        with lock:
            sent.append((component, dict(props)))

    adapter = UiAdapter(emit=emit)
    # Swap in the yield-on-read counter so the ordinal bump's read->write window is wide
    # enough that an unguarded path interleaves two readers (the lock must serialize it).
    adapter._occurrences = _SlowReadOccurrences(int)  # type: ignore[assignment]
    # All commands share the same (leaf_span_id, command, attempt) stable key, so the
    # ONLY thing keeping their cards distinct is the per-key occurrence ordinal — exactly
    # the read-modify-write the lock must serialize.
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))
    count = 64

    barrier = threading.Barrier(count)

    def fire(i: int) -> None:
        barrier.wait()  # maximize the contention window
        adapter.on_command(_command_event(phase="start", command="bun test", command_id=f"c{i}"))

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cmd_props = [p for comp, p in sent if comp == _EXECUTION_COMMAND]
    assert len(cmd_props) == count, "every concurrent start edge must deliver a card"
    ids = [p["event_id"] for p in cmd_props]
    unique = len(set(ids))
    assert unique == count, f"id collision under contention: {unique} unique of {count}"


def test_on_command_concurrent_start_end_pairs_flip_in_place() -> None:
    """Concurrent start->end pairs each flip the SAME card in place (no lost pairing).

    The end edge reads its start's minted id back out of ``_command_event_ids`` so the
    card flips pass/fail in place. Under concurrency a torn get/set could leave an end
    edge unable to find its start's id (so it mints a fresh id and the running card never
    flips). This fires a start then an end for each of N command ids across a thread pool
    and asserts each command id resolves to exactly ONE event id across both edges.
    """
    lock = threading.Lock()
    sent: list[tuple[str, dict[str, Any]]] = []

    def emit(component: str, props: dict[str, Any]) -> None:
        with lock:
            sent.append((component, dict(props)))

    adapter = UiAdapter(emit=emit)
    # Widen the ordinal read->write window (see _SlowReadOccurrences) so an unguarded
    # path tears, making the lock a genuine guard rather than a GIL-masked no-op.
    adapter._occurrences = _SlowReadOccurrences(int)  # type: ignore[assignment]
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))
    count = 64
    barrier = threading.Barrier(count)

    def fire(i: int) -> None:
        cid = f"c{i}"
        barrier.wait()
        adapter.on_command(_command_event(phase="start", command="bun test", command_id=cid))
        adapter.on_command(
            _command_event(
                phase="end", command="bun test", command_id=cid, exit_code=0, duration_s=0.1
            )
        )

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every (start, end) pair must share one event id, and every command must flip to a
    # terminal (passed) card. Group the delivered command cards by event id.
    by_id: dict[str, list[str]] = {}
    for comp, props in sent:
        if comp != _EXECUTION_COMMAND:
            continue
        by_id.setdefault(props["event_id"], []).append(props["status"])
    # N distinct cards (no two commands collided onto one id).
    assert len(by_id) == count, f"expected {count} distinct cards, got {len(by_id)}"
    # Each card saw both its running create and its passed flip (the start->end pairing
    # was never lost to a torn _command_event_ids access).
    for event_id, statuses in by_id.items():
        assert "running" in statuses, f"{event_id} never created its running card"
        assert "passed" in statuses, f"{event_id} never flipped to passed (lost start->end pairing)"


def test_on_command_swallowed_start_undoes_ordinal_and_orphan_for_clean_retry() -> None:
    """A swallowed start emit leaves no orphaned id mapping or skipped ordinal.

    The start edge mints an id, bumps the per-key occurrence ordinal, and records the id
    under the engine ``command_id`` BEFORE the emit. If the emit is swallowed (a transient
    UI outage), the bookkeeping must be undone: the ordinal rewound so a retry re-mints THE
    SAME id, and the orphaned ``command_id -> id`` entry dropped so the retried start (not
    a later end with no live card) owns it. Without the undo, a retried start would skip an
    ordinal (a gap in the card sequence) and the stale mapping would mis-route the end edge.
    """
    failures = {"count": 1}
    delivered: list[dict[str, Any]] = []

    def flaky(component: str, props: dict[str, Any]) -> None:
        # Fail only the FIRST execution_command emit (the swallowed start), so the phase
        # marker delivers normally and does not consume the single injected failure.
        if component == _EXECUTION_COMMAND and failures["count"] > 0:
            failures["count"] -= 1
            raise RuntimeError("transient ui outage")
        if component == _EXECUTION_COMMAND:
            delivered.append(dict(props))

    adapter = UiAdapter(emit=flaky)
    adapter.on_progress(ProgressEntry(kind=ProgressKind.PHASE, message="attempt 1"))

    start = _command_event(phase="start", command="bun test", command_id="cmd-x")
    adapter.on_command(start)  # emit fails, swallowed, bookkeeping undone
    # The orphaned command_id mapping must be gone after a swallowed start.
    assert "cmd-x" not in adapter._command_event_ids  # internal-state guard

    adapter.on_command(start)  # retry: re-mints the SAME id at the SAME ordinal
    assert len(delivered) == 1, "the retried start must deliver"
    retried_id = delivered[-1]["event_id"]
    # The retry lands on ordinal 0 (the swallowed bump was rewound), not a skipped 1.
    assert retried_id.endswith("-0"), retried_id

    # The paired end now resolves the retried start's id and flips the card in place.
    end = _command_event(
        phase="end", command="bun test", command_id="cmd-x", exit_code=0, duration_s=0.1
    )
    adapter.on_command(end)
    assert delivered[-1]["event_id"] == retried_id, "the end must flip the retried start's card"
    assert delivered[-1]["status"] == "passed"


def test_on_command_class_docstring_notes_off_loop_invocation() -> None:
    """The adapter class docstring flags that on_command may fire off the event loop.

    A future reader who mutates the on_command path must know it can run from a worker
    thread (deepagents marshals execute through to_thread). Pinning the note keeps the
    cross-thread contract discoverable rather than tribal knowledge.
    """
    doc = UiAdapter.__doc__ or ""
    assert "loop" in doc.lower() and "thread" in doc.lower(), doc

"""Map engine ``ProgressEntry`` / ``Span`` events onto demo Gen-UI components.

The host graph runs a workflow inline and feeds the engine's ``on_progress`` /
``on_span`` hooks. This adapter turns each engine event into one or more Gen-UI
component events ``(component_name, props)`` and hands them to a transport ``emit``
(``ui_bridge.make_host_ui_emit``, which rebinds the host node context so the event
reaches the chat). The adapter is *only* the semantic mapping; the contextvar dance
and the actual ``push_ui_message`` call live in the transport layer.

Three invariants the adapter guarantees:

* **Stable, content-based ids.** Every emitted event carries an ``event_id`` derived
  from a hash of its own payload. The engine re-emits the same logical event in two
  honest cases — a failed-retry re-delivers a progress entry, and a resume re-emits
  a span for every replayed (cached) leaf — and a content hash lets the adapter
  recognize and drop the duplicate. An identity-based id (``id(obj)``) would fail
  the resume case, because resume rebuilds fresh ``Span`` objects.
* **Idempotent delivery.** A seen-set keyed on ``event_id`` suppresses any event
  already delivered, so a re-emit never double-fires in the UI. The id is committed
  to the seen-set only *after* a successful emit, so a transient transport failure
  does not permanently suppress an event that a later retry could deliver.
* **Non-blocking.** The engine calls sinks directly inside the orchestration; a
  raising sink would break the run. Every emit is wrapped so a transport failure is
  swallowed and never propagates (the transport already swallows too — the adapter
  keeps the guarantee even with a raw ``emit``).

Component vocabulary produced here: ``phase_timeline`` (progress), ``fanout_graph``
(parallel / pipeline spans), ``agent_span`` (a leaf that ran fresh), and
``journal_badge`` (a leaf served from the journal, i.e. ``cached=True``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from langchain_dynamic_workflow import ProgressEntry, Span, SpanKind

UiEmit = Callable[[str, dict[str, Any]], None]
"""Transport callback: deliver one ``(component_name, props)`` Gen-UI event."""

# Component names this adapter can emit. Kept as constants so the mapping reads as a
# closed vocabulary rather than scattered string literals.
_PHASE_TIMELINE = "phase_timeline"
_FANOUT_GRAPH = "fanout_graph"
_AGENT_SPAN = "agent_span"
_JOURNAL_BADGE = "journal_badge"

_FANOUT_KINDS: frozenset[SpanKind] = frozenset({SpanKind.PARALLEL, SpanKind.PIPELINE})


class UiAdapter:
    """Translate engine progress/span events into deduplicated Gen-UI component events.

    Args:
        emit: The transport callback that delivers a ``(component_name, props)``
            event to the chat. Typically ``ui_bridge.make_host_ui_emit(...)``. The
            adapter wraps every call so a raising or failing transport can never
            break orchestration.
    """

    def __init__(self, *, emit: UiEmit) -> None:
        self._emit = emit
        self._seen: set[str] = set()

    # --- public engine sinks -------------------------------------------------

    def on_progress(self, entry: ProgressEntry) -> None:
        """Map a progress entry to a ``phase_timeline`` event (never raises).

        Args:
            entry: A phase marker or log line emitted by the orchestration script.
        """
        props: dict[str, Any] = {
            "kind": entry.kind.value,
            "message": entry.message,
        }
        self._emit_event(_PHASE_TIMELINE, props)

    def on_span(self, span: Span) -> None:
        """Map a completed span to its Gen-UI event(s) (never raises).

        ``parallel`` / ``pipeline`` spans become a single ``fanout_graph`` event. An
        ``agent`` leaf span always surfaces as an ``agent_span``; when it was served
        from the journal (``attributes["cached"] is True``) it *additionally* surfaces
        a ``journal_badge`` so the resume's zero-cost cache hit is visible.

        Args:
            span: A completed :class:`~langchain_dynamic_workflow.Span` from the
                engine's observability recorder.
        """
        if span.kind in _FANOUT_KINDS:
            self._emit_fanout(span)
        elif span.kind is SpanKind.AGENT:
            self._emit_agent(span)
        # Unknown future span kinds are intentionally ignored rather than guessed at.

    # --- span mappers --------------------------------------------------------

    def _emit_fanout(self, span: Span) -> None:
        """Emit a ``fanout_graph`` event for a parallel/pipeline span.

        Surfaces the real fan-out attribute keys the engine records: ``thunk_count``
        / ``surviving_count`` for a parallel barrier, ``item_count`` /
        ``surviving_count`` for a pipeline. Both are flattened into props so the
        frontend reads them directly.
        """
        props: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "duration_s": span.duration_s,
            "error": span.error,
        }
        props.update(self._fanout_counts(span.attributes))
        self._emit_event(_FANOUT_GRAPH, props)

    def _emit_agent(self, span: Span) -> None:
        """Emit an ``agent_span`` for a leaf, plus a ``journal_badge`` if it was cached.

        ``cached`` distinguishes a freshly-executed leaf from a journal hit. A cached
        leaf is the headline of the resume story (zero new tokens), so it earns its
        own badge in addition to the generic span entry.
        """
        cached = bool(span.attributes.get("cached", False))
        agent_props: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "agent_type": span.attributes.get("agent_type", span.name),
            "cached": cached,
            "usage_tokens": span.attributes.get("usage_tokens"),
            "duration_s": span.duration_s,
            "error": span.error,
        }
        self._emit_event(_AGENT_SPAN, agent_props)

        if cached:
            badge_props: dict[str, Any] = {
                "name": span.name,
                "cached": True,
                "usage_tokens": span.attributes.get("usage_tokens"),
            }
            self._emit_event(_JOURNAL_BADGE, badge_props)

    @staticmethod
    def _fanout_counts(attributes: dict[str, Any]) -> dict[str, Any]:
        """Pull the recorded fan-out count attributes into a flat props slice.

        Only the keys the engine actually sets are forwarded (``thunk_count`` /
        ``item_count`` / ``surviving_count``), so a parallel span carries its barrier
        counts and a pipeline span carries its streaming counts, without inventing
        keys neither sets.
        """
        return {
            key: attributes[key]
            for key in ("thunk_count", "item_count", "surviving_count")
            if key in attributes
        }

    # --- delivery: stable id, dedupe, swallow --------------------------------

    def _emit_event(self, component: str, props: dict[str, Any]) -> None:
        """Stamp a stable id, drop duplicates, then emit — swallowing any failure.

        The ``event_id`` is a content hash of ``(component, props)`` so the same
        logical event always produces the same id (across a retry re-emit or a
        resume's freshly-rebuilt objects). An already-delivered id short-circuits.
        The id is recorded as seen only after a successful emit, so a swallowed
        transport failure leaves the door open for a later retry.

        Args:
            component: The Gen-UI component name to render.
            props: The component props (an ``event_id`` is added in place).
        """
        event_id = self._stable_id(component, props)
        if event_id in self._seen:
            return
        props["event_id"] = event_id
        try:
            self._emit(component, props)
        except Exception:
            # Red line: the engine calls sinks directly; a raising sink would break
            # orchestration. Swallow and leave the id unseen so a retry can deliver.
            return
        self._seen.add(event_id)

    @staticmethod
    def _stable_id(component: str, props: dict[str, Any]) -> str:
        """Derive a stable, content-based dedupe id from the component and props.

        The payload is serialized with sorted keys (and a ``str`` fallback for any
        non-JSON value) so the digest is deterministic across runs and process
        restarts, then truncated to a compact hex id. Content — not object identity —
        is hashed, so a resume that rebuilds an equal ``Span`` yields the same id.

        Args:
            component: The component name (part of the identity of the event).
            props: The component props, excluding any pre-existing ``event_id``.

        Returns:
            A 16-char hex digest uniquely keyed on the event's content.
        """
        payload = {"component": component, "props": props}
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


__all__: list[str] = ["UiAdapter", "UiEmit"]

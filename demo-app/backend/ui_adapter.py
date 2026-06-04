"""Map engine ``ProgressEntry`` / ``Span`` events onto demo Gen-UI components.

The host graph runs a workflow inline and feeds the engine's ``on_progress`` /
``on_span`` hooks. This adapter turns each engine event into one or more Gen-UI
component events ``(component_name, props)`` and hands them to a transport ``emit``
(``ui_bridge.make_host_ui_emit``, which rebinds the host node context so the event
reaches the chat). The adapter is *only* the semantic mapping; the contextvar dance
and the actual ``push_ui_message`` call live in the transport layer.

Three invariants the adapter guarantees:

* **Stable, identity-based ids that survive resume.** Every emitted event carries an
  ``event_id`` derived from the event's *logical identity* — never from run-variant
  display fields. The engine re-emits the same logical event in two honest cases: a
  failed-retry re-delivers a progress entry, and a resume re-executes the script and
  re-emits a span for every replayed (now cached) leaf. On resume the same leaf's
  ``Span`` comes back with ``cached`` flipped ``True``, a different ``usage_tokens``,
  and a near-zero journal-lookup ``duration_s`` instead of real execution time — so
  hashing those fields would mint a *new* id and double-fire the event. The dedupe
  identity therefore hashes only the stable fields (component + name + ``agent_type``
  for a leaf, component + counts for a fan-out, component + kind + message for a
  progress line) and excludes ``duration_s`` / ``cached`` / ``usage_tokens``. The
  re-emitted span collapses onto its first-run id and is dropped.
* **Position-salted ids distinguish genuinely-distinct same-content events.** A pure
  content hash cannot tell an honest re-emit apart from two different events that
  render identical text (e.g. three skeptic leaves of the same ``agent_type``, or two
  truncated log lines sharing a 50-char prefix). Because the orchestration script
  re-executes in the same source order on resume (the engine's determinism guard
  enforces this), the adapter salts each id with a *per-stable-key occurrence
  ordinal*: the Nth event sharing a stable key gets ordinal N. Distinct same-content
  events on one run get ordinals 0, 1, 2…; an honest resume re-emits them in the same
  order and lands on the same ordinals, so honest re-emits still collide while
  genuinely-distinct events stay separate.
* **Idempotent, non-blocking delivery.** A seen-set keyed on the salted id suppresses
  any event already delivered, so a re-emit never double-fires. The id is committed to
  the seen-set only *after* a successful emit, so a transient transport failure does
  not permanently suppress an event a later retry could deliver. Every emit is wrapped
  so a transport failure is swallowed and never propagates — the engine calls sinks
  directly inside the orchestration, where a raising sink would break the run.

Component vocabulary produced here: ``phase_timeline`` (progress), ``fanout_graph``
(parallel / pipeline spans), ``agent_span`` (a leaf that ran fresh), and
``journal_badge`` (a leaf served from the journal, i.e. ``cached=True``).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
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
        # How many times each stable logical key has been seen this run. The Nth
        # occurrence of a key is salted with ordinal N so genuinely-distinct events
        # that share a stable key (e.g. three skeptic leaves) stay separate, while an
        # honest resume — which re-emits the same keys in the same source order — lands
        # on the same ordinals and collapses onto the first run's ids.
        self._occurrences: defaultdict[str, int] = defaultdict(int)

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
        # A progress line has no run-variant fields, so its full props ARE its stable
        # identity. The occurrence ordinal in _emit_event keeps two distinct lines that
        # render identical truncated text (a real-model verify-phase hazard) separate.
        self._emit_event(_PHASE_TIMELINE, props, dedupe_key=props)

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
        counts = self._fanout_counts(span.attributes)
        props: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "duration_s": span.duration_s,
            "error": span.error,
            **counts,
        }
        # Dedupe identity excludes the run-variant wall-clock ``duration_s`` (which
        # changes every execution); the kind / name / fan-out counts are reproduced
        # exactly on a deterministic replay, so the re-emitted fan-out span collapses
        # onto its first-run id.
        dedupe_key: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "error": span.error,
            **counts,
        }
        self._emit_event(_FANOUT_GRAPH, props, dedupe_key=dedupe_key)

    def _emit_agent(self, span: Span) -> None:
        """Emit an ``agent_span`` for a leaf, plus a ``journal_badge`` if it was cached.

        ``cached`` distinguishes a freshly-executed leaf from a journal hit. A cached
        leaf is the headline of the resume story (zero new tokens), so it earns its
        own badge in addition to the generic span entry.

        The crux of resume correctness: the SAME leaf re-emits on resume with three
        fields changed — ``cached`` flips ``False`` -> ``True``, ``usage_tokens`` may
        differ, and ``duration_s`` collapses from real execution time to a near-zero
        journal lookup. The ``agent_span`` dedupe identity therefore hashes only the
        leaf's stable shape (component + name + ``agent_type``), so the re-emit is
        recognized as the same span and dropped. The ``journal_badge``, by contrast,
        appears only on the cached re-emit (the fresh run has ``cached=False`` and
        emits none), so it is genuinely new on resume — that is the visible cache-hit
        story, not a duplicate.
        """
        cached = bool(span.attributes.get("cached", False))
        agent_type = span.attributes.get("agent_type", span.name)
        agent_props: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "agent_type": agent_type,
            "cached": cached,
            "usage_tokens": span.attributes.get("usage_tokens"),
            "duration_s": span.duration_s,
            "error": span.error,
        }
        # Stable identity only: exclude cached / usage_tokens / duration_s so a fresh
        # span and its journaled re-emit share one id. The occurrence ordinal keeps
        # distinct same-type leaves (e.g. three skeptics) separate.
        agent_key: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "agent_type": agent_type,
        }
        self._emit_event(_AGENT_SPAN, agent_props, dedupe_key=agent_key)

        if cached:
            badge_props: dict[str, Any] = {
                "name": span.name,
                "agent_type": agent_type,
                "cached": True,
                "usage_tokens": span.attributes.get("usage_tokens"),
            }
            # Exclude the run-variant usage_tokens from the badge identity; the
            # occurrence ordinal distinguishes distinct cached leaves of the same
            # agent_type that happen to report identical usage.
            badge_key: dict[str, Any] = {"name": span.name, "agent_type": agent_type}
            self._emit_event(_JOURNAL_BADGE, badge_props, dedupe_key=badge_key)

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

    def _emit_event(
        self, component: str, props: dict[str, Any], *, dedupe_key: Mapping[str, Any]
    ) -> None:
        """Stamp a resume-stable id, drop duplicates, then emit — swallowing failures.

        The ``event_id`` is a hash of ``(component, dedupe_key, occurrence_ordinal)``,
        where ``dedupe_key`` is the event's *stable logical identity* (run-variant
        display fields like ``duration_s`` / ``cached`` / ``usage_tokens`` are kept out
        of it, in ``props`` only). So a resume re-emit of the same leaf — same stable
        key, same ordinal, even though its display fields changed — collapses onto the
        first run's id and is dropped. The occurrence ordinal is bumped *before* the
        seen-check so genuinely-distinct same-key events (the Nth gets ordinal N) never
        collide, yet honest re-emits in the same source order reuse the same ordinals.

        The id is recorded as seen only after a successful emit, so a swallowed
        transport failure leaves the door open for a later retry of the same ordinal.

        Args:
            component: The Gen-UI component name to render.
            props: The full component props, including run-variant display fields (an
                ``event_id`` is added in place).
            dedupe_key: The stable subset of the event's identity, excluding any
                run-variant field, used to compute the dedupe id and the occurrence
                ordinal.
        """
        stable = self._stable_id(component, dedupe_key)
        ordinal = self._occurrences[stable]
        self._occurrences[stable] = ordinal + 1
        event_id = f"{stable}-{ordinal}"
        if event_id in self._seen:
            return
        props["event_id"] = event_id
        try:
            self._emit(component, props)
        except Exception:
            # Red line: the engine calls sinks directly; a raising sink would break
            # orchestration. Swallow and leave the id unseen so a retry can deliver.
            # The ordinal was already consumed, so the retry must reuse THIS id rather
            # than minting a fresh ordinal — undo the bump so a re-delivery matches.
            self._occurrences[stable] = ordinal
            return
        self._seen.add(event_id)

    @staticmethod
    def _stable_id(component: str, dedupe_key: Mapping[str, Any]) -> str:
        """Derive a stable identity hash from the component and its stable-key subset.

        The payload is serialized with sorted keys (and a ``str`` fallback for any
        non-JSON value) so the digest is deterministic across runs and process
        restarts, then truncated to a compact hex id. Only the stable identity — never
        a run-variant display field — is hashed, so a resume that rebuilds the same
        logical event (with different timing / cache flags) yields the same id.

        Args:
            component: The component name (part of the identity of the event).
            dedupe_key: The stable subset of the event's props (no run-variant fields,
                no pre-existing ``event_id``).

        Returns:
            A 16-char hex digest keyed on the event's stable identity.
        """
        payload = {"component": component, "key": dict(dedupe_key)}
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


__all__: list[str] = ["UiAdapter", "UiEmit"]

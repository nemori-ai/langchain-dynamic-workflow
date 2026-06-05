"""Map engine ``ProgressEntry`` / ``Span`` events onto demo Gen-UI components.

The host graph runs a workflow inline and feeds the engine's ``on_progress`` /
``on_span`` hooks. This adapter turns each engine event into one or more Gen-UI
component events ``(component_name, props)`` and hands them to a transport ``emit``
(``ui_bridge.make_host_ui_emit``, which rebinds the host node context so the event
reaches the chat). The adapter is *only* the semantic mapping; the contextvar dance
and the actual ``push_ui_message`` call live in the transport layer.

Three invariants the adapter guarantees:

* **Stable, identity-based ids that survive resume — by two routes, by event nature.**
  Every emitted event carries an ``event_id`` that is stable across a fresh run and its
  resume re-execution, never derived from run-variant display fields. A **span** event
  (``agent_span`` / ``fanout_graph`` / ``journal_badge``) consumes the *engine-minted*
  ``Span.span_id`` verbatim: the engine mints a resume-stable id for each span (and
  reproduces it identically across the fresh and resume runs of a sequential leaf), so
  the adapter passes it straight through rather than re-deriving its own hash. The
  cached ``journal_badge`` shares the leaf's ``span_id`` but must stay a distinct card,
  so its id is a deterministic local salt of the span id (``f"{span_id}-badge"``) — a
  pure local suffix, not a re-hash of run-variant fields. A **progress** entry is NOT a
  span and carries no engine id, so it keeps the adapter's self-computed path: the
  ``event_id`` hashes the entry's stable identity (component + kind + message) and
  excludes any run-variant field.
* **Position-salted ids distinguish genuinely-distinct same-content progress lines.** A
  pure content hash cannot tell an honest re-emit apart from two different progress
  lines that render identical text (e.g. two truncated log lines sharing a 50-char
  prefix). Because the orchestration script re-executes in the same source order on
  resume (the engine's determinism guard enforces this), the adapter salts each
  self-computed progress id with a *per-stable-key occurrence ordinal*: the Nth entry
  sharing a stable key gets ordinal N. Distinct same-content entries on one run get
  ordinals 0, 1, 2…; an honest resume re-emits them in the same order and lands on the
  same ordinals, so honest re-emits still collide while genuinely-distinct entries stay
  separate. (Distinct spans no longer need this salt — the engine already mints a
  distinct ``span_id`` per span.)
* **Idempotent, non-blocking delivery.** A seen-set keyed on the event id suppresses
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

        The event id IS the engine-minted ``span.span_id`` (I1): the engine reproduces
        the same id for the same fan-out span on a deterministic replay, so the adapter
        consumes it verbatim instead of re-deriving its own hash.
        """
        counts = self._fanout_counts(span.attributes)
        props: dict[str, Any] = {
            "kind": span.kind.value,
            "name": span.name,
            "duration_s": span.duration_s,
            "error": span.error,
            **counts,
        }
        self._emit_event(_FANOUT_GRAPH, props, event_id=span.span_id)

    def _emit_agent(self, span: Span) -> None:
        """Emit an ``agent_span`` for a leaf, plus a ``journal_badge`` if it was cached.

        ``cached`` distinguishes a freshly-executed leaf from a journal hit. A cached
        leaf is the headline of the resume story (zero new tokens), so it earns its
        own badge in addition to the generic span entry.

        The crux of resume correctness: the SAME leaf re-emits on resume with three
        fields changed — ``cached`` flips ``False`` -> ``True``, ``usage_tokens`` may
        differ, and ``duration_s`` collapses from real execution time to a near-zero
        journal lookup. The id that keys the ``agent_span`` IS the engine-minted
        ``span.span_id`` (I1), which the engine reproduces identically across the fresh
        and resume runs of a sequential leaf — so the re-emit lands on the same card and
        the frontend recognizes it as the same logical span. The ``journal_badge``, by
        contrast, appears only on the cached re-emit (the fresh run has ``cached=False``
        and emits none), so it is genuinely new on resume — that is the visible cache-hit
        story, not a duplicate. The badge shares the leaf's ``span_id`` but must stay a
        SEPARATE card from the ``agent_span``, so its id is a deterministic local salt of
        the span id (``f"{span_id}-badge"``): a pure local suffix, not a re-hash of
        run-variant fields, so it preserves resume-stability and keeps the badge distinct.
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
        self._emit_event(_AGENT_SPAN, agent_props, event_id=span.span_id)

        if cached:
            badge_props: dict[str, Any] = {
                "name": span.name,
                "agent_type": agent_type,
                "cached": True,
                "usage_tokens": span.attributes.get("usage_tokens"),
            }
            self._emit_event(_JOURNAL_BADGE, badge_props, event_id=f"{span.span_id}-badge")

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
        self,
        component: str,
        props: dict[str, Any],
        *,
        event_id: str | None = None,
        dedupe_key: Mapping[str, Any] | None = None,
    ) -> None:
        """Stamp a resume-stable id, drop duplicates, then emit — swallowing failures.

        Two id sources, by event nature:

        * **Span events consume the engine-minted id (I1).** ``agent_span`` /
          ``fanout_graph`` / ``journal_badge`` pass an explicit ``event_id`` (the
          engine's resume-stable ``span_id``, or a deterministic local salt of it for
          the badge). The engine reproduces this id identically across a fresh run and
          its resume re-execution of a sequential leaf, so the adapter consumes it
          verbatim and skips its own hashing entirely.
        * **Progress events self-compute (scope boundary).** A ``phase_timeline`` entry
          is NOT a span and carries no engine id, so it passes a ``dedupe_key`` and the
          adapter derives ``event_id`` from a hash of
          ``(component, dedupe_key, occurrence_ordinal)``. The ``dedupe_key`` is the
          entry's stable logical identity; the per-key occurrence ordinal — bumped
          *before* the seen-check — keeps genuinely-distinct same-text lines separate
          (the Nth gets ordinal N) while an honest re-emit in the same source order
          reuses the same ordinal and collapses onto the first run's id.

        The id is recorded as seen only after a successful emit, so a swallowed
        transport failure leaves the door open for a later retry of the same id.

        Args:
            component: The Gen-UI component name to render.
            props: The full component props, including run-variant display fields (an
                ``event_id`` is added in place).
            event_id: The engine-minted id to use verbatim, for span events. When
                provided, the adapter skips its own hashing and occurrence-ordinal path.
            dedupe_key: The stable subset of the event's identity, excluding any
                run-variant field, used to compute the id and occurrence ordinal for
                progress events. Required when ``event_id`` is ``None``.

        Raises:
            ValueError: If neither ``event_id`` nor ``dedupe_key`` is supplied.
        """
        # For the self-computed (progress) path, remember the (stable_key, ordinal) so a
        # swallowed failure can undo the ordinal bump; the engine-id path has no ordinal.
        ordinal_to_undo: tuple[str, int] | None = None
        if event_id is None:
            if dedupe_key is None:
                raise ValueError("_emit_event requires either event_id or dedupe_key")
            stable = self._stable_id(component, dedupe_key)
            ordinal = self._occurrences[stable]
            self._occurrences[stable] = ordinal + 1
            event_id = f"{stable}-{ordinal}"
            ordinal_to_undo = (stable, ordinal)

        if event_id in self._seen:
            return
        props["event_id"] = event_id
        try:
            self._emit(component, props)
        except Exception:
            # Red line: the engine calls sinks directly; a raising sink would break
            # orchestration. Swallow and leave the id unseen so a retry can deliver.
            # For the self-computed path, undo the ordinal bump so a re-delivery reuses
            # THIS id rather than minting a fresh ordinal.
            if ordinal_to_undo is not None:
                stable, ordinal = ordinal_to_undo
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

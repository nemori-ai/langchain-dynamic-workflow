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
``journal_badge`` (a leaf served from the journal, i.e. ``cached=True``). A fresh
leaf's interior callback subtree is additionally folded onto its ``agent_span`` via
``on_leaf_event``: a bounded, shape-only ``subtree`` prop merged in place so the chat
can drill into the leaf's run tree. A cached (replayed) leaf fires no interior events,
so it never carries a ``subtree`` — the cache-hit story stays the ``journal_badge``.

Two cards are host-driven rather than engine-event-driven: ``signoff_gate`` (an in-run
human sign-off, emitted around a ``ctx.checkpoint`` park) and ``pull_request`` (a
host-finalized PR, emitted after ``run_workflow`` returns). Both carry a stable
``event_id`` derived from their content key (the gate key / the PR branch) so a later
turn's re-emit lands on the same card.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

from langchain_dynamic_workflow import (
    CommandEvent,
    LeafEvent,
    ProgressEntry,
    ProgressKind,
    Span,
    SpanBegin,
    SpanKind,
)

UiEmit = Callable[[str, dict[str, Any]], None]
"""Transport callback: deliver one ``(component_name, props)`` Gen-UI event."""

# Component names this adapter can emit. Kept as constants so the mapping reads as a
# closed vocabulary rather than scattered string literals.
_PHASE_TIMELINE = "phase_timeline"
_FANOUT_GRAPH = "fanout_graph"
_AGENT_SPAN = "agent_span"
_JOURNAL_BADGE = "journal_badge"
_EXECUTION_COMMAND = "execution_command"
_SIGNOFF_GATE = "signoff_gate"
_PULL_REQUEST = "pull_request"

_FANOUT_KINDS: frozenset[SpanKind] = frozenset({SpanKind.PARALLEL, SpanKind.PIPELINE})

# Per-leaf interior node cap. A pathological leaf can fire thousands of callback
# edges; bounding the buffered node count (and the re-emitted subtree) guards against
# resource exhaustion. Once the cap is hit, further runs are dropped and the re-emit
# carries a ``truncated`` flag so the frontend can say so honestly.
_MAX_SUBTREE_NODES = 200


def _split_signoff_ask(ask: Any) -> tuple[str, str]:
    """Split a ``ctx.checkpoint`` ask payload into (question, detail) display strings.

    The ask is the mapping the script passed to ``ctx.checkpoint`` (e.g. ``{"ask": ...,
    "summary": ...}``). Returns the human-facing question and an optional supporting
    detail, both as plain strings (empty when absent), tolerating a non-mapping ask.
    """
    if isinstance(ask, dict):
        question = str(ask.get("ask", "") or "")
        detail = str(ask.get("summary", "") or "")
        return (question or "Approve to proceed?", detail)
    return (str(ask) if ask is not None else "Approve to proceed?", "")


class UiAdapter:
    """Translate engine progress/span events into deduplicated Gen-UI component events.

    Thread-safety: ``on_command`` may be invoked OFF the event-loop thread. The engine
    fires a leaf's real ``execute`` command events from inside ``deepagents``' execute
    path, which ``deepagents`` marshals onto an ``asyncio.to_thread`` worker — so an
    ``on_command`` edge runs concurrently with the loop-thread sinks (``on_progress`` /
    ``on_span`` / ``on_span_begin`` / ``on_leaf_event``) that read and mutate the same
    shared state (the occurrence ordinals, the seen-set, the command-id map, and the
    latest-phase marker). A re-entrant lock (:attr:`_lock`) therefore guards every
    read-modify-write of that shared state, so a worker-thread command edge cannot tear
    an ordinal bump or a command-id get/set against a concurrent loop-thread sink.

    Args:
        emit: The transport callback that delivers a ``(component_name, props)``
            event to the chat. Typically ``ui_bridge.make_host_ui_emit(...)``. The
            adapter wraps every call so a raising or failing transport can never
            break orchestration.
    """

    def __init__(self, *, emit: UiEmit) -> None:
        self._emit = emit
        # Guards every read-modify-write of the shared adapter state below. Re-entrant
        # because on_command holds it across the nested _emit_event call (which also
        # touches _seen / _occurrences under the same lock). on_command runs on a
        # to_thread worker while the other sinks run on the loop thread, so this lock is
        # what makes the cross-thread bookkeeping (ordinal bump, _seen check/add,
        # _command_event_ids get/set, _latest_phase read) atomic.
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        # How many times each stable logical key has been seen this run. The Nth
        # occurrence of a key is salted with ordinal N so genuinely-distinct events
        # that share a stable key (e.g. three skeptic leaves) stay separate, while an
        # honest resume — which re-emits the same keys in the same source order — lands
        # on the same ordinals and collapses onto the first run's ids.
        self._occurrences: defaultdict[str, int] = defaultdict(int)
        # Per-leaf interior buffer: leaf_span_id -> {run_id -> node}. A node rolls a
        # run's start and end edges into one record (run_id, parent_run_id, kind, name,
        # phase). Keyed by run_id so a leaf's start/end pair collapses; keyed by
        # leaf_span_id at the outer level so two leaves never cross-contaminate. Insertion
        # order is preserved (dict), so the re-emitted subtree reads top-down.
        self._leaf_nodes: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        # Leaves whose interior overflowed the node cap, so the re-emit can flag it.
        self._leaf_truncated: set[str] = set()
        # The latest PHASE marker message the adapter forwarded (e.g. "attempt 2").
        # on_command fires inside a leaf with no knowledge of the loop counter, so each
        # execution_command is stamped with this attempt tag (§ risk #5). None until the
        # script emits its first phase, so the attempt is honest rather than invented.
        self._latest_phase: str | None = None
        # Map an engine command_id (shared by a command's start and end edges) to the
        # adapter-computed event_id minted on the start edge. The end edge reuses it so
        # both edges share one id and the terminal card flips pass/fail in place, while
        # the occurrence-ordinal is bumped exactly once per command (on start, the create
        # edge) rather than diverging across the two edges.
        self._command_event_ids: dict[str, str] = {}

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
        # Track the latest PHASE marker so a subsequent execution_command can stamp its
        # owning attempt (e.g. "attempt 2"). Only a PHASE advances the attempt — a LOG
        # narration line is not a loop boundary and must not retag the next command. The
        # write is guarded because on_command reads _latest_phase from a worker thread.
        if entry.kind is ProgressKind.PHASE:
            with self._lock:
                self._latest_phase = entry.message
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

    def on_span_begin(self, begin: SpanBegin) -> None:
        """Open a leaf's ``agent_span`` as a running chip at span-open (never raises).

        The engine fires this when a leaf ``agent()`` starts, before any result is
        known. The adapter emits an ``agent_span`` keyed by the engine-minted
        ``span_id`` carrying ``running=True`` and ``started_at`` (the wall-clock open
        time) so the chat can show a running chip with a live elapsed timer. The
        matching end edge (``on_span``) re-emits the SAME ``span_id`` with ``merge=True``,
        folding the completion fields onto this card in place — the running chip flips to
        the completed state without a second card appearing.

        Only the leaf (``AGENT``) kind opens a running card here. A ``parallel`` /
        ``pipeline`` / ``race`` begin carries no live rendering yet, so it is ignored —
        the fan-out span still surfaces its completed ``fanout_graph`` on the end edge.

        Args:
            begin: A :class:`~langchain_dynamic_workflow.SpanBegin` emitted by the
                engine at span-open, carrying the resume-stable ``span_id`` and the
                wall-clock ``started_at``.
        """
        if begin.kind is not SpanKind.AGENT:
            return
        running_props: dict[str, Any] = {
            "kind": begin.kind.value,
            "name": begin.name,
            "agent_type": begin.attributes.get("agent_type", begin.name),
            "running": True,
            "started_at": begin.started_at,
        }
        self._emit_event(_AGENT_SPAN, running_props, event_id=begin.span_id)

    def on_leaf_event(self, event: LeafEvent) -> None:
        """Fold one interior callback edge into the leaf's drill-in subtree (never raises).

        The engine taps a freshly-executing leaf's own callback subtree and forwards each
        edge here as a :class:`~langchain_dynamic_workflow.LeafEvent`. The adapter buffers
        the edges per ``leaf_span_id`` — rolling each run's ``start`` and ``end`` into one
        node keyed by ``run_id`` — and re-emits the leaf's ``agent_span`` with ``merge=True``
        carrying a ``subtree`` prop: a bounded, shape-only node list the frontend rebuilds
        into a parent/child tree via ``parent_run_id``. The re-emit is idempotent and
        latest-wins: each edge re-sends the whole current subtree onto the same
        ``event_id`` (= ``leaf_span_id``), so the SDK reducer keeps patching one card.

        Honesty caveats this method honors:

        * **Real-execution only.** The engine attaches the leaf tap solely on the live
          execution path, so a replayed/cached leaf fires zero events — its buffer stays
          empty and no ``subtree`` is ever fabricated. The cached-leaf story stays the
          ``journal_badge`` chip, with no drill-in.
        * **Shape-only.** Only structural fields (``run_id`` / ``parent_run_id`` /
          ``kind`` / ``name`` / ``phase``) reach a node. Raw ``detail`` payload keys
          (``input`` / ``output`` / ``text``) are never copied in, so the drill-in shows
          the interior's shape, never tool args or model text.
        * **Bounded.** The interior node count is capped; a pathological leaf that fires
          a flood of edges drops further nodes and the re-emit carries ``truncated=True``,
          guarding against resource exhaustion.

        Args:
            event: One normalized interior callback edge from the leaf's run subtree.
        """
        leaf_id = event.leaf_span_id
        nodes = self._leaf_nodes[leaf_id]
        existing = nodes.get(event.run_id)
        if existing is None:
            if len(nodes) >= _MAX_SUBTREE_NODES:
                # Cap reached: drop the new node, remember the leaf overflowed. Edges for
                # already-buffered runs (e.g. a later end edge) still roll in below.
                self._leaf_truncated.add(leaf_id)
            else:
                nodes[event.run_id] = {
                    "run_id": event.run_id,
                    "parent_run_id": event.parent_run_id,
                    "kind": event.kind,
                    "name": event.name,
                    "phase": event.phase,
                }
        else:
            # Roll a later edge of the same run into its node: advance the phase, and keep
            # a non-empty start-edge name when the end edge carries an empty one.
            existing["phase"] = event.phase
            if event.name:
                existing["name"] = event.name

        subtree = list(nodes.values())
        subtree_props: dict[str, Any] = {
            "subtree": subtree,
            "truncated": leaf_id in self._leaf_truncated,
        }
        self._emit_event(_AGENT_SPAN, subtree_props, event_id=leaf_id, merge=True)

    def on_command(self, event: CommandEvent) -> None:
        """Map a real-execution command edge to an ``execution_command`` event (never raises).

        Each real shell ``execute`` fires two edges that share one engine ``command_id``:
        a ``"start"`` edge the instant before the subprocess spawns and an ``"end"`` edge
        once it is reaped. The start edge **creates** a terminal card in the ``running``
        state (``exit_code`` still ``None``); the end edge **patches it in place** (the
        same ``event_id`` with ``merge=True``) flipping the card to ``passed`` (exit 0) or
        ``failed`` (non-zero) and filling the exit code, output tail, truncation flag, and
        duration — the same begin->end in-place flip the span running->complete chip uses.

        The card is correlated to its owning leaf via ``leaf_span_id`` (so the frontend
        nests it beneath that leaf's ``agent_span``) and tagged with the loop ``attempt``
        the adapter most recently forwarded — ``on_command`` fires inside the leaf with no
        knowledge of the loop counter, so the adapter stamps the latest phase marker it
        saw. A replayed (cached) leaf never re-runs its subprocess and so fires no command
        event; terminal cards are fresh-run only and never fabricated on resume.

        Args:
            event: One real-execution command lifecycle edge (``"start"`` then ``"end"``).
        """
        # The whole body runs under the lock because on_command fires from a to_thread
        # worker while the loop-thread sinks touch the same shared state. The lock is
        # re-entrant, so the nested _emit_event below re-acquires it harmlessly.
        with self._lock:
            is_end = event.phase == "end"
            mint_undo: tuple[str, int] | None = None
            if not is_end:
                # Start edge: mint the adapter's resume-stable id (excluding run-variant
                # fields) and remember it under the engine command_id so the end edge
                # reuses it. The id carries attempt so the same command across two attempts
                # stays distinct (else the second attempt's start would collide and be
                # swallowed). The (stable, ordinal) is kept so a swallowed start emit can
                # undo the ordinal bump AND drop the orphaned command-id entry, keeping the
                # mint consistent for a retry.
                event_id, mint_undo = self._mint_command_event_id(event)
                self._command_event_ids[event.command_id] = event_id
                status = "running"
            else:
                # End edge: reuse the start's id so the card flips in place. If the start
                # was never seen (an end with no paired begin), mint a fresh id so the card
                # still renders something honest.
                event_id = self._command_event_ids.get(event.command_id)
                if event_id is None:
                    event_id, _ = self._mint_command_event_id(event)
                status = "passed" if event.exit_code == 0 else "failed"

            props: dict[str, Any] = {
                "leaf_span_id": event.leaf_span_id,
                "command": event.command,
                "attempt": self._latest_phase,
                "status": status,
                "exit_code": event.exit_code,
                "output": event.output,
                "truncated": event.truncated,
                "duration_s": event.duration_s,
            }
            # The end edge patches the running card (same event_id), so it merges — and the
            # merge path bypasses _seen, the same begin/end exemption the span pair uses.
            delivered = self._emit_event(_EXECUTION_COMMAND, props, event_id=event_id, merge=is_end)
            # A swallowed start emit must leave no trace: undo the ordinal bump so a retry
            # re-mints THIS id, and drop the orphaned command-id entry so the retried start
            # (not the missing end) owns the mapping. Without this, a transient UI outage on
            # the start edge would strand the command_id -> id mapping and skip an ordinal.
            if not is_end and not delivered and mint_undo is not None:
                stable, ordinal = mint_undo
                self._occurrences[stable] = ordinal
                self._command_event_ids.pop(event.command_id, None)

    # --- host-driven sign-off card (M4 in-run HITL) --------------------------

    def emit_signoff_request(self, *, gate_key: str, ask: Any) -> None:
        """Emit an awaiting ``signoff_gate`` card for a parked sign-off (never raises).

        Called from the host when an inline run parks at ``ctx.checkpoint``. The card's
        id is derived from the gate's content key so the matching :meth:`emit_signoff_resolved`
        flips this exact card in place — even across the turn boundary between the park and
        the human's answer (the adapter is per-turn, so the gate key, not adapter state, is
        what correlates the two edges).

        Args:
            gate_key: The parked gate's content-hash key (its stable card identity).
            ask: The ask payload the gate surfaced (a mapping with ``ask`` / ``summary``).
        """
        question, detail = _split_signoff_ask(ask)
        props: dict[str, Any] = {
            "status": "awaiting",
            "question": question,
            "detail": detail,
            "attempt": self._latest_phase,
        }
        self._emit_event(_SIGNOFF_GATE, props, event_id=f"signoff-{gate_key}")

    def emit_signoff_resolved(self, *, gate_key: str, ask: Any, decision: Any) -> None:
        """Flip a sign-off card to approved/rejected in place (never raises).

        Re-sends the question/detail (from ``ask``, carried across turns by the host's
        resume lane) alongside the resolved status so the merged card stays complete.

        Args:
            gate_key: The gate key whose awaiting card to resolve (same id as the request).
            ask: The same ask payload the request carried (for the persisted question/detail).
            decision: The human decision (``{"approved": bool, "note": str}`` or a bool).
        """
        question, detail = _split_signoff_ask(ask)
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        note = str(decision.get("note", "") or "") if isinstance(decision, dict) else ""
        props: dict[str, Any] = {
            "status": "approved" if approved else "rejected",
            "question": question,
            "detail": detail,
            "note": note,
            "attempt": self._latest_phase,
        }
        self._emit_event(_SIGNOFF_GATE, props, event_id=f"signoff-{gate_key}", merge=True)

    # --- host-driven pull-request card (M6 host finalization) ----------------

    def emit_pull_request(
        self,
        *,
        number: int,
        branch: str,
        url: str,
        integration_branch: str,
        title: str,
        created: bool,
    ) -> None:
        """Emit a ``pull_request`` card for a host-finalized PR (never raises).

        Called from the host AFTER ``run_workflow`` returns, once it has opened (or
        idempotently re-opened) the workflow's PR intent through a
        ``LocalPullRequestProvider``. The card's id is derived from the PR's source branch
        so a re-finalization on a later turn (the idempotent re-open path) lands on the SAME
        card rather than appending a duplicate — mirroring the stable-``event_id`` pattern
        the sign-off card uses across the turn boundary. Stamped with the latest phase
        marker the adapter saw, for consistency with the other cards.

        Args:
            number: The PR number the provider assigned.
            branch: The source branch the PR was opened from (its stable card identity).
            url: The PR's address (``local://pr/<number>`` for the offline provider).
            integration_branch: The branch the PR targets (not ``main``).
            title: The PR title.
            created: ``True`` when this call newly created the PR, ``False`` on an
                idempotent re-open of an existing one for the same branch.
        """
        props: dict[str, Any] = {
            "number": number,
            "branch": branch,
            "url": url,
            "integration_branch": integration_branch,
            "title": title,
            "created": created,
            "attempt": self._latest_phase,
        }
        self._emit_event(_PULL_REQUEST, props, event_id=f"pr-{branch}")

    def _mint_command_event_id(self, event: CommandEvent) -> tuple[str, tuple[str, int]]:
        """Mint a resume-stable ``execution_command`` id and bump its occurrence ordinal.

        The id hashes the stable identity ``(leaf_span_id, command, attempt)`` — the
        latest forwarded phase as ``attempt`` — and excludes every run-variant field
        (``exit_code`` / ``output`` / ``duration_s`` / ``status``) so a command's start
        and end edges share one id and the card flips in place. The per-key occurrence
        ordinal (bumped once per command, on the create edge) keeps the same command
        across two attempts on distinct cards while an honest resume — which re-emits the
        same keys in the same source order — lands on the same ordinals and collapses.

        Must be called with :attr:`_lock` held (the caller, :meth:`on_command`, holds it):
        the ordinal read-modify-write races a concurrent loop-thread sink otherwise.

        Args:
            event: The command edge whose stable identity seeds the id.

        Returns:
            A ``(event_id, (stable, ordinal))`` pair. The second element lets the caller
            undo the ordinal bump if a swallowed emit means the card never delivered.
        """
        dedupe_key = {
            "leaf_span_id": event.leaf_span_id,
            "command": event.command,
            "attempt": self._latest_phase,
        }
        stable = self._stable_id(_EXECUTION_COMMAND, dedupe_key)
        ordinal = self._occurrences[stable]
        self._occurrences[stable] = ordinal + 1
        return f"{stable}-{ordinal}", (stable, ordinal)

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
            "running": False,
        }
        # The end edge patches the running chip opened by on_span_begin (same span_id):
        # merge=True so the SDK reducer folds these completion fields onto the begin card
        # in place rather than appending a second card. begin-only fields (started_at)
        # are absent here and survive the shallow merge. When a leaf never opened a
        # running card (no begin edge), the merge is a harmless no-op create.
        self._emit_event(_AGENT_SPAN, agent_props, event_id=span.span_id, merge=True)

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
        merge: bool = False,
    ) -> bool:
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

        A ``merge`` event is the exception to seen-suppression: it is a deliberate
        in-place patch of a card that was already created (the leaf's end edge folding
        completion fields onto the running begin card). It therefore bypasses the
        seen-set drop and carries a reserved ``merge`` props flag the transport pops and
        forwards to ``push_ui_message(..., merge=True)`` so the SDK reducer shallow-merges
        these props onto the existing same-``event_id`` card. A merge does not enter the
        seen-set (it targets an id whose create already did).

        Args:
            component: The Gen-UI component name to render.
            props: The full component props, including run-variant display fields (an
                ``event_id`` is added in place).
            event_id: The engine-minted id to use verbatim, for span events. When
                provided, the adapter skips its own hashing and occurrence-ordinal path.
            dedupe_key: The stable subset of the event's identity, excluding any
                run-variant field, used to compute the id and occurrence ordinal for
                progress events. Required when ``event_id`` is ``None``.
            merge: Whether this event patches an existing same-``event_id`` card in place
                (the end edge of a leaf folding onto its running begin card). When ``True``
                the event bypasses seen-suppression and carries the transport merge flag.

        Returns:
            ``True`` if the event was delivered (or was a deduped no-op the seen-set
            suppressed), ``False`` only when the transport raised and the emit was
            swallowed — letting the command path undo its bookkeeping for a clean retry.

        Raises:
            ValueError: If neither ``event_id`` nor ``dedupe_key`` is supplied.
        """
        # The whole body is guarded so the ordinal read-modify-write and the seen-set
        # check/add are atomic against a concurrent sink on another thread (on_command
        # runs on a to_thread worker). The lock is re-entrant, so a caller already holding
        # it (on_command) re-acquires harmlessly.
        with self._lock:
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

            # A merge is a deliberate in-place patch of an already-created card, so it must
            # NOT be dropped by seen-suppression (the create already marked the id seen).
            if not merge and event_id in self._seen:
                return True
            props["event_id"] = event_id
            if merge:
                props["merge"] = True
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
                return False
            self._seen.add(event_id)
            return True

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

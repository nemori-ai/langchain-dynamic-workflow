"""Self-built async background run mechanism for the host-facing workflow tool.

This is the machinery that lets a host agent launch a workflow *without blocking
its own turn*: a coroutine is launched detached with ``asyncio.create_task``, the
caller is handed a slot carrying a placeholder ``run_id`` immediately, and the
host polls (``status``) or is notified (``<workflow_notification>`` injected by
the middleware) when the run settles.

Scope note: this layer is the **host-turn** background wrapper. It is a strictly
different scope from the engine-internal durable execution that runs *inside*
``run_workflow`` (``@task`` / journal / sandbox). The manager owns slot
lifecycle, completion delivery, large-result offload, and idle/hard TTL
reclamation; it knows nothing about what the wrapped coroutine does.

A run is addressed by the composite ``(thread_id, run_id)`` key so the same
``run_id`` on two host threads never collides. Completed results live in a
completed-index so the host can fetch them after the fact; large results are
offloaded to a :class:`ResultStore` and surfaced as a summary plus an opaque
handle rather than inlined.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ._errors import WorkflowSignoffRequired

if TYPE_CHECKING:
    # Type-only references for the transport payload union: the background substrate
    # buffers engine events as opaque payloads and never constructs or inspects them,
    # so these imports stay out of the runtime import graph (the substrate remains
    # runtime-independent of the engine internals).
    from ._leaf_events import LeafEvent
    from ._observability import CommandEvent, Span, SpanBegin
    from ._progress import ProgressEntry


class BgStatus(StrEnum):
    """Lifecycle status of a background run.

    Attributes:
        PENDING: The run has been created but its task has not started executing.
        RUNNING: The run's task is executing.
        DONE: The run finished successfully; its result is available.
        FAILED: The run raised; the error detail is available.
        CANCELLED: The run was cancelled before completing.
        AWAITING_SIGNOFF: The run paused at an in-run ``ctx.checkpoint`` gate and
            waits for a human value (non-terminal); the ask is readable and an
            ``approve`` continues the same run. Still counts as in-flight.
        UNKNOWN: No slot exists for the queried key (never created, or reclaimed).
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_SIGNOFF = "awaiting_signoff"
    UNKNOWN = "unknown"


_TERMINAL_STATUSES: frozenset[BgStatus] = frozenset(
    {BgStatus.DONE, BgStatus.FAILED, BgStatus.CANCELLED}
)
"""Statuses at which a run has settled and its slot is eligible for TTL sweep."""


class BgRunQuotaExceededError(RuntimeError):
    """Raised when a new background run would exceed the concurrent-run quota.

    The manager admits at most ``max_concurrent_runs`` in-flight (non-terminal)
    runs. A host that asks to launch another while the quota is full is refused
    loud rather than having its run fanned out onto the event loop unbounded — the
    bounded-queue / resource-exhaustion guard for host-initiated background work.
    """


class BgRunStateError(RuntimeError):
    """Raised when an operation does not match a run's current lifecycle state.

    For example, approving a run that is not awaiting a sign-off (it is still
    running, already done, or was cancelled): there is nothing to continue, so the
    manager refuses loud rather than relaunching a run in the wrong state.
    """


class CanonicalRunInFlightError(RuntimeError):
    """Raised when a launch would duplicate a canonical journal that is still live.

    The manager enforces an at-most-one-live-run-per-canonical invariant within this
    process: a canonical ``journal_run_id`` identifies one run lineage (its journal
    and its checkpoint thread). Admitting a second live run against an already-live
    canonical would fan two runs onto the same journal sequence and checkpoint thread,
    each with its own (unshared) concurrency gate and token budget — a duplicate the
    manager refuses loud rather than silently double-spending.

    Scope: the live set is process-local. A run launched in a *different* live process
    sharing the same persistent store is invisible here, so a cross-process duplicate
    cannot be detected. That is by design — cross-process ``resume`` is for crash
    recovery (a dead origin), where the canonical is absent from this process's set and
    the resume legitimately proceeds. A durable cross-process lease is future work.
    """


type BufferedPayload = SpanBegin | Span | LeafEvent | ProgressEntry | CommandEvent
"""The verbatim engine event object captured for replay."""


@dataclass(frozen=True, slots=True)
class BufferedEvent:
    """One raw engine event captured from a detached background run for later replay.

    Attributes:
        kind: Which engine sink produced it — ``"span_begin"`` / ``"span"`` /
            ``"leaf_event"`` / ``"progress"`` / ``"command"`` — so a replay
            dispatches to the matching consumer method without type-sniffing.
        payload: The verbatim engine event object.
    """

    kind: str
    payload: BufferedPayload


@dataclass(frozen=True, slots=True)
class RunEventSinks:
    """The five engine sinks that buffer a background run's events on its slot.

    Built by :meth:`BgRunManager.event_sinks` and passed verbatim into
    ``run_workflow``'s keyword-only sink parameters. Each sink is a synchronous,
    bounded, lock-guarded append that never raises and never blocks — satisfying
    the engine's inline-sink contract.

    Attributes:
        on_span_begin: Buffers a span-open edge.
        on_span: Buffers a completed span.
        on_leaf_event: Buffers one leaf interior callback edge.
        on_progress: Buffers a progress entry.
        on_command: Buffers a real-execution command edge (may fire off-loop).
    """

    on_span_begin: Callable[[SpanBegin], None]
    on_span: Callable[[Span], None]
    on_leaf_event: Callable[[LeafEvent], None]
    on_progress: Callable[[ProgressEntry], None]
    on_command: Callable[[CommandEvent], None]


@dataclass(frozen=True, slots=True)
class Notice:
    """A completion notice queued for delivery to a host thread.

    Attributes:
        run_id: The run the notice concerns.
        thread_id: The host thread the run belonged to.
        status: The terminal status the run settled at.
        summary: A short, always-inlinable summary of the outcome (the result
            text truncated for ``DONE``, the error text for ``FAILED``).
        detail: Optional extra detail (the full error string for ``FAILED``),
            or ``None``.
    """

    run_id: str
    thread_id: str
    status: BgStatus
    summary: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of a settled run, as returned to the host by ``get_result``.

    A small result is inlined in ``value``; a large result is offloaded and
    ``value`` is ``None`` while ``handle`` points at the full payload in the
    :class:`ResultStore`. ``summary`` is always a short, inlinable view.

    Attributes:
        status: The terminal status the run settled at.
        value: The full result text when inlined, else ``None`` (offloaded).
        summary: A short summary safe to inline in a status reply.
        handle: An opaque store handle for the full payload when offloaded, else
            ``None``.
        detail: Optional error detail for a failed run, else ``None``.
    """

    status: BgStatus
    value: str | None
    summary: str
    handle: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """A read-only view of one tracked run, for the aggregate runs listing.

    Returned by :meth:`BgRunManager.list_runs` so a host can see all of its runs
    at once without polling each ``run_id``. It deliberately exposes only an
    immutable view (never the mutable :class:`BgRunSlot`).

    Attributes:
        run_id: The run's identifier.
        status: The run's current lifecycle status.
        label: The run's display label (e.g. its workflow name) recorded at launch,
            or ``None`` when the launcher supplied none.
        summary: A short outcome preview for a settled run (result preview, error
            text, or cancellation), or ``None`` while the run is still in flight.
    """

    run_id: str
    status: BgStatus
    label: str | None
    summary: str | None


_SNAPSHOT_SUMMARY_MAX_CHARS = 80
"""Cap for the short per-run outcome preview in an aggregate runs listing."""


def _summarize(text: str, *, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` (no ellipsis past a tiny budget)."""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


class ResultStore:
    """In-memory store for offloaded large run results.

    A run result longer than ``inline_max_chars`` is stashed here behind an
    opaque handle so a ``status`` reply can carry a summary plus the handle
    instead of flooding the host's context with the whole payload. The host
    fetches the full value on demand via :meth:`fetch`.

    Args:
        inline_max_chars: Results at or below this length are inlined by the
            manager; longer ones are offloaded to this store.
    """

    def __init__(self, *, inline_max_chars: int = 2000) -> None:
        self.inline_max_chars = inline_max_chars
        self._payloads: dict[str, str] = {}

    def offload(self, value: str) -> str:
        """Store ``value`` and return an opaque handle for later fetch."""
        handle = f"result://{uuid.uuid4().hex}"
        self._payloads[handle] = value
        return handle

    def fetch(self, handle: str) -> str:
        """Return the full payload for ``handle``.

        Raises:
            KeyError: If ``handle`` is unknown (never offloaded or evicted).
        """
        return self._payloads[handle]

    def discard(self, handle: str) -> None:
        """Drop the payload for ``handle`` if present (idempotent)."""
        self._payloads.pop(handle, None)


def _empty_event_buffer() -> list[BufferedEvent]:
    """Typed default factory for a slot's bounded event buffer."""
    return []


@dataclass
class BgRunSlot:
    """Tracking record for one detached background run.

    Attributes:
        run_id: The run's identifier (host-supplied or generated).
        thread_id: The host thread that launched the run.
        task: The detached ``asyncio.Task`` executing the wrapped coroutine.
        canonical: The canonical ``journal_run_id`` this run holds while live, or
            ``None`` when the launcher reserved no canonical. Used to release the
            process-local single-live-run reservation when the run settles terminal.
        status: The slot's current lifecycle status.
        label: An opaque display label for the run (e.g. its workflow name),
            carried so an aggregate listing can name the run without depending on
            any caller-side bookkeeping. ``None`` when the launcher supplied none.
        created_at: Monotonic timestamp when the slot was created.
        settled_at: Monotonic timestamp when the slot reached a terminal status,
            or ``None`` while still in flight.
        result: The successful result text once ``DONE``, else ``None``.
        handle: The offload handle when the result was offloaded, else ``None``.
        error: The error string once ``FAILED``, else ``None``.
        ask: The sign-off ask payload while ``AWAITING_SIGNOFF`` (what the human is
            deciding), else ``None``. Cleared when the run is approved.
        parked_at: Monotonic timestamp when the run parked at a sign-off gate, else
            ``None``. Drives the park-TTL expiry in ``sweep``; cleared on approve.
        events: Bounded raw-event buffer capturing the run's transport events for
            later replay.
        dropped: How many events were dropped past the buffer cap.
        events_lock: Lock guarding buffer reads/writes — ``on_command`` appends
            from an ``asyncio.to_thread`` worker, off the event loop.
    """

    run_id: str
    thread_id: str
    task: asyncio.Task[Any]
    canonical: str | None = None
    status: BgStatus = BgStatus.PENDING
    label: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    settled_at: float | None = None
    result: str | None = None
    handle: str | None = None
    error: str | None = None
    ask: Any = None
    parked_at: float | None = None
    events: list[BufferedEvent] = field(default_factory=_empty_event_buffer)
    dropped: int = 0
    events_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


def _composite_key(thread_id: str, run_id: str) -> tuple[str, str]:
    """Return the ``(thread_id, run_id)`` registry key isolating runs per thread."""
    return (thread_id, run_id)


class BgRunManager:
    """Launches, tracks, and reclaims detached background workflow runs.

    The manager is the host-turn background substrate: ``start`` detaches a
    coroutine onto the event loop and returns immediately with a slot; ``poll``
    reports lifecycle status; a done callback enqueues a :class:`Notice` for the
    owning thread (drained by the middleware into a ``<workflow_notification>``);
    ``get_result`` returns the settled outcome (inlined or offloaded); ``cancel``
    stops an in-flight run; and ``sweep`` reclaims settled slots past their idle
    TTL.

    Args:
        result_store: Store for offloaded large results; a default in-memory
            store is created when omitted.
        idle_ttl_seconds: How long a settled slot is retained before ``sweep``
            may reclaim it. ``0`` makes a settled slot immediately reclaimable.
        max_concurrent_runs: Cap on the number of in-flight (non-terminal) runs
            the manager will admit at once. ``None`` (the default) leaves it
            unbounded; a positive value makes ``start`` refuse a new run with
            :class:`BgRunQuotaExceededError` once that many runs are in flight,
            bounding host-initiated background fan-out against resource exhaustion.
        park_ttl_seconds: How long a run parked at a sign-off gate is held before
            ``sweep`` expires it (→ ``CANCELLED`` + notice), so an abandoned sign-off
            does not hold a quota slot forever. Generous by default (a human decision
            may take a long time); a live sign-off still completes on approve.
        max_buffered_events: Per-run cap on buffered transport events. Past the
            cap, new events are dropped (drop-newest) and counted in the slot's
            ``dropped`` tally. Must be positive.
    """

    def __init__(
        self,
        *,
        result_store: ResultStore | None = None,
        idle_ttl_seconds: float = 3600.0,
        max_concurrent_runs: int | None = None,
        park_ttl_seconds: float = 86400.0,
        max_buffered_events: int = 2000,
    ) -> None:
        if max_concurrent_runs is not None and max_concurrent_runs <= 0:
            # 0/negative is not a meaningful quota: it would refuse every run
            # (active_run_count() >= 0 is always true). Use None for unbounded, a
            # positive integer for a real cap — reject the ambiguous value loud.
            raise ValueError(
                f"max_concurrent_runs must be a positive integer or None (unbounded); "
                f"got {max_concurrent_runs!r}, which would refuse every run"
            )
        if max_buffered_events <= 0:
            raise ValueError(
                f"max_buffered_events must be a positive integer, got {max_buffered_events!r}; "
                "the per-run event buffer needs a real bound"
            )
        self._max_buffered_events = max_buffered_events
        self._result_store = result_store if result_store is not None else ResultStore()
        self._idle_ttl_seconds = idle_ttl_seconds
        self._max_concurrent_runs = max_concurrent_runs
        # A parked (AWAITING_SIGNOFF) run holds a quota slot until approved/cancelled; an
        # abandoned one would hold it forever. ``sweep`` expires a park idle past this TTL
        # (→ CANCELLED + notice), a defended bound on parked-run resource hold. Generous
        # by default (a human sign-off may legitimately take a long time); host-tunable.
        self._park_ttl_seconds = park_ttl_seconds
        self._slots: dict[tuple[str, str], BgRunSlot] = {}
        # Pending completion notices keyed by host thread, drained FIFO.
        self._notices: dict[str, list[Notice]] = {}
        # The canonical journal_run_ids with a currently-live (non-terminal) run.
        # Enforces at most one live run per canonical lineage IN THIS PROCESS so a
        # launch never fans two runs onto one journal + checkpoint thread (cross
        # host thread, terminal-origin-with-live-resume-child, or a double-fired
        # resume). A canonical is claimed synchronously at ``start`` and released on
        # the terminal ``_settle`` only (a parked/approved run keeps its claim).
        self._live_canonicals: set[str] = set()

    @property
    def result_store(self) -> ResultStore:
        """The store backing offloaded large results."""
        return self._result_store

    @property
    def max_concurrent_runs(self) -> int | None:
        """The concurrent-run quota, or ``None`` when unbounded."""
        return self._max_concurrent_runs

    @property
    def max_buffered_events(self) -> int:
        """Per-run cap on buffered transport events."""
        return self._max_buffered_events

    def active_run_count(self) -> int:
        """Return how many runs are currently in flight (non-terminal)."""
        return sum(1 for slot in self._slots.values() if slot.status not in _TERMINAL_STATUSES)

    def _snapshot_summary(self, slot: BgRunSlot) -> str | None:
        """Render a short outcome preview for a settled slot, else ``None``.

        In-flight runs have no outcome yet (``None``); a settled run gets a short
        preview of its result, error, or cancellation, capped so an aggregate
        listing stays compact.
        """
        if slot.status == BgStatus.DONE:
            full = (
                self._result_store.fetch(slot.handle)
                if slot.handle is not None
                else (slot.result or "")
            )
            return _summarize(full, max_chars=_SNAPSHOT_SUMMARY_MAX_CHARS)
        if slot.status == BgStatus.FAILED:
            return _summarize(slot.error or "run failed", max_chars=_SNAPSHOT_SUMMARY_MAX_CHARS)
        if slot.status == BgStatus.CANCELLED:
            return slot.error or "cancelled"
        if slot.status == BgStatus.AWAITING_SIGNOFF:
            return _summarize(
                f"awaiting sign-off: {slot.ask}", max_chars=_SNAPSHOT_SUMMARY_MAX_CHARS
            )
        return None  # PENDING / RUNNING: no outcome yet

    def list_runs(self, thread_id: str) -> list[RunSnapshot]:
        """Return a read-only snapshot of every run tracked for ``thread_id``.

        The aggregate view behind the tool's ``runs`` command: a host can see all
        of its in-flight and settled runs in one call instead of polling each
        ``run_id``. Runs are listed in creation order (slot insertion order); a
        settled run carries a short ``summary``, an in-flight one carries ``None``.
        Reclaimed slots (swept past their idle TTL) are no longer listed.

        Args:
            thread_id: The host thread whose runs to enumerate.

        Returns:
            One :class:`RunSnapshot` per tracked run on the thread, possibly empty.
        """
        return [
            RunSnapshot(
                run_id=slot.run_id,
                status=slot.status,
                label=slot.label,
                summary=self._snapshot_summary(slot),
            )
            for (slot_thread, _run_id), slot in self._slots.items()
            if slot_thread == thread_id
        ]

    def buffered_events(
        self, run_id: str, *, thread_id: str | None = None
    ) -> tuple[list[BufferedEvent], int]:
        """Return a snapshot copy of a run's buffered events and its dropped count.

        Returns ``([], 0)`` for an unknown run (never created, or reclaimed by
        ``sweep``) so a replay loop can keep polling without guarding KeyError. The
        list is a shallow copy taken under the slot's events lock, so a concurrent
        worker-thread append cannot tear the read.

        Args:
            run_id: The run whose buffer to snapshot.
            thread_id: Optional owning thread to disambiguate the composite key.

        Returns:
            A ``(events, dropped)`` pair: the buffered events in arrival order and
            how many were dropped past the buffer cap.
        """
        slot = self._find(run_id, thread_id)
        if slot is None:
            return ([], 0)
        with slot.events_lock:
            return (list(slot.events), slot.dropped)

    def event_sinks(self, run_id: str, *, thread_id: str) -> RunEventSinks:
        """Build append-only transport sinks bound to ``run_id``'s slot buffer.

        Each sink resolves the slot lazily at append time (so the sinks can be built
        before ``start`` registers the slot, and outlive a swept slot harmlessly),
        then appends under the slot's events lock, dropping past the buffer cap and
        counting the drop. Synchronous, never raises, never blocks.

        Args:
            run_id: The run whose slot buffer receives the events.
            thread_id: The owning host thread (part of the composite slot key).

        Returns:
            A :class:`RunEventSinks` bundle for ``run_workflow``'s sink parameters.
        """
        key = _composite_key(thread_id, run_id)
        cap = self._max_buffered_events

        def _append(kind: str, payload: BufferedPayload) -> None:
            slot = self._slots.get(key)
            if slot is None:
                return  # run never registered, or already swept: drop silently
            with slot.events_lock:
                if len(slot.events) >= cap:
                    slot.dropped += 1
                    return
                slot.events.append(BufferedEvent(kind=kind, payload=payload))

        return RunEventSinks(
            on_span_begin=lambda e: _append("span_begin", e),
            on_span=lambda e: _append("span", e),
            on_leaf_event=lambda e: _append("leaf_event", e),
            on_progress=lambda e: _append("progress", e),
            on_command=lambda e: _append("command", e),
        )

    def start(
        self,
        coro: Coroutine[Any, Any, str],
        *,
        run_id: str | None = None,
        thread_id: str,
        label: str | None = None,
        canonical: str | None = None,
    ) -> BgRunSlot:
        """Detach ``coro`` onto the event loop and return its slot immediately.

        The coroutine is wrapped so that whatever status it settles at, the slot
        is updated and a completion notice is enqueued for ``thread_id``. The
        caller gets the slot back before the coroutine runs, so the host turn is
        never blocked.

        When ``canonical`` is supplied, the launch claims that canonical lineage in
        the process-local live set and refuses if it is already live — enforcing at
        most one live run per canonical journal + checkpoint thread within this
        process, regardless of host thread. The claim is a synchronous check-and-add
        with no ``await`` between the "already live?" test and the add, so two
        concurrent launches of one canonical cannot both pass it (asyncio is
        single-threaded; a no-await critical section is atomic). The claim is
        released on the run's terminal settle only; a parked (awaiting sign-off) run
        keeps it across the human pause and its approve continuation.

        Args:
            coro: The coroutine to run in the background; must resolve to the
                run's result text.
            run_id: Optional explicit run id; a fresh one is generated when
                omitted.
            thread_id: The host thread launching the run (part of the slot key).
            label: Optional opaque display label (e.g. the workflow name) stored on
                the slot so an aggregate listing can name the run.
            canonical: Optional canonical ``journal_run_id`` this run claims while
                live. ``None`` skips the single-live-run reservation (callers that do
                not share a journal/checkpoint lineage, e.g. direct utility starts).

        Returns:
            The :class:`BgRunSlot` tracking the detached run.

        Raises:
            BgRunQuotaExceededError: If a ``max_concurrent_runs`` quota is set and
                that many runs are already in flight. The passed coroutine is
                closed (never scheduled) so a refused run leaks no task or warning.
            BgRunStateError: If the ``(thread_id, run_id)`` composite key already maps
                to a LIVE (non-terminal) slot. Overwriting it would orphan the live
                run and leak its canonical, so the launch is refused; the passed
                coroutine is closed and no claim is added. A settled slot under the
                key may be reused.
            CanonicalRunInFlightError: If ``canonical`` is already live in this
                process. The passed coroutine is closed (never scheduled), and no
                claim is added, so a refused launch leaks no task and holds nothing.
        """
        if (
            self._max_concurrent_runs is not None
            and self.active_run_count() >= self._max_concurrent_runs
        ):
            # Refuse loud before detaching anything: close the un-launched coroutine
            # so the event loop does not warn about a coroutine that was never
            # awaited, then raise so the caller (e.g. the run command) reports it.
            coro.close()
            raise BgRunQuotaExceededError(
                f"background run quota exhausted: {self.active_run_count()} of "
                f"{self._max_concurrent_runs} concurrent runs already in flight; "
                "refusing to launch another (wait for a run to finish or cancel one)"
            )
        resolved_run_id = run_id if run_id is not None else uuid.uuid4().hex
        key = _composite_key(thread_id, resolved_run_id)
        # Reject a launch whose composite key already maps to a LIVE (non-terminal)
        # slot: overwriting it would orphan that run's slot, so the orphaned run's
        # later _settle would resolve the NEW slot and discard the WRONG canonical —
        # leaking the live run's canonical forever (its lineage becomes permanently
        # un-resumable) and prematurely releasing the new one. A settled slot under
        # the key may be reused (a finished run's key is free), so refuse only when
        # the existing slot is still live.
        existing = self._slots.get(key)
        if existing is not None and existing.status not in _TERMINAL_STATUSES:
            coro.close()
            raise BgRunStateError(
                f"run {resolved_run_id!r} on thread {thread_id!r} is already live "
                f"({existing.status.value}); refusing to launch over its slot "
                "(wait for it to settle or cancel it, or use a distinct run_id)"
            )
        # Synchronous canonical reservation: the duplicate-key check above and the
        # "is it live?" check + add here all sit in ONE no-await critical section, so
        # two interleaved launches of the same key or canonical cannot both pass
        # (TOCTOU-free under single-threaded asyncio).
        if canonical is not None and canonical in self._live_canonicals:
            coro.close()
            raise CanonicalRunInFlightError(
                f"canonical run {canonical!r} is already live in this process; refusing "
                "to launch a duplicate against the same journal and checkpoint thread "
                "(wait for the live run to settle or cancel it first)"
            )
        if canonical is not None:
            self._live_canonicals.add(canonical)
        task: asyncio.Task[str] = asyncio.ensure_future(
            self._run_wrapped(coro, key=key, thread_id=thread_id)
        )
        slot = BgRunSlot(
            run_id=resolved_run_id,
            thread_id=thread_id,
            task=task,
            canonical=canonical,
            label=label,
        )
        self._slots[key] = slot
        return slot

    async def _run_wrapped(
        self,
        coro: Coroutine[Any, Any, str],
        *,
        key: tuple[str, str],
        thread_id: str,
    ) -> str:
        """Execute ``coro`` and settle its slot, enqueuing a completion notice.

        Status flips to ``RUNNING`` as soon as the wrapper begins executing
        (which the event loop does only after ``start`` returns), then to a
        terminal status when the coroutine settles. The wrapper never lets an
        exception escape silently: a failure is recorded as ``FAILED`` with its
        error text, a cancellation as ``CANCELLED``.
        """
        slot = self._slots.get(key)
        if slot is not None:
            slot.status = BgStatus.RUNNING
        try:
            result = await coro
        except asyncio.CancelledError:
            self._settle(key, thread_id, BgStatus.CANCELLED, error="cancelled")
            raise
        except WorkflowSignoffRequired as park:
            # The run paused at an in-run sign-off gate: park it (non-terminal) so
            # the host can approve it later, rather than recording it as a failure.
            # Must precede the generic Exception handler (it is a RuntimeError).
            self._park(key, thread_id, park)
            return ""
        except Exception as exc:
            # A background run must never crash the event loop: convert any failure
            # into a FAILED settle carrying the error text for the host.
            self._settle(key, thread_id, BgStatus.FAILED, error=str(exc))
            return ""
        self._settle(key, thread_id, BgStatus.DONE, result=result)
        return result

    def _park(self, key: tuple[str, str], thread_id: str, signoff: WorkflowSignoffRequired) -> None:
        """Park a slot at ``AWAITING_SIGNOFF`` and enqueue a sign-off notice.

        Stores the ask so the host can read what to approve. The slot stays
        non-terminal (``settled_at`` unset) so the TTL sweep never reclaims a run
        that is merely waiting for a person; a host that abandons it can ``cancel``.
        """
        slot = self._slots.get(key)
        if slot is None:
            return
        slot.status = BgStatus.AWAITING_SIGNOFF
        slot.ask = signoff.ask
        slot.parked_at = time.monotonic()
        summary = _summarize(
            f"awaiting sign-off: {signoff.ask}", max_chars=_SNAPSHOT_SUMMARY_MAX_CHARS
        )
        self._notices.setdefault(thread_id, []).append(
            Notice(
                run_id=slot.run_id,
                thread_id=thread_id,
                status=BgStatus.AWAITING_SIGNOFF,
                summary=summary,
            )
        )

    def approve(
        self,
        coro: Coroutine[Any, Any, str],
        *,
        run_id: str,
        thread_id: str,
    ) -> BgRunSlot:
        """Continue a parked (``AWAITING_SIGNOFF``) run with a sign-off continuation.

        Relaunches the run *in place* under the same ``run_id`` (so a host tracking
        the run by id follows it across the human pause) by replacing the slot's
        task with ``coro`` — a fresh ``run_workflow(..., resume=value)`` against the
        same journal that records the human value at the gate and replays completed
        work for free. The continuation may settle the run or park it again at a
        later gate (re-arming the same slot).

        Args:
            coro: The continuation coroutine (a resuming ``run_workflow``); must
                resolve to the run's result text.
            run_id: The parked run to continue.
            thread_id: The owning host thread.

        Returns:
            The re-armed :class:`BgRunSlot`.

        Raises:
            KeyError: If no slot exists for the run (the passed coro is closed).
            BgRunStateError: If the run is not awaiting sign-off (the coro is
                closed so a refused approve leaks no unawaited coroutine).
        """
        key = _composite_key(thread_id, run_id)
        slot = self._slots.get(key)
        if slot is None:
            coro.close()
            raise KeyError(f"unknown run_id {run_id!r}")
        if slot.status != BgStatus.AWAITING_SIGNOFF:
            coro.close()
            raise BgRunStateError(
                f"run {run_id!r} is {slot.status.value}, not awaiting sign-off; nothing to approve"
            )
        # Flip status to RUNNING SYNCHRONOUSLY, before scheduling the continuation, to
        # close the check-then-act window: the new task only sets RUNNING once the loop
        # schedules it (in _run_wrapped), so without this a second approve racing the
        # loop would still see AWAITING_SIGNOFF, pass the guard, and orphan the first
        # continuation — two run_workflow continuations against one journal. With the
        # synchronous flip a duplicate approve sees RUNNING and is refused above.
        slot.ask = None
        slot.parked_at = None
        slot.status = BgStatus.RUNNING
        slot.task = asyncio.ensure_future(self._run_wrapped(coro, key=key, thread_id=thread_id))
        return slot

    def _settle(
        self,
        key: tuple[str, str],
        thread_id: str,
        status: BgStatus,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record a terminal status on a slot and enqueue its completion notice.

        On a successful settle a large result is offloaded to the result store
        and only its summary + handle are kept on the slot; a small result is
        kept inline.
        """
        slot = self._slots.get(key)
        if slot is None:
            return
        slot.status = status
        slot.settled_at = time.monotonic()
        # Release the canonical reservation: this run is now terminal, so the lineage
        # is free for a legitimate resume to re-claim. Only the terminal settle frees
        # it — a parked run (handled by _park) keeps its claim across the human pause.
        if slot.canonical is not None:
            self._live_canonicals.discard(slot.canonical)
        summary: str
        detail: str | None = None
        if status == BgStatus.DONE and result is not None:
            if len(result) > self._result_store.inline_max_chars:
                slot.handle = self._result_store.offload(result)
                slot.result = None
            else:
                slot.result = result
            summary = _summarize(result, max_chars=self._result_store.inline_max_chars)
        elif status == BgStatus.FAILED:
            slot.error = error
            detail = error
            summary = _summarize(
                error or "run failed", max_chars=self._result_store.inline_max_chars
            )
        else:  # CANCELLED
            slot.error = error
            summary = error or "run cancelled"
        self._notices.setdefault(thread_id, []).append(
            Notice(
                run_id=slot.run_id,
                thread_id=thread_id,
                status=status,
                summary=summary,
                detail=detail,
            )
        )

    def _find(self, run_id: str, thread_id: str | None) -> BgRunSlot | None:
        """Resolve the slot for ``run_id`` on ``thread_id`` (or any thread)."""
        if thread_id is not None:
            return self._slots.get(_composite_key(thread_id, run_id))
        # No thread given: scan for a unique match (convenience for single-thread
        # callers). The composite key still isolates collisions when supplied.
        for (_thread, rid), slot in self._slots.items():
            if rid == run_id:
                return slot
        return None

    def poll(self, run_id: str, *, thread_id: str | None = None) -> BgStatus:
        """Return the current status of ``run_id`` (``UNKNOWN`` if no slot)."""
        slot = self._find(run_id, thread_id)
        return slot.status if slot is not None else BgStatus.UNKNOWN

    async def wait(self, run_id: str, *, thread_id: str | None = None) -> None:
        """Await the detached task for ``run_id`` to settle (test/util helper).

        Raises:
            KeyError: If no slot exists for the run.
        """
        slot = self._find(run_id, thread_id)
        if slot is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        # A cancelled task is a settled outcome, not an error to re-raise here.
        with contextlib.suppress(asyncio.CancelledError):
            await slot.task

    def get_result(self, run_id: str, *, thread_id: str | None = None) -> RunResult:
        """Return the settled outcome for ``run_id``.

        Args:
            run_id: The run to fetch.
            thread_id: Optional owning thread to disambiguate the composite key.

        Returns:
            A :class:`RunResult`: inlined ``value`` for a small result, or a
            ``summary`` + ``handle`` for an offloaded large result.

        Raises:
            KeyError: If no slot exists for the run.
        """
        slot = self._find(run_id, thread_id)
        if slot is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        if slot.status == BgStatus.DONE:
            if slot.handle is not None:
                full = self._result_store.fetch(slot.handle)
                return RunResult(
                    status=slot.status,
                    value=None,
                    summary=_summarize(full, max_chars=self._result_store.inline_max_chars),
                    handle=slot.handle,
                )
            value = slot.result or ""
            return RunResult(status=slot.status, value=value, summary=value)
        if slot.status == BgStatus.FAILED:
            return RunResult(
                status=slot.status,
                value=None,
                summary=_summarize(
                    slot.error or "run failed", max_chars=self._result_store.inline_max_chars
                ),
                detail=slot.error,
            )
        # Pending / running / cancelled / awaiting-signoff: no value yet (or never).
        return RunResult(status=slot.status, value=None, summary=slot.status.value)

    def get_signoff(self, run_id: str, *, thread_id: str | None = None) -> Any | None:
        """Return the sign-off ask for a parked run, else ``None``.

        Args:
            run_id: The run to inspect.
            thread_id: Optional owning thread to disambiguate the composite key.

        Returns:
            The ask payload while the run is ``AWAITING_SIGNOFF``, else ``None``
            (the run is not parked, or no slot exists).
        """
        slot = self._find(run_id, thread_id)
        if slot is None or slot.status != BgStatus.AWAITING_SIGNOFF:
            return None
        return slot.ask

    def drain_notifications(self, thread_id: str) -> list[Notice]:
        """Pop and return all pending completion notices for ``thread_id``.

        Draining is destructive: each notice is delivered exactly once, so the
        middleware injects every completion into the host context a single time.
        """
        return self._notices.pop(thread_id, [])

    async def cancel(self, run_id: str, *, thread_id: str | None = None) -> None:
        """Cancel an in-flight run and mark its slot ``CANCELLED``.

        A run that has already settled is left as-is. Cancelling enqueues a
        cancellation notice so the host learns the run will not complete.

        Args:
            run_id: The run to cancel.
            thread_id: Optional owning thread to disambiguate the composite key.

        Raises:
            KeyError: If no slot exists for the run.
        """
        slot = self._find(run_id, thread_id)
        if slot is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        if slot.status in _TERMINAL_STATUSES:
            return
        # Yield once so a freshly-detached task gets a chance to start and enter
        # its wrapper before we cancel it; without this, cancelling a task that
        # the loop has not yet scheduled would leave the inner coroutine never
        # awaited (a resource leak the loop warns about).
        await asyncio.sleep(0)
        if slot.status in _TERMINAL_STATUSES:
            return
        slot.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await slot.task
        # The cancellation handler in _run_wrapped settles the slot; if the task
        # was not yet scheduled it may still be PENDING, so settle defensively.
        if slot.status not in _TERMINAL_STATUSES:
            self._settle(
                _composite_key(slot.thread_id, slot.run_id),
                slot.thread_id,
                BgStatus.CANCELLED,
                error="cancelled",
            )

    def sweep(self, *, now: float | None = None) -> list[str]:
        """Expire stale parked runs and reclaim settled slots past their idle TTL.

        Two-stage: (1) a run parked at a sign-off gate past ``park_ttl_seconds`` is
        EXPIRED — settled to ``CANCELLED`` with an "expired" notice, so an abandoned
        sign-off stops holding a quota slot forever (a defended resource bound; an
        active sign-off still completes on approve). (2) A run in a terminal status
        settled for at least ``idle_ttl_seconds`` is RECLAIMED — its slot and any
        offloaded payload are dropped.

        Args:
            now: Optional monotonic timestamp override (for deterministic tests).

        Returns:
            The ids of the runs reclaimed by this sweep.
        """
        moment = now if now is not None else time.monotonic()
        reclaimed: list[str] = []
        for key in list(self._slots):
            slot = self._slots[key]
            # Stage 1: expire an abandoned parked run so it stops holding a quota slot.
            if (
                slot.status == BgStatus.AWAITING_SIGNOFF
                and slot.parked_at is not None
                and moment - slot.parked_at >= self._park_ttl_seconds
            ):
                slot.parked_at = None
                self._settle(
                    key, slot.thread_id, BgStatus.CANCELLED, error="sign-off expired (park TTL)"
                )
                continue  # now terminal; a later sweep reclaims it past idle_ttl
            if slot.status not in _TERMINAL_STATUSES or slot.settled_at is None:
                continue
            if moment - slot.settled_at >= self._idle_ttl_seconds:
                if slot.handle is not None:
                    self._result_store.discard(slot.handle)
                del self._slots[key]
                reclaimed.append(slot.run_id)
        return reclaimed

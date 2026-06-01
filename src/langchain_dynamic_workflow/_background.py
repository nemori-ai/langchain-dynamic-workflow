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
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BgStatus(StrEnum):
    """Lifecycle status of a background run.

    Attributes:
        PENDING: The run has been created but its task has not started executing.
        RUNNING: The run's task is executing.
        DONE: The run finished successfully; its result is available.
        FAILED: The run raised; the error detail is available.
        CANCELLED: The run was cancelled before completing.
        UNKNOWN: No slot exists for the queried key (never created, or reclaimed).
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


_TERMINAL_STATUSES: frozenset[BgStatus] = frozenset(
    {BgStatus.DONE, BgStatus.FAILED, BgStatus.CANCELLED}
)
"""Statuses at which a run has settled and its slot is eligible for TTL sweep."""


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


@dataclass
class BgRunSlot:
    """Tracking record for one detached background run.

    Attributes:
        run_id: The run's identifier (host-supplied or generated).
        thread_id: The host thread that launched the run.
        task: The detached ``asyncio.Task`` executing the wrapped coroutine.
        status: The slot's current lifecycle status.
        created_at: Monotonic timestamp when the slot was created.
        settled_at: Monotonic timestamp when the slot reached a terminal status,
            or ``None`` while still in flight.
        result: The successful result text once ``DONE``, else ``None``.
        handle: The offload handle when the result was offloaded, else ``None``.
        error: The error string once ``FAILED``, else ``None``.
    """

    run_id: str
    thread_id: str
    task: asyncio.Task[Any]
    status: BgStatus = BgStatus.PENDING
    created_at: float = field(default_factory=time.monotonic)
    settled_at: float | None = None
    result: str | None = None
    handle: str | None = None
    error: str | None = None


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
    """

    def __init__(
        self,
        *,
        result_store: ResultStore | None = None,
        idle_ttl_seconds: float = 3600.0,
    ) -> None:
        self._result_store = result_store if result_store is not None else ResultStore()
        self._idle_ttl_seconds = idle_ttl_seconds
        self._slots: dict[tuple[str, str], BgRunSlot] = {}
        # Pending completion notices keyed by host thread, drained FIFO.
        self._notices: dict[str, list[Notice]] = {}

    @property
    def result_store(self) -> ResultStore:
        """The store backing offloaded large results."""
        return self._result_store

    def start(
        self,
        coro: Coroutine[Any, Any, str],
        *,
        run_id: str | None = None,
        thread_id: str,
    ) -> BgRunSlot:
        """Detach ``coro`` onto the event loop and return its slot immediately.

        The coroutine is wrapped so that whatever status it settles at, the slot
        is updated and a completion notice is enqueued for ``thread_id``. The
        caller gets the slot back before the coroutine runs, so the host turn is
        never blocked.

        Args:
            coro: The coroutine to run in the background; must resolve to the
                run's result text.
            run_id: Optional explicit run id; a fresh one is generated when
                omitted.
            thread_id: The host thread launching the run (part of the slot key).

        Returns:
            The :class:`BgRunSlot` tracking the detached run.
        """
        resolved_run_id = run_id if run_id is not None else uuid.uuid4().hex
        key = _composite_key(thread_id, resolved_run_id)
        task: asyncio.Task[str] = asyncio.ensure_future(
            self._run_wrapped(coro, key=key, thread_id=thread_id)
        )
        slot = BgRunSlot(run_id=resolved_run_id, thread_id=thread_id, task=task)
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
        except Exception as exc:
            # A background run must never crash the event loop: convert any failure
            # into a FAILED settle carrying the error text for the host.
            self._settle(key, thread_id, BgStatus.FAILED, error=str(exc))
            return ""
        self._settle(key, thread_id, BgStatus.DONE, result=result)
        return result

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
        # Pending / running / cancelled: no value yet (or never).
        return RunResult(status=slot.status, value=None, summary=slot.status.value)

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
        """Reclaim settled slots whose idle TTL has elapsed.

        A slot is reclaimable once it is in a terminal status and has been
        settled for at least ``idle_ttl_seconds``. Reclaiming drops the slot and
        any offloaded payload it owns.

        Args:
            now: Optional monotonic timestamp override (for deterministic tests).

        Returns:
            The ids of the runs reclaimed by this sweep.
        """
        moment = now if now is not None else time.monotonic()
        reclaimed: list[str] = []
        for key in list(self._slots):
            slot = self._slots[key]
            if slot.status not in _TERMINAL_STATUSES or slot.settled_at is None:
                continue
            if moment - slot.settled_at >= self._idle_ttl_seconds:
                if slot.handle is not None:
                    self._result_store.discard(slot.handle)
                del self._slots[key]
                reclaimed.append(slot.run_id)
        return reclaimed

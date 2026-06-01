"""Concurrency gating for leaf fan-out — both explicit layers.

The substrate (LangGraph) leaves ``RunnableConfig.max_concurrency`` at ``None``,
which means *unbounded* parallel task scheduling. Fan-out over many leaves must
be bounded on two layers, both set explicitly:

1. An asyncio :class:`~asyncio.Semaphore` (:class:`ConcurrencyGate`) that caps the
   number of in-flight leaf invocations across ``agent`` / ``parallel`` /
   ``pipeline`` within a single workflow run.
2. The substrate ``max_concurrency`` config value, injected via
   :func:`with_max_concurrency`, so the durable executor agrees with the gate.

Both default to ``min(16, cores - 2)`` and are clamped to a hard ceiling of
``1000`` to keep a runaway script from exhausting resources.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from langchain_core.runnables import RunnableConfig

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _GateHolder:
    """Per-context record of who owns a real concurrency slot.

    Attributes:
        owner_task_id: The id of the :class:`asyncio.Task` that acquired the real
            semaphore slot, or ``None`` when acquired outside any task. Reentry is
            free only for this same task; a child task that inherited this record
            via context copy is a new unit and must take its own slot.
        depth: This owner's reentry depth; the real slot is released when it
            drops back to zero.
    """

    owner_task_id: int | None
    depth: int


DEFAULT_SOFT_CAP = 16
"""Soft default ceiling for in-flight leaves when the host has many cores."""

HARD_CEILING = 1000
"""Absolute upper bound on concurrency to guard against runaway fan-out."""


def resolve_max_concurrency(requested: int | None) -> int:
    """Resolve an effective concurrency limit, always bounded and positive.

    Args:
        requested: An explicit limit, or ``None`` to derive a default from the
            host's CPU count.

    Returns:
        A positive integer in ``[1, HARD_CEILING]``. When ``requested`` is
        ``None`` the default is ``min(DEFAULT_SOFT_CAP, cores - 2)`` (at least
        ``1``); explicit values are clamped to ``[1, HARD_CEILING]``.
    """
    if requested is None:
        cores = os.cpu_count() or 1
        derived = min(DEFAULT_SOFT_CAP, cores - 2)
        return max(1, derived)
    return max(1, min(requested, HARD_CEILING))


def with_max_concurrency(config: RunnableConfig, limit: int) -> RunnableConfig:
    """Return a copy of ``config`` with ``max_concurrency`` set to ``limit``.

    Args:
        config: The base runnable config.
        limit: The concurrency cap to inject.

    Returns:
        A shallow copy of ``config`` with ``max_concurrency`` populated; the
        original mapping is left untouched.
    """
    merged: RunnableConfig = {**config}
    merged["max_concurrency"] = limit
    return merged


class ConcurrencyGate:
    """A reentrant asyncio semaphore bounding concurrent leaf invocations.

    The gate is shared across every fan-out path in a single workflow run so
    that ``agent`` / ``parallel`` / ``pipeline`` draw from one global pool rather
    than each opening an unbounded set of tasks.

    It is **reentrant per asyncio task**: a coroutine that already holds the gate
    may re-acquire it *from the same task* without consuming a second slot. This
    is essential because fan-out layers (``parallel`` / ``pipeline``) acquire the
    gate around a unit of work that itself calls ``agent()``, which also gates —
    without reentrancy a non-reentrant semaphore would deadlock once every slot
    is held by an outer acquisition waiting on an inner one.

    Reentrancy is **anchored to the owning task**, not merely to an inherited
    depth counter. State lives in a :class:`~contextvars.ContextVar`, and
    :func:`asyncio.gather` copies that context into each branch task. A naive
    depth counter would let a *child* task inherit ``depth > 0`` and free-ride on
    the parent's slot, so nested fan-out (``parallel`` inside ``parallel``, or a
    ``parallel`` inside a ``pipeline`` stage) would leak concurrency proportional
    to nesting depth. To prevent that, the gate records the id of the task that
    took the real slot. A re-acquisition counts as free only when it comes from
    that same task; a child task that inherited a held context is a genuinely new
    unit and must acquire its own real slot before proceeding.

    Args:
        limit: The maximum number of distinct in-flight units permitted.
    """

    def __init__(self, *, limit: int) -> None:
        self._limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        # Holder state per logical unit: the id of the task that owns the real
        # slot, plus this unit's reentry depth. ``owner_task_id`` is ``None`` when
        # no slot is held in the current context. Both are carried by the
        # ContextVar and therefore copied into gather/child tasks — which is
        # exactly why ``__aenter__`` re-checks the *current* task identity.
        self._holder: ContextVar[_GateHolder | None] = ContextVar(
            "concurrency_gate_holder", default=None
        )

    @property
    def limit(self) -> int:
        """The configured concurrency cap."""
        return self._limit

    async def __aenter__(self) -> ConcurrencyGate:
        current_task = asyncio.current_task()
        current_task_id = id(current_task) if current_task is not None else None
        holder = self._holder.get()
        if holder is not None and holder.owner_task_id == current_task_id:
            # True reentry from the slot-owning task: re-acquire for free.
            self._holder.set(_GateHolder(owner_task_id=current_task_id, depth=holder.depth + 1))
            return self
        # Either no slot is held here, or the held context was inherited by a new
        # child task (holder set by a different task). Both are new units: take a
        # real slot and become this context's owner.
        await self._semaphore.acquire()
        self._holder.set(_GateHolder(owner_task_id=current_task_id, depth=1))
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        holder = self._holder.get()
        if holder is None:
            # Defensive: an exit without a recorded acquisition is a no-op rather
            # than an over-release that would corrupt the semaphore count.
            return
        if holder.depth > 1:
            self._holder.set(
                _GateHolder(owner_task_id=holder.owner_task_id, depth=holder.depth - 1)
            )
            return
        # Releasing the outermost acquisition for this owner: return the slot.
        self._holder.set(None)
        self._semaphore.release()

    async def run(self, factory: Callable[[], Awaitable[T]]) -> T:
        """Run a coroutine produced by ``factory`` while holding the gate.

        The coroutine is created *after* a slot is acquired, so no work begins
        until there is capacity.

        Args:
            factory: A zero-argument callable returning the awaitable to run.

        Returns:
            The awaited result of the coroutine.
        """
        async with self:
            return await factory()

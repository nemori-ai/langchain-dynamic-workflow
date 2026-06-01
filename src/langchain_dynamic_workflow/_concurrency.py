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
from typing import TypeVar

from langchain_core.runnables import RunnableConfig

T = TypeVar("T")

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

    It is **reentrant per logical unit**: a coroutine that already holds the gate
    may re-acquire it without consuming a second slot. This is essential because
    fan-out layers (``parallel`` / ``pipeline``) acquire the gate around a unit
    of work that itself calls ``agent()``, which also gates — without reentrancy
    a non-reentrant semaphore would deadlock once every slot is held by an outer
    acquisition waiting on an inner one. Reentrancy is tracked with a
    :class:`~contextvars.ContextVar`, so each :class:`asyncio.Task` (e.g. each
    ``gather`` branch) carries its own depth counter.

    Args:
        limit: The maximum number of distinct in-flight units permitted.
    """

    def __init__(self, *, limit: int) -> None:
        self._limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self._depth: ContextVar[int] = ContextVar("concurrency_gate_depth", default=0)

    @property
    def limit(self) -> int:
        """The configured concurrency cap."""
        return self._limit

    async def __aenter__(self) -> ConcurrencyGate:
        depth = self._depth.get()
        if depth == 0:
            # Outermost acquisition for this unit: take a real slot.
            await self._semaphore.acquire()
        self._depth.set(depth + 1)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        depth = self._depth.get()
        self._depth.set(depth - 1)
        if depth == 1:
            # Releasing the outermost acquisition: return the slot.
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

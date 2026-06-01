"""The orchestration context (``ctx``) injected into a workflow script.

The context exposes the deterministic fan-out primitives. ``agent()`` runs a
single leaf; ``parallel()`` fans out a list of thunks with a blocking barrier;
``pipeline()`` streams items through stages without a barrier. The content-hash
journal is consulted on every leaf call, so a hit returns the cached result with
zero model calls — that is what makes runs resumable. A shared concurrency gate
bounds the number of in-flight leaves across every fan-out path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from ._concurrency import ConcurrencyGate, resolve_max_concurrency
from ._journal import JournalStore, journal_key
from ._result import fold_result
from ._roster import Roster

LeafRunner = Callable[[str, str], Awaitable[dict[str, Any]]]
"""Callable that actually invokes a leaf: ``(agent_type, prompt) -> raw state``."""

T = TypeVar("T")


class Ctx:
    """Deterministic orchestration context handed to a workflow script.

    Args:
        roster: The leaf registry.
        journal: The content-hash journal store.
        leaf_runner: Callable that invokes a resolved leaf as a durable task.
        gate: Shared concurrency gate bounding in-flight leaves; a bounded
            default is created when omitted.
    """

    def __init__(
        self,
        *,
        roster: Roster,
        journal: JournalStore,
        leaf_runner: LeafRunner,
        gate: ConcurrencyGate | None = None,
    ) -> None:
        self._roster = roster
        self._journal = journal
        self._leaf_runner = leaf_runner
        self._gate = (
            gate if gate is not None else ConcurrencyGate(limit=resolve_max_concurrency(None))
        )

    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        model: str | None = None,
        isolation: str = "shared",
    ) -> str:
        """Run a leaf subagent and return its folded final text.

        Resolves ``agent_type`` against the roster, consults the journal, and on
        a miss invokes the leaf and persists the result (success-only).

        Args:
            prompt: The prompt for the leaf.
            agent_type: The roster name to resolve.
            model: Optional model override (part of the journal key).
            isolation: Isolation mode (part of the journal key).

        Returns:
            The leaf's folded final text.

        Raises:
            KeyError: If ``agent_type`` is not registered.
        """
        self._roster.resolve(agent_type)  # fail fast on unknown agent_type
        key = journal_key(
            prompt=prompt,
            agent_type=agent_type,
            model=model,
            schema=None,
            isolation=isolation,
        )
        cached = await self._journal.get(key)
        if cached is not None:
            return cached
        # The gate bounds the number of leaves actually in flight; a journal hit
        # above never consumes a slot, keeping resume cheap.
        raw = await self._gate.run(lambda: self._leaf_runner(agent_type, prompt))
        folded = fold_result(raw)
        await self._journal.put(key, folded)  # success-only: unreachable if leaf raised
        return folded

    async def parallel(self, thunks: Sequence[Callable[[], Awaitable[T]]]) -> list[T | None]:
        """Fan out a list of thunks concurrently with a blocking barrier.

        Each thunk is a zero-argument callable returning an awaitable (typically
        a closure over an ``agent()`` call). Results are returned in input order.
        A thunk that raises lands as ``None`` at its position; the call as a whole
        never raises, mirroring Claude Code's ``parallel`` semantics — filter the
        ``None`` holes downstream.

        Concurrency is bounded by the shared gate: even with many thunks, only up
        to the gate's limit run at once. The barrier means this returns only once
        every thunk has settled.

        Args:
            thunks: The zero-argument awaitable factories to fan out.

        Returns:
            A list aligned to ``thunks`` input order; each entry is the thunk's
            result, or ``None`` if it raised.
        """

        async def _guarded(thunk: Callable[[], Awaitable[T]]) -> T | None:
            try:
                return await self._gate.run(thunk)
            except Exception:
                # Failure isolation: one bad thunk must not abort the barrier.
                return None

        if not thunks:
            return []
        return await asyncio.gather(*[_guarded(thunk) for thunk in thunks])

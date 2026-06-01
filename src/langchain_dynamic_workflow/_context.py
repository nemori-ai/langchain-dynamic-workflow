"""The orchestration context (``ctx``) injected into a workflow script.

In v1 Phase 1 the context exposes a single primitive, ``agent()``. Later phases
add ``parallel`` / ``pipeline`` / ``phase`` / ``log`` / ``budget`` / ``workflow``.
The content-hash journal is consulted on every ``agent()`` call; a hit returns
the cached result with zero model calls, which is what makes runs resumable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ._journal import JournalStore, journal_key
from ._result import fold_result
from ._roster import Roster

LeafRunner = Callable[[str, str], Awaitable[dict[str, Any]]]
"""Callable that actually invokes a leaf: ``(agent_type, prompt) -> raw state``."""


class Ctx:
    """Deterministic orchestration context handed to a workflow script.

    Args:
        roster: The leaf registry.
        journal: The content-hash journal store.
        leaf_runner: Callable that invokes a resolved leaf as a durable task.
    """

    def __init__(
        self,
        *,
        roster: Roster,
        journal: JournalStore,
        leaf_runner: LeafRunner,
    ) -> None:
        self._roster = roster
        self._journal = journal
        self._leaf_runner = leaf_runner

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
        raw = await self._leaf_runner(agent_type, prompt)
        folded = fold_result(raw)
        await self._journal.put(key, folded)  # success-only: unreachable if leaf raised
        return folded

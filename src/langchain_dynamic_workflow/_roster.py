"""Roster — the named registry of leaf agents.

A roster entry wraps any runnable whose state schema includes a ``messages``
key (e.g. a ``deepagents.create_deep_agent`` compiled graph, or a langchain
``create_agent`` graph). ``agent()`` calls resolve a leaf by name; there is no
ad-hoc escape hatch (decision R1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable


@dataclass(frozen=True)
class RosterEntry:
    """A registered leaf agent.

    Attributes:
        name: The roster key used by ``agent(agent_type=...)``.
        runnable: The leaf runnable; its state must include ``messages``.
        description: Human-readable description.
        needs_execution: Whether this agent type requires an isolated execution
            sandbox (tiered admission) rather than pure in-context reasoning.
        default_model: Default model identifier used as the effective model when an
            ``agent()`` call supplies no ``model`` override. The effective model is
            folded into the journal key and propagated into the leaf config; it is
            honored by config-aware leaves and ignored by leaves whose model is
            bound at construction (in which case it still partitions the cache key).
    """

    name: str
    runnable: Runnable[Any, Any]
    description: str = ""
    needs_execution: bool = False
    default_model: str | None = None


class Roster:
    """A mutable registry mapping agent-type names to leaf runnables."""

    def __init__(self) -> None:
        self._entries: dict[str, RosterEntry] = {}

    def register(
        self,
        name: str,
        runnable: Runnable[Any, Any],
        *,
        description: str = "",
        needs_execution: bool = False,
        default_model: str | None = None,
    ) -> Roster:
        """Register a leaf agent under ``name`` and return ``self`` for chaining."""
        self._entries[name] = RosterEntry(
            name=name,
            runnable=runnable,
            description=description,
            needs_execution=needs_execution,
            default_model=default_model,
        )
        return self

    def resolve(self, name: str) -> RosterEntry:
        """Return the entry for ``name``.

        Raises:
            KeyError: If ``name`` is not registered, listing available names.
        """
        try:
            return self._entries[name]
        except KeyError:
            available = sorted(self._entries)
            raise KeyError(f"unknown agent_type {name!r}; available: {available}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._entries

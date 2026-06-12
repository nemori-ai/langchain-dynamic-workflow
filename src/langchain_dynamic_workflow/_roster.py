"""Roster — the named registry of leaf agents.

A roster entry wraps any runnable whose state schema includes a ``messages``
key (e.g. a ``deepagents.create_deep_agent`` compiled graph, or a langchain
``create_agent`` graph). ``agent()`` calls resolve a leaf by name; there is no
ad-hoc escape hatch (decision R1).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel


@dataclass(frozen=True)
class RosterEntry:
    """A registered leaf agent.

    Exactly one of ``runnable`` (pre-built, schema-less only) or ``builder``
    (constructs a runnable for a given ``response_format``, enabling
    ``agent(schema=...)``) is set.

    Attributes:
        name: The roster key used by ``agent(agent_type=...)``.
        runnable: The pre-built leaf runnable, or ``None`` when a builder is used.
        builder: A factory ``(*, response_format) -> Runnable`` used to construct
            a schema-bound (or schema-less) variant on demand, or ``None``.
        description: Human-readable description.
        needs_execution: Whether this agent type requires an isolated execution
            sandbox (tiered admission) rather than pure in-context reasoning.
        default_model: Default model identifier used as the effective model when
            an ``agent()`` call supplies no ``model`` override. The effective model
            is folded into the journal key and propagated into the leaf config; it
            is honored by config-aware leaves and ignored by leaves whose model is
            bound at construction (in which case it still partitions the cache key).
    """

    name: str
    runnable: Runnable[Any, Any] | None = None
    builder: Callable[..., Runnable[Any, Any]] | None = None
    description: str = ""
    needs_execution: bool = False
    default_model: str | None = None


class Roster:
    """A mutable registry mapping agent-type names to leaf runnables/builders."""

    def __init__(self) -> None:
        self._entries: dict[str, RosterEntry] = {}
        # Process-level cache of built variants keyed by (name, response_format
        # identity). Compiled graphs are stateless across runs, so caching here —
        # next to the builder that owns them — keeps resume cheap and avoids
        # rebuilding per run (decision D-G1b). Concurrency-safe for shared use
        # across runs/threads.
        self._built: dict[tuple[str, str], Runnable[Any, Any]] = {}
        self._build_lock = threading.Lock()

    def register(
        self,
        name: str,
        runnable: Runnable[Any, Any] | None = None,
        *,
        builder: Callable[..., Runnable[Any, Any]] | None = None,
        description: str = "",
        needs_execution: bool = False,
        default_model: str | None = None,
    ) -> Roster:
        """Register a leaf agent under ``name`` and return ``self`` for chaining.

        Provide exactly one of ``runnable`` (pre-built, schema-less) or
        ``builder`` (constructs variants for ``agent(schema=...)``).

        Raises:
            ValueError: If neither or both of ``runnable`` / ``builder`` are given.
        """
        if (runnable is None) == (builder is None):
            raise ValueError(
                f"register({name!r}): provide exactly one of 'runnable' or 'builder' "
                "(a pre-built runnable handles schema-less calls; a builder enables "
                "agent(schema=...))"
            )
        self._entries[name] = RosterEntry(
            name=name,
            runnable=runnable,
            builder=builder,
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

    def runnable_for(self, name: str, *, response_format: Any) -> Runnable[Any, Any]:
        """Return the runnable for ``name`` bound to ``response_format``.

        A ``response_format`` of ``None`` asks for the schema-less variant. A
        builder entry constructs (and caches per response-format identity) the
        variant; a pre-built ``runnable`` entry serves only the schema-less case
        and fails loud if a ``response_format`` is requested.

        Args:
            name: The roster key.
            response_format: The structured-output format to bind, or ``None``.

        Returns:
            The (possibly built and cached) runnable.

        Raises:
            KeyError: If ``name`` is not registered.
            ValueError: If a ``response_format`` is requested for a pre-built
                ``runnable`` entry (no builder to construct a bound variant).
        """
        entry = self.resolve(name)
        if entry.builder is None:
            if response_format is not None:
                raise ValueError(
                    f"agent_type {name!r} was registered with a pre-built runnable and "
                    "cannot produce structured output; register it with a builder "
                    "(builder=lambda *, response_format=None: create_deep_agent(..., "
                    "response_format=response_format)) to use agent(schema=...)"
                )
            assert entry.runnable is not None  # register() guarantees one is set
            return entry.runnable
        cache_key = (name, _response_format_identity(response_format))
        cached = self._built.get(cache_key)
        if cached is not None:
            return cached
        with self._build_lock:
            cached = self._built.get(cache_key)
            if cached is not None:
                return cached
            built = entry.builder(response_format=response_format)
            self._built[cache_key] = built
            return built

    def list_agents(self) -> list[RosterEntry]:
        """Return the registered leaf agents as catalog entries, sorted by name.

        Mirrors the script-level workflow catalog: a host LLM enumerates the
        registered ``agent_type`` names (and their descriptions) to author a script
        without those names being hard-coded in its prompt. The stored
        :class:`RosterEntry` objects are returned as-is, so every field travels with
        the catalog entry.
        """
        return [self._entries[name] for name in sorted(self._entries)]

    def __contains__(self, name: object) -> bool:
        return name in self._entries


def _response_format_identity(response_format: Any) -> str:
    """A stable string identity for a response_format, for the build cache.

    ``None`` (schema-less) maps to a fixed sentinel. A ``ToolStrategy`` over a
    pydantic model is identified by that model's JSON schema; any other value
    falls back to its ``repr``.
    """
    if response_format is None:
        return "__none__"
    schema = getattr(response_format, "schema", None)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        canonical = json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return repr(response_format)

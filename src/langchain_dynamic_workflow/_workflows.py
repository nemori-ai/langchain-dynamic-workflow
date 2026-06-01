"""Named workflow registry — the catalog of runnable orchestration scripts.

A workflow is an async orchestration callable resolved by name, mirroring the
leaf :class:`~langchain_dynamic_workflow._roster.Roster` but at the script level.
Both the host-facing workflow tool (``workflow_tool(run, workflow="...")``) and
the one-level ``ctx.workflow(name, args)`` nesting primitive resolve through this
registry, so a host agent and an orchestration script address workflows the same
way.

The orchestration callable takes the orchestration context plus an arguments
mapping (``async def orchestrate(ctx, args) -> result``); a zero-argument workflow
simply ignores ``args``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ._context import Ctx

WorkflowFn = Callable[[Ctx, dict[str, Any]], Awaitable[Any]]
"""A named workflow: ``async def orchestrate(ctx, args) -> result``."""


class WorkflowRegistry:
    """A mutable registry mapping workflow names to orchestration callables."""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowFn] = {}

    def register(self, name: str, workflow: WorkflowFn) -> WorkflowRegistry:
        """Register ``workflow`` under ``name`` and return ``self`` for chaining."""
        self._workflows[name] = workflow
        return self

    def resolve(self, name: str) -> WorkflowFn:
        """Return the workflow callable registered under ``name``.

        Raises:
            KeyError: If ``name`` is not registered, listing the available names.
        """
        try:
            return self._workflows[name]
        except KeyError:
            available = sorted(self._workflows)
            raise KeyError(f"unknown workflow {name!r}; available: {available}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._workflows

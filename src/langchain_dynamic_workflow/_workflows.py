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
from dataclasses import dataclass
from typing import Any

from ._context import Ctx

WorkflowFn = Callable[[Ctx, dict[str, Any]], Awaitable[Any]]
"""A named workflow: ``async def orchestrate(ctx, args) -> result``."""


@dataclass(frozen=True)
class WorkflowEntry:
    """A registered workflow's catalog entry.

    Attributes:
        name: The registry key used by ``workflow(run, workflow=...)`` and
            ``ctx.workflow(name)``.
        description: A one-line human-readable summary used to render the
            discoverable catalog.
    """

    name: str
    description: str = ""


class WorkflowRegistry:
    """A mutable registry mapping workflow names to orchestration callables."""

    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowFn] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self, name: str, workflow: WorkflowFn, *, description: str = ""
    ) -> WorkflowRegistry:
        """Register ``workflow`` under ``name`` and return ``self`` for chaining.

        Args:
            name: The registry key.
            workflow: The orchestration callable.
            description: A one-line summary for the discoverable catalog. When
                blank, it falls back to the first non-empty line of the callable's
                docstring (or ``""`` when there is no docstring). The stored summary
                is collapsed to a single bounded line so the catalog stays one entry
                per line.

        Returns:
            ``self``, so registrations can be chained.
        """
        self._workflows[name] = workflow
        summary = description.strip() or _docstring_summary(workflow)
        self._descriptions[name] = _one_line_summary(summary)
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

    def list_workflows(self) -> list[WorkflowEntry]:
        """Return the registered workflows as catalog entries, sorted by name."""
        return [
            WorkflowEntry(name=name, description=self._descriptions[name])
            for name in sorted(self._workflows)
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._workflows


def _docstring_summary(workflow: WorkflowFn) -> str:
    """Derive a one-line summary from a workflow callable's docstring.

    Returns the first non-empty, stripped line of ``workflow.__doc__``, or ``""``
    when the callable has no docstring or only blank lines.
    """
    doc = getattr(workflow, "__doc__", None)
    if not isinstance(doc, str):
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


_DESCRIPTION_LIMIT = 200
"""Max characters of a catalog summary, keeping the rendered catalog compact."""


def _one_line_summary(text: str) -> str:
    """Collapse a summary to a single, length-bounded line for the catalog.

    Any run of whitespace (including newlines and tabs) becomes a single space so a
    multi-line description cannot inject extra catalog lines; the result is truncated
    to ``_DESCRIPTION_LIMIT`` characters (with an ellipsis) so one entry stays compact.

    Args:
        text: The raw summary (an explicit description or a docstring-derived line).

    Returns:
        The normalized one-line summary.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) > _DESCRIPTION_LIMIT:
        return collapsed[: _DESCRIPTION_LIMIT - 1].rstrip() + "…"
    return collapsed

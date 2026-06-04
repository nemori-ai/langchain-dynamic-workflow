"""Host-facing run registry abstraction for workflow launches and resumes.

This module hosts the ``WorkflowRunStore`` protocol and its in-memory default,
which back the workflow tool's run registry. The persistent sqlite-backed
implementation lives in the sibling ``_persistence`` module.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._engine import InMemoryJournalStore, JournalStore


@dataclass(frozen=True, slots=True)
class RunSpec:
    """An immutable description of a launched workflow run.

    Carries everything needed to replay a launch on resume, including across a
    process restart: the journal that delivers zero-cost replay is keyed by
    ``run_id`` elsewhere, while this record preserves the label shown in run
    listings and the LangGraph ``thread_id`` so the resumed run rejoins its
    original checkpoint thread.

    Attributes:
        kind: How to resolve the workflow callable: ``"name"`` for a registered
            workflow looked up by name, or ``"script"`` for an authored
            orchestration source compiled on demand.
        name_or_source: The registered workflow name when ``kind == "name"``,
            otherwise the orchestration script source when ``kind == "script"``.
        args: The keyword arguments passed to the workflow at launch.
        label: The human-readable label surfaced in run listings.
        thread_id: The LangGraph thread id this run was launched on; replayed on
            resume so the run rejoins its original checkpoint thread.
    """

    kind: str
    name_or_source: str
    args: dict[str, Any]
    label: str
    thread_id: str


@runtime_checkable
class WorkflowRunStore(Protocol):
    """Persistence boundary for the workflow tool's run registry.

    Implementations map a ``run_id`` to both its launch ``RunSpec`` (so a resume
    can rebuild the original workflow callable, label, and thread) and its
    per-run journal (so completed leaves replay for free). The in-memory default
    keeps the base install dependency-free; the sqlite-backed implementation
    extends durability across process restarts.
    """

    async def save_spec(self, run_id: str, spec: RunSpec) -> None:
        """Persist the launch spec for ``run_id``.

        Args:
            run_id: The unique identifier of the launched run.
            spec: The launch description to persist.
        """
        ...

    async def load_spec(self, run_id: str) -> RunSpec | None:
        """Return the launch spec for ``run_id``, or ``None`` on miss.

        Args:
            run_id: The identifier of a previously launched run.

        Returns:
            The persisted launch spec, or ``None`` if no run was saved under
            ``run_id``.
        """
        ...

    def journal_for(self, run_id: str) -> JournalStore:
        """Return the per-run journal view for ``run_id``.

        This is synchronous: the workflow tool wires the returned journal into a
        launch synchronously, before the run's coroutine starts. A given
        ``run_id`` must always map to the same logical journal so a resume
        replays the leaves a prior run recorded.

        Args:
            run_id: The identifier of the run whose journal is requested.

        Returns:
            The journal store scoped to ``run_id``.
        """
        ...


class InMemoryRunStore:
    """Dependency-free run registry backed by in-process dictionaries.

    This is the default store: specs live in a dict and each run gets exactly
    one cached :class:`InMemoryJournalStore`. Repeated ``journal_for`` calls for
    the same ``run_id`` return the identical instance so a same-session resume
    reuses the journal the original run populated. State is lost on process exit;
    the sqlite-backed store in ``_persistence`` extends durability.
    """

    def __init__(self) -> None:
        self._specs: dict[str, RunSpec] = {}
        self._journals: dict[str, JournalStore] = {}

    async def save_spec(self, run_id: str, spec: RunSpec) -> None:
        """Persist the launch spec for ``run_id`` in the in-process registry.

        Args:
            run_id: The unique identifier of the launched run.
            spec: The launch description to persist.
        """
        self._specs[run_id] = spec

    async def load_spec(self, run_id: str) -> RunSpec | None:
        """Return the launch spec for ``run_id``, or ``None`` on miss.

        Args:
            run_id: The identifier of a previously launched run.

        Returns:
            The persisted launch spec, or ``None`` if none was saved.
        """
        return self._specs.get(run_id)

    def journal_for(self, run_id: str) -> JournalStore:
        """Return the cached journal for ``run_id``, creating it on first use.

        Args:
            run_id: The identifier of the run whose journal is requested.

        Returns:
            The single :class:`InMemoryJournalStore` bound to ``run_id``; the
            same instance is returned on every subsequent call.
        """
        journal = self._journals.get(run_id)
        if journal is None:
            journal = InMemoryJournalStore()
            self._journals[run_id] = journal
        return journal

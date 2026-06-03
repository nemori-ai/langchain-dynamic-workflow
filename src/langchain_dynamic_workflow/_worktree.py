"""Worktree isolation backends — seed a leaf's working copy and collect its changes.

``isolation="worktree"`` gives each file-mutating leaf its own copy of a base
snapshot to edit; the changeset (relative to the seed) is collected afterward so the
orchestration layer can review and merge patches — separating *generating* a change
from *applying* it.

:class:`InMemoryWorktreeProvider` is the offline default. The :class:`WorktreeProvider`
protocol is the seam for a production backend: a real git-worktree provider would
``git worktree add`` a fresh tree per leaf (rooted on a deepagents filesystem
backend so the leaf can run real ``git`` / build / test), then ``git diff`` that
tree for :meth:`WorktreeProvider.collect`. It implements the same two methods and
plugs in wherever the in-memory provider does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorktreeProvider(Protocol):
    """Seeds a worktree leaf's sandbox and collects its changeset versus the seed."""

    def seed(self, leaf_id: str) -> Mapping[str, str]:
        """Return the files to populate a worktree leaf's sandbox with before it runs.

        Args:
            leaf_id: The leaf's stable derived identity.

        Returns:
            A mapping of absolute path -> file content to seed the sandbox with.
        """
        ...

    def collect(self, leaf_id: str, files: Mapping[str, str]) -> dict[str, str]:
        """Return the leaf's changeset: paths added or modified relative to the seed.

        Args:
            leaf_id: The leaf's stable derived identity.
            files: The leaf's sandbox file set after it ran.

        Returns:
            A mapping of absolute path -> new content for every path that differs
            from the seed (added or modified); unchanged paths are omitted.
        """
        ...


class InMemoryWorktreeProvider:
    """A worktree provider backed by an in-memory base snapshot (offline default).

    Seeds every worktree leaf with an isolated copy of the base files and computes a
    leaf's changeset by comparing its final file set against that base. Holds no
    per-leaf state, so distinct leaves are naturally isolated: each ``seed`` hands
    back a fresh copy the leaf can mutate freely.
    """

    def __init__(self, base_files: Mapping[str, str]) -> None:
        # Copy on construction so a later mutation of the caller's dict cannot change
        # the base snapshot every worktree leaf is seeded from.
        self._base: dict[str, str] = dict(base_files)

    def seed(self, leaf_id: str) -> Mapping[str, str]:
        # A fresh copy per leaf; the caller may mutate it without touching the base.
        return dict(self._base)

    def collect(self, leaf_id: str, files: Mapping[str, str]) -> dict[str, str]:
        # Deterministic order (sorted) so the changeset is stable run to run.
        return {
            path: content
            for path, content in sorted(files.items())
            if self._base.get(path) != content
        }

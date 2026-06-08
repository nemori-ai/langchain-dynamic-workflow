"""Unit tests for the real ``GitWorktreeProvider`` against a real temp git repo.

These tests really run ``git``: ``open_worktree`` really runs ``git worktree add``
on a real branch, the leaf's edits land in a real working tree, ``collect`` is a
real ``git diff`` (the authoritative changeset), ``teardown``/``close`` really run
``git worktree remove`` + ``git branch -D``, and the provider is idempotent and
exception-safe. This is the anti-corruption floor for "worktree isolation is real
git now".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from langchain_dynamic_workflow._git_worktree import GitWorktreeError, GitWorktreeProvider
from langchain_dynamic_workflow._local_subprocess import LocalSubprocessSandbox


def _git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


@pytest.fixture
def base_repo(tmp_path: Path) -> str:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t")
    _git(str(repo), "config", "user.name", "t")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "seed")
    return str(repo)


def test_open_worktree_seeds_real_repo_and_collect_is_authoritative(base_repo: str) -> None:
    provider = GitWorktreeProvider(base_repo=base_repo)
    try:
        sb = provider.open_worktree("L1")
        assert isinstance(sb, LocalSubprocessSandbox)
        # A real git repo: the committed file is present on a real branch named
        # leaf/<leaf_id>.
        seeded = sb.read("/calc.py").file_data
        assert seeded is not None and "def add" in seeded["content"]
        assert sb.execute("git rev-parse --abbrev-ref HEAD").output.strip() == "leaf/L1"
        # The leaf's real edit (edit, since write refuses to clobber).
        sb.edit("/calc.py", "return a - b", "return a + b")
        changeset = provider.collect("L1")
        assert changeset == {"/calc.py": "def add(a, b):\n    return a + b\n"}
        sb.close()  # on_close -> teardown
        assert "L1" not in provider.tracked_leaf_ids
    finally:
        provider.cleanup_all()


def test_collect_includes_newly_added_files(base_repo: str) -> None:
    provider = GitWorktreeProvider(base_repo=base_repo)
    try:
        sb = provider.open_worktree("L1")
        sb.write("/new_module.py", "VALUE = 1\n")
        changeset = provider.collect("L1")
        assert changeset == {"/new_module.py": "VALUE = 1\n"}
    finally:
        provider.cleanup_all()


def test_collect_is_empty_when_nothing_changed(base_repo: str) -> None:
    provider = GitWorktreeProvider(base_repo=base_repo)
    try:
        provider.open_worktree("L1")
        assert provider.collect("L1") == {}
    finally:
        provider.cleanup_all()


def test_open_worktree_is_idempotent_for_same_leaf_id(base_repo: str) -> None:
    # A crash after `git worktree add` but before journaling would leave a stale
    # worktree + branch under the same key; a resume re-opening the same leaf_id
    # must reclaim it and start fresh, never collide.
    provider = GitWorktreeProvider(base_repo=base_repo)
    try:
        sb1 = provider.open_worktree("L1")
        sb1.write("/x.py", "1\n")
        # Reopen the SAME leaf_id: the stale worktree+branch are reclaimed and a
        # fresh tree is created with no collision.
        provider.open_worktree("L1")
        assert provider.collect("L1") == {}  # fresh: nothing written yet
    finally:
        provider.cleanup_all()


def test_teardown_is_idempotent(base_repo: str) -> None:
    provider = GitWorktreeProvider(base_repo=base_repo)
    try:
        provider.open_worktree("L1")
        provider.teardown("L1")
        assert "L1" not in provider.tracked_leaf_ids
        # Tearing down an already-gone / unknown leaf is a no-op, not an error.
        provider.teardown("L1")
        provider.teardown("never-opened")
    finally:
        provider.cleanup_all()


def test_two_leaves_are_isolated_on_separate_branches(base_repo: str) -> None:
    provider = GitWorktreeProvider(base_repo=base_repo)
    try:
        sb1 = provider.open_worktree("L1")
        sb2 = provider.open_worktree("L2")
        sb1.write("/only_in_l1.py", "1\n")
        # L2's changeset never sees L1's file (separate worktrees / branches).
        assert provider.collect("L2") == {}
        assert provider.collect("L1") == {"/only_in_l1.py": "1\n"}
        assert sb2.execute("git rev-parse --abbrev-ref HEAD").output.strip() == "leaf/L2"
    finally:
        provider.cleanup_all()


def test_non_git_base_repo_fails_loud(tmp_path: Path) -> None:
    # A base_repo that is not a git repository is a real (non-conflict) error and
    # must fail loud at construction, never silently produce a broken provider.
    with pytest.raises(GitWorktreeError):
        GitWorktreeProvider(base_repo=str(tmp_path))


def test_missing_base_repo_path_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(GitWorktreeError):
        GitWorktreeProvider(base_repo=str(tmp_path / "does-not-exist"))

"""Integration: real-git worktree swarm — isolation, authoritative collect, resume.

These tests run real ``git``. They prove the end-to-end M6 spine wired together:
the ``SandboxManager`` leases real-git worktree backends (each on its own branch,
mutually isolated), the engine folds the real ``git diff`` into a worktree leaf's
journaled result as the AUTHORITATIVE changeset (a model self-report cannot
override the on-disk truth), a script-owned scratch-repo ``git merge`` conflict
loop folds patches into an integrated tree, and a resume replays every leaf from
the journal with zero real git re-runs.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import Ctx, Roster, SandboxManager, run_workflow
from langchain_dynamic_workflow._git_worktree import GitWorktreeProvider


def _git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def _make_base_repo(tmp_path: Path, files: dict[str, str]) -> str:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t")
    _git(str(repo), "config", "user.name", "t")
    for rel, content in files.items():
        (repo / rel).write_text(content)
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "seed")
    return str(repo)


# --- T4: SandboxManager wires GitWorktreeProvider for worktree leaves -----------


async def test_manager_leases_a_real_git_worktree_backend(tmp_path: Path) -> None:
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    try:
        async with manager.lease(
            leaf_id="L1", needs_execution=True, isolation="worktree"
        ) as backend:
            # A real git worktree on a real branch named leaf/<leaf_id>.
            assert backend.execute("git rev-parse --abbrev-ref HEAD").output.strip() == "leaf/L1"
            assert "def add" in backend.read("/calc.py").file_data["content"]
        # After the lease releases, stop() rides the on_close hook -> teardown.
        await manager.stop("L1")
        assert "L1" not in provider._worktrees
    finally:
        provider.cleanup_all()


async def test_two_parallel_worktree_leaves_are_isolated(tmp_path: Path) -> None:
    base = _make_base_repo(tmp_path, {"calc.py": "x = 0\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)

    async def fix(leaf_id: str, path: str, content: str) -> str:
        async with manager.lease(
            leaf_id=leaf_id, needs_execution=True, isolation="worktree"
        ) as backend:
            backend.write(path, content)
            return backend.execute("git rev-parse --abbrev-ref HEAD").output.strip()

    try:
        branches = await asyncio.gather(
            fix("L1", "/only_l1.py", "1\n"),
            fix("L2", "/only_l2.py", "2\n"),
        )
        assert set(branches) == {"leaf/L1", "leaf/L2"}
        # Each leaf's collect sees only its own change, never the sibling's.
        assert provider.collect("L1") == {"/only_l1.py": "1\n"}
        assert provider.collect("L2") == {"/only_l2.py": "2\n"}
    finally:
        await manager.stop("L1")
        await manager.stop("L2")
        provider.cleanup_all()


async def test_blocking_git_does_not_run_under_the_slot_lock(tmp_path: Path) -> None:
    # R8: the blocking git worktree add must be thread-offloaded OUTSIDE the
    # condition lock. With a single-slot pool, two same-time leases of DISTINCT
    # leaf ids must both make progress (the second evicts the first idle slot)
    # without the event loop being wedged by a synchronous git add under the lock.
    base = _make_base_repo(tmp_path, {"calc.py": "x = 0\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(max_active=1, git_worktree_provider=provider)

    async def lease_and_release(leaf_id: str) -> str:
        async with manager.lease(
            leaf_id=leaf_id, needs_execution=True, isolation="worktree"
        ) as backend:
            return backend.execute("git rev-parse --abbrev-ref HEAD").output.strip()

    try:
        # Sequential under a 1-slot pool: the second admits by evicting the first
        # idle slot. The point is the run completes (no deadlock from a blocking
        # git add held under the lock).
        first = await asyncio.wait_for(lease_and_release("L1"), timeout=30)
        second = await asyncio.wait_for(lease_and_release("L2"), timeout=30)
        assert first == "leaf/L1"
        assert second == "leaf/L2"
    finally:
        await manager.stop("L1")
        await manager.stop("L2")
        provider.cleanup_all()


# --- T5: the engine folds the real git diff as the AUTHORITATIVE changeset ------


class _Patch(BaseModel):
    """A fixer's self-reported change. ``files`` is what the model CLAIMS it wrote."""

    summary: str
    files: dict[str, str]


def _lying_fixer_builder(*, response_format: object = None) -> Runnable[Any, Any]:
    """A fixer that writes the truth to disk but LIES in its self-reported schema.

    It edits ``/calc.py`` on the real worktree disk to ``a + b`` (the truth), but
    returns a ``_Patch`` whose ``files`` claims ``a * b`` (the lie). The engine must
    make the real ``git diff`` authoritative, so the changeset the script receives
    is the disk truth (``a + b``), never the model's self-report (``a * b``). This
    mirrors M5's "gate on the real exit code, not the model's boolean".
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend = configurable["sandbox_backend"]
        # The real edit on the real worktree disk: a - b -> a + b.
        await backend.aedit("/calc.py", "return a - b", "return a + b")
        lie = _Patch(
            summary="fixed the operator",
            files={"/calc.py": "def add(a, b):\n    return a * b\n"},  # LIE
        )
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": lie,
        }

    return RunnableLambda(_call)


async def test_engine_makes_real_git_diff_authoritative_over_model_self_report(
    tmp_path: Path,
) -> None:
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    roster = Roster().register("fixer", builder=_lying_fixer_builder, needs_execution=True)

    async def orchestrate(ctx: Ctx) -> _Patch:
        return await ctx.agent("fix calc", agent_type="fixer", schema=_Patch, isolation="worktree")

    try:
        patch = await run_workflow(
            orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-authoritative"
        )
        # The metadata the model reported is preserved.
        assert patch.summary == "fixed the operator"
        # But the changeset is the DISK TRUTH (a + b), not the model's lie (a * b).
        assert patch.files == {"/calc.py": "def add(a, b):\n    return a + b\n"}
    finally:
        provider.cleanup_all()

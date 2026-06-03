"""Integration: SandboxManager seeds and isolates worktree leaves.

``isolation="worktree"`` leases must come pre-seeded from the base snapshot, each
leaf must get its own copy (one leaf's writes are invisible to another), and a
plain ``"shared"`` lease must stay empty (no seed) — the existing behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    InMemoryJournalStore,
    InMemoryWorktreeProvider,
    Roster,
    SandboxManager,
    run_workflow,
)

_BASE = {"/a.py": "print(1)\n", "/b.py": "x = 2\n"}


async def test_worktree_lease_seeds_from_base() -> None:
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as backend:
        a = backend.read("/a.py").file_data
        b = backend.read("/b.py").file_data
        assert a is not None and a["content"] == "print(1)\n"
        assert b is not None and b["content"] == "x = 2\n"


async def test_two_worktree_leaves_are_isolated() -> None:
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))

    # Leaf L1 adds a scratch file to its own seeded copy.
    async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as b1:
        b1.write("/scratch.py", "L1 only")

    # Leaf L2 starts from the same base but never sees L1's write.
    async with manager.lease(leaf_id="L2", needs_execution=True, isolation="worktree") as b2:
        seeded = b2.read("/a.py").file_data
        assert seeded is not None and seeded["content"] == "print(1)\n"  # seeded
        assert b2.read("/scratch.py").error  # L1's write is invisible


async def test_shared_lease_is_not_seeded() -> None:
    # The default "shared" isolation keeps the prior behavior: an empty sandbox.
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    async with manager.lease(leaf_id="L3", needs_execution=True, isolation="shared") as backend:
        assert backend.read("/a.py").error  # not seeded


class _CountingWorktreeProvider:
    """A worktree provider that tallies seed() calls (to prove no re-seeding)."""

    def __init__(self, base: dict[str, str]) -> None:
        self._base = dict(base)
        self.seed_calls = 0

    def seed(self, leaf_id: str) -> Mapping[str, str]:
        self.seed_calls += 1
        return dict(self._base)

    def collect(self, leaf_id: str, files: Mapping[str, str]) -> dict[str, str]:
        return {}


async def test_worktree_seeds_once_across_find_or_create_reuse() -> None:
    # Leasing the same un-evicted leaf_id twice (a retry) reuses the live sandbox and
    # must NOT re-seed — re-seeding would clobber a leaf's in-progress edits.
    provider = _CountingWorktreeProvider(_BASE)
    manager = SandboxManager(worktree_provider=provider)
    async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as b1:
        b1.write("/scratch.py", "in progress")
    async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as b2:
        assert not b2.read("/scratch.py").error  # the reused workspace is preserved
    assert provider.seed_calls == 1  # seeded once on create, never again on reuse


async def test_worktree_without_provider_is_not_seeded() -> None:
    # isolation="worktree" with no provider configured falls back to an empty
    # sandbox (no seeding source), never raising.
    manager = SandboxManager()
    async with manager.lease(leaf_id="L4", needs_execution=True, isolation="worktree") as backend:
        assert backend.read("/a.py").error


def _seed_observing_fixer(seen: dict[str, str]) -> RunnableLambda[Any, Any]:
    """An execution leaf that records the seeded content it sees in its sandbox."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        backend = (config or {}).get("configurable", {})["sandbox_backend"]
        seen["a"] = backend.read("/a.py").file_data["content"]
        return {"messages": [*inp["messages"], AIMessage(content="fixed")]}

    return RunnableLambda(_leaf)


async def test_engine_threads_isolation_so_worktree_leaf_is_seeded() -> None:
    # End-to-end on the real run_workflow path: an execution leaf asked for
    # isolation="worktree" must receive a sandbox seeded from the base (proving the
    # engine threads isolation through leaf_task -> lease, not just the manager API).
    seen: dict[str, str] = {}
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    roster = Roster().register("fixer", _seed_observing_fixer(seen), needs_execution=True)

    async def orchestrate(ctx: Any) -> Any:
        return await ctx.agent("fix /a.py", agent_type="fixer", isolation="worktree")

    await run_workflow(orchestrate, roster=roster, sandbox_manager=manager)

    assert seen["a"] == "print(1)\n"  # the leaf ran against the seeded base


async def test_worktree_on_reasoning_leaf_fails_loud() -> None:
    # isolation="worktree" needs a sandbox to seed into; a reasoning leaf
    # (needs_execution=False) has none, so requesting a worktree must fail loud
    # rather than silently hand back an empty StateBackend keyed as "worktree".
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    roster = Roster().register("thinker", RunnableLambda(lambda inp: inp))  # needs_execution=False

    async def orchestrate(ctx: Any) -> Any:
        return await ctx.agent("think", agent_type="thinker", isolation="worktree")

    with pytest.raises(ValueError, match=r"worktree|needs_execution"):
        await run_workflow(orchestrate, roster=roster, sandbox_manager=manager)


async def test_worktree_leaf_resume_hits_cache() -> None:
    # isolation is part of the journal key, so a worktree leaf's identity is stable
    # across runs: a second run replays from the journal without re-running the leaf.
    calls = {"n": 0}
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    journal = InMemoryJournalStore()

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        calls["n"] += 1
        return {"messages": [*inp["messages"], AIMessage(content="done")]}

    roster = Roster().register("fixer", RunnableLambda(_leaf), needs_execution=True)

    async def orchestrate(ctx: Any) -> Any:
        return await ctx.agent("fix it", agent_type="fixer", isolation="worktree")

    first = await run_workflow(orchestrate, roster=roster, sandbox_manager=manager, journal=journal)
    second = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=manager, journal=journal
    )
    assert first == second == "done"
    assert calls["n"] == 1  # second run replayed from the journal, leaf ran once

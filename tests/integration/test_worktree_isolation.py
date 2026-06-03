"""Integration: SandboxManager seeds and isolates worktree leaves.

``isolation="worktree"`` leases must come pre-seeded from the base snapshot, each
leaf must get its own copy (one leaf's writes are invisible to another), and a
plain ``"shared"`` lease must stay empty (no seed) — the existing behavior.
"""

from __future__ import annotations

from langchain_dynamic_workflow import InMemoryWorktreeProvider, SandboxManager

_BASE = {"/a.py": "print(1)\n", "/b.py": "x = 2\n"}


async def test_worktree_lease_seeds_from_base() -> None:
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as backend:
        assert backend.read("/a.py").file_data["content"] == "print(1)\n"
        assert backend.read("/b.py").file_data["content"] == "x = 2\n"


async def test_two_worktree_leaves_are_isolated() -> None:
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))

    # Leaf L1 adds a scratch file to its own seeded copy.
    async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as b1:
        b1.write("/scratch.py", "L1 only")

    # Leaf L2 starts from the same base but never sees L1's write.
    async with manager.lease(leaf_id="L2", needs_execution=True, isolation="worktree") as b2:
        assert b2.read("/a.py").file_data["content"] == "print(1)\n"  # seeded
        assert b2.read("/scratch.py").error  # L1's write is invisible


async def test_shared_lease_is_not_seeded() -> None:
    # The default "shared" isolation keeps the prior behavior: an empty sandbox.
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(_BASE))
    async with manager.lease(leaf_id="L3", needs_execution=True, isolation="shared") as backend:
        assert backend.read("/a.py").error  # not seeded


async def test_worktree_without_provider_is_not_seeded() -> None:
    # isolation="worktree" with no provider configured falls back to an empty
    # sandbox (no seeding source), never raising.
    manager = SandboxManager()
    async with manager.lease(leaf_id="L4", needs_execution=True, isolation="worktree") as backend:
        assert backend.read("/a.py").error

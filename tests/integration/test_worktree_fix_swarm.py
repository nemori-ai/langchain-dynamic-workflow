"""Phase G2 integration: the worktree fix-swarm example runs end to end.

Loads ``examples.features.worktree`` and drives its ``fix_swarm`` workflow
through ``run_workflow`` with a real SandboxManager wired with an
InMemoryWorktreeProvider, using the example's OWN offline fixer/reviewer leaves (so
the real example code is exercised, not a duplicate fake). Pins the swarm shape —
one isolated worktree fixer per target, a 2-vote review — and that each fixer's
patch actually fixes its file's bug (read from its seeded worktree).
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from langchain_dynamic_workflow import (
    InMemoryWorktreeProvider,
    Roster,
    SandboxManager,
    run_workflow,
)


def _load_example() -> ModuleType:
    """Import the worktree fix-swarm feature demo as a module."""
    return importlib.import_module("examples.features.worktree")


async def test_fix_swarm_produces_one_correct_isolated_patch_per_target() -> None:
    module = _load_example()
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(module.BASE_REPO))
    # Use the example's real offline leaves (real_model() is None without the env var).
    roster = (
        Roster()
        .register("fixer", builder=module._fixer_builder, needs_execution=True)
        .register("reviewer", builder=module._reviewer_builder)
    )
    targets = sorted(module.BASE_REPO)

    async def orchestrate(ctx: Any) -> Any:
        return await module.fix_swarm(ctx, {"targets": targets})

    approved = await run_workflow(orchestrate, roster=roster, sandbox_manager=manager)

    # One approved patch per target, each scoped to its own file.
    assert sorted(p["summary"] for p in approved) == sorted(f"fixed {t}" for t in targets)
    by_path = {f["path"]: f["new_content"] for patch in approved for f in patch["files"]}
    assert set(by_path) == set(targets)

    # Each patch actually fixes its file's bug (not just strips a comment).
    assert "a + b" in by_path["/calc.py"] and "a - b" not in by_path["/calc.py"]
    assert "s.upper()" in by_path["/strutil.py"] and "s.lower()" not in by_path["/strutil.py"]

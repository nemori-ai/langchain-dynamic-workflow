"""Phase G2 integration: the worktree fix-swarm example runs end to end.

Loads ``examples/10_worktree_fix_swarm.py`` and drives its ``fix_swarm`` workflow
through ``run_workflow`` with deterministic fakes + a real SandboxManager wired with
an InMemoryWorktreeProvider. Pins the swarm shape — one isolated worktree fixer per
target file (each seeded from the base), a Patch per fixer, and a 2-vote review —
and that each fixer ran against a seeded worktree (not an empty sandbox).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    InMemoryWorktreeProvider,
    Roster,
    SandboxManager,
    run_workflow,
)


def _load_example() -> ModuleType:
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "10_worktree_fix_swarm.py"
    spec = importlib.util.spec_from_file_location("_ldw_fix_swarm_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register for nested-model forward refs
    spec.loader.exec_module(module)
    return module


def _fixer_builder(module: ModuleType, seeded_paths: dict[str, list[str]]) -> Any:
    """Fake fixer: records the files it saw seeded, returns a Patch for its target."""

    def builder(*, response_format: Any = None) -> Any:
        assert response_format is not None
        schema = response_format.schema

        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            prompt = inp["messages"][-1].text
            backend = (config or {}).get("configurable", {})["sandbox_backend"]
            target = next(p for p in module.BASE_REPO if p in prompt)
            # Record the seeded file set this fixer's isolated worktree exposes.
            seeded_paths[target] = sorted(p for p in module.BASE_REPO if not backend.read(p).error)
            patch = schema.model_validate(
                {
                    "summary": f"fixed {target}",
                    "files": [{"path": target, "new_content": "# fixed\n"}],
                }
            )
            return {
                "messages": [*inp["messages"], AIMessage(content="patched")],
                "structured_response": patch,
            }

        return RunnableLambda(_leaf)

    return builder


def _reviewer_builder() -> Any:
    """Fake reviewer: always approves."""

    def builder(*, response_format: Any = None) -> Any:
        assert response_format is not None
        schema = response_format.schema

        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            vote = schema.model_validate({"approve": True, "reason": "looks correct"})
            return {
                "messages": [*inp["messages"], AIMessage(content="reviewed")],
                "structured_response": vote,
            }

        return RunnableLambda(_leaf)

    return builder


async def test_fix_swarm_produces_one_isolated_patch_per_target() -> None:
    module = _load_example()
    seeded_paths: dict[str, list[str]] = {}
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(module.BASE_REPO))
    roster = (
        Roster()
        .register("fixer", builder=_fixer_builder(module, seeded_paths), needs_execution=True)
        .register("reviewer", builder=_reviewer_builder())
    )
    targets = sorted(module.BASE_REPO)

    async def orchestrate(ctx: Any) -> Any:
        return await module.fix_swarm(ctx, {"targets": targets})

    approved = await run_workflow(orchestrate, roster=roster, sandbox_manager=manager)

    # One approved patch per target, each scoped to its own file.
    assert sorted(p["summary"] for p in approved) == sorted(f"fixed {t}" for t in targets)
    for patch in approved:
        assert len(patch["files"]) == 1

    # Every fixer ran in a worktree seeded with the full base repo (isolation +
    # seeding held on the real parallel engine path).
    assert len(seeded_paths) == len(targets)
    for seen in seeded_paths.values():
        assert seen == sorted(module.BASE_REPO)

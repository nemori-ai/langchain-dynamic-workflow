"""Phase G2 demo: a parallel fix swarm over isolated worktrees.

Mirrors the shape of Claude Code's flagship migration workflow (one fixer per file,
each in its own git worktree, returning a patch the orchestration reviews):

    fan out one fixer per target file
      -> each runs in its OWN worktree, seeded from the base repo (isolation="worktree")
      -> each returns a Patch (schema-as-handoff), separating "generate" from "apply"
      -> 2-vote review per patch -> keep the approved patches

Each fixer is leased a sandbox seeded with an isolated copy of the base repo, so the
fixers cannot see one another's edits; the patch each returns is the unit the
orchestration layer reviews and would merge.

Run it:

    uv sync --group example
    # credentials + model come from a local .env (OPENROUTER_API_KEY); see _demo_models
    export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
    uv run python examples/10_worktree_fix_swarm.py

With ``LDW_DEMO_REAL_MODEL`` unset the demo runs fully offline: deterministic fake
fixers read their seeded worktree and return a patch, with no API key (the path the
integration test pins).
"""

from __future__ import annotations

import asyncio
from typing import Any

from _demo_models import demo_cache_middleware, load_demo_env, real_leaf_model, real_model
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    InMemoryWorktreeProvider,
    Roster,
    SandboxManager,
    run_workflow,
)

REVIEWERS_PER_PATCH = 2
APPROVALS_TO_KEEP = 2

# A tiny repo with one genuine bug per file; each fixer gets its own seeded copy.
BASE_REPO: dict[str, str] = {
    "/calc.py": "def add(a, b):\n    return a - b  # bug: should be a + b\n",
    "/strutil.py": "def shout(s):\n    return s.lower() + '!'  # bug: should be s.upper()\n",
}


# ── structured leaf contracts (schema-as-handoff) ─────────────────────────────


class FilePatch(BaseModel):
    """One file's replacement content in a patch."""

    path: str
    new_content: str


class Patch(BaseModel):
    """A fixer's proposed change to one file, returned for review."""

    summary: str
    files: list[FilePatch]


class Vote(BaseModel):
    """One reviewer's ruling on a patch."""

    approve: bool
    reason: str


# ── the workflow ──────────────────────────────────────────────────────────────


def _fix_prompt(target: str) -> str:
    return (
        f"The repository has a bug in {target}. Read the file in your worktree, fix "
        f"ONLY {target}, and return a patch: a short summary plus the file's full "
        f"corrected new_content.\n\n{target}:\n{BASE_REPO[target]}"
    )


def _review_prompt(voter: int, patch: Patch) -> str:
    changes = "\n".join(
        f"--- {f.path} ---\nBEFORE:\n{BASE_REPO.get(f.path, '(new file)')}\nAFTER:\n{f.new_content}"
        for f in patch.files
    )
    return (
        f"Reviewer #{voter + 1}: comparing BEFORE and AFTER, does this patch correctly "
        f"fix a real bug and touch only the intended file? Approve only if so.\n"
        f"Summary: {patch.summary}\n{changes}"
    )


async def fix_swarm(ctx: Any, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Fan out one isolated-worktree fixer per target, then 2-vote review each patch.

    Each fixer returns a ``Patch`` (the generated change); the orchestration layer
    reviews and keeps the approved ones — generation and application stay separate.
    """
    targets: list[str] = sorted(args["targets"])

    ctx.phase("fix")
    patches = await ctx.parallel(
        [
            lambda t=t: ctx.agent(
                _fix_prompt(t), agent_type="fixer", isolation="worktree", schema=Patch
            )
            for t in targets
        ]
    )
    collected = [patch for patch in patches if patch is not None]
    ctx.log(f"collected {len(collected)}/{len(targets)} patches")

    ctx.phase("review")
    approved: list[dict[str, Any]] = []
    for patch in collected:
        votes = await ctx.parallel(
            [
                lambda p=patch, v=v: ctx.agent(
                    _review_prompt(v, p), agent_type="reviewer", schema=Vote
                )
                for v in range(REVIEWERS_PER_PATCH)
            ]
        )
        approvals = sum(1 for vote in votes if vote is not None and vote.approve)
        kept = approvals >= APPROVALS_TO_KEEP
        mark = "kept" if kept else "dropped"
        ctx.log(f"patch {mark} ({approvals}/{REVIEWERS_PER_PATCH}): {patch.summary}")
        if kept:
            approved.append(
                {
                    "summary": patch.summary,
                    "files": [{"path": f.path, "new_content": f.new_content} for f in patch.files],
                }
            )
    return approved


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fixer_builder(*, response_format: Any = None) -> Any:
    """Fixer leaf: reads its seeded worktree, returns a Patch for its target file."""
    model = real_leaf_model()
    if model is not None:
        return create_deep_agent(
            model=model, response_format=response_format, middleware=demo_cache_middleware()
        )
    schema: Any = response_format.schema if response_format is not None else Patch

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text
        backend = (config or {}).get("configurable", {})["sandbox_backend"]
        target = next(path for path in BASE_REPO if path in prompt)
        original = backend.read(target).file_data  # proves the worktree was seeded
        base = original["content"] if original is not None else ""
        # Apply the actual fix (and drop the now-resolved bug comment), line by line.
        fixed_lines = []
        for line in base.splitlines():
            line = line.replace("a - b", "a + b").replace("s.lower()", "s.upper()")
            fixed_lines.append(line.split("  # bug")[0].rstrip())
        fixed = "\n".join(fixed_lines) + "\n"
        patch = schema.model_validate(
            {"summary": f"fixed {target}", "files": [{"path": target, "new_content": fixed}]}
        )
        return {
            "messages": [*inp["messages"], AIMessage(content="patched")],
            "structured_response": patch,
        }

    return RunnableLambda(_leaf)


def _reviewer_builder(*, response_format: Any = None) -> Any:
    """Reviewer leaf: approves a patch (offline always approves)."""
    model = real_leaf_model()
    if model is not None:
        return create_deep_agent(
            model=model, response_format=response_format, middleware=demo_cache_middleware()
        )
    schema: Any = response_format.schema if response_format is not None else Vote

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        vote = schema.model_validate({"approve": True, "reason": "patch fixes the stated bug"})
        return {
            "messages": [*inp["messages"], AIMessage(content="reviewed")],
            "structured_response": vote,
        }

    return RunnableLambda(_leaf)


# ── driver ───────────────────────────────────────────────────────────────────


async def main() -> None:
    load_demo_env()
    mode = "REAL (OpenRouter)" if real_model() is not None else "offline (fake)"
    manager = SandboxManager(worktree_provider=InMemoryWorktreeProvider(BASE_REPO))
    roster = (
        Roster()
        .register(
            "fixer",
            builder=_fixer_builder,
            description="Fixes one file in its worktree",
            needs_execution=True,
        )
        .register("reviewer", builder=_reviewer_builder, description="Reviews a patch")
    )

    async def orchestrate(ctx: Any) -> list[dict[str, Any]]:
        return await fix_swarm(ctx, {"targets": sorted(BASE_REPO)})

    print(f"mode: {mode}")
    approved = await run_workflow(orchestrate, roster=roster, sandbox_manager=manager)
    print(f"approved patches ({len(approved)}):")
    for patch in approved:
        paths = [f["path"] for f in patch["files"]]
        print(f"  - {patch['summary']} -> {paths}")


if __name__ == "__main__":
    asyncio.run(main())

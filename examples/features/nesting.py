"""``ctx.workflow()`` named nesting — a parent inlines a registered child workflow.

A parent orchestration resolves a named child workflow from the registry and runs
it inline with ``ctx.workflow("gather", {...})``. The child fans out one research
leaf per topic with ``ctx.parallel`` and synthesizes them; its result folds
straight back into the parent, while sharing the parent's journal, budget, and
concurrency gate. Named nesting is currently one level deep: a ``ctx.workflow()``
call from inside an already-nested workflow fails loud with
``WorkflowNestingError``, which this demo provokes on purpose. Runs fully offline
with a deterministic fake.

    uv run python -m examples.features.nesting
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent

from examples._shared.offline_models import ScriptedModel
from langchain_dynamic_workflow import (
    Ctx,
    Roster,
    WorkflowNestingError,
    WorkflowRegistry,
    run_workflow,
)

TOPICS = ["batteries", "solar", "wind"]


async def gather(ctx: Ctx, args: dict[str, Any]) -> str:
    """Child workflow: fan out a research leaf per topic and synthesize them."""
    topics: list[str] = args["topics"]
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    surviving = [f for f in findings if f is not None]
    return f"synthesis of {len(surviving)} findings: " + "; ".join(surviving)


async def nest_once(ctx: Ctx, args: dict[str, Any]) -> str:
    """An already-nested child that tries to nest a second level — must fail loud."""
    return await ctx.workflow("gather", {"topics": TOPICS})


async def main() -> None:
    roster = Roster().register(
        "researcher",
        create_deep_agent(model=ScriptedModel(prefix="research")),
        description="Researches a single topic",
    )
    workflows = WorkflowRegistry().register("gather", gather).register("nest_once", nest_once)

    # The parent inlines the named child workflow and folds in its synthesized result.
    async def orchestrate(ctx: Ctx) -> str:
        inner = await ctx.workflow("gather", {"topics": TOPICS})
        return f"parent folded child result -> {inner}"

    parent_result = await run_workflow(
        orchestrate,
        roster=roster,
        workflows=workflows,
        thread_id="nesting",
        max_concurrency=4,
    )
    print(f"one-level nesting:\n  {parent_result}")
    assert parent_result.startswith("parent folded child result -> synthesis of 3 findings:")

    # The one-level cap, made concrete: a second nesting level fails loud.
    async def too_deep(ctx: Ctx) -> str:
        return await ctx.workflow("nest_once", {})

    nesting_error: WorkflowNestingError | None = None
    try:
        await run_workflow(
            too_deep, roster=roster, workflows=workflows, thread_id="nesting-too-deep"
        )
    except WorkflowNestingError as exc:
        nesting_error = exc
    print(f"two-level nesting refused: {type(nesting_error).__name__}")
    assert nesting_error is not None, "a second nesting level must raise WorkflowNestingError"
    print("OK: ctx.workflow() inlined a named child one level deep; a second level was refused.")


if __name__ == "__main__":
    asyncio.run(main())

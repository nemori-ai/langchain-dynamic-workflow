"""Budget-guarded loop — accumulate leaves until the shared token pool nears exhaustion.

The script checks ``ctx.budget.remaining()`` before each iteration and stops
gracefully once the pool nears the threshold (loop-until-budget). A
usage-metering fake draws the budget down so the loop terminates with no real
API call. The sibling loop-until-dry shape (stop when no new work appears) is
shown in ``examples.features.reduce``.

    uv run python -m examples.features.budget
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent

from examples._shared.offline_models import ScriptedModel
from langchain_dynamic_workflow import Ctx, Roster, run_workflow

TOPICS = ["batteries", "solar", "wind", "hydrogen", "geothermal", "nuclear", "tidal"]
TOTAL_BUDGET = 50
THRESHOLD = 10
TOKENS_PER_LEAF = 10


async def orchestrate(ctx: Ctx) -> dict[str, Any]:
    findings: list[str] = []
    for topic in TOPICS:
        if not (ctx.budget.total and ctx.budget.remaining() > THRESHOLD):
            break
        findings.append(await ctx.agent(f"Research {topic}", agent_type="researcher"))
    return {"findings": findings, "spent": ctx.budget.spent()}


async def main() -> None:
    roster = Roster().register(
        "researcher",
        create_deep_agent(model=ScriptedModel(prefix="note", tokens_per_call=TOKENS_PER_LEAF)),
        description="Researches a single topic",
    )
    result = await run_workflow(orchestrate, roster=roster, thread_id="budget", budget=TOTAL_BUDGET)
    print(f"findings: {len(result['findings'])}, spent {result['spent']} of {TOTAL_BUDGET}")
    # The loop stopped before exhausting the pool: spent + one leaf would cross the line.
    assert result["spent"] <= TOTAL_BUDGET
    assert result["spent"] + TOKENS_PER_LEAF > TOTAL_BUDGET - THRESHOLD
    print("OK: the loop stopped gracefully on the budget threshold.")


if __name__ == "__main__":
    asyncio.run(main())

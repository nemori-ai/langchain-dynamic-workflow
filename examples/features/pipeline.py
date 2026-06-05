"""``pipeline()`` no-barrier stage flow — each item travels research -> summarize alone.

Stream topics through a two-stage pipeline with no barrier between stages; each
item advances independently. The companion ``examples.features.parallel`` shows
the blocking-barrier fan-out shape.

    uv run python -m examples.features.pipeline
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent

from examples._shared.offline_models import ScriptedModel
from langchain_dynamic_workflow import Ctx, Roster, run_workflow

TOPICS = ["batteries", "solar", "wind", "hydrogen", "geothermal"]


async def main() -> None:
    roster = (
        Roster()
        .register(
            "researcher",
            create_deep_agent(model=ScriptedModel(prefix="research")),
            description="Researches a single topic",
        )
        .register(
            "summarizer",
            create_deep_agent(model=ScriptedModel(prefix="summary")),
            description="Condenses research into a brief",
        )
    )

    async def orchestrate(ctx: Ctx) -> list[str]:
        async def research_stage(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"Deep-dive {item}", agent_type="researcher")

        async def summarize_stage(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"Summarize: {prev}", agent_type="summarizer")

        return await ctx.pipeline(TOPICS, research_stage, summarize_stage)

    briefs = await run_workflow(orchestrate, roster=roster, thread_id="pipeline", max_concurrency=4)
    print(f"pipeline briefs ({len(briefs)}):")
    for brief in briefs:
        print(f"  - {brief!r}")
    assert len(briefs) == len(TOPICS), "every item must traverse the pipeline"
    print("OK: pipeline() streamed each item through both stages without a barrier.")


if __name__ == "__main__":
    asyncio.run(main())

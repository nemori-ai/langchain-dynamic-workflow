"""``parallel()`` barrier fan-out — N leaves run, the call blocks until all settle.

Fan out one research leaf per topic with a blocking barrier; a failed leaf lands
as ``None`` and is filtered out (``parallel`` itself never raises). The companion
``examples.features.pipeline`` shows the no-barrier streaming shape.

    uv run python -m examples.features.parallel
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent

from examples._shared.offline_models import ScriptedModel
from langchain_dynamic_workflow import Ctx, Roster, run_workflow

TOPICS = ["batteries", "solar", "wind", "hydrogen", "geothermal"]


async def main() -> None:
    roster = Roster().register(
        "researcher",
        create_deep_agent(model=ScriptedModel(prefix="research")),
        description="Researches a single topic",
    )

    async def orchestrate(ctx: Ctx) -> list[str]:
        findings = await ctx.parallel(
            [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in TOPICS]
        )
        return [f for f in findings if f is not None]

    surviving = await run_workflow(
        orchestrate, roster=roster, thread_id="parallel", max_concurrency=4
    )
    print(f"parallel findings ({len(surviving)}):")
    for finding in surviving:
        print(f"  - {finding!r}")
    assert len(surviving) == len(TOPICS), "the barrier must join all leaves"
    print("OK: parallel() fanned out and barrier-joined every leaf.")


if __name__ == "__main__":
    asyncio.run(main())

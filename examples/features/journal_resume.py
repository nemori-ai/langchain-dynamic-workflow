"""In-process resume — the content-hash journal replays completed leaves for free.

Run the same workflow twice against one journal (fresh thread each time): the
second run serves every leaf from the journal at zero model cost yet rebuilds an
identical result, and a per-leaf live-call counter proves the replay re-paid for
nothing. The fail-loud determinism guard backstops a divergent replay.

    uv run python -m examples.features.journal_resume
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    WorkflowDeterminismError,
    run_workflow,
)

TOPICS = ["batteries", "solar", "wind"]


class _Counter:
    """Tallies live leaf invocations across both runs."""

    def __init__(self) -> None:
        self.calls = 0


def _counting_leaf(counter: _Counter) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        counter.calls += 1
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"finding({last})")]}

    return RunnableLambda(_leaf)


async def orchestrate(ctx: Ctx) -> list[str]:
    return await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in TOPICS]
    )


async def main() -> None:
    counter = _Counter()
    roster = Roster().register("researcher", _counting_leaf(counter))
    journal = InMemoryJournalStore()

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="run-a")
    calls_after_first = counter.calls
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="run-b")

    print(f"first run findings: {first}")
    print(f"resume findings:    {second}")
    print(f"live calls: first={calls_after_first}, after resume={counter.calls}")
    assert first == second, "resume must rebuild an identical result"
    assert counter.calls == calls_after_first, "resume re-ran a leaf (not free)"

    # The determinism guard is not optional: a divergent replay fails loud.
    diverge_journal = InMemoryJournalStore()

    async def script_a(ctx: Ctx) -> str:
        a = await ctx.agent("first", agent_type="researcher")
        b = await ctx.agent("second", agent_type="researcher")
        return a + b

    async def script_b(ctx: Ctx) -> str:
        b = await ctx.agent("second", agent_type="researcher")
        a = await ctx.agent("first", agent_type="researcher")
        return b + a

    await run_workflow(script_a, roster=roster, journal=diverge_journal, thread_id="d-a")
    tripped = False
    try:
        await run_workflow(script_b, roster=roster, journal=diverge_journal, thread_id="d-b")
    except WorkflowDeterminismError:
        tripped = True
    assert tripped, "a divergent replay must trip the determinism guard"
    print("OK: journal replay was free and the determinism guard fired on divergence.")


if __name__ == "__main__":
    asyncio.run(main())

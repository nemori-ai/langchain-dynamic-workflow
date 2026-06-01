"""Phase 2 integration: parallel + pipeline fan-out through ``run_workflow``.

These tests drive the full ``run_workflow`` -> ``@entrypoint`` -> shared gate ->
leaf path with fake leaves (no API keys), asserting the locked Phase 2 semantics:
ordered results, None-on-error, no-barrier pipeline, a bounded concurrency cap
shared across fan-out paths, and half-completed resume hitting the journal.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


def _instrumented_leaf(tracker: dict[str, int], *, delay: float) -> Runnable[Any, Any]:
    """A leaf that tracks peak concurrency and echoes its prompt as the reply."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        tracker["in_flight"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["in_flight"])
        await asyncio.sleep(delay)
        tracker["in_flight"] -= 1
        prompt = inp["messages"][-1].content
        return {"messages": [*inp["messages"], AIMessage(content=f"done:{prompt}")]}

    return RunnableLambda(_call)


async def test_parallel_fanout_returns_ordered_results() -> None:
    tracker = {"in_flight": 0, "peak": 0}
    roster = Roster().register("worker", _instrumented_leaf(tracker, delay=0.0))

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [lambda i=i: ctx.agent(f"q{i}", agent_type="worker") for i in range(5)]
        )

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert results == [f"done:q{i}" for i in range(5)]


async def test_parallel_failed_leaf_lands_none(make_fake_leaf: FakeLeafFactory) -> None:
    ok_leaf, _ = make_fake_leaf("ok")
    boom_leaf, _ = make_fake_leaf("never", fail_times=99)
    roster = Roster().register("ok", ok_leaf).register("boom", boom_leaf)

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("a", agent_type="ok"),
                lambda: ctx.agent("b", agent_type="boom"),
                lambda: ctx.agent("c", agent_type="ok"),
            ]
        )

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert results == ["ok", None, "ok"]


async def test_concurrency_cap_is_enforced_across_fanout() -> None:
    tracker = {"in_flight": 0, "peak": 0}
    roster = Roster().register("worker", _instrumented_leaf(tracker, delay=0.02))

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [lambda i=i: ctx.agent(f"q{i}", agent_type="worker") for i in range(12)]
        )

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1", max_concurrency=3)
    assert len(results) == 12
    assert tracker["peak"] <= 3
    assert tracker["peak"] == 3  # the cap actually saturates


async def test_pipeline_two_stage_through_workflow() -> None:
    tracker = {"in_flight": 0, "peak": 0}
    roster = Roster().register("worker", _instrumented_leaf(tracker, delay=0.0))

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        async def research(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"research {item}", agent_type="worker")

        async def summarize(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"summarize {prev}", agent_type="worker")

        return await ctx.pipeline(["x", "y"], research, summarize)

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert results == [
        "done:summarize done:research x",
        "done:summarize done:research y",
    ]


async def test_halfway_resume_hits_journal_zero_calls(make_fake_leaf: FakeLeafFactory) -> None:
    # First run: one of the two parallel leaves fails, so its journal entry is
    # never written (success-only). Resume with the failing leaf fixed: the
    # already-completed leaf is served from the journal (zero new calls), only
    # the previously-failed one runs live.
    ok_leaf, ok_state = make_fake_leaf("good")
    flaky_leaf, flaky_state = make_fake_leaf("recovered", fail_times=1)
    roster = Roster().register("ok", ok_leaf).register("flaky", flaky_leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("stable", agent_type="ok"),
                lambda: ctx.agent("unstable", agent_type="flaky"),
            ]
        )

    # parallel never raises; the flaky leaf lands as None on the first pass.
    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert first == ["good", None]
    assert ok_state.calls == 1
    assert flaky_state.calls == 1  # failed once, not journaled

    # Resume on a fresh thread with the SAME journal: the ok leaf is a cache hit.
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == ["good", "recovered"]
    assert ok_state.calls == 1  # zero additional calls: served from journal
    assert flaky_state.calls == 2  # ran live to recover

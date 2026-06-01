"""Phase 2 integration: parallel + pipeline fan-out through ``run_workflow``.

These tests drive the full ``run_workflow`` -> ``@entrypoint`` -> shared gate ->
leaf path with fake leaves (no API keys), asserting the locked Phase 2 semantics:
ordered results, None-on-error, no-barrier pipeline, a bounded concurrency cap
shared across fan-out paths, and half-completed resume hitting the journal.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    journal_key,
    run_workflow,
)

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


async def test_nested_parallel_does_not_leak_concurrency_cap() -> None:
    # Regression for the nested fan-out cap-leak: an outer parallel branch that
    # itself fans out an inner parallel must NOT free-ride the cap. The gate is
    # acquired only by leaves (agent()), so an orchestration frame never holds a
    # slot while awaiting its children. With cap=3 and 4x3=12 leaves nested two
    # layers deep, peak in-flight leaves must stay at the cap, not multiply by
    # nesting depth (the bug produced peak=9 here).
    tracker = {"in_flight": 0, "peak": 0}
    roster = Roster().register("worker", _instrumented_leaf(tracker, delay=0.02))

    async def orchestrate(ctx: Ctx) -> list[list[str | None] | None]:
        async def outer_branch(i: int) -> list[str | None]:
            return await ctx.parallel(
                [lambda i=i, j=j: ctx.agent(f"q{i}-{j}", agent_type="worker") for j in range(3)]
            )

        return await ctx.parallel([lambda i=i: outer_branch(i) for i in range(4)])

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1", max_concurrency=3)
    # All 12 leaves completed, returned as 4 branches of 3.
    assert results == [[f"done:q{i}-{j}" for j in range(3)] for i in range(4)]
    assert tracker["peak"] <= 3
    assert tracker["peak"] == 3  # the cap saturates without leaking past it


async def test_parallel_inside_pipeline_stage_does_not_leak_cap() -> None:
    # Composition across primitives: a pipeline stage that fans out an inner
    # parallel must also respect the single global cap. This exercises the
    # parallel-inside-a-pipeline-stage path, the other place the old free-ride
    # leaked concurrency proportional to nesting depth.
    tracker = {"in_flight": 0, "peak": 0}
    roster = Roster().register("worker", _instrumented_leaf(tracker, delay=0.02))

    async def orchestrate(ctx: Ctx) -> list[Any | None]:
        async def fan_stage(prev: str, item: str, index: int) -> list[str | None]:
            return await ctx.parallel(
                [
                    lambda item=item, j=j: ctx.agent(f"{item}-{j}", agent_type="worker")
                    for j in range(3)
                ]
            )

        return await ctx.pipeline(["a", "b", "c"], fan_stage)

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1", max_concurrency=4)
    assert results == [[f"done:{item}-{j}" for j in range(3)] for item in ("a", "b", "c")]
    assert tracker["peak"] <= 4
    assert tracker["peak"] == 4  # cap saturates across composed primitives


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


async def test_inflight_interrupt_then_resume_replays_completed_runs_unfinished_live() -> None:
    # A genuine mid-flight interruption (not a failed leaf): one parallel leaf
    # completes and journals while a second is still in flight when the run is
    # torn down. On resume with the SAME journal, the completed leaf is served
    # from the journal (zero new calls) and only the never-finished leaf runs live.
    fast_calls = {"n": 0}
    slow_calls = {"n": 0}
    slow_release = asyncio.Event()

    async def fast_leaf(inp: dict[str, Any]) -> dict[str, Any]:
        fast_calls["n"] += 1
        return {"messages": [*inp["messages"], AIMessage(content="fast-done")]}

    async def slow_leaf(inp: dict[str, Any]) -> dict[str, Any]:
        slow_calls["n"] += 1
        # First pass: park here so the run can be interrupted while in flight.
        if not slow_release.is_set():
            await slow_release.wait()
        return {"messages": [*inp["messages"], AIMessage(content="slow-done")]}

    roster = (
        Roster()
        .register("fast", RunnableLambda(fast_leaf))
        .register("slow", RunnableLambda(slow_leaf))
    )
    journal = InMemoryJournalStore()
    fast_key = journal_key(
        prompt="a", agent_type="fast", model=None, schema=None, isolation="shared"
    )

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("a", agent_type="fast"),
                lambda: ctx.agent("b", agent_type="slow"),
            ]
        )

    # First pass: launch the run. The fast leaf completes and journals while the
    # slow leaf parks unfinished. Wait until the fast leaf is genuinely journaled
    # (its put has run) so the interrupt lands with one leaf done and one still in
    # flight — a real partial completion, not a race.
    run_task = asyncio.create_task(
        run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    )

    async def _fast_journaled() -> None:
        while await journal.get(fast_key) is None:
            await asyncio.sleep(0)

    await asyncio.wait_for(_fast_journaled(), timeout=2.0)
    # Tear the whole run down mid-flight (cancellation = a real interrupt) while
    # the slow leaf is still parked and unjournaled.
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task

    # The fast leaf finished and was journaled; the slow leaf was interrupted
    # mid-flight and never journaled (success-only persistence).
    assert fast_calls["n"] == 1
    assert slow_calls["n"] == 1
    assert await journal.get(fast_key) == "fast-done"

    # Resume on a fresh thread with the SAME journal; let the slow leaf complete.
    slow_release.set()
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == ["fast-done", "slow-done"]
    assert fast_calls["n"] == 1  # zero new calls: completed leaf served from journal
    assert slow_calls["n"] == 2  # the never-finished leaf ran live on resume


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

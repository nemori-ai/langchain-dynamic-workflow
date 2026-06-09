"""Integration: ``ctx.batch_map`` through the full ``run_workflow`` resume loop.

Drives the engine with deterministic fake leaves (no API keys) to prove the
streaming map's headline contract end to end: results are collected in input
order, a failing ``fn`` lands ``None`` at its position without aborting the
barrier, an engine control-flow signal raised inside ``fn`` fails loud (never
masked as ``None``), a journaled batch reproduces every result on resume and
dispatches no leaf, and the transient ``BATCH`` count/ETA progress is a live
view that is NOT recorded — so a resume re-emits it (regenerated each run) yet
replays the leaves at zero dispatch.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    ProgressEntry,
    ProgressKind,
    Roster,
    WorkflowBudgetExceededError,
    run_workflow,
)


def _counting_leaf(calls: dict[str, int]) -> Runnable[Any, Any]:
    """A leaf that counts dispatches and echoes the prompt text back as its reply."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        prompt = str(inp["messages"][0].content)
        return {"messages": [*inp["messages"], AIMessage(content=f"audited:{prompt}")]}

    return RunnableLambda(_call)


async def test_batch_map_collects_in_order_and_isolates_failures() -> None:
    # The streaming map collects results aligned to INPUT order regardless of
    # completion order, and a leaf that raises lands None at its position without
    # aborting the barrier (mirroring parallel/pipeline failure isolation).
    calls = {"n": 0}

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        prompt = str(inp["messages"][0].content)
        # The middle item's leaf fails -> None hole at index 1, others survive.
        if prompt.endswith("b"):
            raise RuntimeError("finder boom on b")
        return {"messages": [*inp["messages"], AIMessage(content=f"ok:{prompt}")]}

    roster = Roster().register("finder", RunnableLambda(_call))

    async def orchestrate(ctx: Ctx) -> list[Any]:
        items = ["a", "b", "c"]
        return await ctx.batch_map(
            items,
            lambda x: ctx.agent(f"audit {x}", agent_type="finder"),
            max_in_flight=2,
        )

    result = await run_workflow(orchestrate, roster=roster)
    # Order preserved; index 1 ("b") is a None hole; the other two survive in order.
    assert result == ["ok:audit a", None, "ok:audit c"]
    assert calls["n"] == 3  # every item dispatched once


async def test_batch_map_control_flow_signal_fails_loud() -> None:
    # An engine control-flow signal raised inside fn must NOT be masked as a None
    # hole like an ordinary leaf failure: it surfaces out of batch_map after the
    # drain, exactly as parallel/pipeline re-raise it.
    roster = Roster().register("finder", RunnableLambda(lambda inp: inp))

    async def orchestrate(ctx: Ctx) -> list[Any]:
        async def _fn(x: str) -> str:
            if x == "b":
                raise WorkflowBudgetExceededError("budget exhausted mid-batch")
            return x

        return await ctx.batch_map(["a", "b", "c"], _fn, max_in_flight=2)

    with pytest.raises(WorkflowBudgetExceededError, match="exhausted"):
        await run_workflow(orchestrate, roster=roster)


async def test_batch_map_replay_reproduces_results_and_dispatches_nothing() -> None:
    # A journaled batch resumes on a fresh thread with the SAME journal: every leaf
    # is a cache hit (zero new dispatches) and the ordered result list is identical.
    calls = {"n": 0}
    roster = Roster().register("finder", _counting_leaf(calls))
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[Any]:
        items = ["x0", "x1", "x2", "x3"]
        return await ctx.batch_map(
            items,
            lambda x: ctx.agent(f"audit {x}", agent_type="finder"),
            max_in_flight=2,
        )

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched_on_first = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert first == second
    assert first == [
        "audited:audit x0",
        "audited:audit x1",
        "audited:audit x2",
        "audited:audit x3",
    ]
    assert dispatched_on_first == 4  # all four ran live on the fresh run
    assert calls["n"] == dispatched_on_first  # replay dispatched NOTHING


async def test_batch_map_resume_re_emits_live_progress_but_never_records_it() -> None:
    # Live count/ETA is a transient BATCH entry: delivered to the sink but never
    # appended to ProgressLog._entries / never journaled (pinned by the Task 3 unit
    # test). It is a LIVE VIEW regenerated each run, NOT replayed-from-record — so a
    # resume RE-EMITS it (the script re-executes batch_map), while the leaves replay
    # from the journal at zero dispatch. The determinism boundary is "never recorded",
    # NOT "never re-emitted".
    calls = {"n": 0}
    roster = Roster().register("finder", _counting_leaf(calls))
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[Any]:
        items = [f"f{i}" for i in range(8)]
        return await ctx.batch_map(
            items,
            lambda x: ctx.agent(f"audit {x}", agent_type="finder"),
            max_in_flight=4,
        )

    first_batch: list[ProgressEntry] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t1",
        on_progress=lambda e: first_batch.append(e) if e.kind is ProgressKind.BATCH else None,
    )
    # The fresh run advanced live work; the forced final settled entry always arrives.
    assert first_batch, "expected a transient BATCH entry on the fresh run"
    assert first_batch[-1].metrics is not None
    assert first_batch[-1].metrics.completed == 8

    second_batch: list[ProgressEntry] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t2",
        on_progress=lambda e: second_batch.append(e) if e.kind is ProgressKind.BATCH else None,
    )
    # The resume RE-EXECUTES the script, so batch_map re-runs and live BATCH progress
    # is re-emitted (regenerated, never replay-suppressed) — NOT empty.
    assert second_batch, "resume re-emits live BATCH progress (regenerated each run)"
    assert second_batch[-1].metrics is not None
    assert second_batch[-1].metrics.completed == 8
    # Yet the resume genuinely replayed: NO new leaf dispatched (zero-cost journal
    # replay). The transient progress carries no journal/determinism weight.
    assert calls["n"] == 8

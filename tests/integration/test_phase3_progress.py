"""Phase 3 integration: ``phase`` / ``log`` capture + replay idempotency.

The progress sink captures every newly-delivered entry through ``run_workflow``.
A resumed run re-executes the script (re-emitting the same narration), but the
already-delivered entries are suppressed so progress is not repeated; only
genuinely new entries reach the sink.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_phase_log_entries_are_captured(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    captured: list[str] = []

    async def orchestrate(ctx: Ctx) -> str:
        ctx.phase("research")
        ctx.log("starting")
        result = await ctx.agent("q", agent_type="worker")
        ctx.log(f"got {result}")
        return result

    result = await run_workflow(
        orchestrate,
        roster=roster,
        thread_id="t1",
        on_progress=lambda e: captured.append(f"{e.kind.value}:{e.message}"),
    )
    assert result == "answer"
    assert captured == ["phase:research", "log:starting", "log:got answer"]


async def test_replay_does_not_redeliver_progress(make_fake_leaf: FakeLeafFactory) -> None:
    # First run delivers three entries; the resume (same journal) re-executes the
    # script — re-emitting the same three — but they must NOT be re-delivered.
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        ctx.phase("research")
        ctx.log("starting")
        result = await ctx.agent("q", agent_type="worker")
        ctx.log("done")
        return result

    first: list[str] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t1",
        on_progress=lambda e: first.append(e.message),
    )
    assert first == ["research", "starting", "done"]

    # Resume on a fresh thread with the SAME journal: the leaf is a cache hit and
    # the already-delivered progress is suppressed (idempotent narration).
    second: list[str] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t2",
        on_progress=lambda e: second.append(e.message),
    )
    assert second == []


async def test_replay_delivers_only_new_progress(make_fake_leaf: FakeLeafFactory) -> None:
    # The script emits more progress on the resume than on the first run; only the
    # genuinely new entries (beyond the recorded count) reach the sink.
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()
    extra = {"emit": False}

    async def orchestrate(ctx: Ctx) -> str:
        ctx.phase("research")
        ctx.log("starting")
        if extra["emit"]:
            ctx.log("new-on-resume")
        return await ctx.agent("q", agent_type="worker")

    first: list[str] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t1",
        on_progress=lambda e: first.append(e.message),
    )
    assert first == ["research", "starting"]

    extra["emit"] = True
    second: list[str] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t2",
        on_progress=lambda e: second.append(e.message),
    )
    # Only the new line is delivered; the two already-delivered entries stay quiet.
    assert second == ["new-on-resume"]

"""Background-run transport: a real detached run_workflow buffers replayable events."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import (
    BgRunManager,
    Ctx,
    InMemoryJournalStore,
    Roster,
    run_workflow,
)


def _counting_leaf(calls: dict[str, int]) -> Runnable[Any, Any]:
    """A leaf that counts dispatches and echoes the prompt text back as its reply."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        prompt = str(inp["messages"][0].content)
        return {"messages": [*inp["messages"], AIMessage(content=f"found:{prompt}")]}

    return RunnableLambda(_call)


def _make_roster(calls: dict[str, int]) -> Roster:
    """Build a roster with one counting fake worker leaf."""
    return Roster().register("worker", _counting_leaf(calls))


async def _orchestrate(ctx: Ctx) -> list[Any]:
    """Fan two leaves out through a parallel barrier."""
    return await ctx.parallel(
        [
            lambda: ctx.agent("find a", agent_type="worker"),
            lambda: ctx.agent("find b", agent_type="worker"),
        ]
    )


def _detached_run(
    manager: BgRunManager,
    *,
    run_id: str,
    thread_id: str,
    roster: Roster,
    journal: InMemoryJournalStore,
):
    """Wrap a sink-wired run_workflow as a detached coroutine for ``manager``."""
    sinks = manager.event_sinks(run_id, thread_id=thread_id)

    async def _run() -> str:
        result = await run_workflow(
            _orchestrate,
            roster=roster,
            journal=journal,
            on_span_begin=sinks.on_span_begin,
            on_span=sinks.on_span,
            on_leaf_event=sinks.on_leaf_event,
            on_progress=sinks.on_progress,
            on_command=sinks.on_command,
        )
        return str(result)

    return _run()


async def test_detached_run_buffers_span_and_leaf_events() -> None:
    manager = BgRunManager()
    run_id = "bg-1"
    calls = {"n": 0}

    slot = manager.start(
        _detached_run(
            manager,
            run_id=run_id,
            thread_id="t1",
            roster=_make_roster(calls),
            journal=InMemoryJournalStore(),
        ),
        run_id=run_id,
        thread_id="t1",
    )
    await manager.wait(slot.run_id, thread_id="t1")

    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    assert dropped == 0
    kinds = [e.kind for e in events]
    assert "span_begin" in kinds and "span" in kinds  # leaf begin + end edges
    # begin precedes its matching end (arrival order preserved)
    assert kinds.index("span_begin") < kinds.index("span")
    assert calls["n"] == 2  # both leaves dispatched live


async def test_buffer_is_transient_across_resume() -> None:
    # Same journal, fresh manager (≈ process restart): the buffer does NOT replay;
    # a resumed run re-generates only what actually re-executes (cached leaves are
    # journal hits and fire no interior events).
    journal = InMemoryJournalStore()
    calls = {"n": 0}
    roster = _make_roster(calls)

    first_manager = BgRunManager()
    first_slot = first_manager.start(
        _detached_run(
            first_manager, run_id="bg-first", thread_id="t1", roster=roster, journal=journal
        ),
        run_id="bg-first",
        thread_id="t1",
    )
    await first_manager.wait(first_slot.run_id, thread_id="t1")
    first_events, _first_dropped = first_manager.buffered_events("bg-first", thread_id="t1")
    assert first_events, "the fresh run must have buffered transport events"
    assert calls["n"] == 2  # both leaves dispatched live on the fresh run

    second_manager = BgRunManager()
    second_slot = second_manager.start(
        _detached_run(
            second_manager, run_id="bg-second", thread_id="t1", roster=roster, journal=journal
        ),
        run_id="bg-second",
        thread_id="t1",
    )
    await second_manager.wait(second_slot.run_id, thread_id="t1")

    second_events, _second_dropped = second_manager.buffered_events("bg-second", thread_id="t1")
    # The resume replayed every leaf from the journal: no leaf re-executed, so no
    # interior leaf callback edges were re-fired into the new buffer.
    assert calls["n"] == 2, "resume must dispatch no leaf (journal replay)"
    assert [e.kind for e in second_events if e.kind == "leaf_event"] == []
    # The old run's buffer is gone with the old manager instance: the new manager
    # never tracked it, so the snapshot is the empty ([], 0) — not a replay source.
    assert second_manager.buffered_events("bg-first", thread_id="t1") == ([], 0)

"""Unit tests for the background-run event transport substrate (BufferedEvent buffer)."""

import asyncio
import threading

import pytest

from langchain_dynamic_workflow._background import (
    BgRunManager,
    BufferedEvent,
)
from langchain_dynamic_workflow._observability import SpanBegin, SpanKind
from langchain_dynamic_workflow._progress import ProgressEntry, ProgressKind


async def _start_noop_run(manager: BgRunManager, *, run_id: str = "r1") -> str:
    """Start a trivial background run so a slot (and its buffer) exists."""

    async def _noop() -> str:
        return "ok"

    slot = manager.start(_noop(), run_id=run_id, thread_id="t1")
    await manager.wait(slot.run_id, thread_id="t1")
    return slot.run_id


async def test_slot_buffer_starts_empty_and_buffered_events_unknown_run() -> None:
    manager = BgRunManager()
    run_id = await _start_noop_run(manager)
    events: list[BufferedEvent]
    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    assert events == []
    assert dropped == 0
    # Unknown run: empty snapshot, never a KeyError (the run may have been swept).
    assert manager.buffered_events("nope", thread_id="t1") == ([], 0)


async def test_manager_rejects_non_positive_max_buffered_events() -> None:
    for bad in (0, -5):
        with pytest.raises(ValueError, match="max_buffered_events"):
            BgRunManager(max_buffered_events=bad)


def _span_begin(name: str = "leaf") -> SpanBegin:
    return SpanBegin(
        span_id=f"sid-{name}",
        kind=SpanKind.AGENT,
        name=name,
        attributes={},
        started_at=0.0,
        monotonic_start=0.0,
    )


async def test_event_sinks_append_in_arrival_order() -> None:
    manager = BgRunManager()
    run_id = await _start_noop_run(manager)
    sinks = manager.event_sinks(run_id, thread_id="t1")

    sinks.on_span_begin(_span_begin("a"))
    sinks.on_progress(ProgressEntry(kind=ProgressKind.LOG, message="hello"))

    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    assert dropped == 0
    assert [e.kind for e in events] == ["span_begin", "progress"]
    assert isinstance(events[0].payload, SpanBegin)
    assert events[0].payload.name == "a"


async def test_event_sinks_drop_newest_past_cap_and_count_dropped() -> None:
    manager = BgRunManager(max_buffered_events=2)
    run_id = await _start_noop_run(manager)
    sinks = manager.event_sinks(run_id, thread_id="t1")

    for i in range(5):
        sinks.on_span_begin(_span_begin(f"n{i}"))

    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    # drop-newest: the earliest (structural) events survive, later ones are counted.
    assert [e.payload.name for e in events] == ["n0", "n1"]  # type: ignore[union-attr]
    assert dropped == 3


async def test_event_sinks_for_swept_run_are_silent_noops() -> None:
    # A sink may outlive its slot (the run settled and was swept while the detached
    # task still flushes a last event). It must drop silently — never raise.
    manager = BgRunManager(idle_ttl_seconds=0.0)
    run_id = await _start_noop_run(manager)
    sinks = manager.event_sinks(run_id, thread_id="t1")
    manager.sweep()
    sinks.on_span_begin(_span_begin("late"))  # must not raise
    assert manager.buffered_events(run_id, thread_id="t1") == ([], 0)


async def test_cross_thread_appends_do_not_tear_reads() -> None:
    # on_command fires from an asyncio.to_thread worker concurrently with loop-thread
    # sinks and buffered_events reads — hammer both sides and assert consistency.
    manager = BgRunManager(max_buffered_events=10_000)
    run_id = await _start_noop_run(manager)
    sinks = manager.event_sinks(run_id, thread_id="t1")

    def _worker() -> None:
        for i in range(500):
            sinks.on_progress(ProgressEntry(kind=ProgressKind.LOG, message=f"w{i}"))

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    # Interleave reads while workers append.
    for _ in range(50):
        events, dropped = manager.buffered_events(run_id, thread_id="t1")
        assert dropped == 0
        assert len(events) <= 2000
        await asyncio.sleep(0)
    for t in threads:
        t.join()
    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    assert len(events) == 2000 and dropped == 0


async def test_sweep_reclaims_buffer_with_slot() -> None:
    manager = BgRunManager(idle_ttl_seconds=0.0)
    run_id = await _start_noop_run(manager)
    sinks = manager.event_sinks(run_id, thread_id="t1")
    sinks.on_span_begin(_span_begin("x"))
    assert manager.buffered_events(run_id, thread_id="t1")[0] != []
    reclaimed = manager.sweep()
    assert run_id in reclaimed
    assert manager.buffered_events(run_id, thread_id="t1") == ([], 0)

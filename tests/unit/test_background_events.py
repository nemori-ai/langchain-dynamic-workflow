"""Unit tests for the background-run event transport substrate (BufferedEvent buffer)."""

import pytest

from langchain_dynamic_workflow._background import (
    BgRunManager,
    BufferedEvent,
)


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

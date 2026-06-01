"""Unit tests for the self-built async background run mechanism.

These tests pin the host-facing background machinery without any host agent or
model: a coroutine is launched detached, the slot reports a placeholder
immediately, the done callback enqueues a completion notice, the result store
offloads large payloads behind a handle, and idle/hard TTL reclaims finished
slots.
"""

from __future__ import annotations

import asyncio

import pytest

from langchain_dynamic_workflow._background import (
    BgRunManager,
    BgStatus,
    ResultStore,
)


async def test_start_returns_immediately_with_pending_or_running_slot() -> None:
    manager = BgRunManager()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow() -> str:
        started.set()
        await release.wait()
        return "done-value"

    slot = manager.start(slow(), run_id="r1", thread_id="t1")
    # The slot is handed back without awaiting the coroutine: it is not yet done.
    assert slot.run_id == "r1"
    assert slot.thread_id == "t1"
    assert manager.poll("r1") in {BgStatus.PENDING, BgStatus.RUNNING}

    await started.wait()
    assert manager.poll("r1") == BgStatus.RUNNING
    release.set()
    await manager.wait("r1")
    assert manager.poll("r1") == BgStatus.DONE


async def test_done_enqueues_notification_drained_per_thread() -> None:
    manager = BgRunManager()

    async def quick() -> str:
        return "answer-42"

    manager.start(quick(), run_id="r1", thread_id="tA")
    await manager.wait("r1")

    # Notices are keyed by thread; a different thread sees nothing.
    assert manager.drain_notifications("tB") == []
    notices = manager.drain_notifications("tA")
    assert len(notices) == 1
    assert notices[0].run_id == "r1"
    assert notices[0].status == BgStatus.DONE
    # Draining is destructive: a second drain on the same thread is empty.
    assert manager.drain_notifications("tA") == []


async def test_failed_run_reports_failed_status_and_failure_notice() -> None:
    manager = BgRunManager()

    async def boom() -> str:
        raise RuntimeError("kaboom")

    manager.start(boom(), run_id="rx", thread_id="tF")
    await manager.wait("rx")

    assert manager.poll("rx") == BgStatus.FAILED
    notices = manager.drain_notifications("tF")
    assert len(notices) == 1
    assert notices[0].status == BgStatus.FAILED
    # The error text is surfaced for the host, not swallowed.
    assert "kaboom" in (notices[0].detail or "")


async def test_get_result_returns_completed_value_and_offloads_large_payload() -> None:
    store = ResultStore(inline_max_chars=8)
    manager = BgRunManager(result_store=store)

    async def small() -> str:
        return "tiny"

    async def large() -> str:
        return "x" * 200

    manager.start(small(), run_id="rs", thread_id="t1")
    manager.start(large(), run_id="rl", thread_id="t1")
    await manager.wait("rs")
    await manager.wait("rl")

    # A small result is returned inline (no handle).
    small_result = manager.get_result("rs")
    assert small_result.value == "tiny"
    assert small_result.handle is None
    assert small_result.summary == "tiny"

    # A large result is offloaded: status carries a summary + handle, and the
    # full value is fetched from the store by handle.
    large_result = manager.get_result("rl")
    assert large_result.value is None
    assert large_result.handle is not None
    assert len(large_result.summary) <= 8
    assert store.fetch(large_result.handle) == "x" * 200


async def test_unknown_run_polls_as_unknown() -> None:
    manager = BgRunManager()
    assert manager.poll("nope") == BgStatus.UNKNOWN
    with pytest.raises(KeyError):
        manager.get_result("nope")


async def test_cancel_marks_slot_cancelled() -> None:
    manager = BgRunManager()
    release = asyncio.Event()

    async def slow() -> str:
        await release.wait()
        return "never"

    manager.start(slow(), run_id="rc", thread_id="t1")
    await manager.cancel("rc")
    assert manager.poll("rc") == BgStatus.CANCELLED
    # A cancellation notice is delivered to the thread.
    notices = manager.drain_notifications("t1")
    assert len(notices) == 1
    assert notices[0].status == BgStatus.CANCELLED


async def test_composite_key_isolates_same_run_id_across_threads() -> None:
    manager = BgRunManager()

    async def reply(value: str) -> str:
        return value

    # Same run_id on two different threads must not collide.
    manager.start(reply("from-A"), run_id="shared", thread_id="tA")
    manager.start(reply("from-B"), run_id="shared", thread_id="tB")
    await manager.wait("shared", thread_id="tA")
    await manager.wait("shared", thread_id="tB")

    assert manager.get_result("shared", thread_id="tA").value == "from-A"
    assert manager.get_result("shared", thread_id="tB").value == "from-B"


async def test_idle_ttl_sweeps_completed_slots() -> None:
    # A zero idle TTL means a completed slot is reclaimable immediately.
    manager = BgRunManager(idle_ttl_seconds=0.0)

    async def quick() -> str:
        return "ok"

    manager.start(quick(), run_id="r1", thread_id="t1")
    await manager.wait("r1")
    assert manager.poll("r1") == BgStatus.DONE

    reclaimed = manager.sweep()
    assert "r1" in reclaimed
    # After reclamation the slot is gone (polls as unknown).
    assert manager.poll("r1") == BgStatus.UNKNOWN

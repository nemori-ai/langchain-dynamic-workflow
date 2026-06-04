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
    BgRunQuotaExceededError,
    BgStatus,
    ResultStore,
    RunSnapshot,
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


async def test_max_concurrent_runs_quota_refuses_when_full() -> None:
    # Resource-exhaustion guard: with a quota of 2, a third in-flight run is
    # refused loud rather than fanned out unbounded onto the event loop.
    manager = BgRunManager(max_concurrent_runs=2)
    release = asyncio.Event()

    async def slow(value: str) -> str:
        await release.wait()
        return value

    s1 = manager.start(slow("a"), run_id="r1", thread_id="t1")
    s2 = manager.start(slow("b"), run_id="r2", thread_id="t1")
    assert {s1.run_id, s2.run_id} == {"r1", "r2"}

    # Both r1/r2 are still in flight (gated closed), so the quota is full.
    with pytest.raises(BgRunQuotaExceededError) as excinfo:
        manager.start(slow("c"), run_id="r3", thread_id="t1")
    # The refusal names the quota so the host gets an actionable message.
    assert "2" in str(excinfo.value)
    # Nothing was launched for the refused run: it never created a slot.
    assert manager.poll("r3", thread_id="t1") == BgStatus.UNKNOWN

    # Drain a slot: once r1 settles, the quota frees up and a new run is admitted.
    release.set()
    await manager.wait("r1", thread_id="t1")
    await manager.wait("r2", thread_id="t1")
    assert manager.poll("r1", thread_id="t1") == BgStatus.DONE

    # A settled run no longer counts against the quota — a fresh start succeeds.
    s4 = manager.start(slow("d"), run_id="r4", thread_id="t1")
    assert s4.run_id == "r4"
    release.set()
    await manager.wait("r4", thread_id="t1")


async def test_list_runs_enumerates_thread_runs_with_status() -> None:
    # The aggregate view: list every run on a thread with its live status, so the
    # host need not poll each run_id. A settled run carries a short summary; an
    # in-flight run's summary is None.
    manager = BgRunManager()
    release = asyncio.Event()

    async def slow() -> str:
        await release.wait()
        return "later"

    async def quick() -> str:
        return "done-value"

    manager.start(slow(), run_id="r1", thread_id="t1")  # stays in flight, no label
    manager.start(quick(), run_id="r2", thread_id="t1", label="quick-job")  # settles
    await manager.wait("r2", thread_id="t1")

    snapshots = manager.list_runs("t1")
    assert all(isinstance(s, RunSnapshot) for s in snapshots)
    by_id = {s.run_id: s for s in snapshots}
    assert set(by_id) == {"r1", "r2"}
    assert by_id["r1"].status in {BgStatus.PENDING, BgStatus.RUNNING}
    assert by_id["r1"].summary is None  # in flight: no outcome yet
    assert by_id["r1"].label is None  # launched without a label
    assert by_id["r2"].status == BgStatus.DONE
    assert by_id["r2"].summary == "done-value"  # settled: short outcome preview
    assert by_id["r2"].label == "quick-job"  # label recorded at launch flows through

    release.set()
    await manager.wait("r1", thread_id="t1")


async def test_list_runs_isolated_per_thread() -> None:
    # list_runs is scoped to the host thread, mirroring the composite-key isolation.
    manager = BgRunManager()

    async def reply(value: str) -> str:
        return value

    manager.start(reply("a"), run_id="r1", thread_id="tA")
    manager.start(reply("b"), run_id="r2", thread_id="tB")
    await manager.wait("r1", thread_id="tA")
    await manager.wait("r2", thread_id="tB")

    assert [s.run_id for s in manager.list_runs("tA")] == ["r1"]
    assert [s.run_id for s in manager.list_runs("tB")] == ["r2"]
    # A thread with no runs gets an empty list.
    assert manager.list_runs("tC") == []


def test_max_concurrent_runs_rejects_non_positive() -> None:
    # 0/negative is not a meaningful quota (it would refuse every run); only None
    # (unbounded) or a positive cap are valid, rejected loud at construction.
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive integer or None"):
            BgRunManager(max_concurrent_runs=bad)
    # None and a positive cap are accepted.
    assert BgRunManager().max_concurrent_runs is None
    assert BgRunManager(max_concurrent_runs=1).max_concurrent_runs == 1


async def test_unbounded_quota_is_the_default() -> None:
    # The default keeps the existing behavior: no quota, many runs admitted.
    manager = BgRunManager()
    release = asyncio.Event()

    async def slow() -> str:
        await release.wait()
        return "ok"

    for i in range(5):
        manager.start(slow(), run_id=f"r{i}", thread_id="t1")
    # All five are in flight; none was refused.
    assert all(manager.poll(f"r{i}", thread_id="t1") != BgStatus.UNKNOWN for i in range(5))
    release.set()
    for i in range(5):
        await manager.wait(f"r{i}", thread_id="t1")

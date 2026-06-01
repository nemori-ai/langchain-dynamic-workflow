"""Unit tests for the no-barrier bounded-queue pipeline scheduler."""

from __future__ import annotations

import asyncio

import pytest

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._pipeline import run_pipeline


async def test_pipeline_empty_stages_raises() -> None:
    # A pipeline with no stages is a programming error; the documented guard must
    # fail loudly rather than silently return the items unprocessed.
    gate = ConcurrencyGate(limit=2)
    with pytest.raises(ValueError, match="at least one stage"):
        await run_pipeline([1, 2], (), gate=gate)


async def test_pipeline_preserves_input_order() -> None:
    gate = ConcurrencyGate(limit=8)

    async def stage_one(prev: int, item: int, index: int) -> int:
        # Later items finish faster, so completion order differs from input order.
        await asyncio.sleep((10 - item) * 0.001)
        return prev + 1

    async def stage_two(prev: int, item: int, index: int) -> str:
        return f"{item}:{prev}"

    results = await run_pipeline(
        list(range(5)),
        [stage_one, stage_two],
        gate=gate,
    )
    # stage_one threads prev=item -> item+1; stage_two formats "item:prev".
    assert results == ["0:1", "1:2", "2:3", "3:4", "4:5"]


async def test_pipeline_has_no_barrier_between_stages() -> None:
    # Prove A reaches the last stage while B is still in the first stage.
    gate = ConcurrencyGate(limit=8)
    reached_stage_two: list[int] = []
    stage_one_started = asyncio.Event()
    item_b_held_in_stage_one = asyncio.Event()

    async def stage_one(prev: int, item: int, index: int) -> int:
        stage_one_started.set()
        if item == 1:
            # Item B parks in stage one until item A has already cleared stage two.
            await item_b_held_in_stage_one.wait()
        return prev

    async def stage_two(prev: int, item: int, index: int) -> int:
        reached_stage_two.append(item)
        if item == 0:
            # Item A is in stage two; release B (still in stage one) only now,
            # which is impossible under a barrier (B would have to finish first).
            item_b_held_in_stage_one.set()
        return prev

    results = await run_pipeline([0, 1], [stage_one, stage_two], gate=gate)
    assert results == [0, 1]
    # A (item 0) reached stage two before B (item 1) left stage one.
    assert reached_stage_two[0] == 0


async def test_pipeline_stage_error_drops_item_to_none_and_skips_rest() -> None:
    gate = ConcurrencyGate(limit=4)
    stage_two_seen: list[int] = []

    async def stage_one(prev: int, item: int, index: int) -> int:
        if item == 1:
            raise RuntimeError("stage one boom for item 1")
        return item

    async def stage_two(prev: int, item: int, index: int) -> int:
        stage_two_seen.append(item)
        return prev * 10

    results = await run_pipeline([0, 1, 2], [stage_one, stage_two], gate=gate)
    # Item 1 dropped to None and never reached stage two; others survive.
    assert results == [0, None, 20]
    assert 1 not in stage_two_seen


async def test_pipeline_does_not_deadlock_when_all_items_fail() -> None:
    gate = ConcurrencyGate(limit=2)

    async def boom(prev: int, item: int, index: int) -> int:
        raise RuntimeError("always fails")

    async def never_runs(prev: int, item: int, index: int) -> int:  # pragma: no cover
        raise AssertionError("downstream stage must not run for a dropped item")

    results = await asyncio.wait_for(
        run_pipeline([0, 1, 2, 3], [boom, never_runs], gate=gate),
        timeout=2.0,
    )
    assert results == [None, None, None, None]


async def test_pipeline_respects_concurrency_gate_across_stages() -> None:
    # The global gate is shared across all stages and bounds in-flight LEAF work.
    # Real stages call agent(), which wraps the leaf in gate.run; here the stages
    # acquire the same gate directly to model that single chokepoint. Total leaf
    # work in flight across BOTH stages must never exceed the limit, even though
    # each stage runs its own worker group.
    gate = ConcurrencyGate(limit=2)
    in_flight = 0
    peak = 0

    async def busy(prev: int, item: int, index: int) -> int:
        nonlocal in_flight, peak

        async def _leaf() -> int:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return prev

        # Mirror agent(): the leaf invocation is what acquires the shared gate.
        return await gate.run(_leaf)

    results = await run_pipeline(
        list(range(8)),
        [busy, busy],
        gate=gate,
    )
    assert results == list(range(8))
    assert peak <= 2
    assert peak == 2  # the cap saturates across both stages, not just one


async def test_pipeline_tiny_queue_backpressure_does_not_deadlock() -> None:
    # A queue smaller than the item count must apply backpressure, not deadlock:
    # more items than capacity still flow through every stage to completion.
    gate = ConcurrencyGate(limit=4)

    async def stage_one(prev: int, item: int, index: int) -> int:
        await asyncio.sleep(0.001)
        return item + 1

    async def stage_two(prev: int, item: int, index: int) -> int:
        await asyncio.sleep(0.001)
        return prev * 2

    results = await asyncio.wait_for(
        run_pipeline(
            list(range(20)),
            [stage_one, stage_two],
            gate=gate,
            queue_maxsize=2,
        ),
        timeout=3.0,
    )
    assert results == [(i + 1) * 2 for i in range(20)]


async def test_pipeline_empty_items_returns_empty() -> None:
    gate = ConcurrencyGate(limit=2)

    async def stage(prev: int, item: int, index: int) -> int:
        return prev

    assert await run_pipeline([], [stage], gate=gate) == []


async def test_pipeline_is_faster_than_sequential() -> None:
    # Wall-clock proof that the pipeline overlaps stage work.
    gate = ConcurrencyGate(limit=8)
    per_stage = 0.02

    async def slow(prev: int, item: int, index: int) -> int:
        await asyncio.sleep(per_stage)
        return prev

    items = list(range(6))
    start = asyncio.get_event_loop().time()
    results = await run_pipeline(items, [slow, slow], gate=gate)
    elapsed = asyncio.get_event_loop().time() - start

    assert results == items
    sequential = len(items) * 2 * per_stage
    # Overlap must beat naive sequential by a wide margin.
    assert elapsed < sequential * 0.6

"""Unit tests for the no-barrier bounded-queue pipeline scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError
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


async def test_pipeline_control_flow_error_fails_loud_across_workers_and_drains() -> None:
    # An engine control-flow signal (budget/determinism) raised by a stage must NOT
    # be masked as a dropped None like an ordinary leaf failure: it propagates out
    # of run_pipeline. This stresses the abort path under multiple workers (gate
    # limit 4), two stages, and backpressure (queue_maxsize 2 << 8 items): the
    # pipeline must drain queued items without deadlocking and re-raise the signal.
    gate = ConcurrencyGate(limit=4)
    stage_two_seen: list[int] = []

    async def stage_one(prev: int, item: int, index: int) -> int:
        await asyncio.sleep(0.005)
        if item == 3:
            raise WorkflowBudgetExceededError("budget exhausted mid-pipeline")
        return item

    async def stage_two(prev: int, item: int, index: int) -> int:
        stage_two_seen.append(item)
        return prev * 10

    with pytest.raises(WorkflowBudgetExceededError, match="exhausted"):
        await asyncio.wait_for(
            run_pipeline(
                list(range(8)),
                [stage_one, stage_two],
                gate=gate,
                queue_maxsize=2,
            ),
            timeout=3.0,
        )


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


async def test_pipeline_accepts_sync_generator_without_len() -> None:
    # A generator is Iterable but NOT Sized: it has no __len__. The generalized
    # run_pipeline must stream it through without ever calling len(items) and
    # collect results in the generator's yield order.
    gate = ConcurrencyGate(limit=4)

    def gen() -> Iterator[int]:  # generator: Iterable, not Sized
        yield from (10, 20, 30, 40, 50)

    async def stage(prev: int, item: int, index: int) -> int:
        # Later items finish faster so completion order != input order.
        await asyncio.sleep((50 - item) * 0.0001)
        return item + index

    results = await run_pipeline(gen(), [stage], gate=gate)
    # index threads the yield position: 10+0, 20+1, 30+2, 40+3, 50+4.
    assert results == [10, 21, 32, 43, 54]


async def test_pipeline_accepts_async_iterable() -> None:
    # An async source (paged API / streaming reader) is the point of streaming
    # admission. The feeder must drive it via `async for` and preserve order.
    gate = ConcurrencyGate(limit=4)

    async def asource():
        for value in range(6):
            await asyncio.sleep(0.0005)
            yield value

    async def stage(prev: int, item: int, index: int) -> int:
        await asyncio.sleep((6 - item) * 0.001)
        return item * 100

    results = await asyncio.wait_for(
        run_pipeline(asource(), [stage], gate=gate),
        timeout=3.0,
    )
    assert results == [0, 100, 200, 300, 400, 500]


async def test_pipeline_empty_async_iterable_returns_empty() -> None:
    # An empty AsyncIterable has no __len__ and no __bool__: the old
    # `if not items: return []` early return would have crashed on it. The
    # generalized run must drain zero items and return [] via poison-pill teardown.
    gate = ConcurrencyGate(limit=2)

    async def empty_asource():
        return
        yield  # pragma: no cover  -- makes this an async generator

    async def stage(prev: int, item: int, index: int) -> int:  # pragma: no cover
        raise AssertionError("stage must not run for an empty input")

    results = await asyncio.wait_for(
        run_pipeline(empty_asource(), [stage], gate=gate),
        timeout=2.0,
    )
    assert results == []


async def test_pipeline_async_iterable_preserves_order_under_backpressure() -> None:
    # An AsyncIterable feeding more items than the queue capacity must apply
    # backpressure (not deadlock) and still return results in input order, proving
    # the dict-keyed collection reconstructs order without a pre-allocated length.
    gate = ConcurrencyGate(limit=4)

    async def asource():
        for value in range(20):
            yield value

    async def stage_one(prev: int, item: int, index: int) -> int:
        await asyncio.sleep(0.001)
        return item + 1

    async def stage_two(prev: int, item: int, index: int) -> int:
        await asyncio.sleep(0.001)
        return prev * 2

    results = await asyncio.wait_for(
        run_pipeline(asource(), [stage_one, stage_two], gate=gate, queue_maxsize=2),
        timeout=3.0,
    )
    assert results == [(i + 1) * 2 for i in range(20)]

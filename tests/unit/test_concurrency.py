"""Unit tests for the concurrency gate (semaphore + config injection)."""

from __future__ import annotations

import asyncio

from langchain_dynamic_workflow._concurrency import (
    ConcurrencyGate,
    resolve_max_concurrency,
    with_max_concurrency,
)


def test_resolve_default_is_bounded_and_at_least_one() -> None:
    # The substrate default is None (unbounded); the gate must always be bounded.
    limit = resolve_max_concurrency(None)
    assert isinstance(limit, int)
    assert 1 <= limit <= 16


def test_resolve_honours_explicit_limit() -> None:
    assert resolve_max_concurrency(4) == 4


def test_resolve_clamps_to_hard_ceiling() -> None:
    # A runaway request is clamped to the 1000 hard ceiling (§11).
    assert resolve_max_concurrency(10_000) == 1000


def test_resolve_rejects_non_positive() -> None:
    # Zero / negative would deadlock the semaphore; coerce to at least 1.
    assert resolve_max_concurrency(0) == 1
    assert resolve_max_concurrency(-5) == 1


def test_with_max_concurrency_injects_into_config() -> None:
    cfg = with_max_concurrency({"configurable": {"thread_id": "t"}}, 7)
    assert cfg.get("max_concurrency") == 7
    # Original keys are preserved.
    assert cfg.get("configurable") == {"thread_id": "t"}


async def test_gate_caps_max_in_flight() -> None:
    gate = ConcurrencyGate(limit=3)
    in_flight = 0
    peak = 0

    async def work() -> None:
        nonlocal in_flight, peak
        async with gate:
            in_flight += 1
            peak = max(peak, in_flight)
            # Yield so other tasks get a chance to pile up if the gate were leaky.
            await asyncio.sleep(0.01)
            in_flight -= 1

    await asyncio.gather(*[work() for _ in range(12)])
    assert peak <= 3
    assert peak == 3  # the gate should actually saturate, not serialize


async def test_gate_run_wraps_a_coroutine_factory() -> None:
    gate = ConcurrencyGate(limit=2)

    async def make(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    result = await gate.run(lambda: make(21))
    assert result == 42


async def test_gate_is_reentrant_within_one_task() -> None:
    # Nested acquisition by the same logical unit must not consume a second slot,
    # otherwise fan-out layers (parallel/pipeline) that wrap agent() — which also
    # gates — would deadlock when every slot is held by an outer acquisition.
    gate = ConcurrencyGate(limit=3)

    async def inner() -> int:
        async with gate:  # re-entry from the same task
            await asyncio.sleep(0.01)
            return 1

    async def outer() -> int:
        async with gate:  # outer acquisition
            return await inner()

    results = await asyncio.wait_for(
        asyncio.gather(*[outer() for _ in range(12)]),
        timeout=3.0,
    )
    assert results == [1] * 12


async def test_gate_reentrancy_does_not_inflate_peak() -> None:
    # With reentrancy, nested acquisitions by the same unit count once, so the
    # observed peak still respects the limit.
    gate = ConcurrencyGate(limit=2)
    peak = 0
    in_flight = 0

    async def unit() -> None:
        nonlocal peak, in_flight
        async with gate:
            in_flight += 1
            peak = max(peak, in_flight)
            async with gate:  # re-entry: no extra slot
                await asyncio.sleep(0.01)
            in_flight -= 1

    await asyncio.gather(*[unit() for _ in range(8)])
    assert peak == 2

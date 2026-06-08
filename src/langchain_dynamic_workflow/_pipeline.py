"""No-barrier streaming pipeline scheduler — a substrate structural blind spot.

LangGraph offers no streaming, no-barrier fan-out primitive (``Send`` is a
map-reduce barrier), so the engine builds its own. Items flow through the stages
*independently*: item A can be in the last stage while item B is still in the
first, with no synchronization point between stages.

Mechanics:

- Each stage owns a bounded :class:`~asyncio.Queue`, providing backpressure so a
  flood of items cannot exhaust memory by materializing every stage at once.
- Each stage runs a worker group that pulls an envelope, runs the stage function,
  and forwards the envelope to the next stage's queue. The worker count per stage
  is itself bounded by the shared
  :class:`~langchain_dynamic_workflow._concurrency.ConcurrencyGate` limit, and the
  leaf ``agent()`` calls inside the stages acquire the gate — so leaf concurrency
  is capped at the gate limit without an orchestration frame ever holding a slot.
- A stage that raises drops that item to ``None``; the item skips all remaining
  stages and its result slot is filled immediately.
- Results are collected by the item's original input index, so the returned list
  is order-preserving regardless of completion order.
- Workers are torn down with poison pills, so a mid-pipeline failure or an empty
  input drains the queues gracefully without deadlocking.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ._concurrency import ConcurrencyGate
from ._errors import WORKFLOW_CONTROL_FLOW_SIGNALS

Stage = Callable[[Any, Any, int], Awaitable[Any]]
"""A pipeline stage: ``(prev_result, original_item, index) -> next_result``."""

DEFAULT_QUEUE_MAXSIZE = 32
"""Default bounded-queue capacity per stage (backpressure against item floods)."""


@dataclass(slots=True)
class _Envelope:
    """An item travelling through the pipeline.

    Attributes:
        index: The item's original input position; drives ordered result collection.
        original: The original input item, passed unchanged to every stage.
        payload: The running result threaded from one stage to the next.
    """

    index: int
    original: Any
    payload: Any


# Sentinel pushed through a stage's queue to retire its workers once drained.
_POISON = object()


def _workers_per_stage(gate: ConcurrencyGate, item_count: int) -> int:
    """Pick a worker count per stage, bounded by the gate and the item count."""
    return max(1, min(gate.limit, item_count))


async def run_pipeline(
    items: Sequence[Any],
    stages: Sequence[Stage],
    *,
    gate: ConcurrencyGate,
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
) -> list[Any | None]:
    """Stream ``items`` through ``stages`` without a barrier and collect results.

    Args:
        items: The input items; each travels through every stage independently.
        stages: The ordered stage functions, each ``(prev, original, index)``.
        gate: The shared concurrency gate. Its limit bounds the per-stage worker
            count, and the leaf ``agent()`` calls inside the stages acquire it, so
            leaf concurrency across all stages stays within the cap.
        queue_maxsize: Bounded capacity of each stage's queue (backpressure).

    Returns:
        A list aligned to ``items`` input order. Each entry is the item's final
        stage result, or ``None`` if any stage raised for that item.

    Raises:
        ValueError: If ``stages`` is empty.
    """
    if not stages:
        raise ValueError("pipeline requires at least one stage")
    if not items:
        return []

    item_count = len(items)
    results: list[Any | None] = [None] * item_count
    worker_count = _workers_per_stage(gate, item_count)
    # First engine control-flow signal (budget/determinism) raised by any stage.
    # Once set, workers drain remaining items WITHOUT running their stages and the
    # run re-raises it after a clean teardown — re-raising from inside a worker
    # would hang the feeder's queue.join() and deadlock the drain.
    aborted: list[BaseException] = []

    # One bounded queue per stage; envelopes enter stage 0's queue and graduate
    # to the next stage's queue after each stage succeeds.
    queues: list[asyncio.Queue[Any]] = [asyncio.Queue(maxsize=queue_maxsize) for _ in stages]

    def drop(envelope: _Envelope) -> None:
        """Record a dropped item; it skips all remaining stages."""
        results[envelope.index] = None

    async def stage_worker(stage_index: int) -> None:
        in_queue = queues[stage_index]
        stage_fn = stages[stage_index]
        is_last = stage_index == len(stages) - 1
        while True:
            envelope = await in_queue.get()
            try:
                if envelope is _POISON:
                    return
                if aborted:
                    # Fail-loud teardown in progress: drain queued items without
                    # running their stages so the feeder's join() completes.
                    drop(envelope)
                    continue
                try:
                    next_payload = await stage_fn(
                        envelope.payload, envelope.original, envelope.index
                    )
                except WORKFLOW_CONTROL_FLOW_SIGNALS as exc:
                    # Engine control-flow signal: do NOT mask as a leaf failure.
                    # Record the first one, drop this item, and let the run drain
                    # and re-raise after teardown (fail loud).
                    if not aborted:
                        aborted.append(exc)
                    drop(envelope)
                    continue
                except Exception:
                    # Failure isolation: drop this item, skip remaining stages.
                    drop(envelope)
                    continue
                if is_last:
                    results[envelope.index] = next_payload
                else:
                    envelope.payload = next_payload
                    await queues[stage_index + 1].put(envelope)
            finally:
                in_queue.task_done()

    async def feeder() -> None:
        """Inject every item into stage 0, then propagate poison pills downstream."""
        for index, item in enumerate(items):
            await queues[0].put(_Envelope(index=index, original=item, payload=item))
        # Retire each stage's workers in order. Each stage drains fully (every
        # live envelope already forwarded) before its poison pills are consumed,
        # because a worker only pulls a pill after all real envelopes ahead of it.
        for stage_index in range(len(stages)):
            await queues[stage_index].join()
            for _ in range(worker_count):
                await queues[stage_index].put(_POISON)

    workers: list[asyncio.Task[None]] = [
        asyncio.create_task(stage_worker(stage_index))
        for stage_index in range(len(stages))
        for _ in range(worker_count)
    ]
    feeder_task = asyncio.create_task(feeder())
    try:
        await feeder_task
        await asyncio.gather(*workers)
    finally:
        # Defensive cleanup: cancel anything still pending so we never leak tasks.
        for worker in workers:
            if not worker.done():
                worker.cancel()
        if not feeder_task.done():
            feeder_task.cancel()
    if aborted:
        # A stage tripped an engine control-flow signal; surface it loud now that
        # the pipeline has drained and torn down cleanly.
        raise aborted[0]
    return results

"""No-barrier streaming pipeline scheduler — a substrate structural blind spot.

LangGraph offers no streaming, no-barrier fan-out primitive (``Send`` is a
map-reduce barrier), so the engine builds its own. Items flow through the stages
*independently*: item A can be in the last stage while item B is still in the
first, with no synchronization point between stages.

The scheduler admits items *lazily* through a bounded queue, so the input may be
any ``Iterable`` or ``AsyncIterable`` — a list, a generator, or an async source
(a paged API, a streaming line reader) — and the number of live envelopes/tasks
stays bounded by the admission window, decoupled from the total item count. The
length is never required up front; results are keyed by input index and the
ordered list is reconstructed once every item has settled.

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
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
    Sized,
)
from contextlib import aclosing
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


async def _drain(
    items: Iterable[Any] | AsyncIterable[Any],
) -> AsyncGenerator[tuple[int, Any], None]:
    """Yield ``(index, item)`` from either a sync or an async input source.

    The ``AsyncIterable`` check comes first: a ``list`` is ``Iterable`` but not
    ``AsyncIterable``, whereas an async generator is both, so probing for the
    async protocol first routes each source down its correct branch. The async
    branch counts manually because there is no async ``enumerate``.

    Args:
        items: The input source, an ``Iterable`` or an ``AsyncIterable``.

    Yields:
        ``(index, item)`` pairs, index ascending from zero in source order.
    """
    if isinstance(items, AsyncIterable):
        index = 0
        async for item in items:
            yield index, item
            index += 1
    else:
        for index, item in enumerate(items):
            yield index, item


def _workers_per_stage(gate: ConcurrencyGate, length: int | None, max_in_flight: int) -> int:
    """Pick a worker count per stage, bounded by the gate, length, and window.

    Args:
        gate: The shared concurrency gate; its limit is an upper bound.
        length: The input length when known (a ``Sized`` source), else ``None``.
        max_in_flight: The admission window (the per-stage queue capacity).

    Returns:
        The per-stage worker count, at least one.
    """
    if length is not None:
        return max(1, min(gate.limit, length, max_in_flight))
    return max(1, min(gate.limit, max_in_flight))


async def run_pipeline(
    items: Iterable[Any] | AsyncIterable[Any],
    stages: Sequence[Stage],
    *,
    gate: ConcurrencyGate,
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
) -> list[Any | None]:
    """Stream ``items`` through ``stages`` without a barrier and collect results.

    The input is admitted lazily through the bounded queue, so it may be any
    ``Iterable`` or ``AsyncIterable``: a list takes the length fast path, while a
    generator or an async source streams without its length ever being read. The
    number of live envelopes stays bounded by the admission window, decoupled from
    the total item count.

    Args:
        items: The input items, an ``Iterable`` or an ``AsyncIterable``; each one
            travels through every stage independently.
        stages: The ordered stage functions, each ``(prev, original, index)``.
        gate: The shared concurrency gate. Its limit bounds the per-stage worker
            count, and the leaf ``agent()`` calls inside the stages acquire it, so
            leaf concurrency across all stages stays within the cap.
        queue_maxsize: Bounded capacity of each stage's queue. It is the admission
            window: the feeder blocks on a full queue, so the live envelope count
            stays near ``worker_count + queue_maxsize``, independent of the input
            size.

    Returns:
        A list aligned to ``items`` input order. Each entry is the item's final
        stage result, or ``None`` if any stage raised for that item.

    Raises:
        ValueError: If ``stages`` is empty.
    """
    if not stages:
        raise ValueError("pipeline requires at least one stage")

    length = len(items) if isinstance(items, Sized) else None
    worker_count = _workers_per_stage(gate, length, queue_maxsize)
    # Order-preserving collection keyed by input index. A length is never required
    # up front, so a streaming (generator / AsyncIterable) source is collected the
    # same way a list is, then flattened once every item has settled.
    results: dict[int, Any | None] = {}
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
                except BaseException as exc:
                    # A non-Exception BaseException escaped both clauses above.
                    # CancelledError is the headline case and is ambiguous, so the
                    # CURRENT task's cancelling() count discriminates the two origins:
                    #   - cancelling() > 0: THIS worker's own task is being cancelled
                    #     externally (the whole run is being torn down). Propagate the
                    #     CancelledError so teardown stays clean; the run_pipeline
                    #     finally cancels + awaits every sibling worker/feeder.
                    #   - cancelling() == 0: a stray/interior cancellation (a child the
                    #     stage awaited was cancelled) or another propagating
                    #     BaseException (KeyboardInterrupt / SystemExit). The worker is
                    #     NOT being torn down, so killing it here would strand stage 0's
                    #     queue and deadlock the feeder's join() when this is the sole
                    #     worker.
                    current = asyncio.current_task()
                    if current is not None and current.cancelling() > 0:
                        raise
                    if isinstance(exc, asyncio.CancelledError):
                        # Interior cancellation == a leaf failure: drop this one item,
                        # keep draining (siblings survive, run does not hang).
                        drop(envelope)
                        continue
                    # A genuinely-propagating BaseException (KeyboardInterrupt /
                    # SystemExit): it must surface loud, but a bare re-raise here would
                    # kill the worker and deadlock the feeder. Mirror the control-flow
                    # abort path — record the first, drop this item, and keep the worker
                    # alive to drain remaining items + poison pills so join() completes;
                    # run_pipeline re-raises it after a clean teardown (fail loud, no
                    # silent swallow, no deadlock).
                    if not aborted:
                        aborted.append(exc)
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
        # aclosing closes the (possibly unbounded / paged) async source even when we break
        # out early on a control-flow abort, releasing its resources.
        async with aclosing(_drain(items)) as stream:
            async for index, item in stream:
                # A stage tripped a fail-loud control-flow signal: stop pulling the source.
                # Envelopes already queued are still drained + task_done by the workers
                # (which drop them under ``aborted``), so the join() and poison-pill teardown
                # below still complete; we only stop admitting new items, so an unbounded
                # source cannot keep paging after the abort.
                if aborted:
                    break
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
        # Defensive cleanup: cancel anything still pending, then AWAIT it so teardown
        # is complete before run_pipeline returns/propagates — no leaked pending tasks
        # and no in-flight stage/source cleanup left detached. Mirrors dag/race/parallel
        # (cancel + await gather); a bare cancel without awaiting would leave the
        # cancelled worker/feeder (and the source's aclosing finally) pending on an
        # external cancel mid-flight. return_exceptions swallows their CancelledError.
        pending_teardown = [task for task in (*workers, feeder_task) if not task.done()]
        for task in pending_teardown:
            task.cancel()
        if pending_teardown:
            await asyncio.gather(*pending_teardown, return_exceptions=True)
    if aborted:
        # A stage tripped an engine control-flow signal; surface it loud now that
        # the pipeline has drained and torn down cleanly.
        raise aborted[0]
    # Flatten the index-keyed results into an ordered list. An empty run yields [];
    # otherwise positions absent from the dict are filled with None.
    if not results:
        return []
    return [results.get(i) for i in range(max(results) + 1)]

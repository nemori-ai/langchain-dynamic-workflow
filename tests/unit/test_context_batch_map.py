"""Unit tests for ``Ctx.batch_map`` — streaming fan-out + live count/ETA progress.

Mirrors the ``_CountingLeaf`` + ctx harness from ``test_dag.py`` /
``test_context_pipeline.py``: a thin map over an (async) iterable, results
collected in input order, a failing ``fn`` lands ``None`` (no abort), identical
``agent()`` calls dedup through the journal, and a ``BATCH`` span plus transient
``BATCH`` progress entries are emitted as the batch advances.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import (
    _FANOUT_DEPTH,  # pyright: ignore[reportPrivateUsage] - fan-out frame under test
    Ctx,
    LeafOutcome,
)
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._observability import Span, SpanKind, SpanRecorder
from langchain_dynamic_workflow._progress import (
    BatchMetrics,
    ProgressEntry,
    ProgressKind,
    ProgressLog,
)
from langchain_dynamic_workflow._roster import Roster


class _CountingLeaf:
    """A leaf runner that records prompts and counts invocations."""

    def __init__(self, *, prefix: str) -> None:
        self.calls = 0
        self.prefix = prefix

    async def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
    ) -> LeafOutcome:
        self.calls += 1
        return LeafOutcome(
            state={"messages": [AIMessage(content=f"{self.prefix}:{prompt}")]}, usage=0
        )


def _batch_ctx(
    leaf: _CountingLeaf,
    journal: InMemoryJournalStore,
    *,
    spans: SpanRecorder | None = None,
    progress: ProgressLog | None = None,
) -> Ctx:
    roster = Roster()
    roster.register("worker", object())  # type: ignore[arg-type]
    return Ctx(
        roster=roster,
        journal=journal,
        leaf_runner=leaf,
        gate=ConcurrencyGate(limit=8),
        spans=spans if spans is not None else SpanRecorder(),
        progress=progress,
    )


async def test_batch_map_maps_fn_over_items_in_input_order() -> None:
    leaf = _CountingLeaf(prefix="R")
    ctx = _batch_ctx(leaf, InMemoryJournalStore())

    results = await ctx.batch_map(
        ["alpha", "beta", "gamma"],
        lambda item: ctx.agent(f"audit {item}", agent_type="worker"),
    )

    assert results == ["R:audit alpha", "R:audit beta", "R:audit gamma"]
    assert leaf.calls == 3


async def test_batch_map_failing_fn_lands_none_and_does_not_abort() -> None:
    leaf = _CountingLeaf(prefix="R")
    ctx = _batch_ctx(leaf, InMemoryJournalStore())

    async def fn(item: str) -> str:
        if item == "bad":
            raise RuntimeError("fn blew up")
        return await ctx.agent(f"ok {item}", agent_type="worker")

    results = await ctx.batch_map(["good", "bad", "fine"], fn)

    # The bad item is a None hole; its neighbours survive (no barrier abort).
    assert results == ["R:ok good", None, "R:ok fine"]


async def test_batch_map_dedups_identical_agent_calls_via_journal() -> None:
    # Same journal => a repeated identical agent() prompt is a single leaf call.
    # agent() has no in-flight lock between its journal get/put, so this exact
    # count (1) is a property of synchronous-completing fakes; for real, slow
    # leaves in-batch dedup manifests as journal reuse on resume, not within one run.
    leaf = _CountingLeaf(prefix="R")
    ctx = _batch_ctx(leaf, InMemoryJournalStore())

    results = await ctx.batch_map(
        ["a", "b", "c"],
        lambda _item: ctx.agent("constant prompt", agent_type="worker"),
    )

    assert results == ["R:constant prompt"] * 3
    # First item missed and ran the leaf; the rest hit the journal.
    assert leaf.calls == 1


async def test_batch_map_leaves_run_inside_a_fanout_frame() -> None:
    # An agent() dispatched from inside batch_map sees a non-zero fan-out depth,
    # so it is excluded from the determinism sequence guard (its completion order
    # is wall-clock dependent) — exactly like parallel/pipeline/dag leaves.
    leaf = _CountingLeaf(prefix="R")
    ctx = _batch_ctx(leaf, InMemoryJournalStore())
    observed_depths: list[int] = []

    async def fn(item: str) -> str:
        observed_depths.append(_FANOUT_DEPTH.get())
        return await ctx.agent(f"q {item}", agent_type="worker")

    await ctx.batch_map(["x", "y"], fn)

    assert observed_depths == [1, 1]


async def test_batch_map_emits_a_batch_span_with_counts() -> None:
    leaf = _CountingLeaf(prefix="R")
    collected: list[Span] = []
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), spans=SpanRecorder(sink=collected.append))

    async def fn(item: str) -> str:
        if item == "bad":
            raise RuntimeError("boom")
        return await ctx.agent(f"q {item}", agent_type="worker")

    await ctx.batch_map(["a", "bad", "c"], fn)

    batch_spans = [s for s in collected if s.kind is SpanKind.BATCH]
    assert len(batch_spans) == 1
    attrs = batch_spans[0].attributes
    # A Sized input sets total = len(items) up front; admitted = all settled;
    # surviving = the non-None results (the bad item dropped to None).
    assert attrs["total"] == 3
    assert attrs["admitted_count"] == 3
    assert attrs["surviving_count"] == 2


async def test_batch_map_delivers_batch_progress_with_monotonic_counts() -> None:
    leaf = _CountingLeaf(prefix="R")
    delivered: list[ProgressEntry] = []
    # A small K (total // 100 floors to 0 -> max(1, ...) == 1) makes every
    # settled item cross the throttle, so each emits a BATCH entry.
    progress = ProgressLog(delivered_count=0, sink=delivered.append)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    results = await ctx.batch_map(
        ["a", "b", "c", "d"],
        lambda item: ctx.agent(f"q {item}", agent_type="worker"),
    )

    assert results == ["R:q a", "R:q b", "R:q c", "R:q d"]
    batch_entries = [e for e in delivered if e.kind is ProgressKind.BATCH]
    # Transient BATCH entries reached the sink and carry metrics...
    assert batch_entries, "expected at least one BATCH progress entry"
    for entry in batch_entries:
        assert isinstance(entry.metrics, BatchMetrics)
    # ...the completed counter never goes backwards and the last entry is exact.
    counts = [e.metrics.completed for e in batch_entries if e.metrics is not None]
    assert counts == sorted(counts)
    assert counts[-1] == 4
    last_metrics = batch_entries[-1].metrics
    assert last_metrics is not None
    assert last_metrics.total == 4
    # Transient entries are delivered but NOT recorded (the determinism boundary):
    # the append-only entries list stays empty for a pure batch_map run.
    assert progress.entries == []


async def test_batch_map_empty_sized_input_returns_empty_list() -> None:
    leaf = _CountingLeaf(prefix="R")
    ctx = _batch_ctx(leaf, InMemoryJournalStore())

    results = await ctx.batch_map([], lambda item: ctx.agent("q", agent_type="worker"))

    assert results == []
    assert leaf.calls == 0


async def test_batch_map_generator_input_without_len_has_no_eta() -> None:
    # An async generator is not Sized and no total= hint is given, so ETA is not
    # computable: every BATCH metric reports completed-only, eta_seconds is None.
    leaf = _CountingLeaf(prefix="R")
    delivered: list[ProgressEntry] = []
    progress = ProgressLog(delivered_count=0, sink=delivered.append)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    async def gen() -> AsyncIterator[str]:
        for token in ("a", "b", "c"):
            yield token

    results = await ctx.batch_map(gen(), lambda item: ctx.agent(f"q {item}", agent_type="worker"))

    assert results == ["R:q a", "R:q b", "R:q c"]
    batch_entries = [e for e in delivered if e.kind is ProgressKind.BATCH]
    assert batch_entries
    for entry in batch_entries:
        assert entry.metrics is not None
        assert entry.metrics.total is None
        assert entry.metrics.eta_seconds is None
    # No len() was ever taken on the generator (it would consume it); the span's
    # total is therefore None.

    # Sanity: the async source was fully consumed, not partially.
    assert len(results) == 3


async def test_batch_map_respects_max_in_flight_window() -> None:
    # The admission window bounds concurrency: with max_in_flight=2 over 6 items,
    # at most 2 fn bodies are ever in flight at once. The in-flight bound is the
    # worker_count (= min(gate, len, max_in_flight)), with the bounded queue
    # providing feeder backpressure; N is decoupled from the live-task count.
    journal = InMemoryJournalStore()

    class _SlowLeaf(_CountingLeaf):
        def __init__(self) -> None:
            super().__init__(prefix="R")
            self.in_flight = 0
            self.peak = 0
            self._lock = asyncio.Lock()

        async def __call__(self, *args: Any, **kwargs: Any) -> LeafOutcome:
            async with self._lock:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
            try:
                await asyncio.sleep(0.01)
                return await super().__call__(*args, **kwargs)
            finally:
                async with self._lock:
                    self.in_flight -= 1

    leaf = _SlowLeaf()
    ctx = _batch_ctx(leaf, journal)

    results = await ctx.batch_map(
        [f"item-{i}" for i in range(6)],
        lambda item: ctx.agent(f"q {item}", agent_type="worker"),
        max_in_flight=2,
    )

    assert len(results) == 6
    assert all(r is not None for r in results)
    assert leaf.peak <= 2, f"window breached: peak {leaf.peak} > 2"


async def test_batch_map_rejects_non_positive_max_in_flight() -> None:
    # A non-positive admission window is meaningless; batch_map fails fast before
    # opening a span or admitting any work — the guard fires before fn ever runs.
    leaf = _CountingLeaf(prefix="R")
    ctx = _batch_ctx(leaf, InMemoryJournalStore())
    for bad_window in (0, -3):
        with pytest.raises(ValueError, match="max_in_flight"):
            await ctx.batch_map(
                ["a", "b"],
                lambda item: ctx.agent(f"audit {item}", agent_type="worker"),
                max_in_flight=bad_window,
            )
    assert leaf.calls == 0  # the guard fired before any fn / leaf ran


async def test_batch_map_interior_sink_failure_does_not_corrupt_successful_result() -> None:
    # A progress sink that raises on an INTERIOR success emit must never corrupt the
    # genuinely-successful result at that position. Progress is best-effort, out-of-band
    # observability (delivered, never recorded/journaled/replayed); a host sink that
    # raises is the host's telemetry bug, isolated on every emit so it cannot demote a
    # real result to a None hole (which run_pipeline's `except Exception` would do if the
    # fault propagated out of _stage) nor abort the batch. With max_in_flight=1 settles
    # are strictly sequential, so the sink fault lands exactly on the item at completed==2
    # (index 1) — and that result must still be the correct value, not None.
    leaf = _CountingLeaf(prefix="R")

    def exploding_sink(entry: ProgressEntry) -> None:
        if (
            entry.kind is ProgressKind.BATCH
            and entry.metrics is not None
            and entry.metrics.completed == 2
        ):
            raise RuntimeError("interior sink boom")

    progress = ProgressLog(delivered_count=0, sink=exploding_sink)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    # The run returns normally (the sink fault neither raises nor aborts) AND the
    # correct result survives at the position whose emit raised.
    results = await ctx.batch_map(
        ["a", "b", "c", "d", "e"],
        lambda item: ctx.agent(f"q {item}", agent_type="worker"),
        max_in_flight=1,
    )
    assert results == ["R:q a", "R:q b", "R:q c", "R:q d", "R:q e"]
    assert results[1] is not None  # the sink fault did NOT corrupt the real result


async def test_batch_map_final_forced_emit_sink_failure_does_not_abort_batch() -> None:
    # The post-pipeline FINAL forced emit (fired once run_pipeline returns, to make the
    # last settled state exact) is a distinct emit site from the per-item _stage finally.
    # A sink that raises ONLY on that final emit must not turn a computationally-successful
    # batch into a propagated failure — the isolation is centralized at the emit chokepoint
    # so every emit, per-item AND final, is covered. The sink raises at completed == 3 (the
    # terminal settled count for a 3-item input), which the forced final emit always fires;
    # the batch must still RETURN its results, not raise RuntimeError.
    leaf = _CountingLeaf(prefix="R")

    def exploding_sink(entry: ProgressEntry) -> None:
        if (
            entry.kind is ProgressKind.BATCH
            and entry.metrics is not None
            and entry.metrics.completed == 3
        ):
            raise RuntimeError("final sink boom")

    progress = ProgressLog(delivered_count=0, sink=exploding_sink)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    results = await ctx.batch_map(
        ["a", "b", "c"],
        lambda item: ctx.agent(f"q {item}", agent_type="worker"),
        max_in_flight=1,
    )
    assert results == ["R:q a", "R:q b", "R:q c"]  # batch returned normally, did not raise


async def test_batch_map_fn_control_flow_signal_propagates_past_suppressed_emit() -> None:
    # The symmetric finally suppress wraps ONLY the emit, never `await fn(item)`. A
    # control-flow signal raised by fn ITSELF (not the sink) must still surface loud —
    # a budget / determinism breach is part of the computation and is never masked. The
    # progress sink also raising must not change that: the signal propagates from the
    # try body, the sink fault is independently suppressed in the finally.
    leaf = _CountingLeaf(prefix="R")

    def exploding_sink(entry: ProgressEntry) -> None:
        if entry.kind is ProgressKind.BATCH:
            raise RuntimeError("sink boom")

    progress = ProgressLog(delivered_count=0, sink=exploding_sink)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    async def fn(_item: str) -> str:
        raise WorkflowBudgetExceededError("budget breached")

    with pytest.raises(WorkflowBudgetExceededError):
        await ctx.batch_map(["a"], fn)


async def test_batch_map_underestimated_total_drops_to_unknown_no_negative_eta() -> None:
    # A non-Sized source with a `total=` hint that under-counts: once completed exceeds
    # the hint, the hint is proven wrong, so the emit drops to unknown-total mode (total
    # None, eta None, message shows completed only) rather than reporting a negative ETA
    # or a misleading "10/3". Every emitted BATCH entry must therefore satisfy: eta is
    # None-or-non-negative, total is None or completed <= total, and no message shows
    # completed > total. throttle_step == max(1, 3 // 100) == 1, so every settle emits
    # and the would-be-bad entries (completed 4..10) all reach the sink.
    leaf = _CountingLeaf(prefix="R")
    delivered: list[ProgressEntry] = []
    progress = ProgressLog(delivered_count=0, sink=delivered.append)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    async def gen() -> AsyncIterator[str]:
        for i in range(10):
            yield f"item-{i}"

    await ctx.batch_map(
        gen(),
        lambda item: ctx.agent(f"q {item}", agent_type="worker"),
        total=3,
    )

    batch_entries = [
        e for e in delivered if e.kind is ProgressKind.BATCH and isinstance(e.metrics, BatchMetrics)
    ]
    assert batch_entries, "expected at least one BATCH progress entry"
    for entry in batch_entries:
        metrics = entry.metrics
        assert metrics is not None
        assert metrics.eta_seconds is None or metrics.eta_seconds >= 0
        assert metrics.total is None or metrics.completed <= metrics.total
        # Once the hint is exceeded the message is completed-only (no "/N"); while the
        # hint still holds it stays "completed/total".
        if metrics.total is None:
            assert entry.message == f"batch_map: {metrics.completed}"
        else:
            assert entry.message == f"batch_map: {metrics.completed}/{metrics.total}"


async def test_batch_map_accurate_total_keeps_k_over_n_with_nonnegative_eta() -> None:
    # Guard against regressing the happy path: a Sized input (len() known up front, an
    # always-accurate total) keeps the "k/N" message, a preserved total, completed <= N
    # throughout, and a non-negative ETA — byte-identical to the pre-fix behavior.
    leaf = _CountingLeaf(prefix="R")
    delivered: list[ProgressEntry] = []
    progress = ProgressLog(delivered_count=0, sink=delivered.append)
    ctx = _batch_ctx(leaf, InMemoryJournalStore(), progress=progress)

    await ctx.batch_map(
        [f"item-{i}" for i in range(5)],
        lambda item: ctx.agent(f"q {item}", agent_type="worker"),
    )

    batch_entries = [
        e for e in delivered if e.kind is ProgressKind.BATCH and isinstance(e.metrics, BatchMetrics)
    ]
    assert batch_entries, "expected at least one BATCH progress entry"
    for entry in batch_entries:
        metrics = entry.metrics
        assert metrics is not None
        assert metrics.total == 5  # the Sized total is preserved on every emit
        assert metrics.completed <= metrics.total
        assert metrics.eta_seconds is None or metrics.eta_seconds >= 0
        assert entry.message == f"batch_map: {metrics.completed}/5"

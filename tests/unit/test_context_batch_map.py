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

from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import (
    _FANOUT_DEPTH,  # pyright: ignore[reportPrivateUsage] - fan-out frame under test
    Ctx,
    LeafOutcome,
)
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

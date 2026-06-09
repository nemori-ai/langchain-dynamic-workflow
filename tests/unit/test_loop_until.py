"""Unit tests for ``ctx.loop_until`` — the measured-stop loop helper."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._progress import ProgressKind, ProgressLog
from langchain_dynamic_workflow._roster import Roster


class _SeqLeaf:
    """Returns a scripted reply per call, so a stop predicate can be exercised."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

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
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return LeafOutcome(state={"messages": [AIMessage(content=reply)]}, usage=0)


def _ctx(leaf: _SeqLeaf, progress: ProgressLog) -> Ctx:
    roster = Roster()
    roster.register("worker", object())  # type: ignore[arg-type]
    return Ctx(
        roster=roster,
        journal=InMemoryJournalStore(),
        leaf_runner=leaf,
        gate=ConcurrencyGate(limit=4),
        progress=progress,
    )


async def test_loop_until_stops_when_done_satisfied() -> None:
    leaf = _SeqLeaf(["no", "no", "STOP", "no"])
    delivered: list[Any] = []
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=delivered.append))

    async def body(i: int, acc: list[str]) -> str:
        # `acc` is the accumulated-so-far (dedup-against-all-seen); prompt varies by i.
        return await ctx.agent(f"try {i} (seen {len(acc)})", agent_type="worker")

    out = await ctx.loop_until(body, done=lambda acc: "STOP" in acc, max_iters=10)
    assert out == ["no", "no", "STOP"]  # stopped the iteration that produced STOP
    assert leaf.calls == 3


async def test_loop_until_caps_and_logs_without_convergence() -> None:
    leaf = _SeqLeaf(["no"])
    delivered: list[Any] = []
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=delivered.append))

    async def body(i: int, acc: list[str]) -> str:
        return await ctx.agent(f"try {i}", agent_type="worker")

    out = await ctx.loop_until(body, done=lambda acc: False, max_iters=3)
    assert out == ["no", "no", "no"]
    assert leaf.calls == 3
    # The cap-without-convergence is surfaced as a (replay-idempotent) log line.
    # ProgressEntry fields: .kind (ProgressKind) and .message (str).
    logs = [e for e in delivered if e.kind is ProgressKind.LOG and "max_iters" in e.message]
    assert len(logs) == 1


async def test_loop_until_rejects_nonpositive_max_iters() -> None:
    leaf = _SeqLeaf(["x"])
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=lambda _e: None))

    async def body(i: int, acc: list[str]) -> str:
        return await ctx.agent("x", agent_type="worker")

    with pytest.raises(ValueError, match="max_iters"):
        await ctx.loop_until(body, done=lambda acc: True, max_iters=0)

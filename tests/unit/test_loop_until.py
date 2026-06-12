"""Unit tests for ``ctx.loop_until`` — the measured-stop loop helper."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError
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


async def test_loop_until_body_failure_preserves_accumulated_on_partial() -> None:
    """A mid-loop body raise re-raises the SAME exception, carrying the survivors.

    Iterations 0 and 1 complete and journal their results; iteration 2 raises a
    ``RuntimeError``. The original exception type must propagate (so a caller's
    ``except RuntimeError`` still catches it) with a ``.partial`` attribute carrying
    the two results accumulated before the failure — fail-fast-but-recoverable, the
    same graceful-degradation channel parallel (None-on-failure) and race provide.
    """
    leaf = _SeqLeaf(["R:try 0", "R:try 1", "R:try 2"])
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=lambda _e: None))

    async def body(i: int, acc: list[str]) -> str:
        if i == 2:
            raise RuntimeError("body boom on iteration 2")
        return await ctx.agent(f"try {i}", agent_type="worker")

    with pytest.raises(RuntimeError) as exc_info:
        await ctx.loop_until(body, done=lambda acc: False, max_iters=3)
    assert getattr(exc_info.value, "partial", None) == ["R:try 0", "R:try 1"]
    # Two leaves ran before the failure; the failing iteration issued no agent() call.
    assert leaf.calls == 2


async def test_loop_until_control_flow_signal_aborts_without_partial() -> None:
    """A control-flow signal raised by body propagates UNTOUCHED, with no ``.partial``.

    A clean engine abort (an exhausted budget, a determinism break, a checkpoint
    reached from a fan-out) is structurally distinct from a recoverable body failure:
    it tears the run down rather than handing back survivors, so it must NOT acquire a
    ``.partial`` channel. This pins that the clean-abort path stays separate from the
    fail-fast-but-recoverable path.
    """
    leaf = _SeqLeaf(["R:try 0", "R:try 1"])
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=lambda _e: None))

    async def body(i: int, acc: list[str]) -> str:
        if i == 1:
            raise WorkflowBudgetExceededError("budget exhausted mid-loop")
        return await ctx.agent(f"try {i}", agent_type="worker")

    with pytest.raises(WorkflowBudgetExceededError) as exc_info:
        await ctx.loop_until(body, done=lambda acc: False, max_iters=5)
    assert not hasattr(exc_info.value, "partial")


@dataclasses.dataclass(frozen=True)
class _FrozenBodyError(Exception):
    """An immutable body exception whose instances reject attribute assignment.

    A frozen dataclass exception cannot carry a ``.partial`` attribute — assigning
    one raises ``FrozenInstanceError`` (a ``dataclasses`` subclass of ``AttributeError``).
    It stands in for any immutable exception (frozen dataclass, ``__slots__`` without
    ``partial``, or a blocking ``__setattr__``) a body might raise.
    """

    detail: str


async def test_loop_until_immutable_body_exception_degrades_to_no_partial() -> None:
    """An immutable body exception propagates by its ORIGINAL type, without ``.partial``.

    The partial-attach must never mask the original failure: if the exception cannot
    carry attributes, the attach is skipped and the original exception propagates
    cleanly (no partial channel) rather than being replaced by a
    ``FrozenInstanceError`` / ``AttributeError`` / ``TypeError``. That replacement
    would lose BOTH the original type (breaking a caller's ``except _FrozenBodyError``)
    and the partial — strictly worse than the pre-fix baseline. This pins the graceful
    degradation: type preserved, no partial.
    """
    leaf = _SeqLeaf(["R:try 0", "R:try 1"])
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=lambda _e: None))

    async def body(i: int, acc: list[str]) -> str:
        if i == 2:
            raise _FrozenBodyError("body boom on iteration 2")
        return await ctx.agent(f"try {i}", agent_type="worker")

    with pytest.raises(_FrozenBodyError) as exc_info:
        await ctx.loop_until(body, done=lambda acc: False, max_iters=3)
    # The original type propagated (not an AttributeError/FrozenInstanceError/TypeError),
    # and no partial channel was forced onto the immutable instance.
    assert getattr(exc_info.value, "partial", None) is None
    assert leaf.calls == 2  # two leaves ran before the failing iteration

"""Unit tests for ``Ctx.parallel`` fan-out semantics (barrier + None-on-error)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._errors import WorkflowNestingError
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._roster import Roster


def _noop_runnable() -> Runnable[Any, Any]:
    """A roster placeholder; the cap test drives a custom leaf_runner instead."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        return {"messages": []}

    return RunnableLambda(_call)


def _ctx() -> Ctx:
    """Build a Ctx with a no-op leaf runner (parallel tests drive thunks directly)."""

    async def _leaf(
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
        return LeafOutcome(state={"messages": []}, usage=0)

    return Ctx(
        roster=Roster(),
        journal=InMemoryJournalStore(),
        leaf_runner=_leaf,
        gate=ConcurrencyGate(limit=8),
    )


async def test_parallel_returns_results_in_input_order() -> None:
    ctx = _ctx()

    async def make(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    # Reverse the completion order via delays; result order must still follow input order.
    results = await ctx.parallel(
        [
            lambda: make(0, 0.03),
            lambda: make(1, 0.0),
            lambda: make(2, 0.02),
        ]
    )
    assert results == [0, 1, 2]


async def test_parallel_failed_thunk_becomes_none_and_does_not_raise() -> None:
    ctx = _ctx()

    async def ok(value: int) -> int:
        return value

    async def boom() -> int:
        raise RuntimeError("thunk exploded")

    results = await ctx.parallel([lambda: ok(10), boom, lambda: ok(30)])
    # Failure lands as None in-place; the call as a whole never raises.
    assert results == [10, None, 30]
    # Idiomatic downstream filtering works.
    assert [r for r in results if r is not None] == [10, 30]


async def test_parallel_empty_returns_empty_list() -> None:
    ctx = _ctx()
    assert await ctx.parallel([]) == []


async def test_parallel_reraises_nesting_error_loud() -> None:
    # A WorkflowNestingError (a structural depth-cap breach) raised inside a thunk
    # must fail loud through the barrier, not be masked as a None hole.
    # This is the regression for the M7 Codex-review BLOCKER fix.
    ctx = _ctx()

    async def _breach() -> str:
        raise WorkflowNestingError("too deep")

    with pytest.raises(WorkflowNestingError):
        await ctx.parallel([_breach])


async def test_parallel_respects_concurrency_gate() -> None:
    # The shared gate caps in-flight LEAVES (agent() calls), the real unit of
    # work — not orchestration frames. Thunks fan out through ctx.agent, whose
    # leaf runner is the single chokepoint that acquires the gate.
    gate = ConcurrencyGate(limit=2)
    in_flight = 0
    peak = 0

    async def _leaf(
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
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return LeafOutcome(state={"messages": [], "result": prompt}, usage=0)

    ctx = Ctx(
        roster=Roster().register("worker", _noop_runnable()),
        journal=InMemoryJournalStore(),
        leaf_runner=_leaf,
        gate=gate,
    )

    results = await ctx.parallel(
        [lambda i=i: ctx.agent(f"q{i}", agent_type="worker") for i in range(10)]
    )
    assert len(results) == 10
    assert peak <= 2
    assert peak == 2  # the cap actually saturates, not over- or under-shoots

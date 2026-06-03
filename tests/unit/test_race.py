"""Unit tests for ``Ctx.race`` — fresh-path mechanics + loser cancellation.

These build a ``Ctx`` directly with a fake ``leaf_runner`` (mirroring
``tests/unit/test_parallel.py``) so each candidate's result, usage, and failure is
controlled without a real model. Replay / nesting / footgun behaviour is covered by
``tests/integration/test_race.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow._budget import Budget
from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._race_types import RaceCandidate, RaceResult
from langchain_dynamic_workflow._roster import Roster


def _noop_runnable() -> Runnable[Any, Any]:
    """A roster placeholder; race tests drive results through a custom leaf_runner."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        return {"messages": []}

    return RunnableLambda(_call)


def _text_runner(
    results: dict[str, str], *, raise_on: frozenset[str] = frozenset(), usage: int = 0
) -> Any:
    """A fake leaf_runner returning a text result keyed by prompt (or raising)."""

    async def _leaf(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
    ) -> LeafOutcome:
        if prompt in raise_on:
            raise RuntimeError("fake leaf boom")
        return LeafOutcome(state={"messages": [AIMessage(content=results[prompt])]}, usage=usage)

    return _leaf


def _ctx(
    leaf_runner: Any,
    *,
    budget: Budget | None = None,
    gate: ConcurrencyGate | None = None,
    journal: InMemoryJournalStore | None = None,
) -> Ctx:
    return Ctx(
        roster=Roster()
        .register("inv", _noop_runnable())
        .register("winner", _noop_runnable())
        .register("loser", _noop_runnable()),
        journal=journal if journal is not None else InMemoryJournalStore(),
        leaf_runner=leaf_runner,
        gate=gate if gate is not None else ConcurrencyGate(limit=8),
        budget=budget,
    )


async def test_race_first_to_satisfy_win_wins() -> None:
    ctx = _ctx(_text_runner({"h0": "lose", "h1": "WIN", "h2": "lose"}))
    result: RaceResult[str] = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1", "h2")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.won is True
    assert result.winner == "WIN"
    assert result.winner_index == 1


async def test_race_ascending_index_tiebreak() -> None:
    # Every candidate satisfies win; the lowest input index must win regardless of
    # completion / set-iteration order, so the winner is deterministic.
    ctx = _ctx(_text_runner({"h0": "WIN", "h1": "WIN", "h2": "WIN"}))
    result: RaceResult[str] = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1", "h2")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.winner_index == 0


async def test_race_no_winner_returns_unwon_result() -> None:
    ctx = _ctx(_text_runner({"h0": "lose", "h1": "lose"}))
    result: RaceResult[str] = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.won is False
    assert result.winner is None and result.winner_index is None


async def test_race_failed_candidate_is_skipped_others_continue() -> None:
    # The lower-index candidate's leaf raises; it cannot win, and the next candidate
    # that satisfies win takes the race.
    ctx = _ctx(_text_runner({"h1": "WIN"}, raise_on=frozenset({"h0"})))
    result: RaceResult[str] = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.winner == "WIN"
    assert result.winner_index == 1


async def test_race_empty_candidates_raises() -> None:
    ctx = _ctx(_text_runner({}))
    with pytest.raises(ValueError, match="at least one candidate"):
        await ctx.race([], win=lambda text: True, win_tag="t")


async def test_race_mixed_schema_raises() -> None:
    # One schema-less candidate + one schema candidate would make the winner type
    # ambiguous; the homogeneity guard fails fast before any dispatch.
    ctx = _ctx(_text_runner({"h0": "x", "h1": "y"}))
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="homogeneous"):
        await ctx.race(
            [
                RaceCandidate(prompt="h0", agent_type="inv"),
                RaceCandidate(prompt="h1", agent_type="inv", schema=schema),
            ],
            win=lambda obj: True,
            win_tag="t",
        )


async def test_race_predicate_raise_fails_loud() -> None:
    # The win predicate is script logic; a raise is a bug, not a leaf failure — it
    # must propagate (after the in-flight losers are torn down), never be swallowed.
    ctx = _ctx(_text_runner({"h0": "WIN", "h1": "lose"}))

    def boom(_text: str) -> bool:
        raise ValueError("predicate boom")

    with pytest.raises(ValueError, match="predicate boom"):
        await ctx.race(
            [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
            win=boom,
            win_tag="t",
        )


async def test_race_budget_signal_fails_loud() -> None:
    # An exhausted budget makes each candidate's agent() raise the engine
    # control-flow signal; the race must fail loud rather than mask it as a loser.
    ctx = _ctx(_text_runner({"h0": "WIN", "h1": "lose"}), budget=Budget(total=0))
    with pytest.raises(WorkflowBudgetExceededError):
        await ctx.race(
            [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
            win=lambda text: text == "WIN",
            win_tag="t",
        )


async def test_race_replay_records_winner_under_leaf_key_not_race_key() -> None:
    # Replay must reconstruct the winner's spend under the winner's LEAF key (the key
    # the fresh run counted it under), never the race-key. A single winning candidate
    # is journaled on the fresh run; on replay the race dispatches nothing but rebuilds
    # spend. If it recorded under the race-key, a later agent() with the winner's exact
    # params would record the same usage AGAIN under the leaf key (rkey != leaf_key),
    # double-counting one leaf's tokens and breaking fresh-vs-replay spend agreement.
    journal = InMemoryJournalStore()
    runner = _text_runner({"h0": "WIN"}, usage=7)
    cands = [RaceCandidate(prompt="h0", agent_type="inv")]

    # Fresh run: the winner's agent() records 7 under its leaf key; rkey is not a budget
    # key. A repeated identical agent() is a journal hit, idempotent -> spent stays 7.
    fresh_budget = Budget(total=None)
    fresh = _ctx(runner, budget=fresh_budget, journal=journal)
    r1: RaceResult[str] = await fresh.race(cands, win=lambda text: text == "WIN", win_tag="t")
    await fresh.agent("h0", agent_type="inv")  # winner's exact params -> same leaf key
    assert r1.winner == "WIN"
    assert fresh_budget.spent() == 7

    # Replay run (shares the journal -> rkey hit, zero dispatch). Spend is rebuilt under
    # the winner's leaf key, so the later identical agent() stays idempotent and the
    # winner's 7 tokens are counted ONCE — not 7 (rkey) + 7 (leaf key) = 14.
    replay_budget = Budget(total=None)
    replay = _ctx(runner, budget=replay_budget, journal=journal)
    r2: RaceResult[str] = await replay.race(cands, win=lambda text: text == "WIN", win_tag="t")
    await replay.agent("h0", agent_type="inv")
    assert r2.winner == "WIN"
    assert replay_budget.spent() == 7


async def test_race_cancels_losers_and_releases_gate_slots() -> None:
    # The winner completes immediately; the losers block forever until cancelled.
    # On a win the race must cancel them, await their teardown (no orphans), and
    # release every gate slot they held.
    gate = ConcurrencyGate(limit=4)
    cancelled = {"count": 0}

    async def _leaf(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
    ) -> LeafOutcome:
        if agent_type == "winner":
            return LeafOutcome(state={"messages": [AIMessage(content="WIN")]}, usage=0)
        try:
            await asyncio.Event().wait()  # block until cancelled
        except asyncio.CancelledError:
            cancelled["count"] += 1
            raise
        return LeafOutcome(state={"messages": []}, usage=0)  # pragma: no cover

    ctx = _ctx(_leaf, gate=gate)
    result: RaceResult[str] = await ctx.race(
        [
            RaceCandidate(prompt="w", agent_type="winner"),
            RaceCandidate(prompt="l1", agent_type="loser"),
            RaceCandidate(prompt="l2", agent_type="loser"),
        ],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.winner == "WIN" and result.winner_index == 0
    assert cancelled["count"] == 2  # both losers received CancelledError during teardown

    # No slot leaked: all `limit` slots must be acquirable SIMULTANEOUSLY again. If a
    # cancelled loser had leaked its slot, this all-in-flight barrier would dead-lock
    # — wait_for bounds it so the test fails loud instead of hanging.
    entered = asyncio.Semaphore(0)
    release = asyncio.Event()

    async def _occupy() -> None:
        async with gate:
            entered.release()
            await release.wait()

    occupants = [asyncio.ensure_future(_occupy()) for _ in range(gate.limit)]
    try:
        for _ in range(gate.limit):
            await asyncio.wait_for(entered.acquire(), timeout=1.0)
    finally:
        release.set()
        await asyncio.gather(*occupants)

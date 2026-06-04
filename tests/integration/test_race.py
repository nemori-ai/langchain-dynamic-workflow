"""Integration: ``ctx.race`` through the full ``run_workflow`` resume loop.

Drives the engine with deterministic fake leaves (no API keys) to prove: a
journaled race decision reproduces the winner on resume and dispatches no
candidate; a race nested inside ``parallel`` resumes by content hash even though
its key is excluded from the determinism sequence; and the win_tag footgun — two
races over identical candidates with the same (default) tag alias one decision.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow
from langchain_dynamic_workflow._race_types import RaceCandidate, RaceResult


def _counting_runnable(calls: dict[str, int]) -> Runnable[Any, Any]:
    """A leaf that counts dispatches and replies WIN to the prompt ending in '0'."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        prompt = inp["messages"][0].content
        content = "WIN" if str(prompt).endswith("0") else "lose"
        return {"messages": [*inp["messages"], AIMessage(content=content)]}

    return RunnableLambda(_call)


async def test_race_replay_reproduces_winner_and_dispatches_nothing() -> None:
    calls = {"n": 0}
    roster = Roster().register("inv", _counting_runnable(calls))
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> tuple[int | None, Any]:
        result: RaceResult[str] = await ctx.race(
            [RaceCandidate(prompt=f"h{i}", agent_type="inv") for i in range(3)],
            win=lambda text: text == "WIN",
            win_tag="x",
        )
        return (result.winner_index, result.winner)

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched_on_first = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    # h0 is the only WIN, so the winner is index 0 regardless of completion order.
    assert first == second == (0, "WIN")
    assert dispatched_on_first >= 1  # at least the winner ran on the fresh run
    assert calls["n"] == dispatched_on_first  # replay dispatched NOTHING


async def test_nested_race_in_parallel_resumes_by_content_hash() -> None:
    # A race inside parallel runs at fan-out depth > 0, so its race-key is excluded
    # from the determinism sequence — yet the decision is still journaled by content
    # hash, so the whole workflow resumes and the nested race dispatches nothing.
    calls = {"n": 0}
    roster = Roster().register("inv", _counting_runnable(calls))
    journal = InMemoryJournalStore()
    items = ["a", "b"]

    async def orchestrate(ctx: Ctx) -> list[Any]:
        async def race_for(item: str) -> Any:
            result: RaceResult[str] = await ctx.race(
                [RaceCandidate(prompt=f"{item}-h{j}", agent_type="inv") for j in range(2)],
                win=lambda text: text == "WIN",
                win_tag="nested",
            )
            return result.winner

        return await ctx.parallel([lambda item=item: race_for(item) for item in sorted(items)])

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched_on_first = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert first == second == ["WIN", "WIN"]
    assert calls["n"] == dispatched_on_first  # nested race replayed with zero dispatch


async def test_same_default_win_tag_aliases_the_decision_footgun() -> None:
    # The footgun: two races over identical candidates with the same (default """)
    # win_tag share one race-key, so the second replays the first's decision even
    # though its predicate differs. A distinct win_tag keeps them independent.
    calls = {"n": 0}
    roster = Roster().register("inv", _counting_runnable(calls))

    async def orchestrate(ctx: Ctx) -> tuple[Any, Any, Any]:
        cands = [RaceCandidate(prompt=f"h{i}", agent_type="inv") for i in range(2)]
        first: RaceResult[str] = await ctx.race(
            cands, win=lambda text: text == "WIN"
        )  # default win_tag ""
        aliased: RaceResult[str] = await ctx.race(
            cands, win=lambda text: text == "lose"
        )  # SAME key -> aliases
        independent: RaceResult[str] = await ctx.race(
            cands, win=lambda text: text == "lose", win_tag="distinct"
        )
        return (first.winner, aliased.winner, independent.won)

    out = await run_workflow(orchestrate, roster=roster)
    first_winner, aliased_winner, independent_won = out
    # The aliased race returns the first race's winner (WIN), NOT a "lose" result.
    assert first_winner == "WIN"
    assert aliased_winner == "WIN"
    # The distinctly-tagged race runs for real: no candidate replies "lose" at h0, so
    # h0 (the only one the WIN-style winner would pick) is "WIN" not "lose" -> the
    # winner is the candidate replying "lose", which is h1; it found a winner.
    assert independent_won is True

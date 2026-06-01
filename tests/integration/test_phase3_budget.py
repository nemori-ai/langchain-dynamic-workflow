"""Phase 3 integration: shared budget enforcement, replay spend, model override.

These tests drive ``run_workflow`` with fake leaves that meter token usage (no
API keys), proving: the shared budget meters per-leaf usage and enforces a cap
(:class:`WorkflowBudgetExceededError`), ``spent()`` rebuilds identically on
resume from the journal, and an ``agent(model=...)`` override actually reaches
leaf execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    WorkflowBudgetExceededError,
    run_workflow,
)

UsageLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]
ModelEchoLeafFactory = Callable[[], Runnable[Any, Any]]


async def test_budget_meters_usage_and_exposes_spent(make_usage_leaf: UsageLeafFactory) -> None:
    # Each leaf reports 10 tokens; two distinct leaves => spent() == 20.
    leaf, _model = make_usage_leaf("ok", tokens_per_call=10)
    roster = Roster().register("worker", leaf)
    captured: dict[str, int] = {}

    async def orchestrate(ctx: Ctx) -> None:
        await ctx.agent("a", agent_type="worker")
        await ctx.agent("b", agent_type="worker")
        captured["spent"] = ctx.budget.spent()
        captured["remaining"] = int(ctx.budget.remaining())

    await run_workflow(orchestrate, roster=roster, thread_id="t1", budget=100)
    assert captured["spent"] == 20
    assert captured["remaining"] == 80


async def test_budget_cap_blocks_new_leaf_inflight_results_kept(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # total=20, each leaf=10: two leaves exhaust the pool; the third agent() call
    # must raise. The two completed leaves keep their results (the run reaches the
    # raising third call only after the first two succeeded and journaled).
    leaf, _model = make_usage_leaf("ok", tokens_per_call=10)
    roster = Roster().register("worker", leaf)
    completed: list[str] = []

    async def orchestrate(ctx: Ctx) -> None:
        completed.append(await ctx.agent("a", agent_type="worker"))
        completed.append(await ctx.agent("b", agent_type="worker"))
        # Pool is now exhausted (spent 20 of 20); the next dispatch is refused.
        await ctx.agent("c", agent_type="worker")

    with pytest.raises(WorkflowBudgetExceededError, match="exhausted"):
        await run_workflow(orchestrate, roster=roster, thread_id="t1", budget=20)
    # The two leaves that ran before the cap tripped kept their results.
    assert completed == ["ok", "ok"]


async def test_spent_rebuilds_identically_on_resume(make_usage_leaf: UsageLeafFactory) -> None:
    # The backstop's premise: spent() reconstructed on resume from the journal
    # equals the first run's total. The first run journals two leaves; the resume
    # serves both from cache (zero model calls) yet rebuilds the same spend.
    leaf, model = make_usage_leaf("ok", tokens_per_call=10)
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()
    spends: dict[str, int] = {}

    async def orchestrate(ctx: Ctx) -> None:
        await ctx.agent("a", agent_type="worker")
        await ctx.agent("b", agent_type="worker")
        spends.setdefault("first" if "first" not in spends else "second", ctx.budget.spent())

    await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1", budget=100)
    first_spent = spends["first"]
    calls_after_first = model.calls

    await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2", budget=100)
    second_spent = spends["second"]

    assert first_spent == 20
    assert second_spent == first_spent  # rebuilt identically from journal usage
    assert model.calls == calls_after_first  # zero new model calls on resume


async def test_no_total_budget_remaining_is_infinite(make_usage_leaf: UsageLeafFactory) -> None:
    # Loop-until-budget needs an unbounded remaining() when no total is set, so a
    # script can run as many leaves as it likes without ever tripping the cap.
    leaf, _model = make_usage_leaf("ok", tokens_per_call=1000)
    roster = Roster().register("worker", leaf)
    observed: dict[str, Any] = {}

    async def orchestrate(ctx: Ctx) -> None:
        assert ctx.budget.total is None
        for i in range(5):
            await ctx.agent(f"q{i}", agent_type="worker")
        observed["remaining"] = ctx.budget.remaining()

    await run_workflow(orchestrate, roster=roster, thread_id="t1")  # no budget=
    assert observed["remaining"] == float("inf")


async def test_model_override_reaches_leaf_execution(
    make_model_echo_leaf: ModelEchoLeafFactory,
) -> None:
    # Regression for the key-vs-execution gap (Phase 2 review minor #5): agent()
    # folds model into the journal key, but the leaf must actually run with that
    # model. The echo leaf reports the model override it received; different
    # overrides must yield different results (distinct execution, distinct keys).
    roster = Roster().register("worker", make_model_echo_leaf())

    async def orchestrate(ctx: Ctx) -> list[str]:
        return [
            await ctx.agent("q", agent_type="worker", model="opus"),
            await ctx.agent("q", agent_type="worker", model="haiku"),
            await ctx.agent("q", agent_type="worker"),
        ]

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert results == ["ran-with:opus", "ran-with:haiku", "ran-with:default"]

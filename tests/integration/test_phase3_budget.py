"""Phase 3 integration: shared budget enforcement, replay spend, model override.

These tests drive ``run_workflow`` with fake leaves that meter token usage (no
API keys), proving: the shared budget meters per-leaf usage and enforces a cap
(:class:`WorkflowBudgetExceededError`), ``spent()`` rebuilds identically on
resume from the journal, and an ``agent(model=...)`` override actually reaches
leaf execution.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import pytest
from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import PrivateAttr

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


async def test_default_model_is_used_when_no_override(
    make_model_echo_leaf: ModelEchoLeafFactory,
) -> None:
    # A leaf registered with default_model and called with no model= override must
    # run with the default (config-aware leaf honors it). Pins that default_model
    # is live configuration, not dead — the effective model resolves to the
    # registered default and reaches leaf execution.
    roster = Roster().register("worker", make_model_echo_leaf(), default_model="sonnet")

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("q", agent_type="worker")

    result = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert result == "ran-with:sonnet"


async def test_override_beats_default_model_and_shares_key_with_explicit_call() -> None:
    # The effective model is "override else default_model", and it is what feeds
    # BOTH the journal key and the leaf config. So an explicit model="sonnet" call
    # and a no-override call against a default_model="sonnet" leaf resolve to the
    # same effective model: one journal entry, one model run, the second served
    # from cache. This pins that key and execution are derived from one value and
    # cannot disagree. The leaf counts its runs and echoes the model it received.
    runs: dict[str, int] = {"n": 0}

    async def _counting(
        inp: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        runs["n"] += 1
        configurable = (config or {}).get("configurable", {})
        model = configurable.get("model", "default")
        return {"messages": [*inp["messages"], AIMessage(content=f"ran-with:{model}")]}

    roster = Roster().register("worker", RunnableLambda(_counting), default_model="sonnet")

    async def orchestrate(ctx: Ctx) -> list[str]:
        return [
            await ctx.agent("q", agent_type="worker", model="sonnet"),  # explicit
            await ctx.agent("q", agent_type="worker"),  # falls back to default_model
        ]

    results = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert results == ["ran-with:sonnet", "ran-with:sonnet"]
    assert runs["n"] == 1  # the second call was a journal cache hit (same effective model)


class _BarrierUsageModel(BaseChatModel):
    """A usage-metering chat model that parks every call at a barrier first.

    Awaiting a shared :class:`asyncio.Barrier` in ``_agenerate`` forces all
    concurrently-dispatched leaves to be in flight *simultaneously* before any of
    them returns and records usage. That makes the budget pre-dispatch check
    (``ensure_within_cap``) run for every leaf in the barrier before a single
    ``record`` lands — deterministically exercising the soft-cap concurrent
    overshoot path rather than relying on a scheduling race.

    Attributes:
        parties: The number of concurrent calls the barrier waits for.
        tokens_per_call: Total tokens reported per generation.
        model_name: The model name emitted so the usage callback aggregates it.
    """

    parties: int
    tokens_per_call: int = 10
    model_name: str = "barrier-usage-model"
    _barrier: asyncio.Barrier | None = PrivateAttr(default=None)
    _calls: int = PrivateAttr(default=0)

    @property
    def calls(self) -> int:
        """Number of generations performed."""
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "barrier-usage-fake"

    def _get_barrier(self) -> asyncio.Barrier:
        # Lazily create the barrier on the running loop the first call lands on, so
        # construction does not depend on an event loop already being active.
        if self._barrier is None:
            self._barrier = asyncio.Barrier(self.parties)
        return self._barrier

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("barrier model is async-only")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Block until every party in the barrier has arrived: this guarantees all
        # concurrent leaves are in flight before any returns and records usage.
        await self._get_barrier().wait()
        self._calls += 1
        usage = UsageMetadata(
            input_tokens=self.tokens_per_call,
            output_tokens=0,
            total_tokens=self.tokens_per_call,
        )
        message = AIMessage(
            content="note",
            usage_metadata=usage,
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        """Ignore tools and return self (the fake never emits tool calls)."""
        return self


async def test_parallel_barrier_overshoots_soft_cap_keeping_inflight_results() -> None:
    # The budget is a SOFT cap, not a hard ceiling: under parallel() all N thunks
    # pass the pre-dispatch ensure_within_cap() before any has recorded usage, so a
    # single barrier can overshoot total by up to the combined usage of the leaves
    # admitted in that window. With total=10 (one leaf's worth) but a barrier of 4
    # leaves dispatched while the pool still reads as having headroom, all 4 are
    # admitted and complete, spending 40 — a 4x overshoot. The acceptance criterion
    # "in-flight leaves keep their results" must hold concurrently, not just for the
    # trivial sequential one-extra-call case. The next agent() after the barrier
    # settles is what finally refuses.
    parties = 4
    model = _BarrierUsageModel(parties=parties, tokens_per_call=10)
    leaf = cast(Runnable[Any, Any], create_deep_agent(model=model))
    roster = Roster().register("worker", leaf)
    captured: dict[str, Any] = {}

    async def orchestrate(ctx: Ctx) -> None:
        # total=10 means the pool has room for exactly one leaf, yet the barrier of
        # 4 all pass ensure_within_cap() before any records and overshoot to 40.
        results = await ctx.parallel(
            [lambda i=i: ctx.agent(f"q{i}", agent_type="worker") for i in range(parties)]
        )
        captured["results"] = results
        captured["spent"] = ctx.budget.spent()
        # After the barrier settles the pool is over budget; remaining floors at 0.
        captured["remaining"] = ctx.budget.remaining()

    await run_workflow(
        orchestrate,
        roster=roster,
        thread_id="t1",
        budget=10,
        max_concurrency=parties,  # let all four be in flight at once
        on_progress=lambda _e: None,
    )
    # All four leaves were admitted and kept their results (none cancelled to claw
    # back tokens): the soft cap overshot to 4x its total.
    assert captured["results"] == ["note", "note", "note", "note"]
    assert captured["spent"] == parties * 10  # 40: a four-leaf overshoot of a one-leaf cap
    assert captured["remaining"] == 0  # floored, never negative
    assert model.calls == parties


async def test_post_overshoot_dispatch_is_refused(make_usage_leaf: UsageLeafFactory) -> None:
    # The other half of the soft-cap contract: once a barrier has overshot the cap,
    # the NEXT sequential agent() sees the overshot spent() and refuses loud. This
    # pins that the overshoot is bounded to a single barrier — the cap is not
    # disabled, it is enforced again the moment a new dispatch is attempted.
    leaf, _model = make_usage_leaf("note", tokens_per_call=10)
    roster = Roster().register("worker", leaf)

    async def orchestrate(ctx: Ctx) -> None:
        # A small sequential barrier overshoots total=10 to 20...
        await ctx.parallel([lambda i=i: ctx.agent(f"q{i}", agent_type="worker") for i in range(2)])
        # ...and the next dispatch is refused now that spent() >= total.
        await ctx.agent("after", agent_type="worker")

    with pytest.raises(WorkflowBudgetExceededError, match="exhausted"):
        await run_workflow(
            orchestrate,
            roster=roster,
            thread_id="t1",
            budget=10,
            on_progress=lambda _e: None,
        )

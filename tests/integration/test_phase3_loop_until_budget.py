"""Phase 3 integration: the loop-until-budget end-to-end demo behaviour.

A budget-guarded loop accumulates research leaves until the shared token pool is
nearly exhausted, narrating progress as it goes. This is the M3 milestone demo
shape; the tests assert the three locked guarantees: the loop terminates at the
cap (it does not over-run), ``spent()`` rebuilds identically on resume from the
journal, and a hard cap mid-loop raises :class:`WorkflowBudgetExceededError`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    run_workflow,
)

UsageLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


def _echo_prompt_leaf(prefix: str) -> Runnable[Any, Any]:
    """A content-deterministic leaf that returns ``{prefix}{prompt}``.

    The engine invokes a leaf with the prompt as the final human message and folds
    the final AIMessage content as the agent() result, so the reply is a pure
    function of the prompt — and thus of the content-hash journal key. This is what
    lets a resume serve every leaf from the journal and reproduce the same result.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].content
        return {"messages": [*inp["messages"], AIMessage(content=f"{prefix}{prompt}")]}

    return RunnableLambda(_call)


def _loop_until_budget(
    *, threshold: int, topics: list[str]
) -> Callable[[Ctx], Awaitable[dict[str, Any]]]:
    """Build a loop-until-budget orchestration over a fixed topic list.

    The script fans out one research leaf per iteration while the budget has more
    than ``threshold`` tokens of headroom, narrating each step, and stops
    gracefully once the remaining pool drops to the threshold.
    """

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        ctx.phase("budgeted research")
        findings: list[str] = []
        for topic in topics:
            # The loop guard: stop before the pool is exhausted. With no total
            # this is a no-op (remaining() is inf) — but the demo always sets one.
            if ctx.budget.remaining() <= threshold:
                ctx.log(f"stopping: only {int(ctx.budget.remaining())} tokens left")
                break
            ctx.log(f"researching {topic} (remaining={int(ctx.budget.remaining())})")
            findings.append(await ctx.agent(f"Research {topic}", agent_type="researcher"))
        return {"findings": findings, "spent": ctx.budget.spent()}

    return orchestrate


async def test_loop_terminates_at_budget(make_usage_leaf: UsageLeafFactory) -> None:
    # total=50, threshold=10, each leaf=10: the loop runs while remaining > 10,
    # i.e. up to spent 40 (remaining 10), then stops — 4 leaves, never over-run.
    leaf, _model = make_usage_leaf("note", tokens_per_call=10)
    roster = Roster().register("researcher", leaf)
    topics = [f"t{i}" for i in range(10)]

    result = await run_workflow(
        _loop_until_budget(threshold=10, topics=topics),
        roster=roster,
        thread_id="t1",
        budget=50,
        on_progress=lambda _e: None,
    )
    assert len(result["findings"]) == 4
    assert result["spent"] == 40  # stopped gracefully one threshold short of the cap


async def test_loop_spent_rebuilds_identically_on_resume(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # The backstop premise end-to-end: a resumed loop serves every leaf from the
    # journal (zero model calls) yet rebuilds the same spend and the same result.
    leaf, model = make_usage_leaf("note", tokens_per_call=10)
    roster = Roster().register("researcher", leaf)
    journal = InMemoryJournalStore()
    topics = [f"t{i}" for i in range(10)]

    first = await run_workflow(
        _loop_until_budget(threshold=10, topics=topics),
        roster=roster,
        journal=journal,
        thread_id="t1",
        budget=50,
        on_progress=lambda _e: None,
    )
    calls_after_first = model.calls

    second = await run_workflow(
        _loop_until_budget(threshold=10, topics=topics),
        roster=roster,
        journal=journal,
        thread_id="t2",
        budget=50,
        on_progress=lambda _e: None,
    )

    assert first["spent"] == second["spent"] == 40
    assert first["findings"] == second["findings"]
    assert model.calls == calls_after_first  # resume served every leaf from journal


async def test_hard_cap_raises_when_loop_overshoots(make_usage_leaf: UsageLeafFactory) -> None:
    # A script that ignores the soft guard and dispatches past the cap must be
    # stopped loud by the hard enforcement, not allowed to silently over-spend.
    leaf, _model = make_usage_leaf("note", tokens_per_call=10)
    roster = Roster().register("researcher", leaf)

    async def greedy(ctx: Ctx) -> None:
        # No remaining() guard: blindly dispatch more leaves than the pool allows.
        for i in range(10):
            await ctx.agent(f"q{i}", agent_type="researcher")

    with pytest.raises(WorkflowBudgetExceededError, match="exhausted"):
        await run_workflow(
            greedy,
            roster=roster,
            thread_id="t1",
            budget=30,  # 3 leaves exhaust it; the 4th dispatch is refused
            on_progress=lambda _e: None,
        )


async def test_loop_until_fanout_body_iteration_drift_fails_loud_on_resume() -> None:
    """A fan-out body's loop-count drift is caught by the per-iteration loop key.

    The body does its ``agent()`` work INSIDE ``ctx.parallel``, so the leaves run at
    fan-out depth > 0 and are excluded from the depth-0 determinism sequence — the
    loop's leaves contribute zero recorded keys. Without a per-iteration loop key the
    finalize-time count check (``observed < recorded``) sees 0 < 0 and a resume whose
    iteration count drifts (an external/mutable ``done`` stop point) silently returns a
    different-length list. The per-iteration ``loop_key`` makes the drift fail loud:
    the first run records one loop key per iteration; a resume that runs more
    iterations observes a key beyond the recorded sequence and raises.
    """
    roster = Roster().register("worker", _echo_prompt_leaf("done:"))
    journal = InMemoryJournalStore()
    stop_point = {"value": 3}

    async def orchestrate(ctx: Ctx) -> list[Any]:
        async def body(iteration: int, accumulated: list[Any]) -> Any:
            results = await ctx.parallel(
                [lambda it=iteration: ctx.agent(f"iter{it}", agent_type="worker")]
            )
            return results[0]

        return await ctx.loop_until(
            body, done=lambda acc: len(acc) >= stop_point["value"], max_iters=10
        )

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert len(first) == 3

    # Resume the SAME journal with the external stop point moved out: the loop now
    # wants 5 iterations. The drift must be caught by the determinism backstop rather
    # than silently returning a 5-item list.
    stop_point["value"] = 5
    with pytest.raises(WorkflowDeterminismError):
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")


async def test_loop_until_non_fanout_body_resumes_cleanly_at_zero_drift() -> None:
    """The per-iteration loop key does not break a deterministic resume (no false positive).

    A loop with a non-fan-out body (its ``agent()`` calls run at depth 0) and a
    deterministic ``done`` must resume from the journal at zero drift: every leaf AND
    every loop key replays in the same order, so no spurious ``WorkflowDeterminismError``
    fires and the result rebuilds identically. This pins that adding the loop key to the
    determinism sequence leaves the happy path intact.
    """
    roster = Roster().register("worker", _echo_prompt_leaf("done:"))
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[Any]:
        async def body(iteration: int, accumulated: list[Any]) -> Any:
            # Depth-0 agent() call: its leaf key AND the iteration's loop key both
            # land in the determinism sequence, interleaved per iteration.
            return await ctx.agent(f"iter{iteration}", agent_type="worker")

        return await ctx.loop_until(body, done=lambda acc: len(acc) >= 3, max_iters=10)

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert first == ["done:iter0", "done:iter1", "done:iter2"]

    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == first  # resume rebuilds identically, no spurious determinism error

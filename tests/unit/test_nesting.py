"""Unit tests for ``ctx.workflow`` N-level nesting with a depth cap + cycle guard.

A workflow may inline other workflows up to ``max_workflow_depth`` levels; exceeding
the cap raises ``WorkflowNestingError`` and re-entering a name already on the inlining
stack (a cycle) raises ``WorkflowCycleError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Ctx, Roster, run_workflow
from langchain_dynamic_workflow._errors import WorkflowCycleError, WorkflowNestingError
from langchain_dynamic_workflow._workflows import WorkflowRegistry

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_two_level_nesting_now_succeeds(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("deep-finding")
    roster = Roster().register("researcher", leaf)

    async def inner(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("leaf", agent_type="researcher")

    async def middle(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow("inner", {})

    workflows = WorkflowRegistry().register("inner", inner).register("middle", middle)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("middle", {})  # depth 1 -> 2: previously refused

    result = await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1")
    assert result == "deep-finding"


async def test_exceeding_max_workflow_depth_raises(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def step(ctx: Ctx, args: dict[str, Any]) -> str:
        # Each level inlines a DISTINCT next level so the cap (not the cycle guard)
        # is what fires. With max_workflow_depth=2, depth 0 -> w1 -> w2 -> w3 breaches.
        nxt = args["next"]
        if nxt is None:
            return await ctx.agent("leaf", agent_type="researcher")
        return await ctx.workflow(nxt, {"next": args["then"], "then": None})

    workflows = WorkflowRegistry().register("w1", step).register("w2", step).register("w3", step)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("w1", {"next": "w2", "then": "w3"})

    with pytest.raises(WorkflowNestingError):
        await run_workflow(
            outer, roster=roster, workflows=workflows, thread_id="t1", max_workflow_depth=2
        )


async def test_name_cycle_raises_cycle_error(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def selfish(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow("selfish", {})  # re-enters itself -> cycle

    workflows = WorkflowRegistry().register("selfish", selfish)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("selfish", {})

    with pytest.raises(WorkflowCycleError):
        await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1")


async def test_deep_nesting_shares_parent_budget(
    make_usage_leaf: Callable[..., tuple[Runnable[Any, Any], Any]],
) -> None:
    leaf, _model = make_usage_leaf("ok", tokens_per_call=10)
    roster = Roster().register("researcher", leaf)
    spent: dict[str, float] = {}

    async def inner(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    async def middle(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow("inner", {})

    workflows = WorkflowRegistry().register("inner", inner).register("middle", middle)

    async def outer(ctx: Ctx) -> str:
        out = await ctx.workflow("middle", {})
        spent["after"] = ctx.budget.spent()
        return out

    await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1", budget=1000)
    assert spent["after"] == 10  # inner leaf's tokens visible on the parent budget


async def test_workflow_without_registry_raises_lookuperror(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("missing", {})

    with pytest.raises(LookupError):
        await run_workflow(outer, roster=roster, thread_id="t1")

"""Unit tests for one-level ``ctx.workflow(name, args)`` nesting.

A workflow may inline another workflow exactly one level deep; a second-level
nest must fail loud. These tests drive ``Ctx.workflow`` directly with fake leaves
(no host model): one level returns the inner result and shares the parent's
journal/budget, while a nested ``workflow()`` inside an already-nested workflow
raises ``WorkflowNestingError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Ctx, Roster, run_workflow
from langchain_dynamic_workflow._errors import WorkflowNestingError
from langchain_dynamic_workflow._workflows import WorkflowRegistry

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_one_level_nesting_returns_inner_result(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("inner-finding")
    roster = Roster().register("researcher", leaf)

    async def inner(ctx: Ctx, args: dict[str, Any]) -> str:
        topic = args["topic"]
        return await ctx.agent(f"Research {topic}", agent_type="researcher")

    workflows = WorkflowRegistry().register("inner", inner)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("inner", {"topic": "batteries"})

    result = await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1")
    assert result == "inner-finding"


async def test_one_level_nesting_shares_parent_budget(
    make_usage_leaf: Callable[..., tuple[Runnable[Any, Any], Any]],
) -> None:
    # The inner workflow's leaf usage must count against the parent's shared
    # budget — nesting must not open a fresh, separate pool.
    leaf, _model = make_usage_leaf("ok", tokens_per_call=10)
    roster = Roster().register("researcher", leaf)
    spent: dict[str, float] = {}

    async def inner(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("inner", inner)

    async def outer(ctx: Ctx) -> str:
        out = await ctx.workflow("inner", {})
        spent["after"] = ctx.budget.spent()
        return out

    await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1", budget=1000)
    # The inner leaf's 10 tokens are visible on the parent ctx budget.
    assert spent["after"] == 10


async def test_second_level_nesting_raises(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def leaf_wf(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    async def middle(ctx: Ctx, args: dict[str, Any]) -> str:
        # Trying to nest a third level: workflow() inside an already-nested wf.
        return await ctx.workflow("leaf_wf", {})

    workflows = WorkflowRegistry().register("leaf_wf", leaf_wf).register("middle", middle)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("middle", {})

    with pytest.raises(WorkflowNestingError):
        await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1")


async def test_workflow_without_registry_raises_lookuperror(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("missing", {})

    # No workflow registry wired -> resolving any name is a loud lookup error.
    with pytest.raises(LookupError):
        await run_workflow(outer, roster=roster, thread_id="t1")

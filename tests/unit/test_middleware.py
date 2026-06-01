"""Unit tests for the workflow middleware (tool contribution + notify injection).

The middleware packages the workflow tool, carries the shared BgRunManager, and
injects an in-band ``<workflow_notification>`` before the host's next model call
when a background run has settled. These tests exercise the middleware surface
directly (no host model loop): they assert it contributes exactly the workflow
tool, that ``abefore_model`` drains completion notices into a notification
message scoped to the host thread, and that a thread with no completed runs gets
no injection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.runtime import Runtime

from langchain_dynamic_workflow import Ctx, Roster
from langchain_dynamic_workflow._background import BgRunManager
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.middleware import (
    WORKFLOW_NOTIFICATION_TAG,
    create_workflow_middleware,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


def _config(thread_id: str) -> RunnableConfig:
    """A minimal RunnableConfig carrying the host thread id."""
    return {"configurable": {"thread_id": thread_id}}


def _runtime() -> Runtime[Any]:
    """A bare Runtime; abefore_model reads the thread id from config, not runtime."""
    return Runtime(context=None)


def test_middleware_contributes_the_workflow_tool() -> None:
    roster = Roster()
    workflows = WorkflowRegistry()
    middleware = create_workflow_middleware(roster, workflows=workflows)

    tool_names = [t.name for t in middleware.tools]
    assert tool_names == ["workflow"]


async def test_abefore_model_injects_completion_notification(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("the-result")
    roster = Roster().register("researcher", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    # Launch a run on the host thread and let it settle so a notice is enqueued.
    slot = manager.start(orchestrate_runner(roster), run_id="r1", thread_id="host-1")
    await manager.wait(slot.run_id, thread_id="host-1")

    update = await middleware.abefore_model({"messages": []}, _runtime(), _config("host-1"))
    assert update is not None
    injected = update["messages"]
    assert len(injected) == 1
    text = injected[0].content
    assert WORKFLOW_NOTIFICATION_TAG in text
    assert "r1" in text
    # Draining is one-shot: a second pass on the same thread injects nothing.
    again = await middleware.abefore_model({"messages": []}, _runtime(), _config("host-1"))
    assert again is None


async def test_abefore_model_no_notice_for_other_thread(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    workflows = WorkflowRegistry()
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    manager.start(orchestrate_runner(roster), run_id="r1", thread_id="host-1")
    await manager.wait("r1", thread_id="host-1")

    # A different host thread has no completed runs -> no injection.
    update = await middleware.abefore_model({"messages": []}, _runtime(), _config("host-OTHER"))
    assert update is None


def test_middleware_declares_workflow_runs_state_channel() -> None:
    roster = Roster()
    workflows = WorkflowRegistry()
    middleware = create_workflow_middleware(roster, workflows=workflows)
    # The dedicated channel must be on the middleware state schema so launched
    # runs survive context compaction (mirrors deepagents' async_tasks).
    annotations = middleware.state_schema.__annotations__
    assert "workflow_runs" in annotations


async def orchestrate_runner(roster: Roster) -> str:
    """A trivial settled coroutine standing in for a launched workflow."""
    return "done"

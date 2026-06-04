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

import pytest
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain_dynamic_workflow import Ctx, Roster
from langchain_dynamic_workflow._background import BgRunManager
from langchain_dynamic_workflow._run_store import InMemoryRunStore
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.middleware import (
    WORKFLOW_NOTIFICATION_TAG,
    create_workflow_middleware,
    merge_workflow_runs,
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


def test_merge_workflow_runs_upserts_by_run_id() -> None:
    # The settle-aware reducer: a status-only update merges field-wise into the
    # existing launch record (label kept, status overwritten, position preserved);
    # a new run_id appends. This is what makes the workflow_runs channel reflect
    # terminal status instead of staying stuck at the launch-time RUNNING.
    existing = [
        {"run_id": "r1", "workflow": "alpha", "status": "running"},
        {"run_id": "r2", "workflow": "beta", "status": "running"},
    ]
    merged = merge_workflow_runs(existing, [{"run_id": "r1", "status": "done"}])
    by_id = {r["run_id"]: r for r in merged}
    assert by_id["r1"] == {"run_id": "r1", "workflow": "alpha", "status": "done"}
    assert by_id["r2"] == {"run_id": "r2", "workflow": "beta", "status": "running"}
    # No duplication; first-seen order preserved.
    assert [r["run_id"] for r in merged] == ["r1", "r2"]
    # A new run_id appends after the existing ones.
    merged2 = merge_workflow_runs(
        merged, [{"run_id": "r3", "workflow": "gamma", "status": "running"}]
    )
    assert [r["run_id"] for r in merged2] == ["r1", "r2", "r3"]


def test_merge_workflow_runs_handles_empty_base() -> None:
    # The reducer must tolerate an empty/absent accumulator (first write).
    assert merge_workflow_runs([], [{"run_id": "r1", "status": "running"}]) == [
        {"run_id": "r1", "status": "running"}
    ]
    assert merge_workflow_runs(None, [{"run_id": "r1", "status": "running"}]) == [
        {"run_id": "r1", "status": "running"}
    ]


async def test_abefore_model_emits_settled_workflow_runs_update(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # On drain, the middleware not only injects the notification but also emits a
    # settle update so the workflow_runs channel is rewritten from RUNNING to the
    # terminal status (merged by run_id by the channel reducer).
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    workflows = WorkflowRegistry()
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    manager.start(orchestrate_runner(roster), run_id="r1", thread_id="host-1")
    await manager.wait("r1", thread_id="host-1")

    update = await middleware.abefore_model({"messages": []}, _runtime(), _config("host-1"))
    assert update is not None
    runs_update = update["workflow_runs"]
    assert any(r["run_id"] == "r1" and r["status"] == "done" for r in runs_update)


def test_middleware_raises_on_manager_plus_quota_conflict() -> None:
    # No silent failure: max_concurrent_runs only applies to a factory-built default
    # manager. Passing it alongside an explicit manager used to be silently ignored;
    # now it fails loud so the host does not believe a quota took effect when it did not.
    roster = Roster()
    workflows = WorkflowRegistry()
    manager = BgRunManager(max_concurrent_runs=2)
    with pytest.raises(ValueError, match="max_concurrent_runs"):
        create_workflow_middleware(
            roster, workflows=workflows, manager=manager, max_concurrent_runs=5
        )


def test_middleware_default_manager_honors_quota() -> None:
    # The default-manager path still wires the quota through (regression guard).
    roster = Roster()
    workflows = WorkflowRegistry()
    middleware = create_workflow_middleware(roster, workflows=workflows, max_concurrent_runs=3)
    assert middleware.manager.max_concurrent_runs == 3


async def test_middleware_forwards_injected_store_to_the_tool(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The middleware threads an injected run store through to the tool factory, so
    # a launch persists its spec into that store (the cross-process persistence
    # seam wired at the host edge).
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("researcher", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager()
    store = InMemoryRunStore()
    middleware = create_workflow_middleware(
        roster, workflows=workflows, manager=manager, store=store
    )

    tool: Any = middleware.tools[0]
    runtime: ToolRuntime[Any, Any] = ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": "host-1"}},
        stream_writer=lambda _chunk: None,
        tool_call_id="call-1",
        store=None,
    )
    out = await tool.coroutine(runtime=runtime, command="run", workflow="wf")
    assert isinstance(out, Command)
    update: dict[str, Any] = out.update or {}
    runs: list[dict[str, Any]] = update["workflow_runs"]
    run_id = runs[-1]["run_id"]
    assert isinstance(run_id, str)
    await manager.wait(run_id, thread_id="host-1")

    spec = await store.load_spec(run_id)
    assert spec is not None
    assert spec.name_or_source == "wf"
    assert spec.thread_id == "host-1"


async def orchestrate_runner(roster: Roster) -> str:
    """A trivial settled coroutine standing in for a launched workflow."""
    return "done"

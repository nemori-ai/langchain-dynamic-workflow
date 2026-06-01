"""Unit tests for the multi-command host-facing workflow tool.

The tool is the agent's single runtime surface (a multi-command ``run`` /
``status`` / ``resume`` / ``cancel`` tool). These tests drive it directly with a
constructed ``ToolRuntime`` (no host model) and a fake-leaf roster, asserting that
``run`` returns a placeholder run_id immediately without blocking, that
``status`` reports progress and then the settled result, and that ``cancel``
stops an in-flight run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command

from langchain_dynamic_workflow import Ctx, Roster
from langchain_dynamic_workflow._background import BgRunManager, BgStatus
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.tool import create_workflow_tool

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


def _runtime(*, thread_id: str, tool_call_id: str = "call-1") -> ToolRuntime[Any, Any]:
    """Build a minimal ToolRuntime carrying the host thread id and call id."""
    return ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda _chunk: None,
        tool_call_id=tool_call_id,
        store=None,
    )


async def _ainvoke_command(tool: Any, args: dict[str, Any], runtime: ToolRuntime[Any, Any]) -> Any:
    """Invoke the tool's async implementation with an injected runtime."""
    return await tool.coroutine(runtime=runtime, **args)


def _launched_run_id(command_out: Any) -> str:
    """Extract the launched run_id from a `run`/`resume` Command update."""
    assert isinstance(command_out, Command)
    update: dict[str, Any] = command_out.update or {}
    runs: list[dict[str, Any]] = update["workflow_runs"]
    run_id = runs[-1]["run_id"]
    assert isinstance(run_id, str)
    return run_id


def _placeholder_text(command_out: Any) -> str:
    """Extract the placeholder ToolMessage text from a Command update."""
    assert isinstance(command_out, Command)
    update: dict[str, Any] = command_out.update or {}
    messages: list[Any] = update["messages"]
    content = messages[0].content
    assert isinstance(content, str)
    return content


async def test_run_returns_run_id_placeholder_immediately(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("research-output")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Research X", agent_type="researcher")

    workflows = WorkflowRegistry().register("research", slow_orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "run", "workflow": "research"}, runtime)
    # A run returns a Command carrying a placeholder ToolMessage with a run_id,
    # while the workflow is still blocked on `release`.
    assert isinstance(out, Command)
    assert "run_id" in _placeholder_text(out)
    # The run is in flight, not done — the host turn was not blocked.
    run_id = _launched_run_id(out)
    assert manager.poll(run_id, thread_id="host-1") in {BgStatus.PENDING, BgStatus.RUNNING}

    release.set()
    await manager.wait(run_id, thread_id="host-1")
    assert manager.poll(run_id, thread_id="host-1") == BgStatus.DONE


async def test_status_reports_running_then_result(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("the-answer")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    run_id = _launched_run_id(run_out)

    # Before release: status reports it is still in flight.
    status_running = await _ainvoke_command(tool, {"command": "status", "run_id": run_id}, runtime)
    assert "running" in status_running.lower() or "pending" in status_running.lower()

    release.set()
    await manager.wait(run_id, thread_id="host-1")

    status_done = await _ainvoke_command(tool, {"command": "status", "run_id": run_id}, runtime)
    assert "the-answer" in status_done


async def test_cancel_stops_in_flight_run(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("never")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    run_id = _launched_run_id(run_out)

    cancel_out = await _ainvoke_command(tool, {"command": "cancel", "run_id": run_id}, runtime)
    assert "cancel" in cancel_out.lower()
    assert manager.poll(run_id, thread_id="host-1") == BgStatus.CANCELLED


async def test_resume_replays_journal_zero_model_calls(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # resume must reuse the same journal so completed leaves replay at zero cost.
    leaf, model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    workflows = WorkflowRegistry().register("geo", orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "geo"}, runtime)
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-1")
    calls_after_first = model.calls

    # resume the same run_id: the journal is reused, so the leaf replays from cache.
    resume_out = await _ainvoke_command(tool, {"command": "resume", "run_id": run_id}, runtime)
    resumed_run_id = _launched_run_id(resume_out)
    await manager.wait(resumed_run_id, thread_id="host-1")

    status = await _ainvoke_command(tool, {"command": "status", "run_id": resumed_run_id}, runtime)
    assert "Paris" in status
    assert model.calls == calls_after_first  # zero additional model calls on resume


async def test_unknown_workflow_name_is_a_loud_tool_error(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "run", "workflow": "nope"}, runtime)
    # An unknown workflow is reported back to the host as a plain error string,
    # never silently launched.
    assert isinstance(out, str)
    assert "nope" in out


async def test_unknown_command_is_rejected(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "frobnicate"}, runtime)
    assert isinstance(out, str)
    assert "frobnicate" in out


async def test_status_unknown_run_id_reports_unknown(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "status", "run_id": "ghost"}, runtime)
    assert isinstance(out, str)
    assert "ghost" in out

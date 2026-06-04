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
from langchain_dynamic_workflow._background import BgRunManager, BgStatus, ResultStore
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


async def test_status_offloads_large_result_with_summary_and_handle(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Acceptance: a large result is offloaded behind a handle, and the HOST-FACING
    # status reply carries a summary + handle rather than inlining the full payload.
    # inline_max_chars=8 forces the offload branch deterministically.
    leaf, _state = make_fake_leaf("a-long-research-conclusion-well-over-the-inline-limit")
    roster = Roster().register("researcher", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager(result_store=ResultStore(inline_max_chars=8))
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-1")

    status = await _ainvoke_command(tool, {"command": "status", "run_id": run_id}, runtime)
    # The host sees the offload surface: an "offloaded" notice + a fetchable handle,
    # not the full inlined value.
    assert "offload" in status.lower()
    assert "handle:" in status
    assert "result://" in status


async def test_resume_after_partial_run_replays_completed_leaf_and_runs_rest_live(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Mid-run/partial resume through the tool surface: a parallel run where one leaf
    # journals and the other fails the first pass (lands None, success-only so it is
    # NOT journaled). Resuming the same run_id reuses the journal: the completed leaf
    # replays at zero cost while only the previously-failed leaf runs live.
    ok_leaf, ok_state = make_fake_leaf("good")
    flaky_leaf, flaky_state = make_fake_leaf("recovered", fail_times=1)
    roster = Roster().register("ok", ok_leaf).register("flaky", flaky_leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("stable", agent_type="ok"),
                lambda: ctx.agent("unstable", agent_type="flaky"),
            ]
        )

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-1")
    # First pass: ok leaf ran and journaled; flaky leaf failed once (not journaled).
    assert ok_state.calls == 1
    assert flaky_state.calls == 1

    resume_out = await _ainvoke_command(tool, {"command": "resume", "run_id": run_id}, runtime)
    resumed_id = _launched_run_id(resume_out)
    await manager.wait(resumed_id, thread_id="host-1")
    # Completed leaf served from the journal (zero new calls); the failed one ran live.
    assert ok_state.calls == 1
    assert flaky_state.calls == 2
    status = await _ainvoke_command(tool, {"command": "status", "run_id": resumed_id}, runtime)
    assert "good" in status and "recovered" in status


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


_AUTHORED_SCRIPT = """\
async def orchestrate(ctx, args):
    return await ctx.agent(f"Summarize {args['topic']}", agent_type="writer")
"""


async def test_run_script_launches_an_authored_script(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The meta-layer surface: a host authors an ad-hoc script and submits it via
    # run_script — no registered workflow name needed. It launches in the
    # background like a named run and status fetches its result.
    leaf, _state = make_fake_leaf("the-summary")
    roster = Roster().register("writer", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(
        tool,
        {"command": "run_script", "script": _AUTHORED_SCRIPT, "args": {"topic": "batteries"}},
        runtime,
    )
    assert isinstance(out, Command)
    run_id = _launched_run_id(out)
    await manager.wait(run_id, thread_id="host-1")

    status = await _ainvoke_command(tool, {"command": "status", "run_id": run_id}, runtime)
    assert "the-summary" in status


async def test_run_script_rejects_gate_violation_as_plain_string(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # A script that fails the AST gate must come back as a plain error string
    # enumerating the violation (the feed-back-and-retry channel) — never a
    # Command, and nothing is launched.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("writer", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    bad_script = "import os\nasync def orchestrate(ctx, args):\n    return 1\n"
    out = await _ainvoke_command(tool, {"command": "run_script", "script": bad_script}, runtime)
    assert isinstance(out, str)
    assert "import" in out.lower()


async def test_run_script_requires_a_script(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("writer", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "run_script"}, runtime)
    assert isinstance(out, str)
    assert "script" in out.lower()


async def test_run_script_resume_recompiles_and_replays_journal(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # Resume of an ad-hoc run re-forges the callable from the persisted source and
    # re-runs against the same journal, so the completed leaf replays at zero cost.
    leaf, model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    script = (
        "async def orchestrate(ctx, args):\n"
        '    return await ctx.agent("Capital of France?", agent_type="geographer")\n'
    )
    run_out = await _ainvoke_command(tool, {"command": "run_script", "script": script}, runtime)
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-1")
    calls_after_first = model.calls

    resume_out = await _ainvoke_command(tool, {"command": "resume", "run_id": run_id}, runtime)
    resumed_id = _launched_run_id(resume_out)
    await manager.wait(resumed_id, thread_id="host-1")

    status = await _ainvoke_command(tool, {"command": "status", "run_id": resumed_id}, runtime)
    assert "Paris" in status
    assert model.calls == calls_after_first  # zero additional model calls on resume


async def test_runs_command_lists_all_runs_with_labels_and_status(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The aggregate view: `runs` lists every run on the host thread with its
    # workflow label and live status, so the host need not poll each run_id.
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Q", agent_type="researcher")

    async def quick(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("slow_wf", slow).register("quick_wf", quick)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    slow_out = await _ainvoke_command(tool, {"command": "run", "workflow": "slow_wf"}, runtime)
    slow_id = _launched_run_id(slow_out)
    quick_out = await _ainvoke_command(tool, {"command": "run", "workflow": "quick_wf"}, runtime)
    quick_id = _launched_run_id(quick_out)
    await manager.wait(quick_id, thread_id="host-1")

    runs_out = await _ainvoke_command(tool, {"command": "runs"}, runtime)
    assert isinstance(runs_out, str)
    # Both runs appear, each with its workflow label and live status.
    assert slow_id in runs_out and quick_id in runs_out
    assert "slow_wf" in runs_out and "quick_wf" in runs_out
    assert "running" in runs_out.lower() or "pending" in runs_out.lower()
    assert "done" in runs_out.lower()

    release.set()
    await manager.wait(slow_id, thread_id="host-1")


async def test_runs_command_reports_no_runs_when_empty(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "runs"}, runtime)
    assert isinstance(out, str)
    assert "no runs" in out.lower()


async def test_run_refused_when_concurrent_run_quota_full(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The run command surfaces the manager's concurrent-run quota as a clear,
    # non-Command refusal string instead of fanning out unbounded.
    leaf, _state = make_fake_leaf("out")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", slow_orchestrate)
    manager = BgRunManager(max_concurrent_runs=1)
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    # First run is admitted (and held in flight by the closed gate).
    first = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    assert isinstance(first, Command)
    first_id = _launched_run_id(first)

    # Second run is refused with a plain string (no Command, nothing launched).
    refused = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    assert isinstance(refused, str)
    assert "quota" in refused.lower()

    release.set()
    await manager.wait(first_id, thread_id="host-1")
    assert manager.poll(first_id, thread_id="host-1") == BgStatus.DONE

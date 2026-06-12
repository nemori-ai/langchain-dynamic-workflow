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
from langchain_dynamic_workflow._run_store import InMemoryRunStore, RunSpec
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


async def test_launch_saves_spec_with_run_id_label_and_args(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The store contract on launch: the tool persists a RunSpec under the SAME
    # run_id the manager slot reports, carrying the kind/label/args/journal lineage
    # that a (possibly fresh-process) resume needs to rebuild the launch. A fresh
    # launch stamps its OWN run_id as the canonical journal_run_id so a resume keys
    # the same journal — the host thread is NOT persisted (it is the caller's).
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("researcher", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", orchestrate)
    manager = BgRunManager()
    store = InMemoryRunStore()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows, store=store)
    runtime = _runtime(thread_id="host-7")

    run_out = await _ainvoke_command(
        tool, {"command": "run", "workflow": "wf", "args": {"topic": "batteries"}}, runtime
    )
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-7")

    # The spec was saved under the manager's run_id (run_id generated up front,
    # before manager.start) with the full launch description; the canonical journal
    # lineage is stamped to the run's own id (no host thread persisted).
    spec = await store.load_spec(run_id)
    assert spec == RunSpec(
        kind="name",
        name_or_source="wf",
        args={"topic": "batteries"},
        label="wf",
        journal_run_id=run_id,
    )


async def test_run_script_launch_saves_a_script_spec(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # An ad-hoc script launch persists its SOURCE as the spec (kind='script') so a
    # resume recompiles it; the label comes from the script's optional meta name.
    leaf, _state = make_fake_leaf("the-summary")
    roster = Roster().register("writer", leaf)
    manager = BgRunManager()
    store = InMemoryRunStore()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry(), store=store)
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(
        tool,
        {"command": "run_script", "script": _AUTHORED_SCRIPT, "args": {"topic": "batteries"}},
        runtime,
    )
    run_id = _launched_run_id(out)
    await manager.wait(run_id, thread_id="host-1")

    spec = await store.load_spec(run_id)
    assert spec is not None
    assert spec.kind == "script"
    assert spec.name_or_source == _AUTHORED_SCRIPT
    assert spec.args == {"topic": "batteries"}
    # The host thread is not persisted; the canonical journal lineage is the run's
    # own id so a resume keys the same journal.
    assert spec.journal_run_id == run_id


async def test_resume_reads_spec_from_injected_store(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # Cross-process simulation: a SECOND tool instance, sharing only the injected
    # store, resumes a run launched through a FIRST tool. The resume must rebuild
    # the callable from the store's RunSpec (not in-process closure state) and
    # reuse the run's journal so the completed leaf replays at zero model cost.
    leaf, model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    workflows = WorkflowRegistry().register("geo", orchestrate)
    manager = BgRunManager()
    store = InMemoryRunStore()

    launching_tool = create_workflow_tool(roster, manager=manager, workflows=workflows, store=store)
    runtime = _runtime(thread_id="host-1")
    run_out = await _ainvoke_command(launching_tool, {"command": "run", "workflow": "geo"}, runtime)
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-1")
    calls_after_first = model.calls

    # A fresh tool over the same shared store + manager (stands in for a restarted
    # host process pointed at the same persistent store).
    resuming_tool = create_workflow_tool(roster, manager=manager, workflows=workflows, store=store)
    resume_out = await _ainvoke_command(
        resuming_tool, {"command": "resume", "run_id": run_id}, runtime
    )
    resumed_id = _launched_run_id(resume_out)
    await manager.wait(resumed_id, thread_id="host-1")

    status = await _ainvoke_command(
        resuming_tool, {"command": "status", "run_id": resumed_id}, runtime
    )
    assert "Paris" in status
    assert model.calls == calls_after_first  # journal replayed; zero new model calls


async def test_resume_unknown_run_id_reports_unknown(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # A resume against a run_id the store never saw is a plain refusal string, not
    # a Command — nothing is relaunched.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "resume", "run_id": "ghost"}, runtime)
    assert isinstance(out, str)
    assert "ghost" in out


async def test_resume_of_a_still_running_origin_is_refused(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The resume guard (mirrors approve's poll): resuming a run whose origin is STILL
    # in flight on this process must be refused with a plain string, never relaunched.
    # Without the guard, resume launches a concurrent DUPLICATE against the same
    # canonical journal + checkpoint thread, each with a FRESH ConcurrencyGate and
    # Budget — so a budget=N run can spend ~2N and both write the same journal.
    leaf, _state = make_fake_leaf("research-output")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()  # the single leaf blocks until the test releases it
        return await ctx.agent("Research X", agent_type="researcher")

    workflows = WorkflowRegistry().register("research", slow_orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "research"}, runtime)
    run_id = _launched_run_id(run_out)
    # The origin is in flight (still blocked on release), never settled.
    assert manager.poll(run_id, thread_id="host-1") in {BgStatus.PENDING, BgStatus.RUNNING}

    try:
        out = await _ainvoke_command(tool, {"command": "resume", "run_id": run_id}, runtime)
        # A still-running origin is REFUSED with a string (as approve does for a
        # non-parked run), never relaunched as a concurrent duplicate.
        assert isinstance(out, str)
        assert "running" in out.lower()
    finally:
        release.set()
        await manager.wait(run_id, thread_id="host-1")


async def test_resume_of_a_live_origin_from_a_different_host_thread_is_refused(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Cross-thread duplicate: the origin runs on host-A (still live). A resume issued
    # on host-B polls UNKNOWN there (its slot is keyed under host-A), so the friendly
    # per-(thread, run_id) poll cannot see it. The AUTHORITATIVE canonical reservation
    # in the manager still catches it: the canonical journal lineage is live, so the
    # cross-thread resume is refused — never relaunched as a concurrent duplicate.
    leaf, _state = make_fake_leaf("research-output")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Research X", agent_type="researcher")

    workflows = WorkflowRegistry().register("research", slow_orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)

    run_out = await _ainvoke_command(
        tool, {"command": "run", "workflow": "research"}, _runtime(thread_id="host-A")
    )
    run_id = _launched_run_id(run_out)
    assert manager.poll(run_id, thread_id="host-A") in {BgStatus.PENDING, BgStatus.RUNNING}
    # The resuming thread (host-B) cannot see the origin's slot by poll.
    assert manager.poll(run_id, thread_id="host-B") == BgStatus.UNKNOWN

    try:
        out = await _ainvoke_command(
            tool, {"command": "resume", "run_id": run_id}, _runtime(thread_id="host-B")
        )
        # Refused despite the UNKNOWN poll on host-B: the canonical reservation is the
        # real enforcer, not the per-thread slot poll.
        assert isinstance(out, str)
        assert "canonical" in out.lower()
    finally:
        release.set()
        await manager.wait(run_id, thread_id="host-A")


async def test_resume_of_terminal_origin_with_a_live_resume_child_is_refused(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Terminal-origin-with-live-child: the origin settles DONE (poll is terminal, the
    # friendly guard would wave it through), but a PRIOR resume's child is still live
    # against the same canonical journal. A second resume must be refused so it does
    # not duplicate the live resume-child onto the shared journal + checkpoint thread.
    leaf, _state = make_fake_leaf("research-output")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def gated_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()  # blocks every pass until the test releases it
        return await ctx.agent("Research X", agent_type="researcher")

    workflows = WorkflowRegistry().register("research", gated_orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    # Origin runs to DONE (release, then re-gate for the resume-child).
    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "research"}, runtime)
    origin_id = _launched_run_id(run_out)
    release.set()
    await manager.wait(origin_id, thread_id="host-1")
    assert manager.poll(origin_id, thread_id="host-1") == BgStatus.DONE
    release.clear()

    # R1 resume of the now-terminal origin: it claims the canonical and blocks live.
    r1_out = await _ainvoke_command(tool, {"command": "resume", "run_id": origin_id}, runtime)
    r1_id = _launched_run_id(r1_out)
    assert manager.poll(r1_id, thread_id="host-1") in {BgStatus.PENDING, BgStatus.RUNNING}

    try:
        # A SECOND resume of the (terminal) origin: poll is terminal/UNKNOWN, but R1's
        # child still holds the canonical, so this is refused — no concurrent duplicate.
        out = await _ainvoke_command(tool, {"command": "resume", "run_id": origin_id}, runtime)
        assert isinstance(out, str)
        assert "canonical" in out.lower()
    finally:
        release.set()
        await manager.wait(r1_id, thread_id="host-1")


async def test_two_concurrent_resumes_of_one_canonical_admit_exactly_one(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # TOCTOU double-resume: two resumes of the same settled origin fire concurrently.
    # The canonical claim in the manager is a synchronous, no-await check-and-add, so
    # exactly ONE wins the canonical and launches; the other is refused. Without it,
    # both poll the terminal origin, both pass, both launch a duplicate.
    leaf, _state = make_fake_leaf("research-output")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def gated_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Research X", agent_type="researcher")

    workflows = WorkflowRegistry().register("research", gated_orchestrate)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "research"}, runtime)
    origin_id = _launched_run_id(run_out)
    release.set()
    await manager.wait(origin_id, thread_id="host-1")
    release.clear()  # the resume-child re-gates so the winner stays live

    # Fire both resumes concurrently against the one settled canonical.
    out_a, out_b = await asyncio.gather(
        _ainvoke_command(tool, {"command": "resume", "run_id": origin_id}, runtime),
        _ainvoke_command(tool, {"command": "resume", "run_id": origin_id}, runtime),
    )
    outs: list[Any] = [out_a, out_b]
    commands: list[Command[Any]] = [o for o in outs if isinstance(o, Command)]
    refusals: list[str] = [o for o in outs if isinstance(o, str)]
    try:
        # Exactly one launched (a Command), exactly one refused (a string).
        assert len(commands) == 1, outs
        assert len(refusals) == 1
        assert "canonical" in refusals[0].lower()
    finally:
        release.set()
        winner_id = _launched_run_id(commands[0])
        await manager.wait(winner_id, thread_id="host-1")


async def test_resume_from_different_host_thread_is_pollable_by_that_thread(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # Thread identity (fix #2): the resume's background slot is keyed by the
    # CURRENT caller's host thread, not the original launch thread, so the caller
    # who issued the resume can poll its new run. The LangGraph CHECKPOINT thread
    # is a per-run canonical id carried in the spec — distinct from the host thread
    # used for slot bookkeeping — so cross-thread resume stays pollable.
    leaf, _model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    workflows = WorkflowRegistry().register("geo", orchestrate)
    manager = BgRunManager()
    store = InMemoryRunStore()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows, store=store)

    run_out = await _ainvoke_command(
        tool, {"command": "run", "workflow": "geo"}, _runtime(thread_id="host-A")
    )
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-A")

    # Resume issued from a DIFFERENT host thread ("host-B"): its slot lands under
    # host-B so that caller can poll the new run, and NOT under host-A.
    resume_out = await _ainvoke_command(
        tool, {"command": "resume", "run_id": run_id}, _runtime(thread_id="host-B")
    )
    resumed_id = _launched_run_id(resume_out)
    assert manager.poll(resumed_id, thread_id="host-B") != BgStatus.UNKNOWN
    assert manager.poll(resumed_id, thread_id="host-A") == BgStatus.UNKNOWN
    await manager.wait(resumed_id, thread_id="host-B")

    # And the resuming caller can fetch the result via status on its own thread.
    status = await _ainvoke_command(
        tool, {"command": "status", "run_id": resumed_id}, _runtime(thread_id="host-B")
    )
    assert "Paris" in status


async def test_resume_of_a_resume_replays_journal_at_zero_new_invocations(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Headline HIGH (fix #1): resuming a resume-issued run_id must still replay the
    # journal at zero new leaf invocations. Before the fix the resume minted a NEW
    # run_id whose saved spec carried no journal lineage, so resuming THAT id keyed
    # an empty journal and re-ran every leaf live (invocations 1 -> 1 -> 2).
    leaf, state = make_fake_leaf("Paris")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    workflows = WorkflowRegistry().register("geo", orchestrate)
    manager = BgRunManager()
    store = InMemoryRunStore()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows, store=store)
    runtime = _runtime(thread_id="host-1")

    run_out = await _ainvoke_command(tool, {"command": "run", "workflow": "geo"}, runtime)
    run_id = _launched_run_id(run_out)
    await manager.wait(run_id, thread_id="host-1")
    assert state.calls == 1  # the leaf ran live exactly once on the first run

    # First resume (R1) of the original run: replays the journal at zero cost.
    resume_out = await _ainvoke_command(tool, {"command": "resume", "run_id": run_id}, runtime)
    r1_id = _launched_run_id(resume_out)
    await manager.wait(r1_id, thread_id="host-1")
    assert state.calls == 1  # still zero new invocations

    # Second resume — of R1's run_id, NOT the original. This is the headline bug:
    # R1's spec must carry the canonical journal lineage so resuming it rejoins the
    # SAME journal and replays the leaf for free.
    resume2_out = await _ainvoke_command(tool, {"command": "resume", "run_id": r1_id}, runtime)
    r2_id = _launched_run_id(resume2_out)
    await manager.wait(r2_id, thread_id="host-1")

    status = await _ainvoke_command(tool, {"command": "status", "run_id": r2_id}, runtime)
    assert "Paris" in status
    assert state.calls == 1  # smoking gun: resume-of-resume added ZERO invocations


async def test_launch_saves_spec_before_admission_and_quota_refusal_deletes_it(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Save-before-start (fix #3): the spec is persisted BEFORE manager.start admits
    # the run, so a crash between admission and save can never strand a run with no
    # spec. A quota refusal then deletes that pre-saved spec so it leaves no
    # unresumable orphan in the registry.
    leaf, _state = make_fake_leaf("out")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow_orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("wf", slow_orchestrate)
    manager = BgRunManager(max_concurrent_runs=1)
    store = InMemoryRunStore()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows, store=store)
    runtime = _runtime(thread_id="host-1")

    # The registry has no public enumeration, so the test reads the in-memory
    # store's backing dict by name to assert on its exact set of saved run ids.
    def saved_run_ids() -> set[str]:
        specs: dict[str, RunSpec] = getattr(store, "_specs")  # noqa: B009 - test introspection
        return set(specs)

    # First run is admitted and held in flight by the closed gate.
    first = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    first_id = _launched_run_id(first)
    # Its spec was persisted (before admission) — provable by load_spec.
    assert await store.load_spec(first_id) is not None

    # Snapshot the saved run ids before the refused launch so we can assert the
    # refused run leaves no NEW orphan spec behind.
    specs_before = saved_run_ids()

    # Second run is refused at the quota; nothing must remain in the registry for it.
    refused = await _ainvoke_command(tool, {"command": "run", "workflow": "wf"}, runtime)
    assert isinstance(refused, str)
    assert "quota" in refused.lower()
    # The refused launch deleted the spec it pre-saved: no orphan spec was added.
    assert saved_run_ids() == specs_before

    release.set()
    await manager.wait(first_id, thread_id="host-1")


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


def _registry_with_descriptions() -> WorkflowRegistry:
    """A registry whose entries carry both explicit and docstring-derived summaries."""

    async def deep_research(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("research", agent_type="researcher")

    async def code_review(ctx: Ctx, args: dict[str, Any]) -> str:
        """Review a diff and report findings."""
        return await ctx.agent("review", agent_type="researcher")

    return (
        WorkflowRegistry()
        .register(
            "deep_research", deep_research, description="Fan out web research and synthesize."
        )
        .register("code_review", code_review)  # docstring-first-line fallback
    )


async def test_built_tool_description_lists_registered_catalog(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Discoverability surface #1: the built tool's description renders the registry
    # catalog (name + one-line description) so a host model sees the menu up front,
    # WITHOUT the names being hard-coded in its prompt.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=_registry_with_descriptions())

    description = tool.description
    assert "deep_research" in description
    assert "Fan out web research and synthesize." in description
    assert "code_review" in description
    assert "Review a diff and report findings." in description  # docstring fallback rendered
    # The catalog is introduced as a launch-by-name menu for command='run'.
    assert "Registered workflows" in description
    assert "command='run'" in description


async def test_built_tool_description_for_empty_registry_points_to_run_script(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # An empty registry renders explicit guidance: no registered workflows, author a
    # script via run_script instead.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())

    description = tool.description
    assert "run_script" in description
    lowered = description.lower()
    assert "no registered workflows" in lowered or "none" in lowered


async def test_built_tool_description_documents_the_catalog_command(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The `catalog` command must be documented in the Commands list of the description.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=_registry_with_descriptions())

    assert "`catalog`" in tool.description


async def test_catalog_command_returns_registered_names_and_descriptions(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Discoverability surface #2: the `catalog` command returns the same catalog on
    # demand. It is a read-only host-tool command (takes no args).
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=_registry_with_descriptions())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "catalog"}, runtime)
    assert isinstance(out, str)
    assert "deep_research" in out
    assert "Fan out web research and synthesize." in out
    assert "code_review" in out
    assert "Review a diff and report findings." in out


async def test_catalog_command_renders_same_text_as_description_section(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # One render helper backs both surfaces: the `catalog` command's text is a
    # substring of the built tool description (the description appends the same
    # rendered catalog section).
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=_registry_with_descriptions())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "catalog"}, runtime)
    assert isinstance(out, str)
    assert out in tool.description


async def test_catalog_command_on_empty_registry_points_to_run_script(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "catalog"}, runtime)
    assert isinstance(out, str)
    assert "run_script" in out
    assert "no registered workflows" in out.lower() or "none" in out.lower()


# --- leaf-agent (roster) discoverability, mirroring the workflow catalog above ---


def _roster_with_descriptions(make_fake_leaf: FakeLeafFactory) -> Roster:
    """A roster whose entries carry distinct, human-readable descriptions."""
    researcher, _ = make_fake_leaf("r")
    writer, _ = make_fake_leaf("w")
    return (
        Roster()
        .register("researcher", researcher, description="Fan out web research and synthesize.")
        .register("writer", writer, description="Draft polished prose from notes.")
    )


async def test_built_tool_description_lists_registered_agents(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Discoverability surface #1: the built tool's description renders the roster
    # catalog (name + description) so a host model knows which agent_type names it
    # may name in a run_script, WITHOUT them being hard-coded in its prompt.
    roster = _roster_with_descriptions(make_fake_leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())

    description = tool.description
    assert "researcher" in description
    assert "Fan out web research and synthesize." in description
    assert "writer" in description
    assert "Draft polished prose from notes." in description
    # The agent catalog is introduced as the agent_type menu for authoring scripts.
    assert "Registered leaf agents" in description
    assert "agent_type" in description


async def test_built_tool_description_for_empty_roster_gives_guidance(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # An empty roster renders one-line guidance: with no leaves, ctx.agent cannot be
    # called at all, so there is nothing to name as agent_type.
    del make_fake_leaf
    roster = Roster()
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())

    description = tool.description
    lowered = description.lower()
    assert "leaf agents" in lowered
    assert "none" in lowered or "no leaf" in lowered


async def test_built_tool_description_documents_the_agents_command(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The `agents` command must be documented in the Commands list of the description.
    roster = _roster_with_descriptions(make_fake_leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())

    assert "`agents`" in tool.description


async def test_agents_command_returns_registered_names_and_descriptions(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Discoverability surface #2: the `agents` command returns the same catalog on
    # demand. It is a read-only host-tool command (takes no args).
    roster = _roster_with_descriptions(make_fake_leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "agents"}, runtime)
    assert isinstance(out, str)
    assert "researcher" in out
    assert "Fan out web research and synthesize." in out
    assert "writer" in out
    assert "Draft polished prose from notes." in out


async def test_agents_command_renders_same_text_as_description_section(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # One render helper backs both surfaces: the `agents` command's text is a
    # substring of the built tool description (the description appends the same
    # rendered agent-catalog section).
    roster = _roster_with_descriptions(make_fake_leaf)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "agents"}, runtime)
    assert isinstance(out, str)
    assert out in tool.description


async def test_agents_command_on_empty_roster_gives_guidance(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    del make_fake_leaf
    roster = Roster()
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "agents"}, runtime)
    assert isinstance(out, str)
    lowered = out.lower()
    assert "leaf agents" in lowered
    assert "none" in lowered or "no leaf" in lowered


async def test_agent_catalog_collapses_multiline_description_to_one_line(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # A multi-line / over-long leaf description must render as exactly ONE catalog
    # line in both the description and the `agents` output — the same normalization
    # the workflow catalog uses (reused _one_line_summary), so a stray newline can
    # never inject a fake catalog entry.
    leaf, _ = make_fake_leaf("x")
    roster = Roster().register(
        "researcher",
        leaf,
        description="line one\nline two\n\tindented three",
    )
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=WorkflowRegistry())
    runtime = _runtime(thread_id="host-1")

    out = await _ainvoke_command(tool, {"command": "agents"}, runtime)
    assert isinstance(out, str)
    # The single entry renders on one line with whitespace collapsed.
    entry_lines = [line for line in out.splitlines() if line.startswith("- researcher")]
    assert entry_lines == ["- researcher — line one line two indented three"]
    # And the same single, collapsed line appears in the built description.
    assert "- researcher — line one line two indented three" in tool.description

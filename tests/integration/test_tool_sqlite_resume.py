"""Tool-path resume over the sqlite store and its persistent checkpointer.

The cross-process worker proves zero-cost replay through the *engine* surface
(``run_workflow`` directly). This file closes the previously untested gap: the
same durability driven through the host-facing ``create_workflow_tool`` with a
``SqliteWorkflowStore`` injected as both the run registry and the LangGraph
checkpointer. It launches a run, waits for its checkpoint to settle, then resumes
by ``run_id`` and asserts the completed leaves replay from the persisted journal
at zero new live invocations — with the per-run canonical thread carrying the
checkpoint, not the host thread.

The leaves are offline, deterministic fakes that count live invocations, so the
assertion is a hard invocation delta, not a model-cost estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command

from langchain_dynamic_workflow import Ctx, Roster, SqliteWorkflowStore
from langchain_dynamic_workflow._background import BgRunManager, BgStatus
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.tool import create_workflow_tool


class _CountingLeafState:
    """Mutable live-invocation counter for an offline tool-path leaf."""

    def __init__(self) -> None:
        self.calls = 0


def _make_counting_leaf(reply: str) -> tuple[Runnable[Any, Any], _CountingLeafState]:
    """Build an offline leaf that counts each live invocation in process.

    Args:
        reply: The text the leaf's terminal ``AIMessage`` carries.

    Returns:
        A runnable that increments its state on each live call and the state it
        increments, so the test can assert a hard invocation delta across resume.
    """
    state = _CountingLeafState()

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        state.calls += 1
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call), state


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


async def _ainvoke(tool: Any, args: dict[str, Any], runtime: ToolRuntime[Any, Any]) -> Any:
    """Invoke the tool's async implementation with an injected runtime."""
    return await tool.coroutine(runtime=runtime, **args)


def _run_id_of(command_out: Any) -> str:
    """Extract the launched run_id from a ``run`` / ``resume`` Command update."""
    assert isinstance(command_out, Command)
    update: dict[str, Any] = command_out.update or {}
    runs: list[dict[str, Any]] = update["workflow_runs"]
    run_id = runs[-1]["run_id"]
    assert isinstance(run_id, str)
    return run_id


async def test_tool_resume_over_sqlite_store_and_checkpointer_replays_free(
    tmp_path: Path,
) -> None:
    """A tool-path resume over the sqlite store + checkpointer replays for free.

    The store is injected as both the run registry and the persistent
    checkpointer, so a launched run journals its leaves and checkpoints on its
    per-run canonical thread. Resuming the run by ``run_id`` reopens the same
    journal and replays both completed leaves at zero new live invocation, with
    the checkpoint carried on the canonical thread (not the host thread).
    """
    db_path = tmp_path / "workflows.db"
    planner, planner_state = _make_counting_leaf("plan")
    writer, writer_state = _make_counting_leaf("draft")
    roster = Roster().register("planner", planner).register("writer", writer)

    async def orchestrate(ctx: Ctx, args: dict[str, Any]) -> str:
        plan = await ctx.agent("Outline the report", agent_type="planner")
        draft = await ctx.agent("Write the report", agent_type="writer")
        return f"{plan}+{draft}"

    workflows = WorkflowRegistry().register("report", orchestrate)
    manager = BgRunManager()
    store = await SqliteWorkflowStore.open(db_path)
    try:
        tool = create_workflow_tool(
            roster,
            manager=manager,
            workflows=workflows,
            checkpointer=store.checkpointer,
            store=store,
        )
        runtime = _runtime(thread_id="host-1")

        run_out = await _ainvoke(tool, {"command": "run", "workflow": "report"}, runtime)
        run_id = _run_id_of(run_out)
        await manager.wait(run_id, thread_id="host-1")
        assert manager.poll(run_id, thread_id="host-1") == BgStatus.DONE
        # Both leaves ran live exactly once on the first pass.
        assert planner_state.calls == 1
        assert writer_state.calls == 1

        # The run's canonical thread (its own run_id) carries the checkpoint, not
        # the host thread — a fresh-process resume keys the journal + checkpoint by
        # this id regardless of which host thread issues the resume.
        spec = await store.load_spec(run_id)
        assert spec is not None
        assert spec.journal_run_id == run_id
        checkpoint_config: RunnableConfig = {"configurable": {"thread_id": run_id}}
        assert await store.checkpointer.aget_tuple(checkpoint_config) is not None
        host_thread_config: RunnableConfig = {"configurable": {"thread_id": "host-1"}}
        assert await store.checkpointer.aget_tuple(host_thread_config) is None

        # Resume by run_id: both completed leaves replay from the persisted journal.
        resume_out = await _ainvoke(tool, {"command": "resume", "run_id": run_id}, runtime)
        resumed_id = _run_id_of(resume_out)
        await manager.wait(resumed_id, thread_id="host-1")

        status = await _ainvoke(tool, {"command": "status", "run_id": resumed_id}, runtime)
        assert "plan+draft" in status
        # Smoking gun: the resume added ZERO new live invocations for either leaf.
        assert planner_state.calls == 1
        assert writer_state.calls == 1
    finally:
        await store.aclose()

"""Integration: a host fans out parallel runs and monitors them via the aggregate view.

This composes M3.5's two capabilities through the full host + middleware + tool
surface (no API key, scripted host model):

1. turn 1 — the host issues several ``workflow(run)`` calls at once; each returns a
   placeholder immediately and is recorded in ``workflow_runs`` as ``running`` while
   the runs are gated in flight (proving the launches did not block the turn).
2. the gated runs are released and settle.
3. turn 2 — a ``<workflow_notification>`` is injected; the host calls
   ``workflow(runs)`` and gets the aggregate listing of every run with its terminal
   status, and the ``workflow_runs`` channel is rewritten from ``running`` to
   ``done`` (settle-aware reducer).

It pins that the aggregate ``runs`` command and the settle-aware ``workflow_runs``
channel work together end to end, not just as isolated unit pieces.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_dynamic_workflow import Ctx, Roster
from langchain_dynamic_workflow._background import BgRunManager, BgStatus
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.middleware import (
    WORKFLOW_NOTIFICATION_TAG,
    create_workflow_middleware,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]

_TOPICS = ["alpha", "beta", "gamma"]


class ScriptedFanoutHost(BaseChatModel):
    """Scripted host: fan out one run per topic, then read the aggregate runs view."""

    @property
    def _llm_type(self) -> str:
        return "scripted-fanout-host"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        launched = [m for m in tool_messages if "Launched workflow" in m.text]
        runs_view = next(
            (m.text for m in reversed(tool_messages) if m.text.startswith("runs:")), None
        )
        notification_seen = any(WORKFLOW_NOTIFICATION_TAG in m.text for m in messages)

        if runs_view is not None:
            return _say(f"FINAL aggregate view:\n{runs_view}")
        if notification_seen:
            return _calls([{"command": "runs"}])
        if launched:
            return _say(f"Launched {len(launched)} parallel runs; monitoring.")
        return _calls(
            [
                {"command": "run", "workflow": "topic_research", "args": {"topic": t}}
                for t in _TOPICS
            ]
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _say(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _calls(calls: list[dict[str, Any]]) -> ChatResult:
    tool_calls = [
        {"name": "workflow", "args": call, "id": f"{call['command']}-{i}"}
        for i, call in enumerate(calls)
    ]
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="", tool_calls=tool_calls))]
    )


async def test_host_fans_out_parallel_runs_and_reads_aggregate_view(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("finding")
    roster = Roster().register("researcher", leaf)

    # Gate the run body so the three runs stay in flight through turn 1 — pins that
    # the launches did not block the host turn and that workflow_runs reads `running`.
    release = asyncio.Event()

    async def topic_research(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent(f"Research {args['topic']}", agent_type="researcher")

    workflows = WorkflowRegistry().register("topic_research", topic_research)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)
    host = create_deep_agent(model=ScriptedFanoutHost(), middleware=[middleware])  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
    config: RunnableConfig = {"configurable": {"thread_id": "obs-thread"}}

    # Turn 1: three parallel runs launched, all recorded as running and in flight.
    state1 = await host.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        {"messages": [{"role": "user", "content": "Research the topics in parallel"}]},
        config=config,
    )
    launched: list[dict[str, Any]] = state1["workflow_runs"]
    run_ids = [record["run_id"] for record in launched]
    assert len(run_ids) == 3
    assert {record["status"] for record in launched} == {BgStatus.RUNNING.value}
    assert all(
        manager.poll(run_id, thread_id="obs-thread") in {BgStatus.PENDING, BgStatus.RUNNING}
        for run_id in run_ids
    )

    # Release and settle every run.
    release.set()
    for run_id in run_ids:
        await manager.wait(run_id, thread_id="obs-thread")

    # Turn 2: notification injected; host reads the aggregate `runs` view and folds it.
    state2 = await host.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        {"messages": [{"role": "user", "content": "How are the runs?"}]}, config=config
    )
    messages: list[BaseMessage] = state2["messages"]
    transcript = "\n".join(m.text for m in messages)

    # The aggregate runs view listed every run by id with its terminal status.
    assert WORKFLOW_NOTIFICATION_TAG in transcript
    final_ai = [m for m in messages if isinstance(m, AIMessage) and m.text]
    assert final_ai and "FINAL aggregate view:" in final_ai[-1].text
    aggregate = final_ai[-1].text
    assert all(run_id in aggregate for run_id in run_ids)
    assert aggregate.count("topic_research") == 3
    assert "done" in aggregate

    # The workflow_runs channel is settle-aware: every record is now terminal, not running.
    final_statuses = {record["run_id"]: record["status"] for record in state2["workflow_runs"]}
    assert set(final_statuses) == set(run_ids)
    assert set(final_statuses.values()) == {BgStatus.DONE.value}

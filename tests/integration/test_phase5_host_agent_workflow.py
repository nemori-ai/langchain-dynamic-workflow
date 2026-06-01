"""Phase 5 integration: a host deepagent drives a background workflow end-to-end.

This is the flagship M5 path with no API key. A scripted tool-calling host model
runs a real ``create_deep_agent`` loop with the workflow middleware attached:

1. turn 1 — the host calls ``workflow(run, workflow="research_fanout")``; the tool
   returns a placeholder run_id immediately (the host turn is not blocked).
2. the background run executes (an internal ``parallel`` fan-out over fake leaves)
   and settles; the middleware drains the completion notice and injects a
   ``<workflow_notification>`` before the host's next model call.
3. turn 2 — having seen the notification, the host calls ``workflow(status,
   run_id=...)`` and gets the result.
4. turn 3 — the host folds the conclusion into its final answer.

The whole thing runs offline on fake models, asserting the placeholder-then-notify
loop, the injected notification, and the final folded conclusion.
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
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import PrivateAttr

from langchain_dynamic_workflow import Ctx, Roster
from langchain_dynamic_workflow._background import BgRunManager, BgStatus
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.middleware import (
    WORKFLOW_NOTIFICATION_TAG,
    create_workflow_middleware,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


def _last_run_id(state: Any) -> str:
    """Extract the most recently launched run_id from an agent state mapping."""
    runs: list[dict[str, Any]] = state.get("workflow_runs", [])
    assert runs, "the host should have launched a background run via the tool"
    run_id = runs[-1]["run_id"]
    assert isinstance(run_id, str)
    return run_id


def _transcript(state: Any) -> str:
    """Join all message text in an agent state mapping."""
    messages: list[BaseMessage] = state["messages"]
    return "\n".join(m.text for m in messages)


class ScriptedHostModel(BaseChatModel):
    """A host model scripted to drive the workflow tool across turns.

    The model is intentionally deterministic and key-free: it inspects the
    conversation so far and emits the next scripted action — first a
    ``workflow(run)`` tool call, then (once a ``<workflow_notification>`` is in
    context) a ``workflow(status)`` tool call, then a final answer that folds in
    the fetched result.
    """

    _run_id_box: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def run_id_box(self) -> dict[str, str]:
        """Holds the launched run_id so a later turn can target status by it."""
        return self._run_id_box

    @property
    def _llm_type(self) -> str:
        return "scripted-host"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        notification_seen = any(WORKFLOW_NOTIFICATION_TAG in m.text for m in messages)
        status_fetched = any("done." in m.text for m in tool_messages)
        launch_seen = any("Launched workflow" in m.text for m in tool_messages)

        if status_fetched:
            # The status result is in context — fold it into the final answer.
            last_status = next(m.text for m in reversed(tool_messages) if "done." in m.text)
            reply = f"FINAL: based on the workflow, {last_status}"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        if notification_seen:
            # A completion notification was injected — call status with the run_id.
            run_id = self._run_id_box.get("run_id", "")
            call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "workflow",
                        "args": {"command": "status", "run_id": run_id},
                        "id": "call-status",
                    }
                ],
            )
            return ChatResult(generations=[ChatGeneration(message=call)])

        if launch_seen:
            # The run was launched but has not finished (no notification yet): end
            # this turn with a plain reply rather than relaunching, mirroring a host
            # that continues other work while the background run executes.
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="Workflow launched; awaiting completion.")
                    )
                ]
            )

        # First action: launch the workflow in the background.
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "workflow",
                    "args": {"command": "run", "workflow": "research_fanout"},
                    "id": "call-run",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Ignore the tool schemas and return self (actions are scripted)."""
        return self


async def test_host_agent_drives_background_workflow_end_to_end(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Leaf roster for the background workflow (offline fake leaves).
    leaf, _state = make_fake_leaf("finding")
    roster = Roster().register("researcher", leaf)

    # The workflow blocks on a release event so it cannot finish during turn 1 —
    # this pins the placeholder-then-notify timing deterministically (no race on
    # whether the run settles before turn 1's last model call).
    release = asyncio.Event()

    async def research_fanout(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        # Internal parallel fan-out: three independent research leaves, joined.
        topics = ["a", "b", "c"]
        findings = await ctx.parallel(
            [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
        )
        surviving = [f for f in findings if f is not None]
        return f"synthesized {len(surviving)} findings: " + ", ".join(surviving)

    workflows = WorkflowRegistry().register("research_fanout", research_fanout)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    host_model = ScriptedHostModel()
    host = create_deep_agent(model=host_model, middleware=[middleware])  # pyright: ignore[reportUnknownVariableType, reportArgumentType]

    config: RunnableConfig = {"configurable": {"thread_id": "host-thread"}}

    # Turn 1: the host launches the workflow. The run is gated closed, so it stays
    # in flight while turn 1 finishes with a plain reply — proving the placeholder
    # returned without blocking the host turn.
    state1 = await host.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        {"messages": [{"role": "user", "content": "Research a, b, c"}]}, config=config
    )
    run_id = _last_run_id(state1)
    host_model.run_id_box["run_id"] = run_id
    assert manager.poll(run_id, thread_id="host-thread") in {BgStatus.PENDING, BgStatus.RUNNING}
    assert WORKFLOW_NOTIFICATION_TAG not in _transcript(state1)  # not finished yet

    # Release and settle the run.
    release.set()
    await manager.wait(run_id, thread_id="host-thread")
    assert manager.poll(run_id, thread_id="host-thread") == BgStatus.DONE

    # Turn 2: resume the host conversation. abefore_model injects the completion
    # notification, the host calls status, then folds the result into a final answer.
    state2 = await host.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        {"messages": [{"role": "user", "content": "Are we done?"}]}, config=config
    )
    final_messages: list[BaseMessage] = state2["messages"]
    transcript = "\n".join(m.text for m in final_messages)

    # The injected notification appeared in the host context.
    assert WORKFLOW_NOTIFICATION_TAG in transcript
    # The host folded the workflow conclusion into its final answer.
    final_ai = [m for m in final_messages if isinstance(m, AIMessage) and m.text]
    assert final_ai
    assert "FINAL:" in final_ai[-1].text
    assert "synthesized 3 findings" in final_ai[-1].text


async def test_placeholder_returns_before_workflow_completes(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Pin the non-blocking guarantee directly: the run tool returns while the
    # workflow is still in flight.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)
    release = asyncio.Event()

    async def slow(ctx: Ctx, args: dict[str, Any]) -> str:
        await release.wait()
        return await ctx.agent("Q", agent_type="researcher")

    workflows = WorkflowRegistry().register("slow", slow)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)
    tool = middleware.tools[0]

    runtime: ToolRuntime[Any, Any] = ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": "ht"}},
        stream_writer=lambda _c: None,
        tool_call_id="c1",
        store=None,
    )
    out = await tool.coroutine(command="run", workflow="slow", runtime=runtime)  # type: ignore[misc]
    assert isinstance(out, Command)
    update: dict[str, Any] = out.update or {}
    run_id = update["workflow_runs"][-1]["run_id"]
    assert isinstance(run_id, str)
    # Still in flight while we hold the gate closed.
    assert manager.poll(run_id, thread_id="ht") in {BgStatus.PENDING, BgStatus.RUNNING}
    release.set()
    await manager.wait(run_id, thread_id="ht")
    assert manager.poll(run_id, thread_id="ht") == BgStatus.DONE

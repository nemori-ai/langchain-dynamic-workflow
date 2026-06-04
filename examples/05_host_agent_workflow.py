"""Phase 5 demo (flagship M5): a host deepagent drives a background workflow.

This is the full outward-form loop, fully offline (no API key):

1. A host ``create_deep_agent`` (scripted fake model) calls the ``workflow`` tool
   with ``command="run"`` to launch a registered workflow **in the background**.
   The tool returns a ``run_id`` placeholder immediately — the host turn is not
   blocked.
2. The background workflow fans out three research leaves with ``ctx.parallel``
   and synthesizes them. While it runs, the host turn ends with a plain reply.
3. Once the run settles, the middleware's ``abefore_model`` injects a
   ``<workflow_notification>`` before the host's next model call.
4. Seeing the notification, the host calls ``command="status"`` to fetch the
   result and folds the conclusion into its final answer.

Set ``LDW_DEMO_REAL_MODEL`` to drive real deepagent leaves inside the workflow
through OpenRouter (model ``anthropic/claude-opus-4.8``; credentials from a local
``.env``); the host model stays scripted so the demo is deterministic. The live
path needs ``uv sync --group example``.

    uv run python examples/05_host_agent_workflow.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from _demo_models import demo_cache_middleware, load_demo_env, real_leaf_model
from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    BgRunManager,
    BgStatus,
    Ctx,
    Roster,
    WorkflowRegistry,
    create_workflow_middleware,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG

TOPICS = ["batteries", "solar", "wind"]

# Holds the launched run_id so the scripted host can target status on a later turn.
_RUN_ID_BOX: dict[str, str] = {}


class ScriptedHost(BaseChatModel):
    """A scripted host model: run -> (notify) -> status -> fold the conclusion."""

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-host"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        notification_seen = any(WORKFLOW_NOTIFICATION_TAG in m.text for m in messages)
        status_done = any("done." in m.text for m in tool_messages)
        launched = any("Launched workflow" in m.text for m in tool_messages)

        if status_done:
            result = next(m.text for m in reversed(tool_messages) if "done." in m.text)
            return _say(f"Here is the synthesized answer — {result}")
        if notification_seen:
            return _call("status", run_id=_RUN_ID_BOX.get("run_id", ""))
        if launched:
            return _say("Workflow launched; I'll report back when it finishes.")
        return _call("run", workflow="energy_research")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _say(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _call(command: str, **args: str) -> ChatResult:
    call = AIMessage(
        content="",
        tool_calls=[{"name": "workflow", "args": {"command": command, **args}, "id": command}],
    )
    return ChatResult(generations=[ChatGeneration(message=call)])


def _build_leaf() -> Any:
    model = real_leaf_model(web_search=True)
    if model is not None:
        return create_deep_agent(model=model, middleware=demo_cache_middleware())

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"finding({last})")]}

    return RunnableLambda(_leaf)


async def main() -> None:
    load_demo_env()
    roster = Roster().register("researcher", _build_leaf(), description="Researches one topic")

    async def energy_research(ctx: Ctx, args: dict[str, Any]) -> str:
        # Background workflow body: parallel fan-out over topics, then synthesize.
        findings = await ctx.parallel(
            [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in TOPICS]
        )
        surviving = [f for f in findings if f is not None]
        return f"synthesized {len(surviving)} findings: " + " | ".join(surviving)

    workflows = WorkflowRegistry().register("energy_research", energy_research)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    host = create_deep_agent(
        model=ScriptedHost(), middleware=[middleware, *demo_cache_middleware()]
    )
    config: RunnableConfig = {"configurable": {"thread_id": "demo-5"}}

    # Turn 1: host launches the background workflow and ends its turn (non-blocking).
    state1 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Research energy storage options"}]},
        config=config,
    )
    run_id = state1["workflow_runs"][-1]["run_id"]
    _RUN_ID_BOX["run_id"] = run_id
    print(f"[turn 1] host launched background run_id={run_id}")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    # Let the background run settle (the host could be doing other work meanwhile).
    await manager.wait(run_id, thread_id="demo-5")
    assert manager.poll(run_id, thread_id="demo-5") == BgStatus.DONE
    print("[background] workflow finished")

    # Turn 2: notification is injected; host fetches status and folds the result.
    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Any update?"}]}, config=config
    )
    print(f"[turn 2] host final answer: {state2['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())

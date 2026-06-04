"""Demo (M3.5): a host fans out several workflows in parallel and monitors them.

The host launches one background research run per topic — several at once — then,
instead of polling each ``run_id`` one by one, uses the aggregate runs view to see
all of them in a single call and reacts once they have all landed:

1. Turn 1: the host issues several ``run`` calls in one go (one per topic). Each
   returns a ``run_id`` placeholder immediately; the host turn is never blocked.
   Every launch is recorded in the ``workflow_runs`` state channel as ``running``.
2. The runs execute concurrently in the background.
3. Turn 2: a ``<workflow_notification>`` is injected once runs settle; the host
   asks for the aggregate runs view (every run on the thread with its label and
   live status) and synthesizes from it. By now the ``workflow_runs`` channel has
   been rewritten from ``running`` to each run's terminal status.

Set ``LDW_DEMO_REAL_MODEL`` to drive real deepagent leaves inside each run through
OpenRouter (model ``anthropic/claude-opus-4.8``; credentials from a local
``.env``); the host model stays scripted so the demo is deterministic. The live
path needs ``uv sync --group example``.

    uv run python examples/14_parallel_runs_observability.py
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
    Ctx,
    Roster,
    WorkflowRegistry,
    create_workflow_middleware,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG

TOPICS = ["grid-scale batteries", "green hydrogen", "small modular reactors"]


class ScriptedHost(BaseChatModel):
    """A scripted host: fan out one run per topic -> (notify) -> aggregate runs view."""

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
        launched = [m for m in tool_messages if "Launched workflow" in m.text]
        runs_view = next(
            (m.text for m in reversed(tool_messages) if m.text.startswith("runs:")), None
        )
        notification_seen = any(WORKFLOW_NOTIFICATION_TAG in m.text for m in messages)

        if runs_view is not None:
            # The aggregate view is in hand: fold it into the final answer.
            return _say(f"All parallel runs accounted for.\n{runs_view}")
        if notification_seen:
            # Runs have settled: ask for the aggregate view rather than polling each.
            return _call("runs")
        if launched:
            return _say(f"Launched {len(launched)} parallel research runs; monitoring them.")
        # Initial turn: fan out one background run per topic, all at once.
        return _call_many(
            [{"command": "run", "workflow": "topic_research", "args": {"topic": t}} for t in TOPICS]
        )

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


def _call_many(calls: list[dict[str, Any]]) -> ChatResult:
    tool_calls = [
        {"name": "workflow", "args": call, "id": f"{call['command']}-{i}"}
        for i, call in enumerate(calls)
    ]
    message = AIMessage(content="", tool_calls=tool_calls)
    return ChatResult(generations=[ChatGeneration(message=message)])


def _build_leaf() -> Any:
    model = real_leaf_model(web_search=True)
    if model is not None:
        return create_deep_agent(model=model, middleware=demo_cache_middleware())

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"finding({last})")]}

    return RunnableLambda(_leaf)


async def topic_research(ctx: Ctx, args: dict[str, Any]) -> str:
    """One background run: research a single topic and return a short finding."""
    topic: str = args["topic"]
    finding = await ctx.agent(
        f"Research {topic} and state one key finding in a sentence.", agent_type="researcher"
    )
    return f"[{topic}] {finding}"


async def main() -> None:
    load_demo_env()
    roster = Roster().register("researcher", _build_leaf(), description="Researches one topic")
    workflows = WorkflowRegistry().register("topic_research", topic_research)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    host = create_deep_agent(
        model=ScriptedHost(), middleware=[middleware, *demo_cache_middleware()]
    )
    config: RunnableConfig = {"configurable": {"thread_id": "demo-14"}}

    # Turn 1: the host fans out one background run per topic (non-blocking).
    state1 = await host.ainvoke(
        {"messages": [{"role": "user", "content": f"Research these in parallel: {TOPICS}"}]},
        config=config,
    )
    launched = state1["workflow_runs"]
    run_ids = [record["run_id"] for record in launched]
    print(f"[turn 1] host launched {len(run_ids)} parallel runs: {run_ids}")
    print(f"[turn 1] workflow_runs statuses: {[r['status'] for r in launched]}")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    # Let all background runs settle (the host could be doing other work meanwhile).
    for run_id in run_ids:
        await manager.wait(run_id, thread_id="demo-14")
    print("[background] all parallel runs finished")

    # Turn 2: notification injected; host asks for the aggregate runs view and folds it.
    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "How are the research runs doing?"}]},
        config=config,
    )
    # The workflow_runs channel is now settle-aware: statuses are terminal, not 'running'.
    final_statuses = {r["run_id"]: r["status"] for r in state2["workflow_runs"]}
    print(f"[turn 2] workflow_runs statuses now: {sorted(set(final_statuses.values()))}")
    print(f"[turn 2] host final answer:\n{state2['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())

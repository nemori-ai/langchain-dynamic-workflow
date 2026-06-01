"""Phase 6 capstone demo (flagship v1.0): a host deepagent drives a multi-stage
adversarial research workflow in the background.

This is the full v1.0 loop with every major feature stacked, fully offline:

1. A host ``create_deep_agent`` (scripted fake model) calls the ``workflow`` tool
   with ``command="run"`` to launch the capstone **in the background**. The tool
   returns a ``run_id`` placeholder immediately — the host turn is not blocked.
2. The background workflow runs the engine's full primitive stack:
   - ``parallel`` fan-out research over N source topics (blocking barrier),
   - ``pipeline`` refinement of each finding (no barrier between stages),
   - adversarial verification: each refined finding is challenged by N skeptic
     leaves in ``parallel``; a finding survives only if a majority vote it valid,
   - synthesis of the survivors into a conclusion.
   The whole run is budgeted (a shared token pool metered through the leaves) and
   sandbox-admitted (the ``needs_execution`` researcher is leased an isolated
   backend that the engine tears down when the run settles).
3. Once the run settles, the middleware's ``abefore_model`` injects a
   ``<workflow_notification>`` before the host's next model call.
4. Seeing the notification, the host calls ``command="status"`` to fetch the
   result and folds the conclusion into its final answer.

By default every leaf is a deterministic fake (no API key). Set
``LDW_DEMO_REAL_MODEL=anthropic:claude-haiku-4-5`` (and a key) to drive real
deepagent leaves inside the workflow; the host model stays scripted so the demo
is deterministic and the leaf split stays reproducible.

    uv run python examples/06_capstone.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

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
    SandboxManager,
    Span,
    WorkflowRegistry,
    create_workflow_middleware,
    run_workflow,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG

TOPICS = ["alpha", "beta", "gamma", "delta"]
SKEPTICS_PER_FINDING = 3

# Holds the launched run_id so the scripted host can target status on a later turn.
_RUN_ID_BOX: dict[str, str] = {}


def _build_leaf(reply: str) -> Any:
    """Build a research/refine leaf: a real deepagent if env-gated, else a fake."""
    spec = os.environ.get("LDW_DEMO_REAL_MODEL")
    if spec:
        from langchain.chat_models import init_chat_model

        return create_deep_agent(model=init_chat_model(spec))

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_leaf)


def _verdict_leaf() -> Any:
    """A deterministic skeptic leaf: votes ``valid``/``invalid`` by topic parity.

    Kept fake even in the real-model variant so the majority split stays
    reproducible: an odd-length topic survives (every skeptic votes ``valid``), an
    even-length topic is rejected (every skeptic votes ``invalid``).
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        topic = prompt.split()[-1] if prompt.split() else ""
        verdict = "valid" if len(topic) % 2 == 1 else "invalid"
        return {"messages": [*inp["messages"], AIMessage(content=verdict)]}

    return RunnableLambda(_call)


async def capstone(ctx: Ctx, args: dict[str, Any]) -> str:
    """research -> refine -> adversarial verify (majority survives) -> synthesize."""
    topics: list[str] = args.get("topics", TOPICS)

    ctx.phase("research")
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    researched = [(t, f) for t, f in zip(topics, findings, strict=True) if f is not None]

    ctx.phase("refine")

    async def refine_stage(prev: Any, item: Any, index: int) -> tuple[str, str]:
        topic, _finding = item
        refined = await ctx.agent(f"Refine {topic}", agent_type="refiner")
        return (topic, refined)

    refined_pairs = await ctx.pipeline(researched, refine_stage)
    refined = [p for p in refined_pairs if p is not None]

    ctx.phase("verify")
    survivors: list[str] = []
    for topic, refined_text in refined:
        verdicts = await ctx.parallel(
            [
                lambda t=topic: ctx.agent(f"Challenge {t}", agent_type="skeptic")
                for _ in range(SKEPTICS_PER_FINDING)
            ]
        )
        valid_votes = sum(1 for v in verdicts if v == "valid")
        if valid_votes * 2 > SKEPTICS_PER_FINDING:  # strict majority survives
            survivors.append(f"{topic}:{refined_text}")

    ctx.phase("synthesize")
    return f"synthesized {len(survivors)} surviving findings: " + " | ".join(sorted(survivors))


class ScriptedHost(BaseChatModel):
    """A scripted host: run -> (notify) -> status -> fold the conclusion."""

    @property
    def _llm_type(self) -> str:
        return "demo-capstone-host"

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
            return _say(f"Here is the verified synthesis — {result}")
        if notification_seen:
            return _call("status", run_id=_RUN_ID_BOX.get("run_id", ""))
        if launched:
            return _say("Capstone launched; I'll report back when it finishes.")
        return _call("run", workflow="capstone")

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


async def main() -> None:
    roster = (
        Roster()
        .register(
            "researcher",
            _build_leaf("finding"),
            description="Researches one source topic",
            needs_execution=True,
        )
        .register("refiner", _build_leaf("refined"), description="Refines one finding")
        .register("skeptic", _verdict_leaf(), description="Adversarially challenges a finding")
    )
    workflows = WorkflowRegistry().register("capstone", capstone)
    manager = BgRunManager(max_concurrent_runs=8)
    # A standalone sandbox manager + budget the launched run uses, plus a span sink
    # so the demo can print the primitive trace the run emitted.
    sandbox_manager = SandboxManager()
    spans: list[Span] = []

    def _launch_capstone(thread_id: str) -> Any:
        async def _orchestrate(ctx: Ctx) -> str:
            return await capstone(ctx, {})

        return run_workflow(
            _orchestrate,
            roster=roster,
            budget=100_000,
            sandbox_manager=sandbox_manager,
            on_span=spans.append,
            thread_id=thread_id,
        )

    # The middleware contributes the workflow tool + completion-notify; it builds
    # the host-turn loop. For the demo we also show a direct (no-host) run so the
    # span trace and sandbox teardown are visible.
    print("=== direct run (no host): full primitive trace ===")
    direct = await _launch_capstone("demo-6-direct")
    print(f"result: {direct}")
    by_kind: dict[str, int] = {}
    for span in spans:
        by_kind[span.kind.value] = by_kind.get(span.kind.value, 0) + 1
    print(f"spans emitted by kind: {by_kind}")
    print(f"active sandboxes still live after the run: {sandbox_manager.active_count}")

    print("\n=== host-driven run (background tool + notify) ===")
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)
    host = create_deep_agent(model=ScriptedHost(), middleware=[middleware])
    config: RunnableConfig = {"configurable": {"thread_id": "demo-6-host"}}

    state1 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Run the verified research capstone"}]},
        config=config,
    )
    run_id = state1["workflow_runs"][-1]["run_id"]
    _RUN_ID_BOX["run_id"] = run_id
    print(f"[turn 1] host launched background run_id={run_id}")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    await manager.wait(run_id, thread_id="demo-6-host")
    assert manager.poll(run_id, thread_id="demo-6-host") == BgStatus.DONE
    print("[background] capstone finished")

    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Any update?"}]}, config=config
    )
    print(f"[turn 2] host final answer: {state2['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())

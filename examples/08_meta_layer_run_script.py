"""Phase 8 demo: the meta layer — a host AUTHORS a script and runs it via ``run_script``.

The earlier demos launch a workflow someone registered ahead of time. This one
shows the meta layer: the host writes an ``async def orchestrate(ctx, args)`` on
the spot and submits the *source* through the ``workflow`` tool's ``run_script``
command. The source crosses a single security seam — an AST gate plus a
restricted-builtins ``exec`` — before it ever runs.

The offline (scripted) host also shows the feed-back-and-retry loop the meta layer
is built around: it first submits a script that reaches for ``import`` (rejected,
with the exact violation handed back), then fixes it and resubmits a clean one
that fans out research and synthesizes — proving the gate turns an unsafe script
into a precise, actionable rejection rather than a silent failure.

Security boundary (A1): the gate stops an accidental slip, not a determined
adversary — an in-process restricted ``exec`` is not a security sandbox. Submit
only scripts the host authors itself.

Run it:

    uv sync --group example
    # credentials + model come from a local .env (OPENROUTER_API_KEY); see _demo_models
    export LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8
    uv run python examples/08_meta_layer_run_script.py

With ``LDW_DEMO_REAL_MODEL`` unset the demo runs fully offline: a scripted host
authors the script and drives deterministic fake leaves, so the meta-layer path is
exercised end to end with no API key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from _demo_models import load_demo_env, real_model
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    BgRunManager,
    BgStatus,
    Roster,
    WorkflowRegistry,
    create_workflow_middleware,
    skills_path,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG

TOPICS = ["grid-scale batteries", "solar PV", "onshore wind"]
TASK = "Compare these clean-energy options and synthesize a recommendation."

# The script the offline host first submits — it reaches for `import`, so the gate
# rejects it and hands back the exact violation (the teachable failure).
REJECTED_SCRIPT = """\
import statistics

async def orchestrate(ctx, args):
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in args["topics"]]
    )
    return statistics.mode(findings)
"""

# The corrected script: a clean parallel-research -> synthesize orchestration that
# uses only ctx primitives and plain builtins (no imports, f-strings not .format).
AUTHORED_SCRIPT = """\
meta = {"name": "ad-hoc-energy-compare"}

async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    ctx.phase("research")
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    surviving = [f for f in findings if f is not None]
    ctx.phase("synthesize")
    joined = "\\n".join(surviving)
    return await ctx.agent(f"Synthesize a recommendation from:\\n{joined}", agent_type="writer")
"""

HOST_SYSTEM_PROMPT = (
    "You orchestrate work with a `workflow` tool. There is NO registered workflow for this "
    "task, so you must AUTHOR one: write a self-contained `async def orchestrate(ctx, args)` "
    "that fans out one `ctx.agent(prompt, agent_type='researcher')` per topic in "
    "args['topics'] using `ctx.parallel`, then folds the findings with a single "
    "`ctx.agent(prompt, agent_type='writer')`. Submit it with command='run_script', "
    "script=<your source>, args={'topics': [...]}. Use only the ctx primitives and plain "
    "builtins — no imports, no dunder access, and use f-strings (never str.format). If the "
    "tool rejects your script, read the listed violations, fix them, and resubmit. The tool "
    "returns a run_id immediately and runs in the background, so end your turn after launching. "
    "When notified it finished, call command='status' with the run_id and present the result."
)

# Holds the launched run_id so the scripted (offline) host can target status later.
_RUN_ID_BOX: dict[str, str] = {}


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_echo_leaf(prefix: str) -> Any:
    """An offline fake leaf that echoes a trimmed prompt behind a role prefix."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"{prefix}: {last.strip()[:80]}")]}

    return RunnableLambda(_leaf)


def _build_leaf(role: str) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model)
    return _fake_echo_leaf(role)


# ── offline scripted host (authors a script; real host replaces it when env-gated) ──


class ScriptedHost(BaseChatModel):
    """A scripted host: bad -> (rejected) -> clean -> (notify) -> status -> present."""

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-host-8"

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
        launched = any("Launched your authored script" in m.text for m in tool_messages)
        rejected = any("was rejected" in m.text for m in tool_messages)

        if status_done:
            report = next(m.text for m in reversed(tool_messages) if "done." in m.text)
            return _say(f"Here is the synthesized recommendation — {report}")
        if notification_seen:
            return _tool_call("status", run_id=_RUN_ID_BOX.get("run_id", ""))
        if launched:
            return _say("Script launched; I'll report back when it finishes.")
        if rejected:
            # Saw the gate's violations — fix the script and resubmit (the retry loop).
            return _run_script_call(AUTHORED_SCRIPT)
        # First attempt: deliberately unsafe, to show the gate hand back the violation.
        return _run_script_call(REJECTED_SCRIPT)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _say(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _run_script_call(script: str) -> ChatResult:
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "workflow",
                "args": {"command": "run_script", "script": script, "args": {"topics": TOPICS}},
                "id": "run_script",
            }
        ],
    )
    return ChatResult(generations=[ChatGeneration(message=call)])


def _tool_call(command: str, *, run_id: str) -> ChatResult:
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "workflow", "args": {"command": command, "run_id": run_id}, "id": command}
        ],
    )
    return ChatResult(generations=[ChatGeneration(message=call)])


# ── driver ───────────────────────────────────────────────────────────────────


async def main() -> None:
    load_demo_env()
    host_model = real_model()
    roster = (
        Roster()
        .register("researcher", _build_leaf("researcher"), description="Researches one topic")
        .register("writer", _build_leaf("writer"), description="Synthesizes the recommendation")
    )
    # No registered workflows: the host must author its own script via run_script.
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=WorkflowRegistry(), manager=manager)

    host_kwargs: dict[str, Any] = {"middleware": [middleware]}
    if host_model is not None:
        host_kwargs["model"] = host_model
        host_kwargs["system_prompt"] = HOST_SYSTEM_PROMPT
        host_kwargs["skills"] = [str(skills_path())]
        host_kwargs["backend"] = FilesystemBackend(root_dir=str(skills_path()), virtual_mode=False)
    else:
        host_kwargs["model"] = ScriptedHost()
    host = create_deep_agent(**host_kwargs)
    config: RunnableConfig = {"configurable": {"thread_id": "demo-8"}}

    print(f"task: {TASK}")
    print(f"topics: {TOPICS}")
    print(f"mode: {'REAL (OpenRouter)' if host_model is not None else 'offline (fake)'}")

    # Turn 1: the host authors a script and submits it via run_script (offline: it
    # is rejected once for an import, then corrected and launched).
    state1 = await host.ainvoke(
        {"messages": [{"role": "user", "content": f"{TASK} Topics: {', '.join(TOPICS)}."}]},
        config=config,
    )
    # Surface the gate's feed-back-and-retry loop: print the first rejection (if any)
    # so the violation the host had to fix is visible, not just the eventual launch.
    for message in state1["messages"]:
        if isinstance(message, ToolMessage) and "was rejected" in message.text:
            print(f"[turn 1] gate rejected the first script:\n  {message.text.splitlines()[-1]}")
            break

    runs = state1.get("workflow_runs", [])
    if not runs:
        print("[turn 1] host did not launch a script. reply:", state1["messages"][-1].text)
        return
    run_id = runs[-1]["run_id"]
    _RUN_ID_BOX["run_id"] = run_id
    print(f"[turn 1] launched run_id={run_id} (workflow label: {runs[-1].get('workflow')!r})")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    # Let the background run settle.
    await manager.wait(run_id, thread_id="demo-8")
    assert manager.poll(run_id, thread_id="demo-8") == BgStatus.DONE
    print("[background] authored script finished")

    # Turn 2: notification is injected; the host fetches the result and presents it.
    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Is it done? Give me the recommendation."}]},
        config=config,
    )
    print(f"[turn 2] final answer:\n{state2['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())

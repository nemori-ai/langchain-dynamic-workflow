"""AST gate — the meta layer's security seam: reject an unsafe authored script.

The other meta-layer demo (the flagship) only walks the happy path: a host
authors an ``async def orchestrate(ctx, args)`` and submits the *source* through
the ``workflow`` tool, where it crosses a single security seam — an AST gate plus
a restricted-builtins ``exec`` — before it ever runs. This offline demo owns the
*rejection* path that makes the seam trustworthy.

A scripted host first submits a script that reaches for ``import`` (rejected, with
the exact violation handed back), then fixes it and resubmits a clean one that
fans out research and synthesizes. The contrast — the gate's precise rejection
followed by the corrected script launching — is the mechanism: the gate turns an
unsafe script into an actionable rejection rather than a silent failure.

Security boundary (A1): the gate stops an accidental slip, not a determined
adversary — an in-process restricted ``exec`` is not a security sandbox. Submit
only scripts the host authors itself.

    uv run python -m examples.features.ast_gate
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig

from examples._shared.offline_models import echo_leaf
from langchain_dynamic_workflow import (
    BgRunManager,
    BgStatus,
    Roster,
    WorkflowRegistry,
    create_workflow_middleware,
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

# Holds the launched run_id so the scripted host can target status later.
_RUN_ID_BOX: dict[str, str] = {}


# ── offline scripted host (authors a script; bad -> rejected -> clean) ─────────


class ScriptedHost(BaseChatModel):
    """A scripted host: bad -> (rejected) -> clean -> (notify) -> status -> present."""

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-host-ast-gate"

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
    roster = (
        Roster()
        .register("researcher", echo_leaf("finding"), description="Researches one topic")
        .register("writer", echo_leaf("synthesis"), description="Synthesizes the recommendation")
    )
    # No registered workflows: the host must author its own script via run_script.
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=WorkflowRegistry(), manager=manager)

    host = create_deep_agent(model=ScriptedHost(), middleware=[middleware])
    config: RunnableConfig = {"configurable": {"thread_id": "ast-gate"}}

    print(f"task: {TASK}")
    print(f"topics: {TOPICS}")

    # Turn 1: the host authors a script and submits it via run_script — it is
    # rejected once for an import, then corrected and launched.
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
    rejected = any(
        isinstance(m, ToolMessage) and "was rejected" in m.text for m in state1["messages"]
    )
    assert rejected, "the unsafe first script must be rejected by the AST gate"
    assert state1.get("workflow_runs"), "the corrected script must then launch"
    run_id = runs[-1]["run_id"]
    _RUN_ID_BOX["run_id"] = run_id
    print(f"[turn 1] launched run_id={run_id} (workflow label: {runs[-1].get('workflow')!r})")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    # Let the background run settle.
    await manager.wait(run_id, thread_id="ast-gate")
    assert manager.poll(run_id, thread_id="ast-gate") == BgStatus.DONE
    print("[background] authored script finished")

    # Turn 2: notification is injected; the host fetches the result and presents it.
    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Is it done? Give me the recommendation."}]},
        config=config,
    )
    print(f"[turn 2] final answer:\n{state2['messages'][-1].text}")
    print("OK: AST gate rejected the unsafe script, the host fixed it, and it ran.")


if __name__ == "__main__":
    asyncio.run(main())

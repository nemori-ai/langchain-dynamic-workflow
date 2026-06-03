"""Phase 7 demo: a REAL host agent runs a registered ``deep_research`` workflow end to end.

Unlike the other demos (which script the host for determinism), this one is built
to run against a **real model through OpenRouter**: a real host agent reads the
``dynamic-workflow`` skill, decides on its own to launch the registered
``deep_research`` workflow via the ``workflow`` tool, waits for the background run,
then fetches and presents the report.

The ``deep_research`` workflow ports the shape of Claude Code's built-in
deep-research dynamic workflow onto this engine's primitives:

    search (parallel fan-out, one researcher per angle)
      -> extract (no-barrier pipeline: one falsifiable claim per finding)
      -> verify (3-vote adversarial skeptics per claim; >=2 refutes kills it; each
                 skeptic is a read-only judge — it can only verify, never edit)
      -> synthesize (one writer folds the surviving claims into a report)

The leaves reason from the model's own knowledge (no live WebSearch/WebFetch tools
are wired in — that is the natural extension point); the value shown here is the
deterministic control-flow inversion driving real model calls.

Run it:

    uv sync --group example
    # credentials + model come from a local .env (OPENROUTER_API_KEY); see _demo_models
    export LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8
    uv run python examples/07_deep_research_real_e2e.py

With ``LDW_DEMO_REAL_MODEL`` unset the demo runs fully offline: a scripted host
drives deterministic fake leaves, so the orchestration is exercised end to end
with no API key (this is the path the integration test pins).
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
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    BgRunManager,
    BgStatus,
    Ctx,
    Roster,
    WorkflowRegistry,
    create_workflow_middleware,
    read_only_leaf,
    skills_path,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG

QUESTION = (
    "What are the main trade-offs between retrieval-augmented generation and long-context LLMs?"
)

# Fixed research angles keep the agent() call sequence deterministic across replays
# (the JS original decomposes dynamically via a structured Scope agent).
ANGLES = [
    "core established findings",
    "supporting evidence and data",
    "contrarian views and limitations",
    "practical implications and cost",
]
SKEPTICS_PER_CLAIM = 3
REFUTATIONS_TO_KILL = 2

# Holds the launched run_id so the scripted (offline) host can target status later.
_RUN_ID_BOX: dict[str, str] = {}

HOST_SYSTEM_PROMPT = (
    "You are a research assistant with a `workflow` tool that runs a deterministic, "
    "multi-agent `deep_research` pipeline in the background. When the user asks you to "
    "research a topic, call the tool with command='run', workflow='deep_research', and "
    "args={'question': <the user's question>}. It returns a run_id immediately and runs "
    "in the background, so end your turn right after launching. When you are later "
    "notified that the run finished, call the tool with command='status' and that run_id "
    "to retrieve the report, then present it clearly."
)


# ── structured leaf contracts (schema-as-handoff) ─────────────────────────────


class Claim(BaseModel):
    """A single falsifiable claim extracted from one angle's research notes."""

    text: str
    checkable: bool


class Verdict(BaseModel):
    """One skeptic's adversarial ruling on a claim."""

    refuted: bool
    reason: str


# ── the registered workflow ──────────────────────────────────────────────────


def _search_prompt(question: str, angle: str) -> str:
    return (
        "You are a researcher. Investigate this question from one specific angle and "
        "report concrete findings.\n"
        f"Question: {question}\nAngle: {angle}\n"
        "Write 2-3 substantive sentences grounded in what you know. Be specific."
    )


def _extract_prompt(question: str, angle: str, finding: str) -> str:
    return (
        "From the research notes below, extract the single most important, falsifiable "
        "claim bearing on the question. Put the claim as ONE concrete sentence in `text`, "
        "and set `checkable` to true only if it is a factual statement that could in "
        f"principle be verified.\nQuestion: {question}\nAngle: {angle}\nNotes: {finding}"
    )


def _verify_prompt(question: str, claim: str, voter: int) -> str:
    return (
        f"You are skeptic #{voter + 1} reviewing a claim for factual accuracy from your own "
        "knowledge. Set `refuted` to true only if the claim is factually wrong, misleading, "
        "or clearly overstated; otherwise set it to false. Give one sentence of `reason`. "
        "These claims are reasoned rather than web-sourced, so judge correctness, not "
        f"citation presence.\nQuestion: {question}\nClaim: {claim}"
    )


def _synthesize_prompt(question: str, confirmed: list[str]) -> str:
    if not confirmed:
        return (
            "Research was inconclusive — no claims survived adversarial verification for: "
            f"{question}. Write 2-3 honest sentences saying so and what sources would help."
        )
    joined = "\n".join(f"- {claim.strip()}" for claim in confirmed)
    return (
        "Write a concise research report answering the question, using ONLY the verified "
        "claims below. Structure: a 2-3 sentence executive summary, then bullet findings, "
        f"then a one-line caveat.\nQuestion: {question}\nVerified claims:\n{joined}"
    )


async def deep_research(ctx: Ctx, args: dict[str, Any]) -> str:
    """search -> extract -> adversarial verify -> synthesize, deep-research style.

    The extract and verify leaves hand back **schema-validated objects** (``Claim``
    / ``Verdict``), so the reduce between phases is plain Python over typed data —
    no brittle string parsing of the leaves' prose.
    """
    question: str = args["question"]

    ctx.phase("search")
    findings = await ctx.parallel(
        [
            lambda a=a: ctx.agent(_search_prompt(question, a), agent_type="researcher")
            for a in ANGLES
        ]
    )
    paired = [(angle, found) for angle, found in zip(ANGLES, findings, strict=True) if found]
    ctx.log(f"researched {len(paired)}/{len(ANGLES)} angles")

    ctx.phase("extract")

    async def _extract(_prev: Any, item: tuple[str, str], _index: int) -> Claim:
        angle, finding = item
        return await ctx.agent(
            _extract_prompt(question, angle, finding), agent_type="extractor", schema=Claim
        )

    claims = [c for c in await ctx.pipeline(paired, _extract) if c is not None and c.checkable]
    ctx.log(f"extracted {len(claims)} checkable claims")

    ctx.phase("verify")
    confirmed: list[str] = []
    for claim in claims:
        verdicts = await ctx.parallel(
            [
                lambda c=claim.text, v=v: ctx.agent(
                    _verify_prompt(question, c, v), agent_type="skeptic", schema=Verdict
                )
                for v in range(SKEPTICS_PER_CLAIM)
            ]
        )
        refutes = sum(1 for verdict in verdicts if verdict is not None and verdict.refuted)
        survived = refutes < REFUTATIONS_TO_KILL
        mark = "kept" if survived else "killed"
        ctx.log(f"claim {mark} ({refutes}/{SKEPTICS_PER_CLAIM} refute): {claim.text.strip()[:50]}")
        if survived:
            confirmed.append(claim.text)

    ctx.phase("synthesize")
    return await ctx.agent(_synthesize_prompt(question, confirmed), agent_type="writer")


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_echo_leaf(prefix: str) -> Any:
    """An offline fake leaf that echoes a trimmed prompt behind a role prefix."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"{prefix}: {last.strip()[:80]}")]}

    return RunnableLambda(_leaf)


def _fake_structured_leaf(structured: BaseModel, *, reply: str) -> Any:
    """An offline fake leaf that attaches a fixed ``structured_response``.

    Stands in for a ``create_deep_agent`` built with ``response_format=ToolStrategy(...)``:
    the leaf appends an ``AIMessage`` and hands back the given validated model
    instance under ``structured_response`` so ``agent(schema=...)`` can fold it out.
    """

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def _build_leaf(role: str) -> Any:
    """Build a schema-less text leaf (researcher / writer)."""
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model)
    return _fake_echo_leaf(role)


def _build_extractor(*, response_format: Any = None) -> Any:
    """Builder for the ``extractor`` leaf, forwarding ``response_format`` (Claim)."""
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline: emit a deterministic checkable Claim so every angle survives extraction.
    return _fake_structured_leaf(
        Claim(
            text="RAG and long-context LLMs trade recall breadth for context cost",
            checkable=True,
        ),
        reply="extracted a checkable claim",
    )


def _build_skeptic(*, response_format: Any = None) -> Any:
    """Builder for the ``skeptic`` leaf (a read-only judge), forwarding ``response_format``.

    The skeptic adversarially verifies a claim — it should only *judge*, never mutate
    state. Building it with ``read_only_leaf`` (a deny-write FilesystemPermission)
    makes that physical: even though it is a full deepagent with file tools, a write
    is refused at the tool boundary, so a hallucinated "fix" can never escape the
    verifier. This is the canonical way a host wires a read-only judge into a
    workflow: register the builder, and ``ctx.agent(agent_type="skeptic", schema=...)``
    yields a structured, read-only verdict.
    """
    model = real_model()
    if model is not None:
        return read_only_leaf(model, response_format=response_format)
    # Offline skeptics never refute, so every claim survives — a deterministic,
    # readable demo. The real path exercises genuine adversarial refutation.
    return _fake_structured_leaf(
        Verdict(refuted=False, reason="consistent with the cited evidence"),
        reply="reviewed the claim",
    )


# ── offline scripted host (deterministic; real host replaces it when env-gated) ──


class ScriptedHost(BaseChatModel):
    """A scripted host: run -> (notify) -> status -> present, mirroring a real host."""

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-host-7"

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
            report = next(m.text for m in reversed(tool_messages) if "done." in m.text)
            return _say(f"Here is the research report — {report}")
        if notification_seen:
            return _tool_call("status", run_id=_RUN_ID_BOX.get("run_id", ""))
        if launched:
            return _say("Deep research launched; I'll report back when it finishes.")
        return _run_call(QUESTION)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _say(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _run_call(question: str) -> ChatResult:
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "workflow",
                "args": {
                    "command": "run",
                    "workflow": "deep_research",
                    "args": {"question": question},
                },
                "id": "run",
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
        .register("researcher", _build_leaf("researcher"), description="Researches one angle")
        .register("extractor", builder=_build_extractor, description="Extracts a falsifiable claim")
        .register("skeptic", builder=_build_skeptic, description="Adversarially verifies a claim")
        .register("writer", _build_leaf("writer"), description="Synthesizes the final report")
    )
    workflows = WorkflowRegistry().register("deep_research", deep_research)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    host_kwargs: dict[str, Any] = {"middleware": [middleware]}
    if host_model is not None:
        # Real host: a live OpenRouter agent that reads the skill and drives the tool.
        host_kwargs["model"] = host_model
        host_kwargs["system_prompt"] = HOST_SYSTEM_PROMPT
        host_kwargs["skills"] = [str(skills_path())]
        host_kwargs["backend"] = FilesystemBackend(root_dir=str(skills_path()), virtual_mode=False)
    else:
        host_kwargs["model"] = ScriptedHost()
    host = create_deep_agent(**host_kwargs)
    config: RunnableConfig = {"configurable": {"thread_id": "demo-7"}}

    print(f"question: {QUESTION}")
    print(f"mode: {'REAL (OpenRouter)' if host_model is not None else 'offline (fake)'}")

    # Turn 1: the host launches the background workflow and ends its turn.
    state1 = await host.ainvoke(
        {
            "messages": [
                {"role": "user", "content": f"Do deep, fact-checked research on: {QUESTION}"}
            ]
        },
        config=config,
    )
    runs = state1.get("workflow_runs", [])
    if not runs:
        print("[turn 1] host did not launch a workflow. reply:", state1["messages"][-1].text)
        return
    run_id = runs[-1]["run_id"]
    _RUN_ID_BOX["run_id"] = run_id
    print(f"[turn 1] launched run_id={run_id}")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    # Let the background run settle.
    await manager.wait(run_id, thread_id="demo-7")
    assert manager.poll(run_id, thread_id="demo-7") == BgStatus.DONE
    print("[background] deep research finished")

    # Turn 2: notification is injected; the host fetches the report and presents it.
    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Is the research done? Give me the report."}]},
        config=config,
    )
    print(f"[turn 2] final answer:\n{state2['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())

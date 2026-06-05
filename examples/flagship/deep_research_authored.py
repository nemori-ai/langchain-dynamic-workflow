"""A host authors a deep-research script live and runs it through ``run_script``.

This flagship combines the deep-research scenario with the meta layer: instead of
launching a workflow someone registered ahead of time, the host writes the
``async def orchestrate(ctx, args)`` source on the spot and submits it through the
``workflow`` tool's ``run_script`` command. The source crosses one security seam —
an AST gate plus a restricted-builtins ``exec`` — and, on the happy path shown
here, passes cleanly and runs against the same real leaf stack as the registered
flagship: research leaves carry OpenRouter's native web search and every agent
runs the Anthropic prompt-caching middleware. The authored script itself uses only
``ctx`` primitives and the reduce helpers (``dedup`` / ``survives``) injected into
the script namespace — no imports.

Security boundary: the gate stops an honest model's slip, not a determined
adversary — an in-process restricted ``exec`` is not a security sandbox. Submit
only scripts the host authors itself. The rejection-and-retry path is the offline
``examples.features.ast_gate`` demo; this flagship walks the happy path only.

With ``LDW_DEMO_REAL_MODEL`` unset the demo runs fully offline: a scripted host
submits the known-good authored script and drives deterministic fake leaves, so
the meta-layer path is exercised end to end with no API key.

    uv run python -m examples.flagship.deep_research_authored
    # real end-to-end:
    export LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8
    uv run python -m examples.flagship.deep_research_authored
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from examples._shared.offline_models import echo_leaf, structured_builder
from examples._shared.real_models import (
    demo_cache_middleware,
    load_demo_env,
    real_leaf_model,
    real_model,
)
from langchain_dynamic_workflow import (
    BgRunManager,
    BgStatus,
    Roster,
    WorkflowRegistry,
    compile_workflow_source,
    create_workflow_middleware,
    read_only_builder,
    skills_path,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG

QUESTION = (
    "What are the main trade-offs between retrieval-augmented generation and long-context LLMs?"
)

# Fixed research angles keep the authored script's agent() call sequence deterministic
# across replays.
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
    "You are a capable analyst. When a task has many independent parts, you think in terms "
    "of decomposing it, working the parts in parallel, and folding the results into a single "
    "synthesis — and when no ready-made procedure fits the task, you compose one yourself "
    "rather than doing everything in a single pass. Make full use of the tools and skills "
    "available to you, and present a clear, well-reasoned result."
)


# ── structured leaf contracts (wired through the roster builders) ─────────────


class Claim(BaseModel):
    """A single falsifiable claim extracted from one angle's research notes."""

    text: str
    checkable: bool


class Verdict(BaseModel):
    """One skeptic's adversarial ruling on a claim."""

    refuted: bool
    reason: str


# The script the offline host authors and submits. It is a clean deep-research-shaped
# orchestration — search -> extract -> adversarial verify -> synthesize — written with
# only ctx primitives and the injected reduce helpers (dedup / survives are available in
# the run_script namespace by name; the AST gate forbids imports). The verify step votes
# over the skeptics' verdict text: a claim is refuted when two or more skeptics say so.
AUTHORED_DEEP_RESEARCH_SCRIPT = """\
meta = {"name": "ad-hoc-deep-research"}

async def orchestrate(ctx, args):
    question = args["question"]
    angles = args["angles"]
    skeptics_per_claim = args["skeptics_per_claim"]
    refutations_to_kill = args["refutations_to_kill"]

    ctx.phase("search")
    findings = await ctx.parallel(
        [
            lambda a=a: ctx.agent(
                f"Research this question from one angle and report concrete findings.\\n"
                f"Question: {question}\\nAngle: {a}",
                agent_type="researcher",
            )
            for a in angles
        ]
    )
    paired = [(a, f) for a, f in zip(angles, findings, strict=True) if f is not None]
    ctx.log(f"researched {len(paired)} of {len(angles)} angles")

    ctx.phase("extract")

    async def extract(prev, item, index):
        angle, finding = item
        return await ctx.agent(
            f"Extract one falsifiable claim bearing on the question.\\n"
            f"Question: {question}\\nAngle: {angle}\\nNotes: {finding}",
            agent_type="extractor",
        )

    extracted = [c for c in await ctx.pipeline(paired, extract) if c is not None]
    claims = dedup(extracted, key=lambda c: c.strip().lower())
    ctx.log(f"extracted {len(claims)} claims ({len(extracted) - len(claims)} dups merged)")

    ctx.phase("verify")
    confirmed = []
    for claim in claims:
        verdicts = await ctx.parallel(
            [
                lambda c=claim, v=v: ctx.agent(
                    f"You are skeptic #{v + 1}. Judge this claim for factual accuracy and "
                    f"say whether you refute it.\\nQuestion: {question}\\nClaim: {c}",
                    agent_type="skeptic",
                )
                for v in range(skeptics_per_claim)
            ]
        )
        survived = survives(
            verdicts,
            against=lambda verdict: "refute" in verdict.lower(),
            kill_at=refutations_to_kill,
        )
        ctx.log(f"claim {'kept' if survived else 'killed'}: {claim.strip()[:50]}")
        if survived:
            confirmed.append(claim)

    ctx.phase("synthesize")
    joined = "\\n".join(f"- {c.strip()}" for c in confirmed)
    return await ctx.agent(
        f"Write a concise research report answering the question using only the verified "
        f"claims below.\\nQuestion: {question}\\nVerified claims:\\n{joined}",
        agent_type="writer",
    )
"""


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _build_leaf(role: str, *, web_search: bool = False) -> Any:
    """Build a schema-less text leaf (researcher / writer / extractor)."""
    model = real_leaf_model(web_search=web_search)
    if model is not None:
        return create_deep_agent(model=model, middleware=demo_cache_middleware())
    return echo_leaf(role)


def _build_skeptic(*, response_format: Any = None) -> Any:
    """Builder for the ``skeptic`` leaf, a read-only adversarial judge.

    The skeptic should only judge, never mutate state. Built live with
    ``read_only_builder`` it is a full deepagent whose write tools are refused at the
    tool boundary, so a hallucinated edit can never escape the verifier. Offline it is
    a structured fake whose verdict text never refutes — every claim survives — keeping
    the happy-path run deterministic; the real path exercises genuine refutation.
    """
    model = real_leaf_model(web_search=True)
    if model is not None:
        return read_only_builder(model, middleware=demo_cache_middleware())(
            response_format=response_format
        )
    return structured_builder(
        lambda: Verdict(refuted=False, reason="consistent with the cited evidence"),
        reply="reviewed the claim; it holds up",
    )(response_format=response_format)


def _build_extractor(*, response_format: Any = None) -> Any:
    """Builder for the ``extractor`` leaf, forwarding ``response_format`` (Claim)."""
    model = real_leaf_model()
    if model is not None:
        return create_deep_agent(
            model=model, response_format=response_format, middleware=demo_cache_middleware()
        )
    return structured_builder(
        lambda: Claim(text="a checkable claim about the trade-offs", checkable=True),
        reply="a checkable claim about the trade-offs",
    )(response_format=response_format)


# ── offline scripted host (authors a script; real host replaces it when env-gated) ──


class ScriptedHost(BaseChatModel):
    """A scripted host: run_script -> (notify) -> status -> present (happy path only)."""

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-host-authored"

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

        if status_done:
            report = next(m.text for m in reversed(tool_messages) if "done." in m.text)
            return _say(f"Here is the research report — {report}")
        if notification_seen:
            return _tool_call("status", run_id=_RUN_ID_BOX.get("run_id", ""))
        if launched:
            return _say("Authored deep research launched; I'll report back when it finishes.")
        return _run_script_call(AUTHORED_DEEP_RESEARCH_SCRIPT)

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
                "args": {
                    "command": "run_script",
                    "script": script,
                    "args": {
                        "question": QUESTION,
                        "angles": ANGLES,
                        "skeptics_per_claim": SKEPTICS_PER_CLAIM,
                        "refutations_to_kill": REFUTATIONS_TO_KILL,
                    },
                },
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
    # Guard the authored source compiles through the gate, so a regression in the
    # script is caught offline before any host turn runs.
    compile_workflow_source(AUTHORED_DEEP_RESEARCH_SCRIPT)

    host_model = real_model()
    roster = (
        Roster()
        .register(
            "researcher",
            _build_leaf("researcher", web_search=True),
            description="Researches one angle",
        )
        .register("extractor", builder=_build_extractor, description="Extracts a falsifiable claim")
        .register("skeptic", builder=_build_skeptic, description="Adversarially verifies a claim")
        .register("writer", _build_leaf("writer"), description="Synthesizes the final report")
    )
    # No registered workflows: the host must author its own script via run_script.
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=WorkflowRegistry(), manager=manager)

    host_kwargs: dict[str, Any] = {"middleware": [middleware, *demo_cache_middleware()]}
    if host_model is not None:
        host_kwargs["model"] = host_model
        host_kwargs["system_prompt"] = HOST_SYSTEM_PROMPT
        host_kwargs["skills"] = [str(skills_path())]
        host_kwargs["backend"] = FilesystemBackend(root_dir=str(skills_path()), virtual_mode=False)
    else:
        host_kwargs["model"] = ScriptedHost()
    host = create_deep_agent(**host_kwargs)
    config: dict[str, Any] = {"configurable": {"thread_id": "demo-authored"}}

    print(f"question: {QUESTION}")
    print(f"mode: {'REAL (OpenRouter)' if host_model is not None else 'offline (fake)'}")

    # Turn 1: the host authors a deep-research script and submits it via run_script.
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
        print("[turn 1] host did not launch a script. reply:", state1["messages"][-1].text)
        return
    run_id = runs[-1]["run_id"]
    _RUN_ID_BOX["run_id"] = run_id
    print(f"[turn 1] launched run_id={run_id} (workflow label: {runs[-1].get('workflow')!r})")
    print(f"[turn 1] host reply: {state1['messages'][-1].text}")

    # Let the background run settle.
    await manager.wait(run_id, thread_id="demo-authored")
    assert manager.poll(run_id, thread_id="demo-authored") == BgStatus.DONE
    print("[background] authored deep research finished")

    # Turn 2: notification is injected; the host fetches the report and presents it.
    state2 = await host.ainvoke(
        {"messages": [{"role": "user", "content": "Is the research done? Give me the report."}]},
        config=config,
    )
    print(f"[turn 2] final answer:\n{state2['messages'][-1].text}")

    assert runs, "host must have launched the authored script"
    print("OK: host authored a deep-research script and ran it via run_script (gate happy path).")


if __name__ == "__main__":
    asyncio.run(main())

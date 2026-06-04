"""Preset roster and named workflows for the demo host.

Two real dynamic workflows ship here, ported from the engine's runnable examples:

* :func:`deep_research` — search (parallel fan-out, one researcher per angle) ->
  extract (no-barrier pipeline, one falsifiable claim per finding) -> verify
  (parallel adversarial skeptics per claim) -> synthesize. Mirrors Claude Code's
  built-in deep-research dynamic workflow.
* :func:`capstone` — research (parallel) -> refine (pipeline) -> adversarial verify
  (parallel majority vote) -> synthesize, the full primitive stack in one run.

The roster these drive is *real*: with a provider key present (``resolve_leaf_model``)
each leaf is a ``create_deep_agent`` (the skeptic a read-only judge); with no key the
roster swaps in deterministic fake leaves so an offline run exercises the full
control-flow inversion with no credentials and reproducible output.

:func:`hello_workflow` is the minimal spike workflow kept for the Gen-UI round-trip
smoke path; it makes no leaf ``agent()`` calls.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from _models import resolve_leaf_model, resolve_openrouter_key
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    Ctx,
    Roster,
    WorkflowRegistry,
    dedup,
    read_only_leaf,
    survives,
)

# Fixed research angles keep the agent() call sequence deterministic across replays.
ANGLES = [
    "core established findings",
    "supporting evidence and data",
    "contrarian views and limitations",
    "practical implications and cost",
]
SKEPTICS_PER_CLAIM = 3
REFUTATIONS_TO_KILL = 2

# Capstone source topics + its skeptic fan-out width. Two refutations out of three
# voters kill a finding (strict-majority adversarial vote), mirroring deep_research.
TOPICS = ["alpha", "beta", "gamma", "delta"]
SKEPTICS_PER_FINDING = 3
CAPSTONE_KILL_AT = 2

DEFAULT_RESEARCH_QUESTION = (
    "What are the main trade-offs between retrieval-augmented generation and long-context LLMs?"
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


# ── deep_research prompts ─────────────────────────────────────────────────────


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


# ── deep_research orchestration ───────────────────────────────────────────────


async def deep_research(ctx: Ctx, args: dict[str, Any]) -> str:
    """search -> extract -> adversarial verify -> synthesize, deep-research style.

    The extract and verify leaves hand back **schema-validated objects** (``Claim`` /
    ``Verdict``), so the reduce between phases is plain Python over typed data rather
    than brittle string parsing of leaf prose. The search and verify phases fan out in
    parallel; extract streams through a no-barrier pipeline.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.
        args: Workflow arguments; ``question`` selects the topic (defaults to a fixed
            RAG-vs-long-context question).

    Returns:
        The synthesized research report (the writer leaf's prose).
    """
    question: str = args.get("question") or DEFAULT_RESEARCH_QUESTION

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

    extracted = [c for c in await ctx.pipeline(paired, _extract) if c is not None and c.checkable]
    claims = dedup(extracted, key=lambda c: c.text.strip().lower())
    ctx.log(
        f"extracted {len(claims)} checkable claims ({len(extracted) - len(claims)} dups merged)"
    )

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
        survived = survives(verdicts, against=lambda v: v.refuted, kill_at=REFUTATIONS_TO_KILL)
        mark = "kept" if survived else "killed"
        ctx.log(f"claim {mark}: {claim.text.strip()[:50]}")
        if survived:
            confirmed.append(claim.text)

    ctx.phase("synthesize")
    return await ctx.agent(_synthesize_prompt(question, confirmed), agent_type="writer")


# ── capstone orchestration ────────────────────────────────────────────────────


def _capstone_challenge_prompt(topic: str) -> str:
    return (
        "You are an adversarial skeptic. Decide whether the refined finding below holds "
        "up. Set `refuted` to true only if it is unsupported, incoherent, or overstated; "
        f"otherwise set it to false. Give one sentence of `reason`.\nFinding: {topic}"
    )


async def capstone(ctx: Ctx, args: dict[str, Any]) -> str:
    """research -> refine -> adversarial verify (majority survives) -> synthesize.

    The full primitive stack in one run: a parallel research barrier, a no-barrier
    pipeline refinement, a parallel skeptic vote per finding (strict majority
    survives), then a synthesis reduce over the survivors.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.
        args: Workflow arguments; ``topics`` selects the source topics (defaults to a
            fixed four-topic set).

    Returns:
        A one-line synthesis naming the surviving findings.
    """
    topics: list[str] = args.get("topics") or TOPICS

    ctx.phase("research")
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    researched = [(t, f) for t, f in zip(topics, findings, strict=True) if f is not None]
    ctx.log(f"researched {len(researched)}/{len(topics)} topics")

    ctx.phase("refine")

    async def _refine(_prev: Any, item: tuple[str, str], _index: int) -> tuple[str, str]:
        topic, _finding = item
        refined = await ctx.agent(f"Refine {topic}", agent_type="refiner")
        return (topic, refined)

    refined = [p for p in await ctx.pipeline(researched, _refine) if p is not None]

    ctx.phase("verify")
    survivors: list[str] = []
    for topic, refined_text in refined:
        verdicts = await ctx.parallel(
            [
                lambda t=topic: ctx.agent(
                    _capstone_challenge_prompt(t), agent_type="capstone_skeptic", schema=Verdict
                )
                for _ in range(SKEPTICS_PER_FINDING)
            ]
        )
        # A finding survives unless a majority of skeptics refute it: with three voters,
        # two refutations kill it (strict-majority adversarial vote over typed Verdicts).
        survived = survives(verdicts, against=lambda v: v.refuted, kill_at=CAPSTONE_KILL_AT)
        ctx.log(f"finding {'kept' if survived else 'killed'}: {topic}")
        if survived:
            survivors.append(f"{topic}:{refined_text}")

    ctx.phase("synthesize")
    return f"synthesized {len(survivors)} surviving findings: " + " | ".join(sorted(survivors))


# ── leaves (real deepagents when a key is present, deterministic fakes offline) ──


def _fake_echo_leaf(prefix: str) -> RunnableLambda[dict[str, Any], dict[str, Any]]:
    """An offline fake leaf that echoes a trimmed prompt behind a role prefix."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"{prefix}: {last.strip()[:80]}")]}

    return RunnableLambda(_leaf)


def _fake_structured_leaf(
    structured: BaseModel, *, reply: str
) -> RunnableLambda[dict[str, Any], dict[str, Any]]:
    """An offline fake leaf that attaches a fixed ``structured_response``.

    Stands in for a ``create_deep_agent`` built with a ``response_format``: appends an
    ``AIMessage`` and hands back the given validated model instance under
    ``structured_response`` so ``agent(schema=...)`` can fold it out.
    """

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def _build_text_leaf(role: str, *, api_key: str | None = None) -> Any:
    """Build a schema-less text leaf (researcher / refiner / writer).

    Args:
        role: The leaf role, used as the offline fake leaf's echo prefix.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key). Threaded into :func:`resolve_leaf_model` so a
            per-session key reaches this leaf.
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        return create_deep_agent(model=model)
    return _fake_echo_leaf(role)


def _build_extractor(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``extractor`` leaf, forwarding ``response_format`` (Claim).

    Args:
        response_format: The structured schema (``Claim``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key).
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)

    # Offline: emit an ANGLE-DISTINCT checkable claim (the prompt names the angle), so
    # each angle yields a distinct claim, dedup() is honestly a no-op, and every angle
    # survives to the verify fan-out — mirroring the real per-angle shape.
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        angle = next((a for a in ANGLES if a in prompt), "general")
        return {
            "messages": [*inp["messages"], AIMessage(content="extracted a checkable claim")],
            "structured_response": Claim(text=f"a checkable claim about {angle}", checkable=True),
        }

    return RunnableLambda(_leaf)


def _build_skeptic(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``skeptic`` leaf (a read-only judge), forwarding ``response_format``.

    A skeptic adversarially verifies a claim — it should only *judge*, never mutate
    state. The real path builds it with ``read_only_leaf`` (a deny-write permission)
    so even a hallucinated "fix" is refused at the tool boundary. Offline it never
    refutes, so every claim survives — a deterministic, readable demo.

    Args:
        response_format: The structured schema (``Verdict``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key).
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        return read_only_leaf(model, response_format=response_format)
    return _fake_structured_leaf(
        Verdict(refuted=False, reason="consistent with the cited evidence"),
        reply="reviewed the claim",
    )


def _build_capstone_skeptic(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the capstone adversarial skeptic, forwarding ``response_format`` (Verdict).

    Like ``deep_research``'s skeptic this is a read-only judge that hands back a
    schema-validated :class:`Verdict`, so the majority vote is plain Python over typed
    data on both paths. The real path is a ``read_only_leaf`` (a deny-write permission)
    so even a hallucinated "fix" is refused at the tool boundary. Offline it refutes by
    topic-name parity — an even-length topic name is unanimously refuted (and dies on
    the strict-majority kill), an odd-length one survives — so the survivor split is
    deterministic and the majority-vote narrative is reproducibly non-trivial.

    Args:
        response_format: The structured schema (``Verdict``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key).
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        return read_only_leaf(model, response_format=response_format)

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        topic = prompt.split()[-1] if prompt.split() else ""
        refuted = len(topic) % 2 == 0
        return {
            "messages": [*inp["messages"], AIMessage(content="reviewed the finding")],
            "structured_response": Verdict(
                refuted=refuted,
                reason="even-length topic name fails the parity check"
                if refuted
                else "consistent with the refined finding",
            ),
        }

    return RunnableLambda(_leaf)


# ── roster + registry ─────────────────────────────────────────────────────────


def make_roster() -> Roster:
    """Build the demo leaf roster shared by both preset workflows.

    Registers the roles ``deep_research`` and ``capstone`` need: ``researcher``,
    ``writer``, ``refiner``, ``extractor`` (structured), ``skeptic`` (a read-only
    adversarial judge over a claim), and ``capstone_skeptic`` (the same kind of
    read-only judge over a refined finding). With an OpenRouter key in force each leaf
    is a real ``create_deep_agent``; with no key the roster serves deterministic fake
    leaves so an offline run is fully reproducible. An ``echo`` leaf is also kept for
    ``hello_workflow``, which makes no ``agent()`` calls but still requires a roster.

    Per-session key capture. The leaf models are baked into their deepagents when this
    roster is built. Since the roster is built inside the host node — where the run
    config is visible — the per-run OpenRouter key is resolved ONCE here and threaded
    into every leaf builder. The schema-aware builders (``extractor`` / ``skeptic`` /
    ``capstone_skeptic``) are invoked LATER by the engine, deep inside the nested
    workflow substrate where the host run config is no longer visible, so the captured
    key is bound into them now via ``functools.partial`` rather than re-resolved at call
    time. The eager leaves (``researcher`` / ``writer`` / ``refiner``) are built here
    directly with the same captured key.

    Returns:
        A :class:`~langchain_dynamic_workflow.Roster` with every preset role.
    """
    # Resolve the per-run OpenRouter key once, here in the host node context, so every
    # leaf — eager or schema-aware — is built with the SAME in-force key. Resolving it
    # inside a builder would be too late: the schema-aware builders run inside the nested
    # engine substrate where the per-run config is not visible.
    api_key = resolve_openrouter_key()
    return (
        Roster()
        .register(
            "researcher",
            _build_text_leaf("researcher", api_key=api_key),
            description="Researches one angle.",
        )
        .register(
            "writer",
            _build_text_leaf("writer", api_key=api_key),
            description="Synthesizes the final report.",
        )
        .register(
            "refiner",
            _build_text_leaf("refiner", api_key=api_key),
            description="Refines one finding.",
        )
        .register(
            "extractor",
            builder=partial(_build_extractor, api_key=api_key),
            description="Extracts a falsifiable claim.",
        )
        .register(
            "skeptic",
            builder=partial(_build_skeptic, api_key=api_key),
            description="Adversarially verifies a claim.",
        )
        .register(
            "capstone_skeptic",
            builder=partial(_build_capstone_skeptic, api_key=api_key),
            description="Adversarially verifies a refined finding (majority vote).",
        )
        .register("echo", _fake_echo_leaf("echo"), description="Trivial echo leaf (hello demo).")
    )


def make_workflows() -> WorkflowRegistry:
    """Register the preset named workflows the host can launch by name.

    Returns:
        A :class:`~langchain_dynamic_workflow.WorkflowRegistry` with ``deep_research``
        and ``capstone``.
    """
    return (
        WorkflowRegistry().register("deep_research", deep_research).register("capstone", capstone)
    )


# ── minimal hello workflow (Gen-UI round-trip smoke path) ─────────────────────


async def hello_workflow(ctx: Ctx) -> str:
    """A minimal workflow that narrates two phases and returns a result.

    Uses ``ctx.phase`` / ``ctx.log`` (synchronous side effects) to drive the engine's
    progress sink, so an inline run streams ``phase_timeline`` events live. Makes no
    leaf ``agent()`` calls.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.

    Returns:
        A short completion string.
    """
    ctx.phase("greeting")
    ctx.log("working...")
    ctx.phase("wrap-up")
    ctx.log("done")
    return "ok"


__all__: list[str] = [
    "Claim",
    "Verdict",
    "capstone",
    "deep_research",
    "hello_workflow",
    "make_roster",
    "make_workflows",
]

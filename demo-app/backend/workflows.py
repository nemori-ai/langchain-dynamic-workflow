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

from _models import cache_middleware, resolve_leaf_model, resolve_openrouter_key
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.config import get_config
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


class EditedFile(BaseModel):
    """One source file the ``code_fixer`` leaf wrote or edited during an attempt.

    The fix loop threads these forward through a SCRIPT VARIABLE: a leaf writes the
    files it was handed into its fresh sandbox, edits them, and returns the edited set so
    the script can seed the NEXT attempt with the accumulated source — cross-attempt state
    lives in the script, not a persistent workspace.

    Attributes:
        path: The file's path relative to the leaf's workspace (e.g. ``src/sum.ts``).
        content: The file's full contents after this attempt's edits.
    """

    path: str
    content: str


class FixResult(BaseModel):
    """One fix-loop attempt's outcome, carrying the REAL build/test exit code.

    The ``code_fixer`` leaf writes the source files it was handed into its fresh sandbox,
    runs the build/test command, transcribes the REAL exit code it observed from its
    execute tool's output, edits the source, and returns this object. The orchestration
    script branches on :attr:`exit_code` (``== 0`` is green) — never on leaf prose — and
    threads :attr:`edited_files` into the next attempt when red.

    Trust boundary: :attr:`exit_code` is the exit code the leaf *transcribed* from its
    real tool output, so it is only as honest as the leaf's transcription; the engine's
    independent ``on_command`` end events are the ground-truth display of what really ran.
    Surfacing the real exit code directly to the orchestration (full determinism) is out
    of scope here.

    Attributes:
        exit_code: The REAL build/test command exit code the leaf observed (``0`` is
            green; non-zero is red). The orchestration gate.
        edited_files: The source files after this attempt's edits, threaded forward into
            the next attempt so the script accumulates the fix across attempts.
        failure_tail: A short tail of the failing build/test output, carried forward to
            the next attempt's prompt when red; empty when green.
        summary: A one-line human-readable summary of what the attempt did.
    """

    exit_code: int
    edited_files: list[EditedFile]
    failure_tail: str
    summary: str


# ── deep_research prompts ─────────────────────────────────────────────────────


def _search_prompt(question: str, angle: str) -> str:
    return (
        "You are a researcher with a web_search tool. Investigate this question from one "
        "specific angle: run web_search to find current, authoritative sources, then "
        "report concrete findings grounded in what you found.\n"
        f"Question: {question}\nAngle: {angle}\n"
        "Write 2-3 substantive sentences citing the specific facts and sources you found."
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
        f"You are skeptic #{voter + 1} fact-checking a claim. Use your web_search tool to "
        "verify it against current, authoritative sources. Set `refuted` to true only if "
        "the sources show the claim is factually wrong, misleading, or clearly overstated; "
        "otherwise set it to false. Give one sentence of `reason` citing what you found.\n"
        f"Question: {question}\nClaim: {claim}"
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


# ── fix_loop orchestration (real in-loop executable verification, M5) ──────────

# The default retry budget for the fix loop. Bounded so an all-red fixer cannot loop
# forever — the loop returns an honest "still red" after this many attempts.
DEFAULT_FIX_ATTEMPTS = 3

# The hard upper bound on the retry budget. A caller-supplied max_attempts is clamped into
# [1, _MAX_FIX_ATTEMPTS] so a hostile or fat-fingered argument can neither disable the loop
# (zero/negative) nor let it run unbounded (a resource-exhaustion guard).
_MAX_FIX_ATTEMPTS = 8

# A tiny TypeScript module with a REAL bug plus a bun test that fails against it: sum()
# ignores its inputs and returns 0, so `sum(-1, -1)` is 0 rather than -2 and the negative
# case fails. The SCRIPT seeds the first attempt with these files; the code_fixer leaf
# writes the files it is handed into its fresh sandbox, runs `bun test` for a real exit
# code (red), edits src/sum.ts to actually add its inputs, and returns the edited files so
# the script threads them into the next attempt — cross-attempt state lives in a script
# variable, not a persistent workspace.
SEED_SUM_MODULE = "export function sum(a: number, b: number): number {\n  return 0;\n}\n"
SEED_SUM_TEST = (
    'import { expect, test } from "bun:test";\n'
    'import { sum } from "./sum";\n\n'
    'test("adds two positives", () => {\n'
    "  expect(sum(2, 3)).toBe(5);\n"
    "});\n\n"
    'test("adds negatives", () => {\n'
    "  expect(sum(-1, -1)).toBe(-2);\n"
    "});\n"
)
FIX_TEST_COMMAND = "bun test"


def _clamp_max_attempts(raw: Any) -> int:
    """Parse and clamp a caller-supplied ``max_attempts`` into a safe bounded range.

    The value reaches here from untrusted workflow args, so it may be missing, non-numeric,
    zero, negative, or absurdly large. A value that parses as an int is clamped into
    ``[1, _MAX_FIX_ATTEMPTS]``; a value that does not parse falls back to the default — so
    the loop can neither be disabled nor run unbounded, and a bad argument never crashes.

    Args:
        raw: The caller-supplied ``max_attempts`` argument (any type).

    Returns:
        A bounded attempt count in ``[1, _MAX_FIX_ATTEMPTS]``.
    """
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = DEFAULT_FIX_ATTEMPTS
    return max(1, min(parsed, _MAX_FIX_ATTEMPTS))


def _render_files(files: list[EditedFile]) -> str:
    """Render the current source files for an attempt's prompt (path + contents block)."""
    return "\n".join(f"{f.path}:\n{f.content}" for f in files)


def _fix_prompt(attempt: int, files: list[EditedFile], last_failure: str) -> str:
    """Build the code_fixer prompt for one attempt, threading the CURRENT source files.

    The current source files live in a SCRIPT VARIABLE and are handed to the leaf EVERY
    attempt (not seeded only on the first): attempt 1 carries the buggy seed; a retry
    carries the prior attempt's edited files plus the real failure tail, so the leaf fixes
    the source as it actually stands. The prompt drives the leaf to write the handed files
    into its fresh sandbox, run the test command, transcribe the REAL exit code, edit the
    source, and return both the edited files and the exit code it observed.

    Args:
        attempt: The 1-based attempt number.
        files: The current source files (seed on attempt 1, threaded edits on a retry).
        last_failure: The previous attempt's real failure tail, or empty on attempt 1.

    Returns:
        The leaf prompt for this attempt.
    """
    retry = (
        ""
        if not last_failure
        else (
            "The previous attempt's test run still failed. Here is the real failure "
            f"tail:\n{last_failure}\n\n"
        )
    )
    return (
        "You are fixing a small TypeScript module until its tests genuinely pass.\n\n"
        + retry
        + "Write these files into your workspace exactly as given, then run the test "
        "command and read its REAL exit code:\n\n"
        + _render_files(files)
        + f"\n\nTest command: {FIX_TEST_COMMAND}\n\n"
        "Use your execute tool to run the test command and read the real exit code "
        "(zero means the tests passed). Edit src/sum.ts so the tests pass — do not edit "
        "the test file. Set `exit_code` to the REAL exit code the test command returned "
        "(0 means green). Return the full contents of every file you wrote or edited in "
        "`edited_files` (so the source carries forward). When the exit code is non-zero, "
        "put the real failure output tail in `failure_tail`. Give a one-line `summary`."
    )


async def fix_loop(ctx: Ctx, args: dict[str, Any]) -> str:
    """Edit code, test it for real, and retry until the REAL exit code goes green.

    A SCRIPT-OWNED loop, the dynamic-workflow thesis made concrete: cross-attempt state —
    the current source files — lives in a SCRIPT VARIABLE seeded from the buggy ``SEED``
    fixture, not a persistent workspace. Each attempt hands the leaf the current files,
    the leaf writes them into its fresh sandbox, runs the real test command, transcribes
    the REAL exit code, edits the source, and returns a :class:`FixResult`. The script
    branches on the REAL ``exit_code`` (``== 0`` is green) — NEVER on a model boolean —
    and threads the leaf's ``edited_files`` into the next attempt's prompt when red, so
    the fix accumulates across attempts. The script formats NO UI; the terminal cards
    appear automatically through the engine's ``on_command`` sink. The loop stops on the
    first green attempt or when the bounded retry budget is spent (an honest "still red"
    return, never a false pass).

    Trust boundary: the gate is the exit code the leaf TRANSCRIBED from its real tool
    output — only as honest as that transcription. The engine's ``on_command`` end events
    are the independent ground-truth display. Surfacing the real exit code directly to the
    orchestration (full determinism) is out of scope here.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.
        args: Workflow arguments; ``max_attempts`` bounds the retry budget (default 3,
            clamped into ``[1, _MAX_FIX_ATTEMPTS]``; a non-numeric value falls back).

    Returns:
        The green attempt's summary, or an honest "still red" message with the last
        failure tail when the budget is exhausted.
    """
    max_attempts = _clamp_max_attempts(args.get("max_attempts", DEFAULT_FIX_ATTEMPTS))
    # The current source files — the cross-attempt state — live in THIS script variable,
    # seeded from the buggy fixture and rethreaded with each attempt's edits below.
    files: list[EditedFile] = [
        EditedFile(path="src/sum.ts", content=SEED_SUM_MODULE),
        EditedFile(path="src/sum.test.ts", content=SEED_SUM_TEST),
    ]
    last_failure = ""
    for attempt in range(1, max_attempts + 1):
        ctx.phase(f"attempt {attempt}")
        # A shared-isolation execution leaf: it writes the handed files into its fresh
        # sandbox, runs the real build/test via execute (exit code is REAL under M5), and
        # hands back the edited files + the exit code it transcribed. Worktree isolation is
        # M6; here the execution runs in the per-leaf LocalSubprocessSandbox temp dir.
        result = await ctx.agent(
            _fix_prompt(attempt, files, last_failure),
            agent_type="code_fixer",
            isolation="shared",
            schema=FixResult,
        )
        # Gate on the REAL exit code, NOT a model boolean: a leaf that lies in its prose
        # cannot false-green because the script reads exit_code, the ground-truth signal.
        if result.exit_code == 0:
            ctx.log(f"green on attempt {attempt}")
            return f"green on attempt {attempt}: {result.summary}"
        # Red: thread the leaf's edited files forward (the script variable accumulates the
        # fix) and carry the failure tail into the next attempt's prompt.
        if result.edited_files:
            files = result.edited_files
        last_failure = result.failure_tail
        ctx.log(
            f"attempt {attempt} red (exit {result.exit_code}): {result.failure_tail.strip()[:50]}"
        )
    return f"still red after {max_attempts} attempts; last failure:\n{last_failure}"


# ── in-run HITL sign-off (M4): pause mid-run for a human decision, then proceed ──

DEFAULT_SIGNOFF_TOPIC = "the staging deployment plan"
"""The thing a sign-off run asks a human to approve when ``args`` names none."""


def _assess_prompt(topic: str) -> str:
    """Prompt the assessment leaf to produce a brief, sign-off-ready risk summary."""
    return (
        f"Assess {topic} and produce a concise risk summary a human reviewer can sign off "
        "on: name the two or three highest-risk steps and give a clear recommendation. "
        "Keep it short."
    )


def _proceed_prompt(topic: str, assessment: str, note: str) -> str:
    """Prompt the report leaf to record the approved go-ahead (folding in any note)."""
    extra = f" Incorporate the reviewer's note: {note}." if note else ""
    return (
        f"The reviewer approved proceeding with {topic}. Write the short go-ahead summary "
        f"that records the decision alongside the assessment.{extra}\n\nAssessment:\n{assessment}"
    )


async def sign_off(ctx: Ctx, args: dict[str, Any]) -> str:
    """Assess a plan, PAUSE for a human sign-off, then proceed or hold on the decision.

    The M4 in-run-HITL thesis made concrete: a reasoning leaf produces a risk
    assessment, then the SCRIPT pauses at ``ctx.checkpoint`` for a person — the run
    parks (``AWAITING_SIGNOFF``), the host surfaces the ask, and an approve feeds the
    decision back as the call's return value. The script branches on the REAL human
    decision (``approved``), never a model guess: an approval proceeds to publish the
    go-ahead, a rejection holds and records the reviewer's note. The pre-gate
    assessment replays from the journal at zero cost on approve; cross-gate state rides
    this script's own variables, not a workspace (worktree persistence is M6). The
    script formats no UI — the sign-off card appears through the host's park handling.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.
        args: Workflow arguments; ``topic`` names what is being signed off (default
            ``"the staging deployment plan"``).

    Returns:
        A short message recording whether the plan was published (approved) or held
        (rejected), with the reviewer's note when one was supplied.
    """
    topic = str(args.get("topic") or DEFAULT_SIGNOFF_TOPIC)
    ctx.phase("assess")
    assessment = await ctx.agent(_assess_prompt(topic), agent_type="researcher")
    ctx.phase("sign-off")
    # The script PAUSES here for a human. The run parks until the host approves with a
    # decision; whatever the host supplies becomes this call's return value.
    decision = await ctx.checkpoint(
        {"ask": f"Approve proceeding with {topic}?", "summary": assessment},
        tag="proceed",
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    note = str(decision.get("note", "")) if isinstance(decision, dict) else ""
    if not approved:
        ctx.log("sign-off declined; holding the plan")
        held = f"held: reviewer did not approve {topic}"
        return f"{held} ({note})" if note else held
    ctx.phase("proceed")
    report = await ctx.agent(_proceed_prompt(topic, assessment, note), agent_type="writer")
    return f"proceeded with {topic}: {report}"


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


def _build_text_leaf(role: str, *, api_key: str | None = None, web_search: bool = False) -> Any:
    """Build a schema-less text leaf (researcher / refiner / writer).

    Args:
        role: The leaf role, used as the offline fake leaf's echo prefix.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key). Threaded into :func:`resolve_leaf_model` so a
            per-session key reaches this leaf.
        web_search: When ``True``, the leaf model carries Anthropic's native web search
            tool so the leaf grounds its work in live web sources. Only takes effect on
            the online path; an offline fake leaf never searches, so determinism is
            preserved.
    """
    model = resolve_leaf_model(api_key=api_key, web_search=web_search)
    if model is not None:
        return create_deep_agent(model=model, middleware=cache_middleware())
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
        return create_deep_agent(
            model=model, response_format=response_format, middleware=cache_middleware()
        )

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


def _build_skeptic(
    *, response_format: Any = None, api_key: str | None = None, web_search: bool = False
) -> Any:
    """Build the ``skeptic`` leaf (a read-only judge), forwarding ``response_format``.

    A skeptic adversarially verifies a claim — it should only *judge*, never mutate
    state. The real path builds it with ``read_only_leaf`` (a deny-write permission)
    so even a hallucinated "fix" is refused at the tool boundary. Offline it never
    refutes, so every claim survives — a deterministic, readable demo.

    Args:
        response_format: The structured schema (``Verdict``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key).
        web_search: When ``True``, the leaf model carries Anthropic's native web search
            tool so the judge fact-checks the claim against live sources before ruling.
            The search runs server-side and is read-only by nature, so it composes with
            the deny-write read-only leaf. Only takes effect on the online path.
    """
    model = resolve_leaf_model(api_key=api_key, web_search=web_search)
    if model is not None:
        return read_only_leaf(model, response_format=response_format, middleware=cache_middleware())
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
        return read_only_leaf(model, response_format=response_format, middleware=cache_middleware())

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


def _build_code_fixer(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``code_fixer`` execution leaf, forwarding ``response_format`` (FixResult).

    The real path is a ``create_deep_agent`` wired with a ``backend`` factory that reads
    the engine-leased per-leaf sandbox from the leaf config at call time
    (``configurable['sandbox_backend']``), so the leaf's ``execute`` tool runs real shell
    commands (``bun test``) in an isolated temp directory for a TRUE exit code. The leaf
    is driven by :func:`_fix_prompt` to write the seeded module, run the test, fix the
    source, and re-run until green, returning a :class:`FixResult`.

    The offline path is a deterministic fake that exercises the SCRIPT-OWNED threading
    loop with no model and no bun: it inspects the source the SCRIPT threaded into the
    prompt, edits ``src/sum.ts`` into the correct ``return a + b`` form, and returns that
    edited file. It transcribes the REAL-shaped ``exit_code`` for the source AS HANDED IN
    — non-zero while the threaded source still carries the seed bug (``return 0``), zero
    once the threaded source already carries the fix. So the loop goes red(attempt 1, seed)
    -> green(attempt 2, threaded edit) genuinely off the exit code and the script variable,
    not a prompt sniff — the real exit-code path is covered by the gated real E2E.

    Args:
        response_format: The structured schema (``FixResult``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``
            to fall back to the env key).

    Returns:
        A real execution deepagent when a key is in force, else the deterministic fake.
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        # The engine leases a per-leaf real sandbox (needs_execution=True) and threads it
        # into the leaf config under 'sandbox_backend'. A backend factory reads it at call
        # time (the backend is per-leaf, resolved at runtime, not at construction), so the
        # leaf's execute tool runs against the real LocalSubprocessSandbox.
        def _backend_factory(_runtime: Any) -> Any:
            # deepagents hands the factory a Runtime with no ``.config``; read the
            # engine-leased per-leaf sandbox off the run-config contextvar via langgraph's
            # get_config instead (the canonical pattern in the engine's real-exec test).
            configurable = (get_config() or {}).get("configurable") or {}
            return configurable["sandbox_backend"]

        return create_deep_agent(
            model=model,
            backend=_backend_factory,
            response_format=response_format,
            middleware=cache_middleware(),
        )

    schema: Any = response_format.schema if response_format is not None else FixResult
    # The correct module: sum actually adds its inputs, so bun test would exit 0.
    fixed_module = "export function sum(a: number, b: number): number {\n  return a + b;\n}\n"

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        # Read the source the SCRIPT threaded into this prompt: if it still carries the seed
        # bug (`return 0`), the real `bun test` would exit non-zero; once the threaded source
        # already carries the fix (`return a + b`), it would exit zero. The fake transcribes
        # the exit code for the source AS HANDED IN, then edits sum.ts into the fixed form and
        # threads it forward — so the loop's red->green emerges from the script variable.
        source_already_fixed = "return a + b;" in prompt
        edited = [EditedFile(path="src/sum.ts", content=fixed_module)]
        if source_already_fixed:
            fix = schema.model_validate(
                {
                    "exit_code": 0,
                    "edited_files": [f.model_dump() for f in edited],
                    "failure_tail": "",
                    "summary": "threaded source adds its inputs; bun test exited 0",
                }
            )
            reply = "tests pass"
        else:
            fix = schema.model_validate(
                {
                    "exit_code": 1,
                    "edited_files": [f.model_dump() for f in edited],
                    "failure_tail": (
                        "FAIL src/sum.test.ts > adds negatives\n"
                        "  expect(sum(-1, -1)).toBe(-2)  // got 0\n"
                        "  1 pass · 1 fail"
                    ),
                    "summary": "edited src/sum.ts to add its inputs; the handed seed was still red",
                }
            )
            reply = "tests still failing"
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": fix,
        }

    return RunnableLambda(_leaf)


# ── roster + registry ─────────────────────────────────────────────────────────


def _register_reasoning_roles(roster: Roster, *, api_key: str | None) -> Roster:
    """Register every PURE-REASONING leaf role onto ``roster`` and return it.

    These roles (``researcher`` / ``writer`` / ``refiner`` / ``extractor`` / ``skeptic``
    / ``capstone_skeptic`` / ``echo``) reason in-context only — none is ``needs_execution``,
    so none can reach the real execution sandbox. Shared by :func:`make_roster` (which
    then adds the ``code_fixer`` execution leaf) and :func:`make_reasoning_roster` (which
    does not), so the reasoning roster is provably a strict subset of the full one.

    Args:
        roster: The roster to register the reasoning roles onto (mutated and returned).
        api_key: The per-run OpenRouter key captured in the host node context, threaded
            into every leaf builder (or ``None`` to fall back to the env key).

    Returns:
        The same ``roster``, with every reasoning role registered, for fluent chaining.
    """
    return (
        roster.register(
            "researcher",
            _build_text_leaf("researcher", api_key=api_key, web_search=True),
            description="Researches one angle (web search).",
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
            builder=partial(_build_skeptic, api_key=api_key, web_search=True),
            description="Adversarially verifies a claim (web search fact-check).",
        )
        .register(
            "capstone_skeptic",
            builder=partial(_build_capstone_skeptic, api_key=api_key),
            description="Adversarially verifies a refined finding (majority vote).",
        )
        .register("echo", _fake_echo_leaf("echo"), description="Trivial echo leaf (hello demo).")
    )


def make_roster() -> Roster:
    """Build the demo leaf roster shared by both preset workflows.

    Registers the reasoning roles (``researcher``, ``writer``, ``refiner``, ``extractor``,
    ``skeptic``, ``capstone_skeptic``, ``echo``) plus ``code_fixer`` (an EXECUTION leaf,
    ``needs_execution=True``, that edits code and runs the real build/test until green —
    the ``fix_loop`` preset's leaf). With an OpenRouter key in force each leaf is a real
    ``create_deep_agent``; with no key the roster serves deterministic fake leaves so an
    offline run is fully reproducible.

    Trust boundary: this FULL roster — which can reach the real ``LocalSubprocessSandbox``
    via ``code_fixer`` — is for the TRUSTED preset path only (``run_workflow_live``, whose
    ``agent()`` calls are fixed in code). The meta layer, which runs an LLM-authored script
    whose ``agent_type`` choices the AST gate does not constrain, must instead use
    :func:`make_reasoning_roster` so an authored script cannot reach real execution.

    Per-session key capture. The leaf models are baked into their deepagents when this
    roster is built. Since the roster is built inside the host node — where the run
    config is visible — the per-run OpenRouter key is resolved ONCE here and threaded
    into every leaf builder. The schema-aware builders (``extractor`` / ``skeptic`` /
    ``capstone_skeptic`` / ``code_fixer``) are invoked LATER by the engine, deep inside the
    nested workflow substrate where the host run config is no longer visible, so the
    captured key is bound into them now via ``functools.partial`` rather than re-resolved
    at call time. The eager leaves (``researcher`` / ``writer`` / ``refiner``) are built
    here directly with the same captured key.

    Returns:
        A :class:`~langchain_dynamic_workflow.Roster` with every preset role, including the
        execution leaf.
    """
    # Resolve the per-run OpenRouter key once, here in the host node context, so every
    # leaf — eager or schema-aware — is built with the SAME in-force key. Resolving it
    # inside a builder would be too late: the schema-aware builders run inside the nested
    # engine substrate where the per-run config is not visible.
    api_key = resolve_openrouter_key()
    roster = _register_reasoning_roles(Roster(), api_key=api_key)
    return roster.register(
        "code_fixer",
        builder=partial(_build_code_fixer, api_key=api_key),
        description="Edits code and runs the real build/test until green (execution leaf).",
        needs_execution=True,
    )


def make_reasoning_roster() -> Roster:
    """Build a REASONING-ONLY roster for the untrusted meta (authored-source) path.

    The AST gate validates an LLM-authored script's *source* but does NOT constrain which
    ``agent_type`` it calls — so an authored script that crossed the gate could otherwise
    call ``ctx.agent(..., agent_type="code_fixer")`` and reach the real shell. This roster
    registers every pure-reasoning role but OMITS ``code_fixer`` (and any other
    ``needs_execution`` leaf), so an authored script has no execution leaf to reach no
    matter what it asks for — a defense-in-depth boundary on top of the gate. The trusted
    preset path uses the full :func:`make_roster` instead.

    Returns:
        A :class:`~langchain_dynamic_workflow.Roster` with only pure-reasoning roles (no
        ``needs_execution`` leaf).
    """
    api_key = resolve_openrouter_key()
    return _register_reasoning_roles(Roster(), api_key=api_key)


def make_workflows() -> WorkflowRegistry:
    """Register the preset named workflows the host can launch by name.

    Returns:
        A :class:`~langchain_dynamic_workflow.WorkflowRegistry` with ``deep_research``,
        ``capstone``, ``fix_loop``, and ``sign_off``.
    """
    return (
        WorkflowRegistry()
        .register("deep_research", deep_research)
        .register("capstone", capstone)
        .register("fix_loop", fix_loop)
        .register("sign_off", sign_off)
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
    "EditedFile",
    "FixResult",
    "Verdict",
    "capstone",
    "deep_research",
    "fix_loop",
    "hello_workflow",
    "make_reasoning_roster",
    "make_roster",
    "make_workflows",
]

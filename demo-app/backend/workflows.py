"""Preset roster and named workflows for the demo host.

Two real dynamic workflows ship here, ported from the engine's runnable examples:

* :func:`deep_research` — search (parallel fan-out, one researcher per angle) ->
  extract (no-barrier pipeline, one falsifiable claim per finding) -> verify
  (parallel adversarial skeptics per claim) -> synthesize. Mirrors Claude Code's
  built-in deep-research dynamic workflow.
* :func:`capstone` — research (parallel) -> refine (pipeline) -> adversarial verify
  (parallel majority vote) -> synthesize, the full primitive stack in one run.
* :func:`refactor_swarm` — fix (parallel real-git worktree fixers, one per target
  file, each on its own ``leaf/<id>`` branch) -> verify (a two-vote read-only judge
  per patch) -> integrate (a script-owned scratch-repo ``git merge`` fold with a real
  conflict -> resolver -> fold loop). It returns the integrated tree plus a PR intent;
  the HOST opens the pull request after ``run_workflow`` returns (host finalization),
  never from inside the orchestration script.

The roster these drive is *real*: with a provider key present (``resolve_leaf_model``)
each leaf is a ``create_deep_agent`` (the skeptic a read-only judge); with no key the
roster swaps in deterministic fake leaves so an offline run exercises the full
control-flow inversion with no credentials and reproducible output.

:func:`hello_workflow` is the minimal spike workflow kept for the Gen-UI round-trip
smoke path; it makes no leaf ``agent()`` calls.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from functools import partial
from pathlib import Path
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


# ── refactor_swarm orchestration (real-git fix swarm + merge conflict loop, M6) ──

# The buggy fixture the swarm refactors: a tiny module with a bug on ONE line of calc.py,
# plus a small helper. TWO fixers target that SAME buggy line of calc.py with DIFFERENT
# corrections, each in its OWN isolated worktree — so neither sees the other's edit, and
# folding their two patches hits a REAL three-way git merge conflict (the SAME line changed
# two ways), the headline path the integrate phase resolves. A third fixer edits a DIFFERENT
# file (helper.py) so the swarm also shows a clean, non-conflicting fold. This mirrors the
# engine integration test's conflict mechanic (two fixers, one overlapping line).
SEED_CALC = "def combine(a, b):\n    return a - b\n"
SEED_HELPER = "VALUE = 0\n"

# The base tree the integration fold starts from (protocol-absolute paths, matching the
# worktree provider's collect() keys), and the target files each parallel fixer owns.
REFACTOR_BASE_TREE: dict[str, str] = {"/calc.py": SEED_CALC, "/helper.py": SEED_HELPER}


class RefactorTarget(BaseModel):
    """One file the swarm targets, with the bug to fix and the corrected line.

    Drives both the real fixer prompt (online) and the deterministic fake fixer
    (offline): the fake replaces ``old`` with ``new`` on the real worktree disk so its
    authoritative ``git diff`` is exactly the edit, with no model involved.

    Attributes:
        path: The protocol-absolute file path the fixer edits (e.g. ``/calc.py``).
        summary: A one-line description of the bug this target fixes.
        old: The exact buggy line to replace.
        new: The corrected line.
    """

    path: str
    summary: str
    old: str
    new: str


# The three fix targets. Targets 0 and 1 both rewrite the SAME line of /calc.py with
# DIFFERENT corrections (in their OWN isolated worktrees), so folding their two patches
# hits a REAL three-way git merge conflict on that line — the headline conflict path.
# Target 2 edits /helper.py cleanly (a non-conflicting fold). Each `summary` is unique so
# the offline fake fixer can tell which target a prompt names.
REFACTOR_TARGETS: list[RefactorTarget] = [
    RefactorTarget(
        path="/calc.py",
        summary="combine() should add its inputs",
        old="    return a - b\n",
        new="    return a + b\n",
    ),
    RefactorTarget(
        path="/calc.py",
        summary="combine() should sum the magnitudes",
        old="    return a - b\n",
        new="    return abs(a) + abs(b)\n",
    ),
    RefactorTarget(
        path="/helper.py",
        summary="VALUE constant is wrong",
        old="VALUE = 0\n",
        new="VALUE = 42\n",
    ),
]

# The PR intent the workflow returns for the host to materialize after the run.
REFACTOR_PR_BRANCH = "ldw/refactor-swarm"
REFACTOR_PR_TITLE = "Refactor swarm: fix calc + helper bugs"


class GitPatch(BaseModel):
    """A single git-worktree fixer's authoritative changeset (engine-folded).

    The engine folds the real ``git diff`` of the fixer's worktree into :attr:`files`
    as the AUTHORITATIVE changeset — a model self-report can never override the on-disk
    truth — so a git-worktree execution leaf's schema MUST declare a ``files: dict[str,
    str]`` field or the fold fails loud. The fixer self-reports a :attr:`summary`; the
    changeset itself is the real diff, not the model's word.

    Attributes:
        summary: The fixer's one-line description of what it changed.
        files: The authoritative changeset (``path -> new content``), folded in by the
            engine from the real ``git diff`` of the leaf's worktree.
    """

    summary: str
    files: dict[str, str]


class MergeResult(BaseModel):
    """The outcome of one scratch-repo three-way ``git merge`` (mirrors the engine test).

    Attributes:
        clean: ``True`` when the merge applied with no conflict.
        files: The merged tree on a clean merge; on a conflict, the working tree
            carrying the real ``<<<<<<<`` markers.
        conflicts: ``path -> conflicted content`` for each file git could not
            auto-merge (empty on a clean merge).
    """

    clean: bool
    files: dict[str, str]
    conflicts: dict[str, str]


class Resolution(BaseModel):
    """A conflict resolver leaf's flattened, marker-free resolution of conflicted files.

    Attributes:
        files: ``path -> resolved content`` for each conflicted file, with every git
            conflict hunk flattened so no ``<<<<<<<`` / ``>>>>>>>`` markers remain.
    """

    files: dict[str, str]


class PrIntent(BaseModel):
    """The pull-request intent a workflow returns for the HOST to materialize (R1).

    Opening a PR is a networked side effect that must NOT live inside the deterministic
    replay (a journaled leaf short-circuits on resume), so the workflow returns this
    pure intent and the host opens the PR once, after ``run_workflow`` returns.

    Attributes:
        branch: The source branch the PR should be opened from.
        title: The PR title.
        body: The PR description body.
    """

    branch: str
    title: str
    body: str


class RefactorResult(BaseModel):
    """The pure, journaled result the ``refactor_swarm`` workflow returns.

    Attributes:
        integrated_tree: The integrated source tree after every approved patch folded in
            (``path -> content``), conflicts resolved — no merge markers remain.
        conflict_resolved: ``True`` when the integrate fold actually hit (and resolved) a
            real ``git merge`` conflict, ``False`` when every fold was clean.
        approved: The summaries of the patches the two-vote judge approved (one per
            integrated patch).
        rejected: The summaries of the patches the judge rejected (folded into nothing).
        pr: The PR intent for the host to materialize after the run (host finalization).
    """

    integrated_tree: dict[str, str]
    conflict_resolved: bool
    approved: list[str]
    rejected: list[str]
    pr: PrIntent


def _scratch_merge(
    base: dict[str, str], ours: dict[str, str], theirs: dict[str, str]
) -> MergeResult:
    """Run a real three-way ``git merge`` in a throwaway repo (pure, resume-safe).

    Mirrors the engine integration test's ``scratch_merge``: builds a disposable git
    repo from the inputs alone — commit ``base``, branch ``ours`` off the base SHA,
    branch ``theirs`` off the base SHA — then runs a real ``git merge ours <- theirs``.
    A clean merge returns the merged tree; a real conflict returns the working tree
    carrying git's real ``<<<<<<<`` markers plus a per-file conflict map. The function is
    a pure rebuild from its inputs (no persisted state across calls), so a merge leaf
    that calls it is resume-safe: every replay reconstructs the identical scratch repo and
    the identical result.

    Args:
        base: The common ancestor tree (``path -> content``).
        ours: The integrated-so-far tree to merge into.
        theirs: The incoming patch tree to merge in.

    Returns:
        A :class:`MergeResult` capturing whether the merge was clean and the resulting
        (merged or conflicted) tree.
    """
    with tempfile.TemporaryDirectory(prefix="ldw-demo-scratch-merge-") as repo:
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")

        def _write_tree(tree: dict[str, str], message: str) -> None:
            # Replace the whole tree deterministically so a removed path is honored.
            for existing in Path(repo).iterdir():
                if existing.name == ".git":
                    continue
                existing.unlink()
            for rel, content in tree.items():
                (Path(repo) / rel.lstrip("/")).write_text(content)
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", message, "--allow-empty")

        # Branch ours/theirs EXPLICITLY off the base commit SHA so the merge is a genuine
        # three-way merge independent of the init default branch name (main vs master).
        _write_tree(base, "base")
        base_sha = _git_out(repo, "rev-parse", "HEAD").strip()
        _git(repo, "checkout", "-qb", "ours", base_sha)
        _write_tree(ours, "ours")
        _git(repo, "checkout", "-qb", "theirs", base_sha)
        _write_tree(theirs, "theirs")
        _git(repo, "checkout", "-q", "ours")
        merge = subprocess.run(
            ["git", "-C", repo, "merge", "--no-edit", "theirs"],
            capture_output=True,
            text=True,
        )
        files: dict[str, str] = {}
        for path in sorted(Path(repo).rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            rel = "/" + str(path.relative_to(repo))
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        if merge.returncode == 0:
            return MergeResult(clean=True, files=files, conflicts={})
        # Real conflict: enumerate the unmerged paths git reports.
        unmerged = _git_out(repo, "diff", "--name-only", "--diff-filter=U")
        conflicts = {"/" + rel: files["/" + rel] for rel in unmerged.splitlines() if rel.strip()}
        return MergeResult(clean=False, files=files, conflicts=conflicts)


def _git(cwd: str, *args: str) -> None:
    """Run a ``git`` command in ``cwd``, raising on failure (used by ``_scratch_merge``)."""
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def _git_out(cwd: str, *args: str) -> str:
    """Run a ``git`` command in ``cwd`` and return its stdout (used by ``_scratch_merge``)."""
    return subprocess.run(
        ["git", "-C", cwd, *args], check=True, capture_output=True, text=True
    ).stdout


def _flatten_conflict_markers(conflicted: str) -> str:
    """Deterministically flatten git conflict hunks, keeping BOTH sides' contributions.

    A real ``git`` conflict hunk looks like::

        <<<<<<< ours
        <ours lines>
        =======
        <theirs lines>
        >>>>>>> theirs

    A deterministic resolver (standing in for an LLM resolver leaf offline) drops the
    marker lines and keeps both bodies concatenated, so the resolution is reproducible
    and contains both contributions. The resolved content IS the merge resolution — it is
    folded directly into the integrated tree (no second merge pass).

    Args:
        conflicted: File content carrying one or more conflict hunks.

    Returns:
        The resolved content with every conflict hunk flattened.
    """
    out: list[str] = []
    for line in conflicted.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        out.append(line)
    return "".join(out)


# Git conflict-hunk marker prefixes. A resolved file that still carries ANY of these is a
# broken (un-resolved) merge and must never be folded into the integrated tree or a PR.
_CONFLICT_MARKERS: tuple[str, ...] = ("<<<<<<<", "=======", ">>>>>>>")

# How many times the conflict resolver leaf is retried before the fold FAILS LOUD. The fold
# gates on the real artifact (no markers, every conflicted file present), not the model's
# word — a small bounded retry tolerates a botched first try without looping forever.
_MAX_RESOLVE_ATTEMPTS = 2


def _has_conflict_markers(text: str) -> bool:
    """Return ``True`` if any line of ``text`` begins with a git conflict marker."""
    return any(
        line.startswith(_CONFLICT_MARKERS) for line in (ln.rstrip("\n") for ln in text.splitlines())
    )


def _invalid_resolution_files(
    conflicts: dict[str, str], resolved: dict[str, str]
) -> dict[str, str]:
    """Return the conflicted files a resolution failed to resolve (missing or marker-laden).

    A resolution is valid only when every conflicted path is present AND carries no remaining
    conflict markers. This returns the offending subset — keyed by path — so a retry can feed
    back exactly which files still need resolving (missing files map to their original
    conflicted content so the retry sees the markers it must remove).

    Args:
        conflicts: The merge's conflicted files (``path -> conflicted content``).
        resolved: The resolver's returned files (``path -> resolved content``).

    Returns:
        A ``path -> content`` map of the still-invalid files (empty when the resolution is
        valid). A missing path maps to its original conflicted content; a present-but-dirty
        path maps to the resolver's still-markered content.
    """
    invalid: dict[str, str] = {}
    for path, conflicted in conflicts.items():
        if path not in resolved:
            invalid[path] = conflicted
        elif _has_conflict_markers(resolved[path]):
            invalid[path] = resolved[path]
    return invalid


# Sentinel delimiting the machine-readable conflicts JSON block in the resolver prompt. The
# real model reads the prose instructions; an offline fake extracts the JSON between these
# markers (see _parse_conflicts_from_prompt) — so one prompt serves both paths.
_CONFLICTS_JSON_BEGIN = "<<<CONFLICTS_JSON>>>"
_CONFLICTS_JSON_END = "<<<END_CONFLICTS_JSON>>>"


def _resolve_conflict_prompt(conflicts: dict[str, str], *, attempt: int, retry_files: str) -> str:
    """Build the prescriptive conflict-resolver prompt for one attempt.

    The prompt is content-distinct per attempt (it carries the attempt number and, on a
    retry, the files still carrying markers), so the engine's content-hash journal does NOT
    short-circuit a retry to the prior attempt's result. It carries prose instructions for a
    real model AND a machine-readable conflicts JSON block (between
    :data:`_CONFLICTS_JSON_BEGIN` / :data:`_CONFLICTS_JSON_END`) an offline fake parses, so
    one prompt serves both the real and offline paths.

    Args:
        conflicts: The merge's conflicted files (``path -> conflicted content with markers``).
        attempt: The 1-based resolver attempt number.
        retry_files: On a retry, a rendered list of the files that still had markers / were
            missing on the prior attempt; empty on the first attempt.

    Returns:
        The resolver leaf prompt for this attempt.
    """
    retry = (
        ""
        if not retry_files
        else (
            "Your previous attempt did NOT fully resolve these files — they still carry "
            f"conflict markers or were missing:\n{retry_files}\n\n"
        )
    )
    rendered = "\n".join(
        f"--- {path} ---\n{content}" for path, content in sorted(conflicts.items())
    )
    conflicts_json = json.dumps(conflicts, sort_keys=True)
    return (
        "You are resolving git merge conflicts. For EACH conflicted file below, return its "
        "fully-merged content in `files` keyed by the same path. Remove EVERY "
        "`<<<<<<<` / `=======` / `>>>>>>>` conflict marker line and keep the correct merged "
        "content from both sides — no marker may remain in any returned file, and every "
        f"conflicted path MUST be present in your `files`.\n\n{retry}"
        f"Attempt {attempt}.\nConflicted files:\n{rendered}\n\n"
        f"{_CONFLICTS_JSON_BEGIN}\n{conflicts_json}\n{_CONFLICTS_JSON_END}"
    )


def _parse_conflicts_from_prompt(prompt: str) -> dict[str, str]:
    """Extract the conflicts JSON block an offline resolver fake resolves.

    Reads the JSON object between :data:`_CONFLICTS_JSON_BEGIN` / :data:`_CONFLICTS_JSON_END`
    in a resolver prompt built by :func:`_resolve_conflict_prompt`. Falls back to parsing the
    whole prompt as JSON (so a caller that passes raw JSON still works) and to an empty map
    when neither yields an object.

    Args:
        prompt: The resolver leaf's prompt text.

    Returns:
        The conflicted files (``path -> conflicted content``), or an empty map when absent.
    """
    begin = prompt.find(_CONFLICTS_JSON_BEGIN)
    end = prompt.find(_CONFLICTS_JSON_END)
    if begin != -1 and end != -1 and end > begin:
        block = prompt[begin + len(_CONFLICTS_JSON_BEGIN) : end].strip()
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            parsed = {}
    else:
        try:
            parsed = json.loads(prompt)
        except json.JSONDecodeError:
            parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _refactor_fix_prompt(target: RefactorTarget) -> str:
    """Build the git fixer prompt for one target file (real-model path)."""
    return (
        "You are fixing a bug in a small Python module that lives in your git worktree.\n\n"
        f"File to fix: {target.path}\n"
        f"Bug: {target.summary}\n"
        f"Replace this exact line:\n{target.old}\n"
        f"with:\n{target.new}\n\n"
        "Use your edit tool to make exactly that change on disk in your worktree, then "
        "give a one-line `summary` of what you changed. The engine will collect the real "
        "git diff of your worktree as the authoritative changeset — make the edit on disk, "
        "do not just describe it."
    )


def _refactor_verify_prompt(summary: str, files: dict[str, str], voter: int) -> str:
    """Build the read-only judge prompt for one patch (one of two voters)."""
    rendered = "\n".join(f"--- {path} ---\n{content}" for path, content in sorted(files.items()))
    return (
        f"You are reviewer #{voter + 1} judging a code patch. Decide whether it is a "
        "sound, safe change. Set `refuted` to true ONLY if the patch is clearly wrong, "
        "unsafe, or introduces a regression; otherwise set it to false. Give one sentence "
        f"of `reason`.\nPatch summary: {summary}\nPatched files:\n{rendered}"
    )


async def _resolve_conflict(ctx: Ctx, conflicts: dict[str, str]) -> dict[str, str]:
    """Drive the conflict resolver leaf, VALIDATING its output, retrying, then failing loud.

    A real resolver can omit a conflicted file or leave ``<<<<<<<`` / ``=======`` /
    ``>>>>>>>`` markers behind. Folding such output would integrate a broken merge and open a
    PR over it, so this gates on the real artifact: after each resolver attempt it checks
    every conflicted path is present and marker-free, and on failure RETRIES the resolver
    (feeding back exactly which files still need work) up to :data:`_MAX_RESOLVE_ATTEMPTS`.
    If no attempt lands a clean resolution, it FAILS LOUD rather than folding markers — the
    same "don't trust the model; verify the artifact" discipline the M5 fix loop uses.

    Each attempt's prompt is content-distinct (it carries the attempt number and the retry
    feedback), so the engine's content-hash journal does not short-circuit a retry to the
    prior attempt's cached result.

    Args:
        ctx: The orchestration context (for the resolver ``ctx.agent`` calls + logging).
        conflicts: The merge's conflicted files (``path -> conflicted content with markers``).

    Returns:
        The validated, marker-free resolution (``path -> resolved content``) for every
        conflicted file.

    Raises:
        ValueError: If the resolver never produces a valid (complete, marker-free)
            resolution within :data:`_MAX_RESOLVE_ATTEMPTS` attempts.
    """
    retry_files = ""
    invalid: dict[str, str] = {}
    for attempt in range(1, _MAX_RESOLVE_ATTEMPTS + 1):
        resolution = await ctx.agent(
            _resolve_conflict_prompt(conflicts, attempt=attempt, retry_files=retry_files),
            agent_type="conflict_resolver",
            schema=Resolution,
        )
        invalid = _invalid_resolution_files(conflicts, resolution.files)
        if not invalid:
            return {path: resolution.files[path] for path in conflicts}
        # Still invalid: feed back which files are unresolved so the retry's prompt is both
        # actionable and content-distinct (so the journal does not replay attempt N's result).
        retry_files = "\n".join(
            f"--- {path} ---\n{content}" for path, content in sorted(invalid.items())
        )
        ctx.log(f"resolver attempt {attempt} left {len(invalid)} file(s) unresolved; retrying")
    raise ValueError(
        "conflict resolution failed: the resolver left "
        f"{len(invalid)} file(s) with conflict markers or missing after "
        f"{_MAX_RESOLVE_ATTEMPTS} attempts ({sorted(invalid)}); refusing to fold a broken "
        "merge or open a PR over it"
    )


async def refactor_swarm(ctx: Ctx, args: dict[str, Any]) -> RefactorResult:
    """Real-git fix swarm -> two-vote judge -> script-owned merge-conflict fold + PR intent.

    The M6 dual-track thesis made concrete, end to end:

    * **fix** — a ``ctx.parallel`` fan-out of git fixers, each ``isolation="worktree"`` so
      it runs in its OWN real ``git worktree`` on its own ``leaf/<id>`` branch, fully
      isolated from its siblings. Each fixer edits one target file on its worktree disk;
      the engine folds the real ``git diff`` into the leaf's :class:`GitPatch` ``files``
      as the AUTHORITATIVE changeset (a model self-report cannot override on-disk truth).
    * **verify** — a two-vote read-only judge per patch. A patch is approved unless a
      voter refutes it; a refuted patch is dropped from the fold (never integrated).
    * **integrate** — a SCRIPT-OWNED fold: cross-leaf state lives in the script variable
      ``integrated`` (seeded from the base tree), and each approved patch is folded by a
      journaled merge leaf running a real scratch-repo ``git merge`` (:func:`_scratch_merge`).
      A clean merge folds its merged tree directly; a real conflict routes through a
      resolver leaf whose flattened resolution is folded straight into the merged working
      tree (completing the merge, no second pass). The script — not the model — owns the
      loop: the control-flow inversion at the heart of the engine.

    The fixture is designed so two fixers edit the SAME overlapping region of ``/calc.py``,
    so the integrate fold ACTUALLY hits (and resolves) a real ``git merge`` conflict — the
    headline conflict path — while a third fixer edits ``/helper.py`` cleanly.

    Host-finalization boundary (R1): opening a PR is a networked side effect that must not
    live in the deterministic replay, so this workflow RETURNS a pure :class:`RefactorResult`
    carrying the integrated tree plus a :class:`PrIntent`; the HOST opens the PR once, after
    ``run_workflow`` returns, never from inside this script.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.
        args: Workflow arguments; unused today (the fixture is fixed for a deterministic
            demo), accepted for signature parity with the other presets.

    Returns:
        A :class:`RefactorResult` with the integrated tree, whether a conflict was
        resolved, the approved/rejected patch summaries, and the PR intent.
    """
    _ = args  # the fixture is fixed; args is accepted for preset-signature parity
    base_tree = dict(REFACTOR_BASE_TREE)

    ctx.phase("fix")
    # Each fixer runs in its OWN real git worktree (isolation="worktree"), so two fixers
    # editing the same file never see each other's edit — the isolation the conflict relies on.
    patches = await ctx.parallel(
        [
            lambda t=target, i=index: ctx.agent(
                _refactor_fix_prompt(t),
                agent_type="git_fixer",
                schema=GitPatch,
                isolation="worktree",
            )
            for index, target in enumerate(REFACTOR_TARGETS)
        ]
    )
    fixed = [p for p in patches if p is not None]
    ctx.log(f"fixed {len(fixed)}/{len(REFACTOR_TARGETS)} target files")

    ctx.phase("verify")
    approved: list[GitPatch] = []
    rejected: list[str] = []
    for patch in fixed:
        verdicts = await ctx.parallel(
            [
                lambda s=patch.summary, f=patch.files, v=v: ctx.agent(
                    _refactor_verify_prompt(s, f, v), agent_type="patch_judge", schema=Verdict
                )
                for v in range(2)
            ]
        )
        # A patch is approved unless ANY voter refutes it (a strict two-vote gate over typed
        # Verdicts) — the script branches on the typed ruling, never on leaf prose.
        survived = survives(verdicts, against=lambda v: v.refuted, kill_at=1)
        if survived:
            approved.append(patch)
            ctx.log(f"patch approved: {patch.summary.strip()[:50]}")
        else:
            rejected.append(patch.summary)
            ctx.log(f"patch rejected: {patch.summary.strip()[:50]}")

    ctx.phase("integrate")
    # Cross-leaf state lives in THIS script variable, seeded from the base tree and folded
    # forward by a journaled merge leaf — deterministic control-flow inversion.
    integrated = dict(base_tree)
    any_conflict = False
    for patch in approved:
        # "theirs" is a real branch: the base tree with this patch's files applied, so a
        # patch touching only some files does not appear to delete the rest.
        theirs = {**base_tree, **patch.files}
        merged = await ctx.agent(
            json.dumps({"base": base_tree, "ours": integrated, "theirs": theirs}, sort_keys=True),
            agent_type="merge",
            schema=MergeResult,
        )
        if merged.clean:
            integrated = merged.files
            continue
        any_conflict = True
        # Real conflict: a resolver leaf flattens git's markers. DON'T trust it blindly —
        # validate the resolution carries no remaining markers and resolves every conflicted
        # file, retrying with feedback up to a bound, and FAIL LOUD if it never lands a clean
        # tree (mirroring M5's "gate on the real artifact, not the model's word"). Only a
        # validated, marker-free resolution is folded into the merged working tree.
        resolved_files = await _resolve_conflict(ctx, merged.conflicts)
        integrated = dict(merged.files)
        integrated.update(resolved_files)
    ctx.log(f"integrated {len(approved)} patches (conflict resolved: {any_conflict})")

    body = (
        "Automated refactor swarm.\n\n"
        + "Approved patches:\n"
        + "\n".join(f"- {s}" for s in (p.summary for p in approved))
        + (f"\n\nResolved {1 if any_conflict else 0} merge conflict." if any_conflict else "")
    )
    return RefactorResult(
        integrated_tree=integrated,
        conflict_resolved=any_conflict,
        approved=[p.summary for p in approved],
        rejected=rejected,
        pr=PrIntent(branch=REFACTOR_PR_BRANCH, title=REFACTOR_PR_TITLE, body=body),
    )


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


def _build_git_fixer(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``git_fixer`` execution leaf, forwarding ``response_format`` (GitPatch).

    The real path is a ``create_deep_agent`` wired with a ``backend`` factory that reads
    the engine-leased per-leaf real GIT WORKTREE backend from the run config at call time
    (``configurable['sandbox_backend']``), so the leaf's edit/write tools mutate a real
    ``git worktree`` on its own ``leaf/<id>`` branch. The engine collects the worktree's
    real ``git diff`` and folds it into :class:`GitPatch` ``files`` as the authoritative
    changeset — so a git-worktree leaf MUST carry a ``files: dict[str, str]`` schema field
    (which :class:`GitPatch` declares) or the fold fails loud.

    The offline path is a deterministic fake that, with no model and no real worktree,
    EDITS the target file on the leased worktree disk itself (so the engine's real
    ``git diff`` collect still yields the authoritative changeset, exactly as a real fixer
    would). It reads the target from its prompt: the prompt names exactly one
    :class:`RefactorTarget`'s ``old`` / ``new`` / ``path``, so the fake replaces ``old``
    with ``new`` on disk and self-reports a ``files={}`` (the engine overrides it with the
    real diff). This exercises the genuine isolation + authoritative-collect + conflict
    path offline, with the real worktree provider driving the diff, not the fake.

    Args:
        response_format: The structured schema (``GitPatch``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``).

    Returns:
        A real execution deepagent when a key is in force, else the deterministic fake.
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:

        def _backend_factory(_runtime: Any) -> Any:
            # The engine leases a per-leaf real git-worktree backend (needs_execution=True,
            # isolation="worktree") and threads it into the run-config contextvar; read it
            # here at call time via get_config (the canonical engine pattern).
            configurable = (get_config() or {}).get("configurable") or {}
            return configurable["sandbox_backend"]

        return create_deep_agent(
            model=model,
            backend=_backend_factory,
            response_format=response_format,
            middleware=cache_middleware(),
        )

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        # The leased real worktree backend is on the run config under 'sandbox_backend';
        # the fake makes the REAL edit on disk so the engine's git diff is authoritative.
        backend = (config or {}).get("configurable", {}).get("sandbox_backend")
        # Identify exactly which target this prompt names. Two targets share a file
        # (/calc.py) AND overlap textually (that overlap is what makes them conflict), so
        # the path or the edited lines are ambiguous; match on the unique per-target
        # `summary` (rendered verbatim as `Bug: ...`) so each fixer makes ITS OWN edit.
        target = next((t for t in REFACTOR_TARGETS if f"Bug: {t.summary}" in prompt), None)
        if backend is not None and target is not None:
            # protocol-absolute path (e.g. "/calc.py") is what the backend edit tool expects.
            await backend.aedit(target.path, target.old, target.new)
        summary = target.summary if target is not None else "edited the target file"
        return {
            "messages": [*inp["messages"], AIMessage(content="edited the worktree file")],
            # files={} — the engine folds the real git diff in as the authoritative changeset.
            "structured_response": GitPatch(summary=summary, files={}),
        }

    return RunnableLambda(_leaf)


def _build_patch_judge(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``patch_judge`` read-only judge leaf, forwarding ``response_format`` (Verdict).

    A two-vote reviewer over a patch: it should only judge, never mutate. The real path is
    a ``read_only_leaf`` (a deny-write permission) so even a hallucinated "fix" is refused
    at the tool boundary. Offline it never refutes, so every patch is approved — a
    deterministic, readable demo (the conflict/fold narrative, not the vote split, is the
    headline; the vote split is exercised in ``capstone``).

    Args:
        response_format: The structured schema (``Verdict``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``).
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        return read_only_leaf(model, response_format=response_format, middleware=cache_middleware())
    return _fake_structured_leaf(
        Verdict(refuted=False, reason="the patch is a sound, targeted fix"),
        reply="reviewed the patch",
    )


def _build_merge_leaf(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``merge`` leaf: it runs a real scratch-repo three-way ``git merge``.

    Always deterministic — on BOTH the online and offline paths — because merging is an
    exact git operation, not a reasoning task: the leaf reads ``{base, ours, theirs}`` from
    its (journal-keyed) prompt JSON and runs :func:`_scratch_merge`, returning a typed
    :class:`MergeResult`. The model is irrelevant to a deterministic git merge, so this
    leaf never builds a deepagent; the conflict outcome is the real git result, not a model
    guess. ``response_format`` / ``api_key`` are accepted for builder-signature parity.

    Args:
        response_format: Unused (the leaf returns a fixed :class:`MergeResult` schema).
        api_key: Unused (the merge is deterministic, model-free).
    """
    _ = (response_format, api_key)

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        payload = json.loads(inp["messages"][-1].text if inp["messages"] else "{}")
        result = _scratch_merge(payload["base"], payload["ours"], payload["theirs"])
        return {
            "messages": [*inp["messages"], AIMessage(content="merged")],
            "structured_response": result,
        }

    return RunnableLambda(_leaf)


def _build_conflict_resolver(*, response_format: Any = None, api_key: str | None = None) -> Any:
    """Build the ``conflict_resolver`` reasoning leaf, forwarding ``response_format`` (Resolution).

    Resolves git conflict markers into clean, marker-free content. The real path is a
    ``create_deep_agent`` (a reasoning leaf — it merges intent, it does not execute), driven
    by the prescriptive :func:`_resolve_conflict_prompt` to return a :class:`Resolution` with
    every conflicted file present and every marker removed. The script does not trust the
    result blindly — :func:`_resolve_conflict` validates it (no markers, all files present),
    retries with feedback, and fails loud otherwise. The offline path deterministically
    flattens the conflict hunks (:func:`_flatten_conflict_markers`) parsed from the prompt's
    machine-readable conflicts block, keeping BOTH sides' contributions so the resolution is
    reproducible — the same mechanic the engine integration test mirrors.

    Args:
        response_format: The structured schema (``Resolution``) the real leaf returns.
        api_key: The per-run OpenRouter key captured at roster-build time (or ``None``).
    """
    model = resolve_leaf_model(api_key=api_key)
    if model is not None:
        return create_deep_agent(
            model=model, response_format=response_format, middleware=cache_middleware()
        )

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        conflicts = _parse_conflicts_from_prompt(prompt)
        resolved = {path: _flatten_conflict_markers(text) for path, text in conflicts.items()}
        return {
            "messages": [*inp["messages"], AIMessage(content="resolved the conflict")],
            "structured_response": Resolution(files=resolved),
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


def _register_refactor_roles(roster: Roster, *, api_key: str | None) -> Roster:
    """Register the ``refactor_swarm`` roles onto ``roster`` (HOST-TRUSTED path only).

    Adds the four leaves the ``refactor_swarm`` preset drives: ``git_fixer`` (an EXECUTION
    leaf, ``needs_execution=True``, that edits a real ``git worktree`` so its diff is the
    authoritative changeset), ``patch_judge`` (a read-only two-vote reviewer), ``merge`` (a
    deterministic real scratch-repo ``git merge`` leaf), and ``conflict_resolver`` (a
    reasoning leaf that flattens conflict markers).

    Trust boundary (R6): ``git_fixer`` is ``needs_execution``, so it can lease a real
    git-worktree backend — exactly what an untrusted authored script must NOT reach. This
    helper is therefore called ONLY from :func:`make_roster` (the trusted preset path),
    never from :func:`make_reasoning_roster`. None of these roles is registered on the
    reasoning roster, so an AST-gated authored script has no ``git_fixer`` to reach no
    matter what ``agent_type`` it asks for.

    Args:
        roster: The roster to register the refactor roles onto (mutated and returned).
        api_key: The per-run OpenRouter key captured in the host node context, threaded
            into every leaf builder (or ``None`` to fall back to the env key).

    Returns:
        The same ``roster``, with every refactor role registered, for fluent chaining.
    """
    return (
        roster.register(
            "git_fixer",
            builder=partial(_build_git_fixer, api_key=api_key),
            description="Fixes one file in a real git worktree (execution leaf).",
            needs_execution=True,
        )
        .register(
            "patch_judge",
            builder=partial(_build_patch_judge, api_key=api_key),
            description="Reviews a code patch read-only (two-vote gate).",
        )
        .register(
            "merge",
            builder=partial(_build_merge_leaf, api_key=api_key),
            description="Runs a real scratch-repo three-way git merge (deterministic).",
        )
        .register(
            "conflict_resolver",
            builder=partial(_build_conflict_resolver, api_key=api_key),
            description="Resolves git merge conflict markers (reasoning leaf).",
        )
    )


def make_roster() -> Roster:
    """Build the demo leaf roster shared by both preset workflows.

    Registers the reasoning roles (``researcher``, ``writer``, ``refiner``, ``extractor``,
    ``skeptic``, ``capstone_skeptic``, ``echo``) plus ``code_fixer`` (the ``fix_loop``
    preset's EXECUTION leaf) and the ``refactor_swarm`` roles — ``git_fixer`` (an EXECUTION
    leaf that edits a real ``git worktree``), ``patch_judge``, ``merge``, and
    ``conflict_resolver``. With an OpenRouter key in force each leaf is a real
    ``create_deep_agent`` (the merge leaf stays deterministic — git merging is exact, not
    reasoning); with no key the roster serves deterministic fake leaves so an offline run is
    fully reproducible.

    Trust boundary: this FULL roster — which can reach the real ``LocalSubprocessSandbox``
    via ``code_fixer`` and a real ``git worktree`` via ``git_fixer`` — is for the TRUSTED
    preset path only (``run_workflow_live``, whose
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
    roster = roster.register(
        "code_fixer",
        builder=partial(_build_code_fixer, api_key=api_key),
        description="Edits code and runs the real build/test until green (execution leaf).",
        needs_execution=True,
    )
    # The refactor_swarm roles — including the needs_execution git_fixer — are registered
    # ONLY here, on the trusted preset roster, never on make_reasoning_roster (R6 boundary).
    return _register_refactor_roles(roster, api_key=api_key)


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
        ``capstone``, ``fix_loop``, ``sign_off``, and ``refactor_swarm``.
    """
    return (
        WorkflowRegistry()
        .register("deep_research", deep_research)
        .register("capstone", capstone)
        .register("fix_loop", fix_loop)
        .register("sign_off", sign_off)
        .register("refactor_swarm", refactor_swarm)
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
    "REFACTOR_BASE_TREE",
    "REFACTOR_PR_BRANCH",
    "REFACTOR_TARGETS",
    "Claim",
    "EditedFile",
    "FixResult",
    "GitPatch",
    "MergeResult",
    "PrIntent",
    "RefactorResult",
    "RefactorTarget",
    "Resolution",
    "Verdict",
    "capstone",
    "deep_research",
    "fix_loop",
    "hello_workflow",
    "make_reasoning_roster",
    "make_roster",
    "make_workflows",
    "refactor_swarm",
    "sign_off",
]

"""Phase G3 demo: a runnable quality-patterns code review.

Composes the community quality patterns into one workflow over this engine's
primitives:

    loop-until-dry hunt (reviewer -> Finding objects)
      -> adversarial verify (3 independent skeptics per fresh finding; >=2 refutes kills it)
      -> reduce in plain Python (keep only the survivors)

The reviewer surfaces findings; each fresh finding faces three skeptics prompted to
*refute by default*, and only a finding that survives a majority is confirmed. The
hunt stops after two dry rounds in a row (no new findings) and never exceeds a hard
``MAX_ROUNDS`` — so a model that keeps "discovering" cannot loop forever.

Run it:

    uv sync --group example
    # credentials + model come from a local .env (OPENROUTER_API_KEY); see _demo_models
    export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
    uv run python examples/09_quality_patterns.py

With ``LDW_DEMO_REAL_MODEL`` unset the demo runs fully offline: deterministic fake
leaves drive the orchestration end to end with no API key (the path the integration
test pins). A real judge ideally cannot edit — once G4 lands, register the skeptic
as a read-only leaf so a hallucinated "fix" can't escape the verifier.
"""

from __future__ import annotations

import asyncio
from typing import Any

from _demo_models import load_demo_env, real_model
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import Roster, run_workflow

MAX_ROUNDS = 4
SKEPTICS_PER_FINDING = 3
REFUTES_TO_KILL = 2

# A small snippet with one genuine bug (SQL injection) and some plausible-but-benign
# bait (an unused import, a plaintext password compare) for the skeptics to weigh.
SAMPLE_CODE = """
import os
import hashlib  # imported but never used

def login(db, username, password):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    row = db.execute(query).fetchone()
    return row is not None and row["password"] == password
"""

# Offline marker: the fake reviewer plants this bait and the fake skeptic refutes it,
# so the offline run deterministically demonstrates a refuted finding being dropped.
_FALSE_MARKER = "FALSE_POSITIVE"


# ── structured leaf contracts (schema-as-handoff) ─────────────────────────────


class Finding(BaseModel):
    """A single code-review finding."""

    title: str
    severity: str


class ReviewBatch(BaseModel):
    """A batch of findings from one review pass."""

    findings: list[Finding]


class Verdict(BaseModel):
    """One skeptic's adversarial ruling on a finding."""

    refuted: bool
    reason: str


# ── the workflow ──────────────────────────────────────────────────────────────


def _review_prompt(target: str, seen: list[str]) -> str:
    return (
        "You are a meticulous code reviewer. List concrete, real bugs in the code "
        f"below that are NOT already in this list: {seen}. For each, give a short "
        "title and a severity (low/medium/high). Only report genuine defects.\n\n"
        f"{target}"
    )


def _verify_prompt(voter: int, title: str) -> str:
    return (
        f"You are skeptic #{voter + 1} reviewing a claimed code-review finding. Set "
        "`refuted` to true unless the finding is a real, exploitable defect; default "
        "to refuted for vague, stylistic, or non-issue claims. Give one sentence of "
        f"`reason`.\nFinding: {title}"
    )


async def code_review(ctx: Any, args: dict[str, Any]) -> list[str]:
    """loop-until-dry hunt -> adversarial verify -> reduce, quality-pattern style.

    The reviewer hands back ``Finding`` objects and each skeptic a ``Verdict``, so the
    reduce between phases is plain Python over typed data. The loop stops after two
    dry rounds in a row and never exceeds ``MAX_ROUNDS``.
    """
    target: str = args["target"]
    seen: set[str] = set()
    confirmed: list[str] = []
    dry_streak = 0

    for _round in range(MAX_ROUNDS):
        ctx.phase("review")
        batch = await ctx.agent(
            _review_prompt(target, sorted(seen)), agent_type="reviewer", schema=ReviewBatch
        )
        fresh = [finding for finding in batch.findings if finding.title not in seen]
        if not fresh:
            dry_streak += 1
            ctx.log(f"dry round ({dry_streak}/2) — no new findings")
            if dry_streak >= 2:  # converged
                break
            continue
        dry_streak = 0
        for finding in fresh:
            seen.add(finding.title)

        ctx.phase("verify")
        for finding in fresh:
            votes = await ctx.parallel(
                [
                    lambda t=finding.title, v=v: ctx.agent(
                        _verify_prompt(v, t), agent_type="skeptic", schema=Verdict
                    )
                    for v in range(SKEPTICS_PER_FINDING)
                ]
            )
            # Fail-safe: a skeptic that failed (None) counts as a refutation, so a
            # finding is never confirmed on absent verification.
            refutes = sum(1 for vote in votes if vote is None or vote.refuted)
            survived = refutes < REFUTES_TO_KILL
            mark = "kept" if survived else "killed"
            ctx.log(f"finding {mark} ({refutes}/{SKEPTICS_PER_FINDING}): {finding.title[:60]}")
            if survived:
                confirmed.append(finding.title)

    return sorted(confirmed)


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_reviewer_builder(*, response_format: Any = None) -> Any:
    """Offline reviewer: always surfaces one real bug + one bait; the loop dedups."""
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    schema: Any = response_format.schema if response_format is not None else ReviewBatch

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        batch = schema.model_validate(
            {
                "findings": [
                    {
                        "title": "Unsanitized username concatenated into the SQL query",
                        "severity": "high",
                    },
                    {
                        "title": f"{_FALSE_MARKER}: an unused import is a security hole",
                        "severity": "low",
                    },
                ]
            }
        )
        return {
            "messages": [*inp["messages"], AIMessage(content="reviewed")],
            "structured_response": batch,
        }

    return RunnableLambda(_leaf)


def _fake_skeptic_builder(*, response_format: Any = None) -> Any:
    """Offline skeptic: refutes the bait (its title carries the marker), keeps the rest."""
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    schema: Any = response_format.schema if response_format is not None else Verdict

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text
        refuted = _FALSE_MARKER in prompt
        verdict = schema.model_validate(
            {"refuted": refuted, "reason": "bait" if refuted else "looks like a real defect"}
        )
        return {
            "messages": [*inp["messages"], AIMessage(content="judged")],
            "structured_response": verdict,
        }

    return RunnableLambda(_leaf)


# ── driver ───────────────────────────────────────────────────────────────────


async def main() -> None:
    load_demo_env()
    mode = "REAL (OpenRouter)" if real_model() is not None else "offline (fake)"
    roster = (
        Roster()
        .register(
            "reviewer", builder=_fake_reviewer_builder, description="Finds code-review findings"
        )
        .register(
            "skeptic", builder=_fake_skeptic_builder, description="Adversarially verifies a finding"
        )
    )

    async def orchestrate(ctx: Any) -> list[str]:
        return await code_review(ctx, {"target": SAMPLE_CODE})

    print(f"mode: {mode}")
    confirmed = await run_workflow(orchestrate, roster=roster)
    print(f"confirmed findings ({len(confirmed)}):")
    for title in confirmed:
        print(f"  - {title}")


if __name__ == "__main__":
    asyncio.run(main())

"""Demo: AI-SRE multi-hypothesis diagnosis with ctx.race (early-exit + cancel).

An incident comes in; the workflow investigates several root-cause hypotheses in
parallel and takes the FIRST one that clears a confidence bar, cancelling the rest:

    investigate (race: one investigator leaf per hypothesis, each returns a
                 structured Diagnosis(root_cause, confidence, evidence))
      -> win = confidence >= 0.8  (first high-confidence hypothesis wins)
      -> the remaining investigators are cancelled

Shows race as a first-class early-exit primitive: latency is bounded by the first
good-enough answer, not the slowest hypothesis, and the decision is journaled so a
resume reproduces the same winner and re-runs nothing.

Run it:

    uv sync --group example
    export LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8
    uv run --group example python examples/13_ai_sre_race_real_e2e.py

With LDW_DEMO_REAL_MODEL unset it runs fully offline on deterministic fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from _demo_models import load_demo_env, real_model
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    Ctx,
    RaceCandidate,
    Roster,
    run_workflow,
)

INCIDENT = (
    "Checkout latency spiked 10x at 02:14 UTC; error rate is flat, CPU is normal, "
    "and the spike began minutes after a routine deploy."
)
HYPOTHESES = [
    "a database connection-pool exhaustion",
    "a slow downstream payment provider",
    "a regression in the deploy that added a synchronous call on the hot path",
    "a noisy-neighbor effect from a co-located batch job",
]


class Diagnosis(BaseModel):
    """One investigator's structured verdict on a single hypothesis."""

    root_cause: str
    confidence: float
    evidence: str


def _investigate_prompt(incident: str, hypothesis: str) -> str:
    return (
        f"You are an SRE investigating an incident. Incident: {incident}\n"
        f"Hypothesis to test: {hypothesis}\n"
        "Assess whether this hypothesis is the root cause. Return `root_cause` (a short "
        "statement), `confidence` in [0, 1], and one-sentence `evidence`."
    )


async def diagnose(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    """investigate (race over hypotheses) -> first high-confidence root cause wins."""
    incident: str = args["incident"]

    ctx.phase("investigate")
    result = await ctx.race(
        [
            RaceCandidate(
                prompt=_investigate_prompt(incident, hypothesis),
                agent_type="investigator",
                schema=Diagnosis,
            )
            for hypothesis in HYPOTHESES
        ],
        win=lambda diagnosis: diagnosis.confidence >= 0.8,
        win_tag="high-confidence-root-cause",
    )

    if result.won and result.winner is not None:
        ctx.log(
            f"confirmed root cause (hypothesis #{result.winner_index}, "
            f"confidence {result.winner.confidence:.2f}): {result.winner.root_cause}"
        )
        return {
            "root_cause": result.winner.root_cause,
            "confidence": result.winner.confidence,
            "evidence": result.winner.evidence,
            "winner_index": result.winner_index,
        }
    ctx.log("no hypothesis reached high confidence")
    return {"root_cause": None, "confidence": None, "evidence": None, "winner_index": None}


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_structured_leaf(structured: BaseModel, *, reply: str) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def _build_investigator(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline: every investigator returns a high-confidence diagnosis, so the race
    # has a clear winner (the lowest-index hypothesis, by the ascending tie-break).
    return _fake_structured_leaf(
        Diagnosis(
            root_cause="synchronous call added on the hot path by the recent deploy",
            confidence=0.9,
            evidence="latency onset aligns with the deploy and tracks the new call",
        ),
        reply="investigated",
    )


async def main() -> None:
    load_demo_env()
    roster = Roster().register(
        "investigator",
        builder=_build_investigator,
        description="Tests one root-cause hypothesis and returns a structured Diagnosis",
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        return await diagnose(ctx, {"incident": INCIDENT})

    print(f"incident: {INCIDENT}")
    print(f"mode: {'REAL (OpenRouter)' if real_model() is not None else 'offline (fake)'}")
    result = await run_workflow(orchestrate, roster=roster)
    print(f"root_cause: {result['root_cause']}")
    print(f"confidence: {result['confidence']}")
    print(f"winning hypothesis index: {result['winner_index']}")


if __name__ == "__main__":
    asyncio.run(main())

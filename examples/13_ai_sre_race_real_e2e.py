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
    "Checkout p99 latency jumped 10x (180ms -> 1.9s) at 02:14 UTC, two minutes after "
    "deploy v412 rolled out. Error rate is flat and CPU / memory are normal across the "
    "fleet. Distributed traces for slow checkout requests show ~1.6s of the 1.9s spent "
    "inside a new synchronous tax-service.calculate() call on the hot path that v412 "
    "added — invoked serially once per line item and absent in v411. Database "
    "connection-pool utilization, downstream payment-provider latency, and co-located "
    "batch-job scheduling are all unchanged from before the deploy."
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

# The one hypothesis the incident's trace evidence actually confirms — the deploy
# regression. Offline, only the investigator testing THIS hypothesis clears the win
# bar; the others stay low and lose. So the race picks it and cancels the rest — the
# same verdict the real model reaches when LDW_DEMO_REAL_MODEL is set.
_CONFIRMED_HYPOTHESIS = HYPOTHESES[2]


def _build_investigator(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)

    # Offline: a deterministic, hypothesis-aware fake. It reads the hypothesis out of
    # the investigate prompt and returns a high-confidence Diagnosis only for the one
    # the trace evidence confirms, a low-confidence one for the rest — so the race
    # demonstrates win-predicate gating (the deploy regression wins, the others lose),
    # not merely an ascending-index tie-break.
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content)
        if _CONFIRMED_HYPOTHESIS in prompt:
            diagnosis = Diagnosis(
                root_cause="a synchronous tax-service.calculate() call the v412 deploy added "
                "on the hot path, invoked once per line item",
                confidence=0.9,
                evidence="traces show ~1.6s of the 1.9s inside the new synchronous call, "
                "absent in v411",
            )
        else:
            diagnosis = Diagnosis(
                root_cause="not supported as root cause by the incident's evidence",
                confidence=0.3,
                evidence="the implicated subsystem is unchanged from before the deploy",
            )
        return {
            "messages": [*inp["messages"], AIMessage(content="investigated")],
            "structured_response": diagnosis,
        }

    return RunnableLambda(_leaf)


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

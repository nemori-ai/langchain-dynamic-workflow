"""Demo: corroborated screening — corroborate() then reconcile().

A workflow that screens claims about a topic:

    gather (parallel: one source-leaf per source, each emits a candidate claim)
      -> corroborate (keep claims >= 2 sources independently produced — cross-leaf backing)
      -> screen (parallel: two independent screeners per corroborated claim, include/exclude)
      -> reconcile (unanimous-include kept, unanimous-exclude dropped, disagreement escalated)

Shows the cross-leaf reduce helpers as first-class, fail-safe functions: corroborate
groups equivalent claims and drops the unsupported; reconcile buckets the dual-blind
screening verdicts (a failed screener -> conflict, never a silent include).

Run it:

    uv sync --group example
    export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
    uv run python examples/12_screening_reconcile_real_e2e.py

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
    ReviewItem,
    Roster,
    corroborate,
    reconcile,
    run_workflow,
)

TOPIC = "the operational trade-offs of microservices vs a monolith"
SOURCES = ["a platform engineering report", "a postmortem blog", "a vendor whitepaper", "a survey"]
SCREENERS = 2


class Candidate(BaseModel):
    """One candidate claim a source-leaf surfaces about the topic."""

    claim: str


class Screen(BaseModel):
    """One screener's include/exclude decision on a corroborated claim."""

    keep: bool
    reason: str


def _gather_prompt(topic: str, source: str) -> str:
    return (
        f"From the perspective of {source}, state THE single most important claim about "
        f"{topic}. Answer with one short, self-contained sentence in `claim`."
    )


def _screen_prompt(topic: str, claim: str, n: int) -> str:
    return (
        f"You are screener #{n + 1}. Decide whether this claim about {topic} is specific and "
        f"defensible enough to INCLUDE in a briefing. Set `keep` and give one-sentence `reason`.\n"
        f"Claim: {claim}"
    )


async def screening(ctx: Ctx, args: dict[str, Any]) -> dict[str, list[str]]:
    """gather -> corroborate -> dual-blind screen -> reconcile."""
    topic: str = args["topic"]

    ctx.phase("gather")
    candidates = await ctx.parallel(
        [
            lambda s=s: ctx.agent(_gather_prompt(topic, s), agent_type="source", schema=Candidate)
            for s in SOURCES
        ]
    )

    # Cross-leaf corroboration: keep claims at least two sources independently produced.
    groups = corroborate(candidates, key=lambda c: c.claim.strip().lower(), min_support=2)
    corroborated = [group.members[0].claim for group in groups]
    ctx.log(f"corroborated {len(corroborated)} of {len(SOURCES)} source claims")

    ctx.phase("screen")
    review: list[ReviewItem[str, Screen]] = []
    for claim in corroborated:
        verdicts = await ctx.parallel(
            [
                lambda c=claim, n=n: ctx.agent(
                    _screen_prompt(topic, c, n), agent_type="screener", schema=Screen
                )
                for n in range(SCREENERS)
            ]
        )
        review.append(ReviewItem(item=claim, verdicts=verdicts))

    result = reconcile(review, include=lambda v: v.keep)
    ctx.log(
        f"included {len(result.included)}, excluded {len(result.excluded)}, "
        f"conflicts {len(result.conflicts)}"
    )
    return {
        "included": result.included,
        "excluded": result.excluded,
        "conflicts": result.conflicts,
    }


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_structured_leaf(structured: BaseModel, *, reply: str) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def _build_source(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline: sources DISCRIMINATE so corroborate visibly drops the unsupported. Two
    # sources agree on one claim (it clears min_support=2 and corroborates); the other
    # two each raise a distinct claim only they hold (each below threshold, dropped). The
    # demo then prints "corroborated 1 of 4" — the reduce genuinely working, not a no-op.
    shared = "Microservices trade local simplicity for operational complexity"

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        index = next((i for i, source in enumerate(SOURCES) if source in prompt), 0)
        claim = shared if index < 2 else f"an unsupported claim only source {index} holds"
        return {
            "messages": [*inp["messages"], AIMessage(content="surfaced a claim")],
            "structured_response": Candidate(claim=claim),
        }

    return RunnableLambda(_leaf)


def _build_screener(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline screeners both include, so the corroborated claim is unanimously kept.
    return _fake_structured_leaf(
        Screen(keep=True, reason="specific and defensible"), reply="screened"
    )


async def main() -> None:
    load_demo_env()
    roster = (
        Roster()
        .register("source", builder=_build_source, description="Surfaces one candidate claim")
        .register("screener", builder=_build_screener, description="Includes/excludes a claim")
    )

    async def orchestrate(ctx: Ctx) -> dict[str, list[str]]:
        return await screening(ctx, {"topic": TOPIC})

    print(f"topic: {TOPIC}")
    print(f"mode: {'REAL (OpenRouter)' if real_model() is not None else 'offline (fake)'}")
    result = await run_workflow(orchestrate, roster=roster)
    print(f"included: {result['included']}")
    print(f"excluded: {result['excluded']}")
    print(f"conflicts: {result['conflicts']}")


if __name__ == "__main__":
    asyncio.run(main())

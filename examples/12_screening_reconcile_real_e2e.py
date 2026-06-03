"""Demo: corroborated screening — corroborate() then reconcile().

A workflow that screens claims about a topic:

    gather (parallel: one source-leaf per source, each emits a claim + a fixed aspect tag)
      -> corroborate (keep the aspects >= 2 sources independently flagged — cross-leaf backing)
      -> screen (parallel: two independent screeners per corroborated claim, include/exclude)
      -> reconcile (unanimous-include kept, unanimous-exclude dropped, disagreement escalated)

Shows the cross-leaf reduce helpers as first-class, fail-safe functions: corroborate
groups on an EXACT structured key (a low-cardinality `aspect`, since free claim prose
never collides across independent model calls) and drops the unsupported; reconcile
buckets the dual-blind screening verdicts (a failed screener -> conflict, never a
silent include).

Run it:

    uv sync --group example
    export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
    uv run python examples/12_screening_reconcile_real_e2e.py

With LDW_DEMO_REAL_MODEL unset it runs fully offline on deterministic fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

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

# A small, fixed vocabulary of aspects. corroborate() groups on an EXACT key, so the
# sources must agree on a low-cardinality structured field for cross-leaf backing to
# register — free-form claim prose never collides across independent model calls. With
# more sources than aspects, the pigeonhole principle guarantees at least one aspect is
# corroborated, so the demo never collapses to an empty result on a real model.
Aspect = Literal["operational complexity", "deployment and scaling", "team and cost"]
ASPECTS: list[Aspect] = ["operational complexity", "deployment and scaling", "team and cost"]


class Candidate(BaseModel):
    """One candidate claim a source-leaf surfaces, tagged with the aspect it bears on."""

    claim: str
    aspect: Aspect


# `from __future__ import annotations` defers the `aspect: Aspect` annotation as a string,
# so pydantic cannot resolve the `Aspect` alias until the model is rebuilt with it in scope.
# Without this, model_json_schema() (called when agent(schema=Candidate) builds its journal
# key) raises "Candidate is not fully defined".
Candidate.model_rebuild()


class Screen(BaseModel):
    """One screener's include/exclude decision on a corroborated claim."""

    keep: bool
    reason: str


def _gather_prompt(topic: str, source: str) -> str:
    return (
        f"From the perspective of {source}, state THE single most important claim about "
        f"{topic}. Put one short, self-contained sentence in `claim`, and classify which "
        f"aspect it bears on in `aspect` (choose exactly one of the allowed values)."
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

    # Cross-leaf corroboration on the structured `aspect` key (free claim prose never
    # collides across independent model calls): keep the aspects at least two sources
    # independently flagged, dropping the rest, and carry one representative claim each.
    groups = corroborate(candidates, key=lambda c: c.aspect, min_support=2)
    corroborated = [group.members[0].claim for group in groups]
    ctx.log(f"corroborated {len(corroborated)} aspect(s) backed by >= 2 of {len(SOURCES)} sources")

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

    # Offline: sources DISCRIMINATE on aspect so corroborate visibly drops the unsupported.
    # Two sources flag the same aspect (it clears min_support=2 and corroborates); the other
    # two each flag a distinct aspect only they raise (each below threshold, dropped). The
    # demo then prints "corroborated 1 aspect(s)" — the reduce genuinely working, not a no-op.
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        index = next((i for i, source in enumerate(SOURCES) if source in prompt), 0)
        aspect: Aspect = ASPECTS[0] if index < 2 else ASPECTS[(index - 1) % len(ASPECTS)]
        return {
            "messages": [*inp["messages"], AIMessage(content="surfaced a claim")],
            "structured_response": Candidate(claim=f"claim from source {index}", aspect=aspect),
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

"""Phase 7 integration: the registered ``deep_research`` workflow runs end to end.

Loads the runnable example's ``deep_research`` workflow and drives it through
``run_workflow`` with deterministic, call-counting fake leaves (no host agent, no
API key). The assertions pin the deep-research *shape* — one researcher per angle,
one extracted claim per finding, a 3-vote adversarial pass per claim, and a single
synthesis — so a regression that collapses a phase or drops the fan-out is caught.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import Roster, run_workflow


def _load_example() -> ModuleType:
    """Import the registered-deep-research flagship as a module."""
    return importlib.import_module("examples.flagship.deep_research_preset")


def _counting_leaf(counter: dict[str, int], role: str, reply: str) -> Any:
    """A fake text leaf that tallies invocations by role and returns a per-call-unique reply.

    The reply carries an incrementing suffix so distinct calls produce distinct
    outputs: identical leaf outputs would collapse to one content-hash journal key
    downstream and dedupe the next stage's calls, masking the real fan-out count.
    """

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        counter[role] = counter.get(role, 0) + 1
        return {"messages": [*inp["messages"], AIMessage(content=f"{reply} #{counter[role]}")]}

    return RunnableLambda(_leaf)


def _counting_structured_builder(
    counter: dict[str, int], role: str, structured: Callable[[int], BaseModel]
) -> Callable[..., Any]:
    """Build a schema leaf that tallies calls and hands back a validated structured object.

    Mirrors a ``create_deep_agent(..., response_format=ToolStrategy(...))`` leaf: the
    workflow calls ``ctx.agent(schema=...)``, so the roster invokes this builder with a
    ``response_format`` and the built leaf must attach a ``structured_response`` for
    ``agent(schema=...)`` to fold out. ``structured`` receives the per-role call index so
    each call yields a distinct object (distinct content-hash key, no journal dedupe).
    """

    def builder(*, response_format: Any = None) -> Any:
        assert response_format is not None, f"{role} leaf must be built with a response_format"

        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            counter[role] = counter.get(role, 0) + 1
            return {
                "messages": [*inp["messages"], AIMessage(content=f"{role} #{counter[role]}")],
                "structured_response": structured(counter[role]),
            }

        return RunnableLambda(_leaf)

    return builder


async def test_deep_research_workflow_executes_the_full_shape() -> None:
    module = _load_example()
    angles = len(module.ANGLES)
    skeptics = module.SKEPTICS_PER_CLAIM

    counts: dict[str, int] = {}
    roster = (
        Roster()
        .register("researcher", _counting_leaf(counts, "researcher", "a finding"))
        # extractor / skeptic are schema-gated (deep_research calls agent(schema=...)),
        # so they must be registered via builders that produce structured leaves.
        .register(
            "extractor",
            builder=_counting_structured_builder(
                counts, "extractor", lambda i: module.Claim(text=f"claim #{i}", checkable=True)
            ),
        )
        # Every skeptic SUPPORTS, so no claim is refuted and all survive to synthesis.
        .register(
            "skeptic",
            builder=_counting_structured_builder(
                counts, "skeptic", lambda i: module.Verdict(refuted=False, reason=f"holds up #{i}")
            ),
        )
        .register("writer", _counting_leaf(counts, "writer", "FINAL REPORT: synthesized"))
    )

    async def orchestrate(ctx: Any) -> str:
        return await module.deep_research(ctx, {"question": "Does RAG beat long context?"})

    report = await run_workflow(orchestrate, roster=roster)

    # The synthesis leaf's output is the workflow's returned result.
    assert report.startswith("FINAL REPORT: synthesized")
    # The deep-research shape: one researcher per angle, one extracted claim per
    # surviving finding, a full skeptic panel per claim, and exactly one synthesis.
    assert counts["researcher"] == angles
    assert counts["extractor"] == angles
    assert counts["skeptic"] == angles * skeptics
    assert counts["writer"] == 1


async def test_deep_research_drops_claims_that_the_skeptics_refute() -> None:
    # When a majority of the panel REFUTES, the claim must not reach synthesis. With
    # every skeptic refuting, all claims are killed and the writer is told the research
    # was inconclusive — proving the adversarial gate actually filters.
    module = _load_example()
    angles = len(module.ANGLES)
    skeptics = module.SKEPTICS_PER_CLAIM

    counts: dict[str, int] = {}
    captured_prompt: list[str] = []

    async def _writer(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        captured_prompt.append(inp["messages"][-1].text if inp["messages"] else "")
        return {"messages": [*inp["messages"], AIMessage(content="done")]}

    roster = (
        Roster()
        .register("researcher", _counting_leaf(counts, "researcher", "a finding"))
        # Extractor yields a checkable claim per angle; the schema-gated skeptics each
        # REFUTE, so the verify loop must actually run and kill every claim.
        .register(
            "extractor",
            builder=_counting_structured_builder(
                counts, "extractor", lambda i: module.Claim(text=f"claim #{i}", checkable=True)
            ),
        )
        .register(
            "skeptic",
            builder=_counting_structured_builder(
                counts,
                "skeptic",
                lambda i: module.Verdict(refuted=True, reason=f"unsupported #{i}"),
            ),
        )
        .register("writer", RunnableLambda(_writer))
    )

    async def orchestrate(ctx: Any) -> str:
        return await module.deep_research(ctx, {"question": "Q?"})

    await run_workflow(orchestrate, roster=roster)

    # The adversarial gate was genuinely exercised: a claim per angle was extracted and
    # the full skeptic panel ran per claim (not skipped because extraction produced zero).
    assert counts["extractor"] == angles
    assert counts["skeptic"] == angles * skeptics
    # Every claim was refuted by a majority -> none reach synthesis -> the writer is
    # handed the inconclusive prompt, proving refutation filtered the survivors out.
    assert captured_prompt and "inconclusive" in captured_prompt[0].lower()

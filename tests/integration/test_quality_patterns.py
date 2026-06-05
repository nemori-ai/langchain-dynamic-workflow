"""Phase G3 integration: the runnable quality-patterns workflow behaves as documented.

Loads ``examples.features.reduce`` and drives its ``code_review`` workflow
through ``run_workflow`` with deterministic, call-counting fake leaves (no host
agent, no API key). The assertions pin the *behavior* the quality patterns promise:
a finding the skeptics refute by majority is dropped (adversarial-verify works), the
surviving real finding is kept, and the loop-until-dry converges within MAX_ROUNDS
instead of spinning forever.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import Roster, run_workflow

# The planted findings the fake reviewer always surfaces: one genuine bug and one
# bait the skeptics must refute. The skeptic fake refutes on this marker.
_REAL_BUG = "Unsanitized user input flows into the SQL login query"
_FALSE_MARKER = "FALSE_POSITIVE"
_BAIT = f"{_FALSE_MARKER}: an unused import is a security vulnerability"


def _load_example() -> ModuleType:
    """Import the reduce feature demo (code_review sub-workflow) as a module."""
    return importlib.import_module("examples.features.reduce")


def _reviewer_builder(module: ModuleType, counter: dict[str, int]) -> Any:
    """Fake reviewer: always surfaces the same real bug + bait; tallies calls.

    The orchestrate dedups by title, so round 1 sees both as fresh and later rounds
    see nothing new — that is what drives loop-until-dry to converge.
    """

    def builder(*, response_format: Any = None) -> Any:
        assert response_format is not None
        model = response_format.schema

        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            counter["reviewer"] = counter.get("reviewer", 0) + 1
            batch = model.model_validate(
                {
                    "findings": [
                        {"title": _REAL_BUG, "severity": "high"},
                        {"title": _BAIT, "severity": "low"},
                    ]
                }
            )
            return {
                "messages": [*inp["messages"], AIMessage(content="reviewed")],
                "structured_response": batch,
            }

        return RunnableLambda(_leaf)

    return builder


def _skeptic_builder(counter: dict[str, int]) -> Any:
    """Fake skeptic: refutes only the bait (its title carries the marker)."""

    def builder(*, response_format: Any = None) -> Any:
        assert response_format is not None
        model = response_format.schema

        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            counter["skeptic"] = counter.get("skeptic", 0) + 1
            prompt = inp["messages"][-1].text
            refuted = _FALSE_MARKER in prompt
            verdict = model.model_validate({"refuted": refuted, "reason": "fake verdict"})
            return {
                "messages": [*inp["messages"], AIMessage(content="judged")],
                "structured_response": verdict,
            }

        return RunnableLambda(_leaf)

    return builder


def _failing_skeptic_builder() -> Any:
    """Fake skeptic that always raises, so ctx.parallel lands a ``None`` vote."""

    def builder(*, response_format: Any = None) -> Any:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            raise RuntimeError("skeptic crashed")

        return RunnableLambda(_leaf)

    return builder


async def test_failed_skeptics_refuse_rather_than_confirm() -> None:
    # Fail-safe: when verification fails (None votes), a finding must NOT be silently
    # confirmed — absent scrutiny counts as refutation, not as a pass.
    module = _load_example()
    counter: dict[str, int] = {}
    roster = (
        Roster()
        .register("reviewer", builder=_reviewer_builder(module, counter))
        .register("skeptic", builder=_failing_skeptic_builder())
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.code_review(ctx, {"target": "login.py"})

    confirmed = await run_workflow(orchestrate, roster=roster)

    assert confirmed == []  # every finding's skeptics failed -> nothing confirmed


async def test_quality_workflow_drops_refuted_finding_and_keeps_the_real_bug() -> None:
    module = _load_example()
    counter: dict[str, int] = {}
    roster = (
        Roster()
        .register("reviewer", builder=_reviewer_builder(module, counter))
        .register("skeptic", builder=_skeptic_builder(counter))
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.code_review(ctx, {"target": "login.py"})

    confirmed = await run_workflow(orchestrate, roster=roster)

    # Adversarial verify: the bait is refuted by majority and dropped; the real bug
    # survives (0 refutes).
    assert _REAL_BUG in confirmed
    assert _BAIT not in confirmed
    # loop-until-dry converged via two dry rounds (reviewer ran fewer times than the
    # hard cap), proving the dry-streak exit fired rather than the MAX_ROUNDS backstop.
    assert 1 <= counter["reviewer"] < module.MAX_ROUNDS


async def test_quality_workflow_runs_three_skeptics_per_finding() -> None:
    # The adversarial pass is a real fan-out: each fresh finding gets the documented
    # 3 independent skeptics (distinct voter index -> distinct journal keys, no dedupe).
    module = _load_example()
    counter: dict[str, int] = {}
    roster = (
        Roster()
        .register("reviewer", builder=_reviewer_builder(module, counter))
        .register("skeptic", builder=_skeptic_builder(counter))
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.code_review(ctx, {"target": "login.py"})

    await run_workflow(orchestrate, roster=roster)

    # 2 fresh findings on the first productive round x 3 skeptics each = 6 verdicts.
    assert counter["skeptic"] == 6

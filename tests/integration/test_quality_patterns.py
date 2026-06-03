"""Phase G3 integration: the runnable quality-patterns workflow behaves as documented.

Loads ``examples/09_quality_patterns.py`` and drives its ``code_review`` workflow
through ``run_workflow`` with deterministic, call-counting fake leaves (no host
agent, no API key). The assertions pin the *behavior* the quality patterns promise:
a finding the skeptics refute by majority is dropped (adversarial-verify works), the
surviving real finding is kept, and the loop-until-dry converges within MAX_ROUNDS
instead of spinning forever.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
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
    """Import ``examples/09_quality_patterns.py`` as a module (sibling import safe)."""
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "09_quality_patterns.py"
    spec = importlib.util.spec_from_file_location("_ldw_quality_patterns_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's nested pydantic models (ReviewBatch ->
    # Finding) can resolve their forward refs under `from __future__ import
    # annotations` (pydantic looks the namespace up via sys.modules[__module__]).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    # loop-until-dry converged within the hard cap rather than spinning forever.
    assert 1 <= counter["reviewer"] <= module.MAX_ROUNDS


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

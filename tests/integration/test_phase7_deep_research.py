"""Phase 7 integration: the registered ``deep_research`` workflow runs end to end.

Loads the runnable example's ``deep_research`` workflow and drives it through
``run_workflow`` with deterministic, call-counting fake leaves (no host agent, no
API key). The assertions pin the deep-research *shape* — one researcher per angle,
one extracted claim per finding, a 3-vote adversarial pass per claim, and a single
synthesis — so a regression that collapses a phase or drops the fan-out is caught.
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


def _load_example() -> ModuleType:
    """Import ``examples/07_deep_research_real_e2e.py`` as a module (sibling import safe)."""
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "07_deep_research_real_e2e.py"
    spec = importlib.util.spec_from_file_location("_ldw_deep_research_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counting_leaf(counter: dict[str, int], role: str, reply: str) -> Any:
    """A fake leaf that tallies invocations by role and returns a per-call-unique reply.

    The reply carries an incrementing suffix so distinct calls produce distinct
    outputs: identical leaf outputs would collapse to one content-hash journal key
    downstream and dedupe the next stage's calls, masking the real fan-out count.
    """

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        counter[role] = counter.get(role, 0) + 1
        return {"messages": [*inp["messages"], AIMessage(content=f"{reply} #{counter[role]}")]}

    return RunnableLambda(_leaf)


async def test_deep_research_workflow_executes_the_full_shape() -> None:
    module = _load_example()
    angles = len(module.ANGLES)
    skeptics = module.SKEPTICS_PER_CLAIM

    counts: dict[str, int] = {}
    roster = (
        Roster()
        .register("researcher", _counting_leaf(counts, "researcher", "a finding"))
        .register("extractor", _counting_leaf(counts, "extractor", "a falsifiable claim"))
        # Every skeptic SUPPORTS, so no claim is refuted and all survive to synthesis.
        .register("skeptic", _counting_leaf(counts, "skeptic", "SUPPORTED: holds up"))
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

    counts: dict[str, int] = {}
    captured_prompt: list[str] = []

    async def _writer(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        captured_prompt.append(inp["messages"][-1].text if inp["messages"] else "")
        return {"messages": [*inp["messages"], AIMessage(content="done")]}

    roster = (
        Roster()
        .register("researcher", _counting_leaf(counts, "researcher", "a finding"))
        .register("extractor", _counting_leaf(counts, "extractor", "a claim"))
        .register("skeptic", _counting_leaf(counts, "skeptic", "REFUTED: unsupported"))
        .register("writer", RunnableLambda(_writer))
    )

    async def orchestrate(ctx: Any) -> str:
        return await module.deep_research(ctx, {"question": "Q?"})

    await run_workflow(orchestrate, roster=roster)

    # All claims refuted -> the synthesis prompt is the inconclusive variant.
    assert captured_prompt and "inconclusive" in captured_prompt[0].lower()

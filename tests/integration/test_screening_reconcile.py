"""Integration: the screening workflow (examples/12) runs corroborate + reconcile end to end.

Loads the runnable example and drives its ``screening`` workflow through ``run_workflow``
with deterministic structured fakes (no host, no API key). Pins the reduce shape: every
source emits the same claim (so it corroborates), both screeners include (so it lands in
``included``), and a third path proves a failed screener routes the claim to ``conflicts``.
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
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "12_screening_reconcile_real_e2e.py"
    spec = importlib.util.spec_from_file_location("_ldw_screening_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structured_builder(make: Any) -> Any:
    def builder(*, response_format: Any = None) -> Any:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            return {
                "messages": [*inp["messages"], AIMessage(content="ok")],
                "structured_response": make(),
            }

        return RunnableLambda(_leaf)

    return builder


async def test_screening_includes_a_corroborated_unanimous_claim() -> None:
    module = _load_example()
    roster = (
        Roster()
        .register(
            "source",
            builder=_structured_builder(
                lambda: module.Candidate(
                    claim="Shared claim across sources", aspect=module.ASPECTS[0]
                )
            ),
        )
        .register(
            "screener",
            builder=_structured_builder(lambda: module.Screen(keep=True, reason="ok")),
        )
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.screening(ctx, {"topic": "T"})

    result = await run_workflow(orchestrate, roster=roster)
    # Every source emits the same claim -> corroborated; both screeners include -> kept.
    assert result["included"] == ["Shared claim across sources"]
    assert result["excluded"] == [] and result["conflicts"] == []


async def test_failed_screener_routes_corroborated_claim_to_conflicts() -> None:
    module = _load_example()

    def _failing_screener_builder(*, response_format: Any = None) -> Any:
        # A screener that returns no structured_response: agent(schema=) cannot fold it,
        # the leaf errors, and parallel() isolates the failure to a None verdict — which
        # reconcile must treat as a conflict (fail-safe), never a silent include/exclude.
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            return {"messages": [*inp["messages"], AIMessage(content="screener crashed")]}

        return RunnableLambda(_leaf)

    roster = (
        Roster()
        .register(
            "source",
            builder=_structured_builder(
                lambda: module.Candidate(claim="Shared claim", aspect=module.ASPECTS[0])
            ),
        )
        .register("screener", builder=_failing_screener_builder)
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.screening(ctx, {"topic": "T"})

    result = await run_workflow(orchestrate, roster=roster)
    # The claim corroborates (every source agrees) but every screener failed -> all
    # verdicts are None -> reconcile escalates it to conflicts, never silently included.
    assert result["conflicts"] == ["Shared claim"]
    assert result["included"] == [] and result["excluded"] == []

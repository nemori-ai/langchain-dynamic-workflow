"""Integration: the AI-SRE workflow (examples/13) runs ctx.race end to end.

Loads the runnable example and drives its ``diagnose`` workflow through
``run_workflow`` with a deterministic structured fake (no host, no API key). Pins
the race shape: a high-confidence diagnosis wins, the winner is the lowest-index
hypothesis (ascending tie-break), and a resume reproduces the same winner.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import InMemoryJournalStore, Roster, run_workflow


def _load_example() -> ModuleType:
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "13_ai_sre_race_real_e2e.py"
    spec = importlib.util.spec_from_file_location("_ldw_ai_sre_example", path)
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


async def test_ai_sre_race_picks_a_high_confidence_winner() -> None:
    module = _load_example()
    roster = Roster().register(
        "investigator",
        builder=_structured_builder(
            lambda: module.Diagnosis(
                root_cause="deploy regression", confidence=0.9, evidence="onset aligns"
            )
        ),
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.diagnose(ctx, {"incident": "I"})

    result = await run_workflow(orchestrate, roster=roster)
    # Every hypothesis clears the 0.8 bar -> the lowest-index hypothesis wins.
    assert result["winner_index"] == 0
    assert result["root_cause"] == "deploy regression"
    assert result["confidence"] == 0.9


async def test_ai_sre_race_resume_reproduces_the_winner() -> None:
    module = _load_example()
    calls = {"n": 0}

    def _counting_builder(*, response_format: Any = None) -> Any:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            calls["n"] += 1
            return {
                "messages": [*inp["messages"], AIMessage(content="ok")],
                "structured_response": module.Diagnosis(
                    root_cause="deploy regression", confidence=0.9, evidence="onset aligns"
                ),
            }

        return RunnableLambda(_leaf)

    roster = Roster().register("investigator", builder=_counting_builder)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any) -> Any:
        return await module.diagnose(ctx, {"incident": "I"})

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert first["winner_index"] == second["winner_index"] == 0
    assert first["root_cause"] == second["root_cause"]
    assert calls["n"] == dispatched  # the resumed race dispatched no investigator

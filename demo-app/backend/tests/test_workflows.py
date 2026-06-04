"""Offline checks for the preset demo workflows and their inline-run UI events.

These run with no model key, so the roster serves deterministic fake leaves and the
orchestration is exercised end to end without credentials. The key property under
test is that ``deep_research`` is a *real* dynamic workflow — it really fans out
parallel researchers and verifiers — and that the engine's progress/span hooks,
fed through a :class:`~ui_adapter.UiAdapter`, surface that fan-out as ordered
Gen-UI events the frontend can render live.

Visual/Gen-UI rendering in the browser is verified separately in Phase 2.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from ui_adapter import UiAdapter
from workflows import capstone, deep_research, make_roster, make_workflows

from langchain_dynamic_workflow import ProgressKind, run_workflow


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the roster serves deterministic fake leaves."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LDW_DEMO_REAL_MODEL"):
        os.environ.pop(key, None)


def _capture_adapter() -> tuple[UiAdapter, list[tuple[str, dict[str, Any]]]]:
    """Build a UiAdapter whose emits are captured in order for assertions."""
    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))
    return adapter, events


def test_make_workflows_registers_presets() -> None:
    """The registry resolves both preset workflows by name."""
    registry = make_workflows()
    assert registry.resolve("deep_research") is deep_research
    assert registry.resolve("capstone") is capstone


async def test_deep_research_offline_completes_with_fanout_and_ordered_phases() -> None:
    """Offline deep_research fans out in parallel and emits ordered phase/log + fan-out.

    Asserts the headline properties of a real dynamic workflow run, captured through
    the same UiAdapter the host graph uses:

    * the run completes and returns a non-empty report;
    * at least one ``fanout_graph`` event is emitted (proves a real ``parallel`` /
      ``pipeline`` fan-out actually happened — a flat sequential run would emit
      none); and
    * the ``phase_timeline`` events arrive in orchestration order
      (search -> extract -> verify -> synthesize), with log lines interleaved.
    """
    adapter, events = _capture_adapter()

    result = await run_workflow(
        lambda ctx: deep_research(ctx, {"question": "What are the trade-offs of RAG?"}),
        roster=make_roster(),
        workflows=make_workflows(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
    )

    # Real, non-empty product.
    assert isinstance(result, str)
    assert result.strip()

    components = [comp for comp, _ in events]

    # Real parallel fan-out happened: the parallel/pipeline spans surface as fan-out
    # events. A flat sequential run would emit zero of these.
    fanout = [props for comp, props in events if comp == "fanout_graph"]
    assert len(fanout) >= 1, f"expected >=1 fanout_graph event, got components={components}"
    # The search phase fans out one researcher per angle: the barrier span reports it.
    assert any(props.get("thunk_count", 0) >= 2 for props in fanout), (
        "expected a parallel barrier spanning multiple researcher thunks"
    )

    # Ordered phase markers: the four deep-research phases in orchestration order.
    phase_titles = [
        props["message"]
        for comp, props in events
        if comp == "phase_timeline" and props["kind"] == ProgressKind.PHASE.value
    ]
    assert phase_titles == ["search", "extract", "verify", "synthesize"]

    # Log lines are interleaved (at least the per-phase narration the workflow emits).
    log_messages = [
        props["message"]
        for comp, props in events
        if comp == "phase_timeline" and props["kind"] == ProgressKind.LOG.value
    ]
    assert log_messages, "expected at least one log narration line"

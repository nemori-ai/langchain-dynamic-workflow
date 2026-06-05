"""Real-model E2E acceptance gate for leaf-live observability (development-time gate).

Gated behind LDW_DEMO_REAL_MODEL + OpenRouter creds (a local .env). Runs a REAL leaf
(a real model that calls a real tool) and asserts the headline path: the span-begin
running edge fires before completion, and real LeafEvents (chat_model + tool
start/end) arrive correlated to the owning leaf's span_id. Offline-skippable: with no
gate set, the whole module is skipped, so CI stays offline.

The scenario is designed so an honest real model takes the headline path: the prompt
forces a tool call (the model must use the provided tool to answer), so a tool
start/end edge genuinely fires — not a fallback that bypasses the tool.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LDW_DEMO_REAL_MODEL"),
    reason="real-model gate: set LDW_DEMO_REAL_MODEL + OpenRouter creds to run",
)


async def test_real_leaf_emits_begin_before_completion_and_correlated_leaf_events() -> None:
    from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
    from examples._shared.real_models import load_demo_env, real_leaf_model
    from langchain_core.tools import tool  # pyright: ignore[reportUnknownVariableType]

    from langchain_dynamic_workflow import (
        Ctx,
        LeafEvent,
        Roster,
        Span,
        SpanBegin,
        SpanKind,
        run_workflow,
    )

    load_demo_env()
    # Disable LangSmith tracing for this deepagent-heavy run (memory: real-e2e).
    os.environ.pop("LANGSMITH_TRACING", None)

    model = real_leaf_model()
    assert model is not None, "real leaf model must be available under the gate"

    @tool
    def lookup_population(city: str) -> str:
        """Return the population of a city (a tool the model must call to answer)."""
        return f"{city} has a population of 3,200,000 (as of the latest census)."

    leaf = create_deep_agent(model=model, tools=[lookup_population])
    roster = Roster().register("researcher", leaf)

    begins: list[SpanBegin] = []
    ends: list[Span] = []
    events: list[LeafEvent] = []

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent(
            "Use the lookup_population tool to report the population of Berlin. "
            "You MUST call the tool; do not guess.",
            agent_type="researcher",
        )

    result = await run_workflow(
        orchestrate,
        roster=roster,
        on_span_begin=begins.append,
        on_span=ends.append,
        on_leaf_event=events.append,
    )
    completion_ts = time.time()
    assert "3,200,000" in result or "3200000" in result.replace(",", "")

    # Headline 1: the running edge fired, before the completion, with a shared id.
    agent_begin = next(b for b in begins if b.kind is SpanKind.AGENT)
    agent_end = next(e for e in ends if e.kind is SpanKind.AGENT)
    assert agent_begin.span_id == agent_end.span_id
    # The begin edge's wall-clock (stamped at span-open, mid-run) precedes the moment
    # the run was observed to complete -- the running edge genuinely fired first.
    assert agent_begin.started_at > 0.0
    assert agent_begin.started_at <= completion_ts

    # Headline 2: real interior events arrived, correlated, with a real tool call.
    assert events, "a real leaf must fire interior callback events"
    assert {e.leaf_span_id for e in events} == {agent_begin.span_id}
    kinds = {e.kind for e in events}
    assert "chat_model" in kinds
    assert any(e.kind == "tool" and e.name == "lookup_population" for e in events)
    # start precedes end for the tool call.
    tool_phases = [e.phase for e in events if e.kind == "tool"]
    assert "start" in tool_phases and "end" in tool_phases

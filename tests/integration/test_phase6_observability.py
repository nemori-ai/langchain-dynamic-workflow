"""Phase 6 integration: observability spans flow through a real workflow run.

``run_workflow`` accepts an ``on_span`` sink; every ``agent`` / ``parallel`` /
``pipeline`` call emits a structured span to it with zero instrumentation in the
orchestration script. These tests assert the spans carry the right kinds and
attributes (cache outcome, token usage, fan-out counts) and that a journal hit on
resume is reported as ``cached`` — the observability-by-default contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow
from langchain_dynamic_workflow._observability import Span, SpanKind

UsageLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_agent_parallel_pipeline_emit_spans_with_usage(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    leaf, _model = make_usage_leaf("finding", tokens_per_call=7)
    roster = Roster().register("researcher", leaf)
    spans: list[Span] = []

    async def orchestrate(ctx: Ctx) -> str:
        one = await ctx.agent("solo", agent_type="researcher")
        fanned = await ctx.parallel(
            [lambda t=t: ctx.agent(f"p{t}", agent_type="researcher") for t in ("a", "b")]
        )

        async def stage(prev: Any, item: Any, index: int) -> str:
            return await ctx.agent(f"s{item}", agent_type="researcher")

        piped = await ctx.pipeline(["x", "y", "z"], stage)
        return f"{one}|{len([f for f in fanned if f])}|{len([p for p in piped if p])}"

    result = await run_workflow(orchestrate, roster=roster, on_span=spans.append)
    assert result == "finding|2|3"

    by_kind: dict[SpanKind, list[Span]] = {}
    for span in spans:
        by_kind.setdefault(span.kind, []).append(span)

    # One agent span per leaf invocation: 1 solo + 2 parallel + 3 pipeline = 6.
    agent_spans = by_kind[SpanKind.AGENT]
    assert len(agent_spans) == 6
    # Every leaf was a fresh miss this run, and each metered its usage onto its span.
    assert all(s.attributes["cached"] is False for s in agent_spans)
    assert all(s.attributes["usage_tokens"] == 7 for s in agent_spans)
    assert all(s.attributes["agent_type"] == "researcher" for s in agent_spans)

    # The fan-out primitives each emit exactly one span carrying their shape.
    assert len(by_kind[SpanKind.PARALLEL]) == 1
    assert by_kind[SpanKind.PARALLEL][0].attributes["thunk_count"] == 2
    assert by_kind[SpanKind.PARALLEL][0].attributes["surviving_count"] == 2
    assert len(by_kind[SpanKind.PIPELINE]) == 1
    assert by_kind[SpanKind.PIPELINE][0].attributes["item_count"] == 3
    assert by_kind[SpanKind.PIPELINE][0].attributes["surviving_count"] == 3


async def test_resume_reports_cached_agent_spans(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # On resume the completed leaf replays from the journal at zero model cost; its
    # span must report cached=True while still attributing the journaled usage, so a
    # trace shows the replay as a hit rather than a fresh call.
    leaf, model = make_usage_leaf("finding", tokens_per_call=5)
    roster = Roster().register("researcher", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("solo", agent_type="researcher")

    first_spans: list[Span] = []
    await run_workflow(orchestrate, roster=roster, journal=journal, on_span=first_spans.append)
    assert model.calls == 1
    assert first_spans[0].attributes["cached"] is False

    # Re-run against the same journal: the leaf is served from cache (no new model
    # call) and its span flips to cached while keeping the journaled token usage.
    second_spans: list[Span] = []
    await run_workflow(orchestrate, roster=roster, journal=journal, on_span=second_spans.append)
    assert model.calls == 1  # no fresh model call on resume
    agent_span = next(s for s in second_spans if s.kind == SpanKind.AGENT)
    assert agent_span.attributes["cached"] is True
    assert agent_span.attributes["usage_tokens"] == 5

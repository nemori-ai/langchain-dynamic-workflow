"""Observability taps — narrate a run with phase()/log() and render its trace.

The script emits ``phase`` / ``log`` narration; an ``on_progress`` sink renders
the narrative and an ``on_span`` sink collects a span per primitive call. These
are the same surfaces the interactive demo app consumes to draw a live trace.

    uv run python -m examples.features.observability
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent

from examples._shared.offline_models import ScriptedModel
from langchain_dynamic_workflow import (
    Ctx,
    ProgressEntry,
    Roster,
    Span,
    run_workflow,
)

TOPICS = ["batteries", "solar", "wind"]


async def orchestrate(ctx: Ctx) -> list[str]:
    ctx.phase("research")
    findings: list[str] = []
    for topic in TOPICS:
        ctx.log(f"researching {topic}")
        findings.append(await ctx.agent(f"Research {topic}", agent_type="researcher"))
    ctx.phase("done")
    return findings


async def main() -> None:
    roster = Roster().register(
        "researcher",
        create_deep_agent(model=ScriptedModel(prefix="finding")),
        description="Researches a single topic",
    )
    progress: list[ProgressEntry] = []
    spans: list[Span] = []

    findings = await run_workflow(
        orchestrate,
        roster=roster,
        thread_id="observability",
        on_progress=progress.append,
        on_span=spans.append,
    )
    print("progress narration:")
    for entry in progress:
        print(f"  [{entry.kind.value}] {entry.message}")
    by_kind: dict[str, int] = {}
    for span in spans:
        by_kind[span.kind.value] = by_kind.get(span.kind.value, 0) + 1
    print(f"spans by kind: {by_kind}")
    assert progress, "phase()/log() must surface progress entries"
    assert spans, "every primitive call must emit a span"
    assert len(findings) == len(TOPICS)
    print("OK: phase/log narration and the span trace were both observable.")


if __name__ == "__main__":
    asyncio.run(main())

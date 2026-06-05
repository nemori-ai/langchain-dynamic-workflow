"""Observability taps — narration, the span trace, the running edge, and a leaf's run tree.

The script emits ``phase`` / ``log`` narration; ``on_progress`` renders the
narrative and ``on_span`` collects a completed span per primitive. Two further taps
surface the *live* picture: ``on_span_begin`` fires the instant each primitive opens
(the running edge + a wall-clock start for an elapsed timer), and ``on_leaf_event``
streams a leaf's own runtime subtree (its model/tool/chain callback edges),
correlated to the owning leaf's span id and reconstructable into a run tree — all
out-of-band, so the host context stays quarantined. These are the same surfaces the
interactive demo app consumes to draw a live trace.

    uv run python -m examples.features.observability
"""

from __future__ import annotations

import asyncio

from deepagents import create_deep_agent

from examples._shared.offline_models import ScriptedModel
from langchain_dynamic_workflow import (
    Ctx,
    LeafEvent,
    ProgressEntry,
    Roster,
    Span,
    SpanBegin,
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
    begins: list[SpanBegin] = []
    spans: list[Span] = []
    leaf_events: list[LeafEvent] = []

    findings = await run_workflow(
        orchestrate,
        roster=roster,
        thread_id="observability",
        on_progress=progress.append,
        on_span_begin=begins.append,
        on_span=spans.append,
        on_leaf_event=leaf_events.append,
    )

    print("progress narration:")
    for entry in progress:
        print(f"  [{entry.kind.value}] {entry.message}")

    print("running edges (begin):")
    for begin in begins:
        print(f"  {begin.kind.value} {begin.name} -> running (span_id={begin.span_id})")

    by_kind: dict[str, int] = {}
    for span in spans:
        by_kind[span.kind.value] = by_kind.get(span.kind.value, 0) + 1
    print(f"completed spans by kind: {by_kind}")

    # One leaf's run-tree subtree, filed under its owning span id.
    first_leaf_id = next(b.span_id for b in begins if b.kind.value == "agent")
    subtree = [e for e in leaf_events if e.leaf_span_id == first_leaf_id]
    print(f"leaf {first_leaf_id} run-tree edges: {[(e.kind, e.phase) for e in subtree]}")

    # Assertions double as the smoke check (offline, deterministic):
    assert progress, "phase()/log() must surface progress entries"
    assert begins, "on_span_begin must fire a running edge for every primitive"
    assert spans, "every primitive call must emit a completed span"
    # begin precedes end: every completed span's id has a matching begin.
    assert {s.span_id for s in spans} <= {b.span_id for b in begins}
    # The leaf's interior subtree is observable and correlated to the owning span.
    assert leaf_events, "on_leaf_event must surface a leaf's runtime subtree"
    assert all(e.leaf_span_id in {b.span_id for b in begins} for e in leaf_events)
    assert len(findings) == len(TOPICS)
    print("OK: narration, running edges, the span trace, and a leaf's run tree were observable.")


if __name__ == "__main__":
    asyncio.run(main())

# uv run python -m examples.features.observability

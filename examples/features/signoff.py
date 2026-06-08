"""In-run HITL sign-off — pause a run for a human decision, then resume it.

A workflow calls ``ctx.checkpoint(ask)`` to pause mid-run for a person. The run raises
``WorkflowSignoffRequired`` carrying the ask; the host surfaces it, obtains a decision,
then resumes with ``run_workflow(..., resume=decision)`` against the SAME journal — the
leaves completed before the gate replay for free, and the script branches on the REAL
human decision (never a model guess). Approve proceeds; reject holds. This is the
host-wiring you lift into your own app: catch the signal, ask a human, resume with their
value. (Cross-gate state rides script variables; the decision is keyed by gate position,
so a second gate parks independently.)

    uv run python -m examples.features.signoff
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    WorkflowSignoffRequired,
    run_workflow,
)


class _Counter:
    """Tallies live leaf invocations so the zero-cost replay across the gate is provable."""

    def __init__(self) -> None:
        self.calls = 0


def _assessor_leaf(counter: _Counter) -> Any:
    """A deterministic offline leaf that 'assesses' the plan (no key, no model)."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        counter.calls += 1
        return {"messages": [*inp["messages"], AIMessage(content="risk: the migration step")]}

    return RunnableLambda(_leaf)


async def gated(ctx: Ctx) -> str:
    """Assess the plan, pause for a human sign-off, then proceed or hold on the decision."""
    assessment = await ctx.agent("Assess the deploy plan's risks.", agent_type="auditor")
    # PAUSE here for a human. The run parks; the returned value is whatever the host
    # supplies on resume, so the script branches on the real decision.
    decision = await ctx.checkpoint(
        {"ask": "Approve the deploy?", "summary": assessment}, tag="deploy"
    )
    if not decision.get("approved"):
        return f"held: {decision.get('note', 'reviewer declined')}"
    return f"proceeding: {assessment}"


async def main() -> None:
    counter = _Counter()
    roster = Roster().register("auditor", _assessor_leaf(counter))

    # ── Approve: the run parks, then a human approval resumes it. ──
    journal = InMemoryJournalStore()
    try:
        await run_workflow(gated, roster=roster, journal=journal, thread_id="approve")
        raise AssertionError("the run should have parked at the sign-off gate")
    except WorkflowSignoffRequired as park:
        print(f"paused — the host shows the human: {park.ask['ask']}")
    calls_at_pause = counter.calls

    approved = await run_workflow(
        gated, roster=roster, journal=journal, thread_id="approve", resume={"approved": True}
    )
    print(f"approved -> {approved}")
    assert approved.startswith("proceeding")
    # The assessment leaf ran once (before the gate) and replayed for FREE on resume.
    assert counter.calls == calls_at_pause, "resume must replay the pre-gate leaf at zero cost"

    # ── Reject: a separate run shows a decline holds and records the note. ──
    journal2 = InMemoryJournalStore()
    with contextlib.suppress(WorkflowSignoffRequired):
        await run_workflow(gated, roster=roster, journal=journal2, thread_id="reject")
    held = await run_workflow(
        gated,
        roster=roster,
        journal=journal2,
        thread_id="reject",
        resume={"approved": False, "note": "too risky this week"},
    )
    print(f"rejected -> {held}")
    assert held.startswith("held") and "too risky" in held

    print("OK: the run paused for a human, approve proceeded (pre-gate leaf free), reject held.")


if __name__ == "__main__":
    asyncio.run(main())

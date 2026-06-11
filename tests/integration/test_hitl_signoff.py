"""In-run HITL sign-off: ``ctx.checkpoint`` parks a run; an approve injects a value.

These drive the real ``run_workflow`` path with offline fake leaves (no host
model, no key). They pin the M4 mechanism: a script pauses at a sign-off gate, the
run raises ``WorkflowSignoffRequired`` carrying the ask, and a resume with a human
value records the decision in the journal and completes the run while completed
leaves before the gate replay from the journal at zero model cost. Multiple gates
approve one at a time, and a fan-out checkpoint fails loud.

The sign-off rides the content-hash journal (not a LangGraph interrupt), so the
load-bearing requirement is the *same journal instance* across the park and the
approve — no checkpointer is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    run_workflow,
)
from langchain_dynamic_workflow._errors import (
    WorkflowCheckpointError,
    WorkflowSignoffRequired,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_checkpoint_parks_then_approve_resumes(make_fake_leaf: FakeLeafFactory) -> None:
    # A two-phase script: research (leaf A) -> human sign-off -> write (leaf B).
    leaf_a, state_a = make_fake_leaf("finding-A")
    leaf_b, state_b = make_fake_leaf("report-B")
    roster = Roster().register("researcher", leaf_a).register("writer", leaf_b)
    # The SAME journal instance must span the park and the approve (zero-cost replay).
    journal = InMemoryJournalStore()
    seen: dict[str, Any] = {}

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        a = await ctx.agent("research the topic", agent_type="researcher")
        decision = await ctx.checkpoint({"ask": "approve report?", "draft": a})
        seen["decision"] = decision
        b = await ctx.agent("write the report", agent_type="writer")
        return {"a": a, "decision": decision, "b": b}

    # First run parks at the gate, surfacing the ask (with leaf A's draft folded in).
    with pytest.raises(WorkflowSignoffRequired) as exc_info:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert exc_info.value.ask == {"ask": "approve report?", "draft": "finding-A"}
    assert state_a.calls == 1
    assert state_b.calls == 0  # the gate stopped the script before leaf B

    # Approve: resume with the human value against the SAME journal.
    result = await run_workflow(
        orchestrate, roster=roster, journal=journal, thread_id="t1", resume="APPROVED"
    )
    assert result == {"a": "finding-A", "decision": "APPROVED", "b": "report-B"}
    assert seen["decision"] == "APPROVED"
    # Leaf A replayed from the journal on approve (NOT re-run); leaf B ran once.
    assert state_a.calls == 1
    assert state_b.calls == 1


async def test_multiple_gates_approve_one_at_a_time() -> None:
    # Two sequential gates: each approve injects exactly one value at the next
    # un-decided gate; already-approved gates replay their decision from the journal.
    roster = Roster()
    journal = InMemoryJournalStore()
    seen: list[Any] = []

    async def orchestrate(ctx: Ctx) -> list[Any]:
        d1 = await ctx.checkpoint({"ask": "gate 1?"}, tag="g1")
        seen.append(d1)
        d2 = await ctx.checkpoint({"ask": "gate 2?"}, tag="g2")
        seen.append(d2)
        return [d1, d2]

    with pytest.raises(WorkflowSignoffRequired) as e1:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")
    assert e1.value.tag == "g1"

    # Approve gate 1 -> the run advances and parks at gate 2.
    with pytest.raises(WorkflowSignoffRequired) as e2:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t", resume="D1")
    assert e2.value.tag == "g2"

    # Approve gate 2 -> the run completes; gate 1 replayed its journaled decision.
    result = await run_workflow(
        orchestrate, roster=roster, journal=journal, thread_id="t", resume="D2"
    )
    assert result == ["D1", "D2"]


async def test_checkpoint_in_fanout_fails_loud(make_fake_leaf: FakeLeafFactory) -> None:
    # A gate is keyed by its ordinal position; inside a fan-out frame that ordinal
    # races across concurrent thunks, so the engine refuses a checkpoint there.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def orchestrate(ctx: Ctx) -> list[Any]:
        async def thunk() -> Any:
            return await ctx.checkpoint({"ask": "inside fan-out?"})

        return await ctx.parallel([thunk])

    with pytest.raises(WorkflowCheckpointError):
        await run_workflow(orchestrate, roster=roster, thread_id="t1")


async def test_checkpoint_returns_resume_value_directly() -> None:
    # The minimal primitive: no leaves before the gate, the resume value comes back
    # as checkpoint()'s return value JSON-normalized (the decision is journaled as
    # JSON and read back the same way). A JSON-stable mapping is unchanged by the
    # round-trip, so == still holds; see test_checkpoint_approve_decision_is_json_
    # normalized for the type-stabilizing case (tuple -> list, int keys -> str keys).
    roster = Roster()
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> Any:
        return await ctx.checkpoint({"ask": "go?"}, tag="gate-1")

    with pytest.raises(WorkflowSignoffRequired) as exc_info:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert exc_info.value.tag == "gate-1"

    result = await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t2",
        resume={"approved": True, "note": "lgtm"},
    )
    assert result == {"approved": True, "note": "lgtm"}


async def test_checkpoint_approve_decision_is_json_normalized() -> None:
    # The approve path must hand the script the SAME JSON-normalized shape a replay
    # reads back (the decision is journaled as JSON), not the raw Python object: a
    # tuple becomes a list and int dict-keys become str keys. Pins normalize-on-approve
    # directly in a single run, without a third replay run.
    roster = Roster()
    journal = InMemoryJournalStore()
    approved: list[Any] = []

    async def orchestrate(ctx: Ctx) -> Any:
        decision = await ctx.checkpoint({"ask": "go?"}, tag="g1")
        approved.append(decision)
        return decision

    with pytest.raises(WorkflowSignoffRequired):
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")

    result = await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t",
        resume={"items": (1, 2, 3), "scores": {1: "a"}},
    )
    # The approve run already returns the normalized shape: tuple -> list, int key -> str.
    assert result == approved[0]
    assert result["items"] == [1, 2, 3] and isinstance(result["items"], list)
    assert list(result["scores"]) == ["1"]


async def test_checkpoint_decision_type_stable_across_approve_and_replay() -> None:
    # Regression for the approve-vs-replay type split: the SAME gate must hand the
    # script a type-identical value on the approving run and on every later replay.
    # Run-1 parks at gate-1; run-2 approves gate-1 with a JSON-UNSTABLE value (a tuple
    # and an int-keyed dict) and re-parks at gate-2 (so the approving run's gate-1
    # value is captured via the side-effect list, NOT the caller return); run-3 replays
    # the same journal and captures gate-1's replayed return. Both must be JSON-normalized.
    roster = Roster()
    journal = InMemoryJournalStore()
    unstable: dict[str, Any] = {"items": (1, 2, 3), "scores": {1: "a"}}
    seen: list[Any] = []

    async def orchestrate(ctx: Ctx) -> list[Any]:
        d1 = await ctx.checkpoint({"ask": "go?"}, tag="g1")
        seen.append(d1)
        d2 = await ctx.checkpoint({"ask": "go again?"}, tag="g2")
        return [d1, d2]

    # Run 1: park at gate-1.
    with pytest.raises(WorkflowSignoffRequired) as e1:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")
    assert e1.value.tag == "g1"

    # Run 2: approve gate-1 with the JSON-unstable value; re-park at gate-2. gate-1's
    # value is captured via `seen` (the approving run never returns past gate-2).
    with pytest.raises(WorkflowSignoffRequired) as e2:
        await run_workflow(
            orchestrate, roster=roster, journal=journal, thread_id="t", resume=unstable
        )
    assert e2.value.tag == "g2"

    # Run 3: a pure replay (no resume value) of the same journal — gate-1 replays its
    # journaled decision via json.loads and the run re-parks at the still-undecided
    # gate-2. (Passing resume="D2" would approve gate-2 and complete the run, so gate-1
    # would never be re-observed: the replay capture requires the no-resume path.)
    with pytest.raises(WorkflowSignoffRequired):
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")

    assert len(seen) >= 2, "gate-1 was not observed on both the approve and the replay run"
    approve, replay = seen[0], seen[1]
    # Type-identical across the human pause: both lists, both str keys.
    assert type(approve["items"]) is type(replay["items"])
    assert isinstance(approve["items"], list)
    assert list(approve["scores"]) == list(replay["scores"]) == ["1"]


async def test_resume_with_no_gate_to_consume_fails_loud() -> None:
    # Review M1/Codex#3 (engine backstop): a sign-off decision injected into a run that
    # has no un-decided gate to consume it must fail loud, never vanish silently.
    roster = Roster()
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return "no gates here"

    with pytest.raises(WorkflowCheckpointError):
        await run_workflow(
            orchestrate, roster=roster, journal=journal, thread_id="t", resume={"approved": True}
        )


async def test_non_json_decision_fails_loud_and_stays_re_approvable() -> None:
    # Review Codex#4: a non-JSON-serializable decision fails with a clear error WITHOUT
    # losing the pending value or journaling a half-gate — so a later valid approve works.
    roster = Roster()
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> Any:
        return await ctx.checkpoint({"ask": "go?"})

    with pytest.raises(WorkflowSignoffRequired):
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")
    # A non-JSON decision → clear WorkflowCheckpointError (not a buried TypeError).
    with pytest.raises(WorkflowCheckpointError):
        await run_workflow(
            orchestrate, roster=roster, journal=journal, thread_id="t", resume={"x": object()}
        )
    # The gate was not journaled and the value not lost: a valid approve still works.
    result = await run_workflow(
        orchestrate, roster=roster, journal=journal, thread_id="t", resume={"approved": True}
    )
    assert result == {"approved": True}


async def test_pre_gate_progress_not_renarrated_on_approve() -> None:
    # Review M4: a sign-off pause is a designed stop, not a crash — the progress count is
    # persisted on park, so an approve does NOT re-deliver pre-gate phase/log narration.
    roster = Roster()
    journal = InMemoryJournalStore()
    emitted: list[str] = []

    async def orchestrate(ctx: Ctx) -> Any:
        ctx.phase("assess")
        ctx.log("assessing the plan")
        decision = await ctx.checkpoint({"ask": "go?"})
        ctx.phase("proceed")
        return decision

    with pytest.raises(WorkflowSignoffRequired):
        await run_workflow(
            orchestrate,
            roster=roster,
            journal=journal,
            thread_id="t",
            on_progress=lambda e: emitted.append(e.message),
        )
    assert emitted == ["assess", "assessing the plan"]
    emitted.clear()

    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t",
        resume="ok",
        on_progress=lambda e: emitted.append(e.message),
    )
    # Pre-gate narration is NOT re-delivered; only the post-gate phase flows.
    assert emitted == ["proceed"]

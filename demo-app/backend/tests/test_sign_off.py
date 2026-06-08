"""Offline checks for the ``sign_off`` preset and its in-run HITL wiring (M4).

These run with no model key, so the roster serves deterministic fake reasoning leaves
(no subprocess, no real model). They pin the sign-off mechanism end to end without a
model:

* the preset PARKS at ``ctx.checkpoint`` (raising ``WorkflowSignoffRequired``) and
  resumes on a decision, branching on the REAL human decision;
* ``run_workflow_live`` catches the park, emits an awaiting ``signoff_gate`` card, and on
  resume flips it in place to resolved (same content ``event_id``, ``merge``);
* the offline host routes a sign-off REQUEST to the ``sign_off`` preset and a sign-off
  RESPONSE (the SignoffGate buttons' natural phrases) back into the paused run; and
* the whole pause -> approve loop works across two real host-graph turns on one thread.

The real-model path (a real host driving the gate live) is the gated real-model E2E.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from _models import OfflineHostModel, _signoff_response
from host_graph import _ResumeLane, run_workflow_live
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ui_adapter import _SIGNOFF_GATE, UiAdapter
from workflows import DEFAULT_SIGNOFF_TOPIC, make_roster, make_workflows, sign_off

from langchain_dynamic_workflow import (
    InMemoryJournalStore,
    Roster,
    WorkflowSignoffRequired,
    run_workflow,
)


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the roster serves deterministic fake leaves."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LDW_DEMO_REAL_MODEL"):
        os.environ.pop(key, None)


def test_make_workflows_registers_sign_off() -> None:
    """The registry resolves the ``sign_off`` preset by name."""
    assert make_workflows().resolve("sign_off") is sign_off


async def test_sign_off_offline_parks_then_approves() -> None:
    """The preset parks at the gate, then an approve proceeds (branching on the decision)."""
    roster = make_roster()
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any) -> str:
        return await sign_off(ctx, {"topic": "the X plan"})

    with pytest.raises(WorkflowSignoffRequired) as exc:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")
    ask = exc.value.ask
    assert isinstance(ask, dict)
    assert ask["ask"] == "Approve proceeding with the X plan?"

    result = await run_workflow(
        orchestrate, roster=roster, journal=journal, thread_id="t", resume={"approved": True}
    )
    assert "proceeded with the X plan" in result


async def test_sign_off_offline_decline_holds() -> None:
    """A declined sign-off holds the plan and records the reviewer's note."""
    roster = make_roster()
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any) -> str:
        return await sign_off(ctx, {"topic": "the Y plan"})

    with pytest.raises(WorkflowSignoffRequired):
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t")
    result = await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="t",
        resume={"approved": False, "note": "too risky"},
    )
    assert result.startswith("held:")
    assert "too risky" in result


async def test_run_workflow_live_parks_awaiting_then_resolves_in_place() -> None:
    """The inline helper emits an awaiting card on the park and flips it to approved on resume."""
    events: list[tuple[str, dict[str, Any]]] = []
    lane = _ResumeLane(thread_id="t::sign_off")

    adapter1 = UiAdapter(emit=lambda c, p: events.append((c, dict(p))))
    paused = await run_workflow_live("sign_off", {"topic": "deploy"}, adapter=adapter1, lane=lane)
    assert paused.startswith("Paused for your sign-off")
    assert lane.awaiting_signoff is True
    awaiting = [p for c, p in events if c == _SIGNOFF_GATE]
    assert awaiting and awaiting[-1]["status"] == "awaiting"
    assert awaiting[-1]["question"] == "Approve proceeding with deploy?"
    awaiting_event_id = awaiting[-1]["event_id"]

    adapter2 = UiAdapter(emit=lambda c, p: events.append((c, dict(p))))
    done = await run_workflow_live(
        "sign_off", {"topic": "deploy"}, adapter=adapter2, lane=lane, resume={"approved": True}
    )
    assert "proceeded with deploy" in done
    assert lane.awaiting_signoff is False
    resolved = [p for c, p in events if c == _SIGNOFF_GATE and p.get("status") == "approved"]
    assert resolved, "approve must emit a resolved signoff_gate card"
    # The resolved card flips the awaiting card IN PLACE: same content id, merge flag set.
    assert resolved[-1]["event_id"] == awaiting_event_id
    assert resolved[-1].get("merge") is True


def test_adapter_emit_signoff_request_and_resolved() -> None:
    """The adapter maps a request/resolved pair to a correlated awaiting/approved card."""
    sent: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda c, p: sent.append((c, dict(p))))
    ask = {"ask": "OK to ship?", "summary": "highest risk: the migration step"}
    adapter.emit_signoff_request(gate_key="abc", ask=ask)
    adapter.emit_signoff_resolved(
        gate_key="abc", ask=ask, decision={"approved": True, "note": "lgtm"}
    )

    cards = [(c, p) for c, p in sent if c == _SIGNOFF_GATE]
    assert len(cards) == 2
    (_, req), (_, res) = cards
    assert req["status"] == "awaiting"
    assert req["question"] == "OK to ship?"
    assert req["detail"] == "highest risk: the migration step"
    assert req["event_id"] == res["event_id"] == "signoff-abc"
    assert res["status"] == "approved"
    assert res["note"] == "lgtm"
    assert res.get("merge") is True


def test_signoff_response_maps_natural_phrases_to_a_decision() -> None:
    """The offline router reads an approve/decline reply as a sign-off decision."""
    approve = _signoff_response([HumanMessage(content="Approved — go ahead and proceed.")])
    assert approve == {"approved": True, "note": ""}
    decline = _signoff_response(
        [HumanMessage(content="Let's hold off — please don't proceed with this.")]
    )
    assert decline is not None and decline["approved"] is False
    # A plain request is NOT a response (it should launch a fresh run, not resume one).
    assert _signoff_response([HumanMessage(content="please sign off on the deploy plan")]) is None


def test_offline_host_routes_signoff_request_and_response() -> None:
    """The scripted host launches sign_off on a request and resumes it on an approve reply."""
    model = OfflineHostModel()

    launch = model._generate(  # pyright: ignore[reportPrivateUsage]
        [HumanMessage(content="I'd like to sign off on the plan before you proceed.")]
    ).generations[0].message
    assert isinstance(launch, AIMessage)
    assert launch.tool_calls[0]["name"] == "run_live"
    assert launch.tool_calls[0]["args"]["workflow"] == "sign_off"
    assert "signoff_decision" not in launch.tool_calls[0]["args"]

    approve = model._generate(  # pyright: ignore[reportPrivateUsage]
        [HumanMessage(content="Approved — go ahead and proceed.")]
    ).generations[0].message
    assert isinstance(approve, AIMessage)
    call = approve.tool_calls[0]
    assert call["name"] == "run_live"
    assert call["args"]["workflow"] == "sign_off"
    assert call["args"]["signoff_decision"] == {"approved": True, "note": ""}


async def test_signoff_pause_then_approve_across_two_host_graph_turns() -> None:
    """Two turns on one thread: turn 1 parks at the gate, turn 2 approves and proceeds.

    The honest multi-turn guard (a fresh-per-turn in-process test would mask the lane
    reuse): the real host graph accumulates messages and the module-scope resume lane
    persists, so the second turn resumes the SAME parked run. Turn 1's ui surfaces an
    awaiting ``signoff_gate``; turn 2's ui shows it resolved to approved in place.
    """
    from host_graph import _RESUME_LANES, make_host_graph

    _RESUME_LANES.clear()
    thread = "test-signoff-two-turns"
    graph = make_host_graph()

    first = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Before you run the staging deployment, I want to sign off on the "
                        "plan myself — pause for my approval before you proceed."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": thread}},
    )
    first_gates = [u for u in first.get("ui", []) if u.get("name") == _SIGNOFF_GATE]
    assert first_gates, "turn 1 must surface an awaiting sign-off card"
    assert (first_gates[-1].get("props") or {}).get("status") == "awaiting"
    # The host reports the pause, not a finish.
    first_tools = [m for m in first["messages"] if isinstance(m, ToolMessage)]
    assert any("Paused for your sign-off" in str(m.content) for m in first_tools)

    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="Approved — go ahead and proceed.")]},
        config={"configurable": {"thread_id": thread}},
    )
    second_gates = [u for u in second.get("ui", []) if u.get("name") == _SIGNOFF_GATE]
    statuses = [(u.get("props") or {}).get("status") for u in second_gates]
    assert "approved" in statuses, statuses
    second_tools = [m for m in second["messages"] if isinstance(m, ToolMessage)]
    assert any("proceeded with" in str(m.content) for m in second_tools)


def test_roster_has_no_signoff_only_leaf() -> None:
    """sign_off reuses the existing reasoning leaves; it adds no new roster role."""
    # A guard against accidental roster bloat: sign_off must resolve against the same
    # researcher/writer reasoning roles the other presets use.
    roster = make_roster()
    assert isinstance(roster, Roster)
    # researcher + writer are the leaves sign_off calls; they must be registered.
    roster.resolve("researcher")
    roster.resolve("writer")


def test_default_signoff_topic_is_used_when_args_omit_it() -> None:
    """An empty args falls back to the default sign-off topic."""
    assert DEFAULT_SIGNOFF_TOPIC == "the staging deployment plan"

"""Real-model E2E acceptance for the M4 in-run HITL sign-off path (gated).

Offline tests prove the ``sign_off`` wiring with deterministic fake leaves; this proves
the REAL thing THROUGH THE PRODUCT SURFACE — and across the real human-pause boundary.
It drives the REAL host graph over TWO turns on one thread from the exact scenario
copy, so the model — not the test — selects the tool on the request AND answers the gate
via the tool's documented ``signoff_decision`` on the approve turn. A pass that bypassed
host tool-selection (calling ``sign_off`` / resuming directly) would skip the very surface
the demo ships and would not prove the 道-prompt + tool description are self-sufficient for
the HITL resume — the project's real-E2E discipline warns against exactly that.

Turn 1 (the "Sign off mid-run" scenario message) → the host routes to ``run_live`` with
``workflow="sign_off"``; the preset assesses the plan and PAUSES at ``ctx.checkpoint``, so
the run parks and a ``signoff_gate`` (awaiting) card lands on the host ``ui`` channel and
the tool reply says it paused. Turn 2 (the SignoffGate "Approve" button's natural message)
→ the host, seeing the pending pause in the accumulated history, calls ``run_live`` again
with ``signoff_decision`` carrying the approval; the paused run resumes (the pre-gate
assessment replays from the journal for free), the card flips to ``approved`` in place, and
the run proceeds.

It asserts the headline properties an offline fake or a host-surface bypass could not
produce:

* turn 1 — a ``run_live`` call naming ``workflow="sign_off"`` (the model routed the
  request to the sign-off preset) and an ``awaiting`` ``signoff_gate`` card on the ui
  channel, with the tool reply reporting the pause;
* turn 2 — a ``run_live`` call carrying a ``signoff_decision`` (the model answered the
  gate THROUGH the documented tool param, the self-sufficiency proof), an ``approved``
  ``signoff_gate`` card on the ui channel, and the run's result reporting it proceeded.

Gating + fail-loud honesty. Skipped unless ``LDW_DEMO_REAL_MODEL`` is set (CI stays
offline). When the gate IS set, an OpenRouter key MUST be in force (the backend ``.env``)
— otherwise the host builds the offline scripted host (which routes deterministically and
would pass on a FAKE run), so the setup FAILS loudly rather than skipping. LangSmith
tracing is kept ON so the real run is captured for usage/billing.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_m4_signoff_real.py -q -s
"""

from __future__ import annotations

import os
from typing import Any

import pytest

_REAL_MODEL_GATE = "LDW_DEMO_REAL_MODEL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_REAL_MODEL_GATE),
    reason=f"{_REAL_MODEL_GATE} not set; real-model sign-off E2E is opt-in",
)

# The exact "Sign off mid-run" scenario message from scenarios.json — the copy the preset
# button sends. Driving the host graph from this proves the model routes the natural
# request to the sign_off preset (no preset name in the text), not a hand-picked call.
_SIGN_OFF_MESSAGE = (
    "Before you actually run the staging deployment, I want to sign off on the plan "
    "myself — walk me through the riskiest steps and pause for my approval before you "
    "proceed."
)
# The SignoffGate "Approve" button's exact natural-language message (provider-key path is
# irrelevant here; the words are what the host reads to answer the gate).
_APPROVE_MESSAGE = "Approved — go ahead and proceed."


def _load_backend_env() -> None:
    """Best-effort load of the backend ``.env`` so the OpenRouter + tracing vars apply."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True))


@pytest.fixture
def _real_signoff_setup() -> None:
    """Load the backend ``.env`` and fail loud unless a REAL sign-off run is possible."""
    _load_backend_env()
    # Keep LangSmith tracing ON (do NOT disable it): the real run must be billable.

    from _models import is_offline

    if is_offline():
        pytest.fail(
            f"{_REAL_MODEL_GATE} is set but no OpenRouter key is in force "
            "(set OPENROUTER_API_KEY in the backend .env). The M4 sign-off HEADLINE path "
            "must run a REAL model THROUGH the host surface across the pause/approve "
            "boundary — a fallback to the offline scripted host routes deterministically, "
            "so it cannot be accepted as proof the real model drove the HITL resume."
        )


def _signoff_gate_props(ui_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull every ``signoff_gate`` event's props off the host ``ui`` channel, in order."""
    return [
        (msg.get("props") or {}) for msg in ui_messages if msg.get("name") == "signoff_gate"
    ]


def _run_live_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Every ``run_live`` tool-call args mapping across the host messages, in order."""
    from langchain_core.messages import AIMessage

    return [
        call["args"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "run_live"
    ]


def _tool_replies(messages: list[Any]) -> list[str]:
    """Every ToolMessage's text content across the host messages, in order."""
    from langchain_core.messages import ToolMessage

    return [str(message.content) for message in messages if isinstance(message, ToolMessage)]


async def test_real_model_signoff_pauses_then_approves_through_host_graph(
    _real_signoff_setup: None,
) -> None:
    """The sign-off scenario drives the real host graph to pause, then approve and proceed.

    Two turns on one thread through ``_build_host_graph`` (a host checkpointer so turn 2
    sees the pending pause in the accumulated history — the same context ``langgraph dev``
    keeps). The real model must select ``sign_off`` on turn 1 and answer the gate via
    ``signoff_decision`` on turn 2; the ui channel and the tool replies carry the proof.
    """
    from host_graph import _RESUME_LANES, _build_host_graph
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    _RESUME_LANES.clear()
    graph = _build_host_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-m4-signoff-real-host"}}

    # ── Turn 1: the request must route to sign_off and PARK at the gate. ──
    turn1 = await graph.ainvoke(
        {"messages": [HumanMessage(content=_SIGN_OFF_MESSAGE)]}, config=config
    )
    turn1_live = _run_live_calls(turn1["messages"])
    assert any(args.get("workflow") == "sign_off" for args in turn1_live), (
        "the host model must route the sign-off request to the sign_off preset; "
        f"run_live calls saw: {turn1_live}"
    )
    turn1_gates = _signoff_gate_props(turn1.get("ui", []))
    assert any(props.get("status") == "awaiting" for props in turn1_gates), (
        f"turn 1 must surface an awaiting signoff_gate card; gates: {turn1_gates}"
    )
    assert any("Paused for your sign-off" in reply for reply in _tool_replies(turn1["messages"])), (
        f"turn 1's tool reply must report the pause; replies: {_tool_replies(turn1['messages'])}"
    )

    # ── Turn 2: the approval must route back through run_live with a signoff_decision. ──
    turn2 = await graph.ainvoke(
        {"messages": [HumanMessage(content=_APPROVE_MESSAGE)]}, config=config
    )
    turn2_live = _run_live_calls(turn2["messages"])
    # Headline self-sufficiency proof: the REAL model answered the gate THROUGH the
    # documented tool param — not by re-launching fresh, not by replying in prose.
    signoff_resumes = [
        args
        for args in turn2_live
        if args.get("workflow") == "sign_off" and args.get("signoff_decision") is not None
    ]
    assert signoff_resumes, (
        "the host model must answer the pending gate by calling run_live with a "
        f"signoff_decision (the documented HITL-resume param); run_live calls: {turn2_live}"
    )
    decisions = [args["signoff_decision"] for args in signoff_resumes]
    assert any(bool((d or {}).get("approved")) for d in decisions), (
        f"the approval must carry approved=true; decisions: {decisions}"
    )

    # The card flipped to approved in place on the ui channel, and the run proceeded.
    turn2_gates = _signoff_gate_props(turn2.get("ui", []))
    assert any(props.get("status") == "approved" for props in turn2_gates), (
        f"turn 2 must resolve the signoff_gate card to approved; gates: {turn2_gates}"
    )
    turn2_replies = _tool_replies(turn2["messages"])
    assert any("proceeded with" in reply for reply in turn2_replies), (
        f"turn 2's result must report the run proceeded; replies: {turn2_replies}"
    )

"""Real-model E2E acceptance for the M3 transport HEADLINE path (gated, opt-in).

Offline tests prove the wiring with deterministic fake leaves; this proves the REAL
thing THROUGH THE PRODUCT SURFACE, across the detached-run / live-turn boundary that is
the whole point of the transport. It drives the REAL host graph over TWO turns on one
thread, from natural scenario copy, so the model — not the test — routes each request:

Turn 1 (the "a few at once" board message) → the host routes to ``run_runs_board``, which
fans out THREE independent ``deep_research`` runs (real sonnet leaves) as DETACHED tasks
and surfaces their aggregate status as one ``run_board`` card. Each detached run's interior
runtime events are buffered on its slot by the transport sinks. The board's final tool
reply now carries a per-run report (M3②) — each run's label, status, and capped result
substance — so the host has real content to relay, not just "3 finished".

Turn 2 (a natural "show me what the RAG one is doing inside" message) → the host routes to
``drill_run`` with a target naming the RAG run; the tool replays that detached run's
buffered events through a fresh adapter, so the SAME card vocabulary an inline run streams
(``agent_span`` / ``fanout_graph`` / ``phase_timeline``) reaches the host ``ui`` channel —
the detached run is no longer UI-dark.

It asserts the headline properties an offline fake or an aggregate-only board could not
produce:

* turn 1 — a ``run_runs_board`` call; a ``run_board`` card with THREE rows all settled to
  ``done``; and a tool reply carrying per-run result substance (a run's label appears in
  the report, beyond the aggregate count line);
* turn 2 — a ``drill_run`` call; and at least one ``agent_span`` card on the ui channel
  (the drilled detached run's real per-leaf interior, replayed from the buffer — a property
  the board's aggregate rows never carry).

Gating + fail-loud honesty. Skipped unless ``LDW_DEMO_REAL_MODEL`` is set (CI stays
offline). When the gate IS set, an OpenRouter key MUST be in force (the backend ``.env``)
— otherwise the host builds the offline scripted host (which routes deterministically and
would pass on a FAKE run), so the setup FAILS loudly rather than skipping. LangSmith
tracing is kept ON so the real run is captured for usage/billing.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_m3_transport_real.py -q -s
"""

from __future__ import annotations

import os
from typing import Any

import pytest

_REAL_MODEL_GATE = "LDW_DEMO_REAL_MODEL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_REAL_MODEL_GATE),
    reason=f"{_REAL_MODEL_GATE} not set; real-model M3 transport E2E is opt-in",
)

# The user's own words for "fan these out together" — the same board scenario copy the
# M3.5 acceptance uses. No tool mechanics: the real host must infer the board route from
# the 道-level prompt + the run_runs_board tool description alone.
_BOARD_MESSAGE = (
    "I've got a few separate things I want looked into at the same time — the trade-offs "
    "of retrieval-augmented generation, the state of long-context models, and how agent "
    "frameworks compare. Start all three off together and keep me posted on how each "
    "one's going; I don't want to wait on them one at a time."
)
# A natural drill request naming the RAG run by a fragment of its board label
# ("RAG vs long-context"). The host must route to drill_run and pass a target the
# resolver matches by unique case-insensitive substring.
_DRILL_MESSAGE = (
    "Can you show me what the RAG one is actually doing inside — its internal steps and "
    "how its sub-agents are progressing? I'd like to look into that run specifically."
)


def _load_backend_env() -> None:
    """Best-effort load of the backend ``.env`` so OpenRouter + tracing vars apply."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True))


@pytest.fixture
def _real_transport_setup() -> None:
    """Load the backend ``.env`` and fail loud unless a REAL run is possible.

    Also clears the module-level background-run manager so a prior test's runs cannot
    pollute this thread's board or make a drill target ambiguous. Tracing stays ON.
    """
    _load_backend_env()

    from _models import is_offline

    if is_offline():
        pytest.fail(
            f"{_REAL_MODEL_GATE} is set but no OpenRouter key is in force "
            "(set OPENROUTER_API_KEY in the backend .env). The M3 transport HEADLINE path "
            "must run a REAL model THROUGH the host surface across the detached-run / "
            "live-turn boundary — a fallback to the offline scripted host routes "
            "deterministically, so it cannot be accepted as proof."
        )

    from host_graph import _BG_MANAGER

    _BG_MANAGER._slots.clear()  # pyright: ignore[reportPrivateUsage] - isolate this thread's runs
    _BG_MANAGER._notices.clear()  # pyright: ignore[reportPrivateUsage]


def _tool_replies(messages: list[Any]) -> list[str]:
    """Every ToolMessage's text content across the host messages, in order."""
    from langchain_core.messages import ToolMessage

    return [str(message.content) for message in messages if isinstance(message, ToolMessage)]


def _ui_props(ui_messages: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """Pull every ``name`` UI event's props off the host ``ui`` channel, in order."""
    return [(msg.get("props") or {}) for msg in ui_messages if msg.get("name") == name]


async def test_real_model_board_fanout_then_drill_replays_interior(
    _real_transport_setup: None,
) -> None:
    """Board fans out three real runs; a drill replays one run's real interior to the ui.

    Two turns on one thread through ``_build_host_graph`` (a host checkpointer so turn 2
    sees the board + its runs in the accumulated history). Turn 1: the real model routes
    to ``run_runs_board``, three ``deep_research`` runs settle ``done``, and the tool reply
    carries per-run substance. Turn 2: the model routes to ``drill_run`` for the RAG run,
    and its detached interior (``agent_span`` cards) reaches the ui channel — the headline
    proof that a detached run's events flowed back through the transport.
    """
    from host_graph import _build_host_graph
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    graph = _build_host_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-m3-transport-real"}}

    # ── Turn 1: board fan-out — three real runs settle done, report carries substance. ──
    turn1 = await graph.ainvoke({"messages": [HumanMessage(content=_BOARD_MESSAGE)]}, config=config)

    # The run_board card is the route proof: only run_runs_board emits it, so its presence
    # means the real host routed the "a few at once" request to the board tool (asserted
    # through the product surface, not via fragile AIMessage tool-call introspection).
    boards = _ui_props(turn1.get("ui", []), "run_board")
    assert boards, (
        "no run_board card reached the ui channel — the host did not route the 'a few at "
        f"once' request to run_runs_board; tool replies were: {_tool_replies(turn1['messages'])}"
    )
    rows: list[dict[str, Any]] = boards[-1].get("runs") or []
    assert len(rows) == 3, f"expected three fanned-out runs, got {len(rows)}: {rows}"
    statuses = [r.get("status") for r in rows]
    assert all(s == "done" for s in statuses), f"every run must settle to done, got {statuses}"

    # M3②: the board's final reply carries per-run substance, not just an aggregate count.
    board_replies = _tool_replies(turn1["messages"])
    labels = [str(r.get("label") or "") for r in rows]
    assert any(label and label in reply for reply in board_replies for label in labels), (
        "the board's final tool reply must carry per-run substance (a run label appears), "
        f"so the host can relay results; replies: {board_replies}"
    )

    # ── Turn 2: drill into the RAG run — its real interior replays onto the ui channel. ──
    turn2 = await graph.ainvoke({"messages": [HumanMessage(content=_DRILL_MESSAGE)]}, config=config)

    # Route proof, through the product surface: drill_run_live's ToolMessage names the
    # replayed event count ("...replayed N events") — drill_run is the only tool that emits
    # it, and a real model cannot fabricate it, so its presence proves the host routed the
    # "show me what the RAG one is doing" request to drill_run and the replay ran.
    drill_replies = [r for r in _tool_replies(turn2["messages"]) if "replayed" in r]
    assert drill_replies, (
        "the host model must route the drill request to drill_run (its reply names the "
        f"replayed events); tool replies were: {_tool_replies(turn2['messages'])}"
    )

    # Headline: the detached run is no longer UI-dark. A drill replay emits the per-leaf
    # interior (agent_span) the board's aggregate rows never carry — so its presence on the
    # final ui channel proves the buffered interior of a detached run flowed back and
    # rendered. (The board route emits only run_board/run_status, never agent_span.)
    agent_spans = _ui_props(turn2.get("ui", []), "agent_span")
    assert agent_spans, (
        "turn 2 must surface at least one agent_span card from the drill replay — the "
        f"drilled detached run's real per-leaf interior; ui names: "
        f"{[m.get('name') for m in turn2.get('ui', [])]}"
    )

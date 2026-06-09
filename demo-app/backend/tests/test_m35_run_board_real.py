"""Real-model E2E acceptance for the M3.5 run board HEADLINE path (gated, opt-in).

Offline tests prove the wiring; this proves the real thing. It drives the FULL host graph
with the "a few at once" scenario message against a real OpenRouter model, so a real opus
host must — from the 道-level prompt + the ``run_runs_board`` tool description + the skill
alone — route the request to the board tool, which fans out THREE independent
``deep_research`` runs (real sonnet leaves) and surfaces their aggregate status as one
``run_board`` card. It asserts the headline properties a single-run or fallback path could
not fake:

* a ``run_board`` card reached the host ``ui`` channel (the host genuinely routed the
  request to the board tool, rather than a single background run or an inline run);
* the board carries THREE rows — the host fanned out three independent runs;
* every row settled to ``done`` (the real runs completed); and
* at least one row carries a real synthesized summary (a real model produced a report).

Gating. The test is skipped unless ``LDW_DEMO_REAL_MODEL`` is set. When it IS set, an
OpenRouter key MUST be in force (``.env`` ``OPENROUTER_API_KEY``) — otherwise the run would
silently fall back to the deterministic offline roster and the assertions would pass on a
FAKE run, defeating the acceptance. So with the gate on and no key, the test FAILS loudly
rather than skipping or passing on a fallback. LangSmith tracing is LEFT ON: the user needs
traces for billing, so a real-model run must not silence telemetry.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_m35_run_board_real.py -q -s
"""

from __future__ import annotations

import os

import pytest

# The opt-in gate. Other repo examples use this same env var as the real-model switch.
_REAL_MODEL_GATE = "LDW_DEMO_REAL_MODEL"


def _is_offline() -> bool:
    """Re-resolve the demo's offline state at call time (after env setup)."""
    from _models import is_offline

    return is_offline()


@pytest.fixture(autouse=True)
def _real_model_setup() -> None:
    """Gate on the real-model flag; require a real key (no silent offline fallback).

    Skips the module unless ``LDW_DEMO_REAL_MODEL`` is set. When it IS set, a real
    OpenRouter key MUST be present — a missing key would silently route to the offline fake
    roster and let the assertions pass on a fake run, so that case FAILS loudly rather than
    skipping. Tracing is deliberately LEFT ON: the user needs LangSmith traces for billing,
    so a real-model run must not silence telemetry.
    """
    if not os.environ.get(_REAL_MODEL_GATE):
        pytest.skip(f"{_REAL_MODEL_GATE} not set; real-model E2E is opt-in")

    if _is_offline():
        pytest.fail(
            f"{_REAL_MODEL_GATE} is set but no OpenRouter key is in force "
            "(set OPENROUTER_API_KEY in the backend .env). The HEADLINE path must run "
            "against a real model — a fallback to the offline roster cannot be accepted."
        )


# The user's own words: several independent investigations, run together, keep me posted.
# No tool mechanics — the real host must infer "fan these out as separate runs and track
# them on a board" from the 道-level prompt + the run_runs_board tool description alone.
_BOARD_MESSAGE = (
    "I've got a few separate things I want looked into at the same time — the trade-offs "
    "of retrieval-augmented generation, the state of long-context models, and how agent "
    "frameworks compare. Start all three off together and keep me posted on how each "
    "one's going; I don't want to wait on them one at a time."
)


async def test_run_board_real_model_fans_out_three_runs_and_settles() -> None:
    """A real opus host fans the request out into three runs and settles the board.

    Drives the full host graph with the "a few at once" message; the real host must route
    to ``run_runs_board``, which launches three independent ``deep_research`` runs (real
    leaves) and emits a ``run_board`` whose three rows all settle to ``done`` with real
    synthesized summaries — properties a single-run or fallback path could not produce.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content=_BOARD_MESSAGE)]},
        config={"configurable": {"thread_id": "m35-real-run-board"}},
    )

    boards = [u for u in out.get("ui", []) if u.get("name") == "run_board"]
    assert boards, (
        "no run_board reached the ui channel — the real host did not route the "
        "'a few at once' request to the board tool (check the 道-level prompt + tool desc)"
    )

    # The board re-emits under one fixed id, so the reducer keeps the latest (settled) card.
    final = boards[-1].get("props") or {}
    rows = final.get("runs") or []
    assert len(rows) == 3, f"expected three fanned-out runs on the board, got {len(rows)}: {rows}"

    statuses = [r.get("status") for r in rows]
    assert all(s == "done" for s in statuses), f"every run must settle to done, got {statuses}"

    assert any((r.get("summary") or "").strip() for r in rows), (
        "at least one settled row must carry a real synthesized summary"
    )

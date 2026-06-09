"""Backend checks for the M3.5 aggregate run board (``run_runs_board`` / ``run_board``).

The board fans out several background runs and surfaces their aggregate status as a single
host-emitted ``run_board`` card, re-emitted in place (a fixed ``event_id``) each poll until
every run settles. These tests pin the mechanism: the live loop fans out the jobs, emits a
board carrying one row per run under a stable id, settles all rows, and summarizes how many
ran — and the offline host routes the "a few at once" scenario to the board tool. They run
through the real tool/offline layers where that applies (no model keys; the scripted host).
"""

from __future__ import annotations

import os
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the host stays on the offline path."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)


_JOBS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("RAG vs long-context", {"question": "What are the trade-offs of RAG vs long-context?"}),
    ("Long-context models", {"question": "What is the state of long-context LLMs?"}),
    ("Agent frameworks", {"question": "How do agent frameworks compare?"}),
)


async def test_run_board_live_emits_one_row_per_run_with_stable_id() -> None:
    """``run_runs_board_live`` fans out the jobs and emits a board that settles all rows.

    Each poll re-emits the SAME ``event_id`` ("run-board-1") so the SDK upserts one card in
    place; the final emit carries one row per launched run, each named by its label and
    settled to ``done`` with an outcome summary. The returned text names how many ran.
    """
    from host_graph import run_runs_board_live

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    emitted: list[tuple[str, dict[str, Any]]] = []

    def fake_emit(component: str, props: dict[str, Any]) -> None:
        emitted.append((component, props))

    summary = await run_runs_board_live(manager, fake_emit, thread_id="board-live", jobs=_JOBS)

    board_emits = [props for name, props in emitted if name == "run_board"]
    assert board_emits, "the board tool must emit at least one run_board card"
    # Every emit upserts the one card in place (a stable id), never stacks a new card.
    assert all(p["event_id"] == "run-board-1" for p in board_emits), board_emits

    final = board_emits[-1]
    rows = final["runs"]
    assert len(rows) == 3, rows
    assert {r["label"] for r in rows} == {label for label, _ in _JOBS}
    assert all(r["status"] == "done" for r in rows), rows
    assert all(r["summary"] for r in rows), "a settled row carries an outcome summary"

    assert isinstance(summary, str) and "3" in summary, summary


async def test_run_board_live_surfaces_quota_cap_honestly() -> None:
    """A concurrency quota caps the fan-out: launch what fits, show only those, say so.

    With ``max_concurrent_runs`` below the job count the manager admits only that many runs.
    The board must stop launching at the cap (never silently drop the rest), render only the
    admitted rows, and the summary must name "launched X of N" honestly rather than implying
    all ran.
    """
    from host_graph import run_runs_board_live

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager(max_concurrent_runs=2)
    emitted: list[tuple[str, dict[str, Any]]] = []

    summary = await run_runs_board_live(
        manager,
        lambda component, props: emitted.append((component, props)),
        thread_id="board-quota",
        jobs=_JOBS,
    )

    final = [props for name, props in emitted if name == "run_board"][-1]
    assert len(final["runs"]) == 2, "only the admitted runs appear on the board"
    assert "2 of 3" in summary, summary


async def test_run_runs_board_tool_layer_emits_board_and_reports() -> None:
    """The "a few at once" message drives ``run_runs_board`` through the real graph layer.

    Runs the full offline host graph the way ``langgraph dev`` invokes it: the offline host
    routes the parallel-runs message to the board tool, which fans out the runs and emits a
    ``run_board`` card into the host ``ui`` channel, then returns a summary naming the jobs.
    This pins the tool registration + the host-emit path end to end (not just the engine
    core ``run_runs_board_live``).
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I've got a few separate things I want looked into at the same "
                        "time — RAG trade-offs, long-context models, and how agent "
                        "frameworks compare. Start all three off together and keep me "
                        "posted; I don't want to wait on them one at a time."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "test-board-tool-layer"}},
    )

    components = {u.get("name") for u in out.get("ui", [])}
    assert "run_board" in components, components

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "run_runs_board did not execute"
    summary = "\n".join(str(m.content) for m in tool_messages)
    assert "jobs" in summary.lower(), summary


def _offline_first_tool_call(prompt: str) -> tuple[str, dict[str, Any]]:
    """Drive the offline host one turn on ``prompt``; return the tool name and its args."""
    from _models import OfflineHostModel
    from langchain_core.messages import HumanMessage

    result = OfflineHostModel()._generate([HumanMessage(content=prompt)])
    message = result.generations[0].message
    call = message.tool_calls[0]  # type: ignore[attr-defined]
    return call["name"], call["args"]


def test_offline_host_routes_a_few_at_once_to_run_runs_board() -> None:
    """A "several independent jobs, all at once, keep me posted" cue routes to the board tool.

    The message names several independent investigations and asks to run them together — a
    parallel-run-board intent. The offline host must route it to ``run_runs_board`` rather
    than ``run_background`` (single detached run) or ``run_live``, even though it also reads
    as "heavy work": the parallel-runs cue takes precedence.
    """
    name, _args = _offline_first_tool_call(
        "I've got a few separate things I want looked into at the same time — the "
        "trade-offs of retrieval-augmented generation, the state of long-context models, "
        "and how agent frameworks compare. Start all three off together and keep me posted."
    )
    assert name == "run_runs_board", name


async def test_run_board_live_scopes_to_its_own_runs_not_the_thread() -> None:
    """The board tracks only the runs IT launched, not pre-existing runs on the thread.

    A prior background run on the same thread (an earlier turn's run) must NOT appear on
    this board or skew its summary; scoping to the launched run ids also keeps a prior
    still-RUNNING run from holding the poll loop open until the cap.
    """
    from host_graph import launch_background_run, run_runs_board_live

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    # A pre-existing settled run on the SAME thread (from an earlier turn).
    prior = launch_background_run(
        manager, thread_id="board-scope", workflow="deep_research", label="prior turn"
    )
    await manager.wait(prior, thread_id="board-scope")

    emitted: list[tuple[str, dict[str, Any]]] = []
    summary = await run_runs_board_live(
        manager,
        lambda component, props: emitted.append((component, props)),
        thread_id="board-scope",
        jobs=_JOBS,
    )

    final = [props for name, props in emitted if name == "run_board"][-1]
    labels = {row["label"] for row in final["runs"]}
    assert len(final["runs"]) == 3, f"board must show only its own 3 runs: {labels}"
    assert labels == {label for label, _ in _JOBS}, (
        f"prior-turn run leaked onto the board: {labels}"
    )
    assert "3 of 3" in summary and "4" not in summary, summary


def test_offline_host_does_not_route_generic_progress_phrase_to_board() -> None:
    """A generic "keep me posted" without the parallel-runs intent must NOT hit the board.

    The board cue must be specific to "several independent jobs at once", not a generic
    progress-update phrase — otherwise an unrelated request would launch three fabricated
    research runs.
    """
    name, _args = _offline_first_tool_call(
        "Please fix this failing module and keep me posted on how it goes."
    )
    assert name != "run_runs_board", (
        f"a generic progress phrase must not route to the board, got {name}"
    )


def test_offline_post_tool_reply_for_board_is_honest_not_streamed() -> None:
    """The offline final reply after the board tool must not claim it 'streamed progress'.

    The board is a UI-dark aggregate (each run's settled status, not a live per-leaf
    stream), so the canned streamed-into-the-panel wording would be dishonest.
    """
    from _models import _RUNS_BOARD_TOOL_NAME, _post_tool_reply
    from langchain_core.messages import ToolMessage

    reply = _post_tool_reply(
        ToolMessage(
            content="Ran 3 of 3 jobs together: 3 finished.",
            name=_RUNS_BOARD_TOOL_NAME,
            tool_call_id="board-call-1",
        )
    )
    assert "stream" not in reply.lower(), f"board reply must not imply live streaming: {reply!r}"

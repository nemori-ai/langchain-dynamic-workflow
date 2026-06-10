"""Backend checks for the M3 background-run transport (drill-in + report-back).

A detached background run used to be event-dark: ``launch_background_run`` wired no
sinks, so nothing about the run's interior ever reached the host. These tests pin the
M3 transport mechanism: the launcher wires the manager's buffer sinks so a settled
run's slot holds replayable span events. They run offline (no model keys; the fake
roster leaves) against the real launcher and manager.
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


# The preset the offline tests launch: the same fan-out-heavy deep-research scenario the
# background path defaults to, served by the deterministic fake leaves.
_OFFLINE_PRESET = "deep_research"


async def test_launch_background_run_wires_event_sinks() -> None:
    """After launch + settle, the run's buffer holds span events — no longer event-dark.

    ``launch_background_run`` must pre-mint the run id, build the manager's buffer
    sinks for it, and thread them into the detached ``run_workflow`` — so the slot's
    bounded buffer captures the run's span edges for a later drill-in replay.
    """
    from host_graph import launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    assert dropped == 0
    assert any(e.kind == "span" for e in events), [e.kind for e in events]


async def test_drill_run_live_replays_buffered_events_as_cards() -> None:
    """A drill replays the run's buffer as the same card vocabulary an inline run shows.

    Launch + settle an offline background run, then drill: the captured emits must
    contain ``agent_span`` (the per-leaf interior, not an aggregate row), keyed by the
    engine-minted stable span_id, and the summary must name the run.
    """
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    captured: list[tuple[str, dict[str, Any]]] = []
    summary = await drill_run_live(
        manager,
        lambda name, props: captured.append((name, props)),
        thread_id="t1",
        target=run_id,
    )
    names = {name for name, _ in captured}
    assert "agent_span" in names, names
    assert run_id in summary, summary


async def test_drill_run_live_is_idempotent_across_ticks() -> None:
    """Two full replays of the same settled run yield the same set of event ids.

    The engine-minted stable span_id makes a replay upsert in place at the SDK reducer
    rather than stacking duplicate cards — so a poll loop re-replaying each tick (or a
    second drill) is idempotent.
    """
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    def _ids(emits: list[tuple[str, dict[str, Any]]]) -> set[str]:
        return {p["event_id"] for _, p in emits if "event_id" in p}

    first: list[tuple[str, dict[str, Any]]] = []
    await drill_run_live(manager, lambda n, p: first.append((n, p)), thread_id="t1", target=run_id)
    second: list[tuple[str, dict[str, Any]]] = []
    await drill_run_live(manager, lambda n, p: second.append((n, p)), thread_id="t1", target=run_id)
    assert _ids(first) == _ids(second) != set()


async def test_drill_run_live_resolves_label_and_reports_unknown_target() -> None:
    """A drill target resolves by label too, and an unknown target gets an honest reply."""
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(
        manager, thread_id="t1", workflow=_OFFLINE_PRESET, label="RAG study"
    )
    await manager.wait(run_id, thread_id="t1")

    summary = await drill_run_live(manager, lambda n, p: None, thread_id="t1", target="RAG study")
    assert run_id in summary, summary

    missing = await drill_run_live(manager, lambda n, p: None, thread_id="t1", target="ghost")
    assert "no background run" in missing.lower(), missing
    assert "RAG study" in missing, missing  # honest: lists what IS available


async def test_drill_run_live_surfaces_dropped_as_truncation_log() -> None:
    """Past the buffer cap, the drill surfaces the dropped count as a timeline log line."""
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager(max_buffered_events=1)
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    captured: list[tuple[str, dict[str, Any]]] = []
    await drill_run_live(
        manager, lambda n, p: captured.append((n, p)), thread_id="t1", target=run_id
    )
    timeline_msgs = [p.get("message", "") for n, p in captured if n == "phase_timeline"]
    assert any("dropped" in m for m in timeline_msgs), timeline_msgs


def test_drill_run_tool_schema_has_target_not_mangled() -> None:
    """The ``drill_run`` tool exists with a ``target`` parameter — checked through the
    tool layer, not the bare function (the ``@tool`` wrapper is what the model sees,
    and LangChain mangles a parameter literally named ``args`` into ``v__args``)."""
    from host_graph import drill_run

    assert drill_run.name == "drill_run"
    fields = set(drill_run.get_input_schema().model_fields)
    assert "target" in fields, sorted(fields)
    assert "v__args" not in fields, sorted(fields)


def _offline_first_tool_call(prompt: str) -> tuple[str, dict[str, Any]]:
    """Drive the offline host one turn on ``prompt``; return the tool name and its args."""
    from _models import OfflineHostModel
    from langchain_core.messages import HumanMessage

    result = OfflineHostModel()._generate([HumanMessage(content=prompt)])
    message = result.generations[0].message
    call = message.tool_calls[0]  # type: ignore[attr-defined]
    return call["name"], call["args"]


def test_offline_host_routes_drill_cue_to_drill_run_with_target() -> None:
    """A "drill into <run>" message routes to ``drill_run`` carrying the named target.

    The target must keep its original casing (run labels are case-sensitive), so the
    scripted host extracts it from the raw message text after the cue phrase.
    """
    name, args = _offline_first_tool_call("Drill into RAG vs long-context")
    assert name == "drill_run", name
    assert args["target"] == "RAG vs long-context", args


def test_offline_host_does_not_route_generic_look_phrase_to_drill() -> None:
    """A generic "look at this" must NOT hit the drill tool — the cue stays narrow."""
    name, _args = _offline_first_tool_call(
        "Have a look at this module and tell me what it is doing."
    )
    assert name != "drill_run", name


async def test_drill_run_tool_layer_executes_and_reports_honestly() -> None:
    """The drill message drives ``drill_run`` through the real graph layer.

    Runs the full offline host graph the way ``langgraph dev`` invokes it: the offline
    host routes the drill message to the registered ``drill_run`` tool. With no
    background run on this fresh thread the tool must reply honestly that nothing
    matches — proving registration + routing + execution end to end.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="Drill into the RAG run")]},
        config={"configurable": {"thread_id": "test-drill-tool-layer"}},
    )

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "drill_run did not execute"
    replies = "\n".join(str(m.content) for m in tool_messages)
    assert "no background run" in replies.lower(), replies


def test_offline_post_tool_reply_for_drill_is_honest() -> None:
    """The offline final reply after a drill must describe a replay, not a live stream."""
    from _models import _DRILL_TOOL_NAME, _post_tool_reply
    from langchain_core.messages import ToolMessage

    reply = _post_tool_reply(
        ToolMessage(
            content="drilled into run abc (RAG study): status=done, replayed 12 events",
            name=_DRILL_TOOL_NAME,
            tool_call_id="drill-call-1",
        )
    )
    assert "drill" in reply.lower(), reply

    no_match = _post_tool_reply(
        ToolMessage(
            content="no background run matches 'ghost'; available runs: (none on this thread)",
            name=_DRILL_TOOL_NAME,
            tool_call_id="drill-call-2",
        )
    )
    assert "couldn't find" in no_match.lower(), no_match


async def test_board_final_report_carries_per_run_results() -> None:
    """The board's return value carries one line per run with label, status, and substance.

    After all runs settle, the host (and the user) must get each run's actual outcome
    fed back — not just an aggregate "2 finished" count.
    """
    from host_graph import run_runs_board_live

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    jobs: list[tuple[str, dict[str, Any]]] = [("alpha", {}), ("beta", {})]
    report = await run_runs_board_live(
        manager,
        lambda n, p: None,
        thread_id="t1",
        jobs=jobs,
        workflow=_OFFLINE_PRESET,
    )
    assert "alpha" in report and "beta" in report, report
    # Substance, not just a count: each line carries the run's status + result text.
    assert report.count("status=done") == 2, report


async def test_large_result_reports_summary_plus_handle() -> None:
    """An offloaded large result surfaces its ``result://`` handle in the board report.

    With a tiny ``inline_max_chars`` the store offloads the run's result; the report
    must name the opaque handle so ``fetch_run_result`` can pull the full payload.
    """
    from host_graph import run_runs_board_live

    from langchain_dynamic_workflow import BgRunManager, ResultStore

    store = ResultStore(inline_max_chars=10)
    manager = BgRunManager(result_store=store)
    report = await run_runs_board_live(
        manager,
        lambda n, p: None,
        thread_id="t1",
        jobs=[("big", {})],
        workflow=_OFFLINE_PRESET,
    )
    assert "result://" in report, report


async def test_fetch_run_result_returns_full_payload() -> None:
    """``fetch_run_result_live`` returns the un-capped payload by handle or by run id."""
    from host_graph import fetch_run_result_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager, ResultStore

    store = ResultStore(inline_max_chars=10)
    manager = BgRunManager(result_store=store)
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    result = manager.get_result(run_id, thread_id="t1")
    assert result.handle is not None

    full = await fetch_run_result_live(manager, thread_id="t1", target=result.handle)
    assert len(full) > 10, full  # the un-capped payload, not the 10-char summary
    by_run = await fetch_run_result_live(manager, thread_id="t1", target=run_id)
    assert by_run == full


async def test_fetch_run_result_tool_schema_has_target_not_mangled() -> None:
    """The ``fetch_run_result`` tool exists with a ``target`` parameter (no v__args)."""
    from host_graph import fetch_run_result

    assert fetch_run_result.name == "fetch_run_result"
    fields = set(fetch_run_result.get_input_schema().model_fields)
    assert "target" in fields, sorted(fields)
    assert "v__args" not in fields, sorted(fields)


async def _two_labelled_runs(manager: Any, labels: tuple[str, str]) -> None:
    """Launch + settle two background runs with the given display labels."""
    from host_graph import launch_background_run

    for label in labels:
        run_id = launch_background_run(
            manager, thread_id="t1", workflow=_OFFLINE_PRESET, label=label
        )
        await manager.wait(run_id, thread_id="t1")


async def test_drill_resolves_unique_case_insensitive_label_substring() -> None:
    """A colloquial target ("RAG") drills the one run whose label contains it.

    A real host turns "look into the RAG one" into ``drill_run(target="RAG")`` — a
    fragment of the board label "RAG vs long-context", not its exact text. Resolution
    must fall back to a unique, case-insensitive label substring so the drill lands on
    the intended run instead of failing on an inexact label.
    """
    from host_graph import drill_run_live

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    await _two_labelled_runs(manager, ("RAG vs long-context", "Agent frameworks"))

    summary = await drill_run_live(manager, lambda name, props: None, thread_id="t1", target="rag")
    assert "RAG vs long-context" in summary, summary
    assert "no background run matches" not in summary, summary


async def test_drill_ambiguous_label_substring_resolves_to_none() -> None:
    """An ambiguous substring matching two labels refuses (honest), never guesses.

    "context" is a substring of both labels; with no unique match the drill must
    report no match (and list what is available) rather than silently picking one.
    """
    from host_graph import drill_run_live

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    await _two_labelled_runs(manager, ("RAG vs long-context", "Long-context models"))

    summary = await drill_run_live(
        manager, lambda name, props: None, thread_id="t1", target="context"
    )
    assert "no background run matches" in summary, summary


async def test_board_then_drill_multiturn_through_host_graph() -> None:
    """Offline multi-turn THROUGH the real host graph: board fan-out, then drill replay.

    Direct-call drill tests bypass host tool-routing and turn accumulation; this drives
    ``_build_host_graph`` over two turns on one thread (a checkpointer accumulates history
    the way ``langgraph dev`` does) so turn B's drill sees turn A's runs on the shared
    background manager. It pins the headline through the product surface: turn A routes to
    ``run_runs_board`` (a ``run_board`` card lands), and turn B's drill replays the picked
    run's per-leaf interior — an ``agent_span`` on the accumulated ui channel, a card the
    board's aggregate rows never carry. Offline (no keys) so it is deterministic and CI-safe.
    """
    from host_graph import _BG_MANAGER, _build_host_graph
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    _BG_MANAGER._slots.clear()  # pyright: ignore[reportPrivateUsage] - isolate this thread's runs
    _BG_MANAGER._notices.clear()  # pyright: ignore[reportPrivateUsage]
    graph = _build_host_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-drill-multiturn"}}

    board_msg = (
        "I've got a few separate things I want looked into at the same time — start all "
        "three off together and keep me posted on how each one is going."
    )
    turn_a = await graph.ainvoke({"messages": [HumanMessage(content=board_msg)]}, config=config)
    boards = [u for u in turn_a.get("ui", []) if u.get("name") == "run_board"]
    assert boards, [u.get("name") for u in turn_a.get("ui", [])]

    # The offline scripted host takes the text AFTER "drill into" as the target verbatim,
    # so it must be the clean run label (the scripted NLU is naive — a real model extracts
    # the label from richer phrasing). The point here is the multi-turn + replay mechanism.
    drill_msg = "Drill into RAG vs long-context"
    turn_b = await graph.ainvoke({"messages": [HumanMessage(content=drill_msg)]}, config=config)

    # drill_run_live's ToolMessage names the replayed event count — proof the drill ran and
    # replayed the detached run's buffered interior (offline _REPLY_DRILL never carries it).
    drill_replies = [
        str(m.content)
        for m in turn_b["messages"]
        if type(m).__name__ == "ToolMessage" and "replayed" in str(m.content)
    ]
    assert drill_replies, "the drill tool must have run (its reply names the replayed events)"

    # Headline: the drilled detached run's per-leaf interior reached the ui channel — an
    # agent_span the board's aggregate rows never carry.
    agent_spans = [u for u in turn_b.get("ui", []) if u.get("name") == "agent_span"]
    assert agent_spans, [u.get("name") for u in turn_b.get("ui", [])]

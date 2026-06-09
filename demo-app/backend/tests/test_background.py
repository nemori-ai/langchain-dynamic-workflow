"""Backend checks for the background-run host tool (``run_background``) and its routing.

HONEST SCOPE of the demo's background mode: ``push_ui_message`` requires the host NODE
context (a captured contextvar). A :meth:`BgRunManager.start` run executes in a DETACHED
asyncio task that does NOT carry that context, so it cannot push to the host ``ui``
channel. The background scenario therefore surfaces lifecycle STATUS plus the final
result (launch -> pending/running -> done -> result), NOT live phase/fan-out streaming.
These tests pin that mechanism: a run launches and returns immediately, its status moves
through the lifecycle, and the result is fetchable once done — exercised through the real
tool layer where it applies.
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


async def test_background_lifecycle_launch_status_settle_result() -> None:
    """A background run launches immediately, moves through the lifecycle, then yields a result.

    Drives the mechanism end to end against a fresh manager: ``launch_background_run``
    returns a run_id without blocking on the work, the manager reports a live (pending
    or running) status before settling, then settles to ``done`` and the final result is
    fetchable. This is the honest background story — lifecycle + final result, not live
    per-leaf streaming from a detached task.
    """
    from host_graph import launch_background_run

    from langchain_dynamic_workflow import BgRunManager, BgStatus

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="bg-test", workflow="deep_research")
    assert isinstance(run_id, str) and run_id

    # Right after launch the work has not finished (it runs in a detached task).
    assert manager.poll(run_id, thread_id="bg-test") in {BgStatus.PENDING, BgStatus.RUNNING}

    await manager.wait(run_id, thread_id="bg-test")
    assert manager.poll(run_id, thread_id="bg-test") is BgStatus.DONE

    result = manager.get_result(run_id, thread_id="bg-test")
    assert result.status is BgStatus.DONE
    payload = result.value if result.value is not None else result.summary
    assert isinstance(payload, str) and payload.strip(), "the settled run must carry a result"


async def test_run_background_tool_layer_reports_done_and_result() -> None:
    """A "delegate / off my hands" message drives ``run_background`` through the tool layer.

    Runs the full offline host graph so ``run_background`` is invoked the way
    ``langgraph dev`` invokes it. The scripted offline host does the simple two-turn
    flow (tool call then final), so the tool itself launches the run, waits for it to
    settle, and returns a lifecycle summary naming the run_id, the ``done`` status, and
    the final result. The fuller notify-and-poll multi-turn loop is a real-model concern;
    the offline host's bounded two-turn shape is documented in the tool.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "This is a heavy, multi-step job — please take it off my hands "
                        "and run the whole thing in the background."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "test-bg-tool-layer"}},
    )

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "run_background did not execute"
    summary = "\n".join(str(m.content) for m in tool_messages)
    assert "done" in summary.lower(), summary
    assert "run" in summary.lower(), summary


async def test_run_background_does_not_stream_fanout_into_ui_channel() -> None:
    """The detached background run must NOT stream fan-out into the host ``ui`` channel.

    This guards the honest limitation: a detached :meth:`BgRunManager.start` task does
    not carry the host node context, so it cannot push UI. The tool intentionally does
    not wire the run's progress/span sinks to a host emit — so the background turn shows
    lifecycle status, never live ``fanout_graph`` / ``agent_span`` events. A regression
    that tried to fake live streaming from the detached task would surface those events
    here and fail.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="Delegate this and run it in the background.")]},
        config={"configurable": {"thread_id": "test-bg-no-stream"}},
    )

    components = {u.get("name") for u in out.get("ui", [])}
    assert "fanout_graph" not in components, components
    assert "agent_span" not in components, components


def _offline_first_tool_call(prompt: str) -> tuple[str, dict[str, Any]]:
    """Drive the offline host one turn on ``prompt``; return the tool name and its args."""
    from _models import OfflineHostModel
    from langchain_core.messages import HumanMessage

    result = OfflineHostModel()._generate([HumanMessage(content=prompt)])
    message = result.generations[0].message
    call = message.tool_calls[0]  # type: ignore[attr-defined]
    return call["name"], call["args"]


def test_offline_host_routes_delegate_message_to_run_background() -> None:
    """A "take it off my hands / run in the background" cue routes to ``run_background``.

    The Delegate-heavy-work scenario asks the host to take a job off the user's hands;
    the offline host must route it to ``run_background`` rather than ``run_live``, even
    though the message also mentions "workflow" / "research" — the background cue wins
    precedence.
    """
    name, _args = _offline_first_tool_call(
        "This is a heavy, multi-step research job — take it off my hands and run the "
        "whole workflow in the background end to end."
    )
    assert name == "run_background", name


async def test_launch_background_run_threads_label_and_args() -> None:
    """``launch_background_run`` records a display label and forwards workflow args.

    The aggregate run board (M3.5) names each row by the label set at launch and runs
    the same preset over distinct questions, so the launcher must thread ``label`` onto
    the manager slot (surfaced by ``list_runs``) and ``workflow_args`` into the preset —
    rather than dropping both as the single-run background path did.
    """
    from host_graph import launch_background_run

    from langchain_dynamic_workflow import BgRunManager, BgStatus

    manager = BgRunManager()
    run_id = launch_background_run(
        manager,
        thread_id="board-test",
        workflow="deep_research",
        label="Agent frameworks",
        workflow_args={"question": "How do agent frameworks compare on durable execution?"},
    )

    snapshots = manager.list_runs("board-test")
    assert len(snapshots) == 1, snapshots
    assert snapshots[0].run_id == run_id
    assert snapshots[0].label == "Agent frameworks", "the launch label must reach the snapshot"

    # The forwarded args must not break the run (deep_research reads args['question']).
    await manager.wait(run_id, thread_id="board-test")
    assert manager.poll(run_id, thread_id="board-test") is BgStatus.DONE

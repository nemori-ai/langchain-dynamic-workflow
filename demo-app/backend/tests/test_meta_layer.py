"""Backend checks for the meta-layer host tool (``run_meta_script``) and its routing.

The meta layer lets the host author an orchestration script on the spot and submit the
*source* across the engine's AST security gate. These tests pin the demo's two headline
shapes — gate-pass (admitted, runs, streams fan-out) and gate-fail (rejected, runs
nothing) — and prove the emitted ``meta_script`` Gen-UI event matches the prop contract
the frontend's MetaScriptViewer renders (``source`` / ``gate`` / optional ``reason`` /
``event_id``). The gate-pass path is driven through the full offline host graph (the
"novel task" cue routes to the tool); both paths are exercised through the real tool
layer, not only the underlying helper.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from _meta_fixtures import AUTHORED_SCRIPT, REJECTED_SCRIPT
from ui_adapter import UiAdapter


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the host stays on the offline path."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)


def _meta_events(ui_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the props of every ``meta_script`` UI message in graph output order."""
    return [u.get("props") or {} for u in ui_messages if u.get("name") == "meta_script"]


async def test_run_meta_script_gate_pass_emits_passed_runs_and_streams_fanout() -> None:
    """The gate-pass path emits ``meta_script`` passed, runs the script, streams fan-out.

    Drives the authored (clean) script through the host helper. The emitted
    ``meta_script`` event must report ``gate="passed"`` with no ``reason`` and carry the
    exact authored source; the script must then actually run (returning a non-empty
    result) and stream a real parallel fan-out (a ``fanout_graph`` event) so its
    orchestration is visible live, not just compiled.
    """
    from host_graph import run_meta_script_live

    events: list[tuple[str, dict[str, Any]]] = []
    sink: Any = lambda comp, props: events.append((comp, dict(props)))  # noqa: E731
    adapter = UiAdapter(emit=sink)

    result = await run_meta_script_live(submit_rejected=False, adapter=adapter, emit=sink)

    meta = [props for comp, props in events if comp == "meta_script"]
    assert len(meta) == 1, meta
    assert meta[0]["gate"] == "passed"
    assert "reason" not in meta[0]
    assert meta[0]["source"] == AUTHORED_SCRIPT
    assert isinstance(meta[0]["event_id"], str) and meta[0]["event_id"]

    assert isinstance(result, str) and result.strip(), "gate-pass must run and return a result"
    assert any(comp == "fanout_graph" for comp, _ in events), [c for c, _ in events]


async def test_run_meta_script_gate_fail_emits_failed_with_reason_and_runs_nothing() -> None:
    """The gate-fail path emits ``meta_script`` failed with a line-numbered reason, runs nothing.

    Drives the rejected (import-bearing) script. The gate rejects it, so the emitted
    ``meta_script`` event reports ``gate="failed"`` with a ``reason`` enumerating the
    violation by line number, carries the rejected source, and NOTHING executes — no
    ``fanout_graph`` event is streamed and the tool returns its rejection text.
    """
    from host_graph import run_meta_script_live

    events: list[tuple[str, dict[str, Any]]] = []
    sink: Any = lambda comp, props: events.append((comp, dict(props)))  # noqa: E731
    adapter = UiAdapter(emit=sink)

    result = await run_meta_script_live(submit_rejected=True, adapter=adapter, emit=sink)

    meta = [props for comp, props in events if comp == "meta_script"]
    assert len(meta) == 1, meta
    assert meta[0]["gate"] == "failed"
    assert meta[0]["source"] == REJECTED_SCRIPT
    assert isinstance(meta[0]["event_id"], str) and meta[0]["event_id"]
    reason = meta[0].get("reason")
    assert isinstance(reason, str) and "line 1" in reason, reason

    assert not any(comp == "fanout_graph" for comp, _ in events), "rejected script must not run"
    assert "rejected" in result.lower()


async def test_run_meta_script_gate_pass_through_tool_layer_via_host_graph() -> None:
    """A "novel task" message drives ``run_meta_script`` through the real tool layer.

    The offline host routes a "no playbook / work out a procedure" request to the meta
    tool (the Phase 2 scenarios.json Novel-task button sends such a message). Running it
    through ``make_host_graph().ainvoke`` proves the tool executes end to end the way
    ``langgraph dev`` invokes it — emitting a passed ``meta_script`` event into the ``ui``
    channel and streaming the authored script's fan-out — guarding the very tool-schema
    path that mangled a param named ``args`` in Phase 2.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "There's no standard playbook for this — please work out a "
                        "procedure yourself: research a few topics and synthesize them."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "test-meta-novel-task"}},
    )

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "the meta tool did not execute"

    meta = _meta_events(out.get("ui", []))
    assert len(meta) == 1, meta
    assert meta[0]["gate"] == "passed"
    assert meta[0]["source"] == AUTHORED_SCRIPT

    components = [u.get("name") for u in out.get("ui", [])]
    assert "fanout_graph" in components, components


def _offline_first_tool_call(prompt: str) -> tuple[str, dict[str, Any]]:
    """Drive the offline host one turn on ``prompt``; return the tool name and its args."""
    from _models import OfflineHostModel
    from langchain_core.messages import HumanMessage

    result = OfflineHostModel()._generate([HumanMessage(content=prompt)])
    message = result.generations[0].message
    call = message.tool_calls[0]  # type: ignore[attr-defined]
    return call["name"], call["args"]


def test_offline_host_routes_novel_task_to_run_meta_script() -> None:
    """A "novel task / no playbook / work out a procedure" cue routes to the meta tool.

    The Novel-task scenario message names no ready-made preset; the offline host must
    route it to ``run_meta_script`` (the gate-pass path) rather than ``run_live``, even
    though the message also mentions "research" — the meta cue must win precedence.
    """
    name, _args = _offline_first_tool_call(
        "There's no standard playbook for this — work out a procedure yourself: "
        "research a few topics and synthesize what survives."
    )
    assert name == "run_meta_script", name

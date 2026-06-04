"""Backend-layer checks for the demo host graph and its inline-run progress path.

These exercise the engine-facing contract directly (no model, no langgraph dev):

* the host graph builds with a ``ui`` state channel and the demo tools;
* an inline ``run_workflow`` drives the progress sink in emission order;
* the generic ``run_workflow_live`` helper resolves a named preset workflow, runs it
  inline, and streams its progress/span events through a :class:`UiAdapter`; and
* the red line — a raising progress sink must never break orchestration.

Visual/Gen-UI rendering is verified separately in the browser.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from ui_adapter import UiAdapter
from workflows import hello_workflow, make_roster

from langchain_dynamic_workflow import ProgressEntry, ProgressKind, run_workflow


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the host stays on the offline path."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)


def test_host_graph_builds_with_ui_channel_and_tools() -> None:
    from host_graph import HostState, make_host_graph, run_hello_demo, run_live

    graph = make_host_graph()
    assert graph is not None
    assert "ui" in HostState.__annotations__
    assert run_hello_demo.name == "run_hello_demo"
    assert run_live.name == "run_live"


async def test_run_workflow_live_streams_named_preset_with_fanout() -> None:
    """The generic live runner resolves a named preset and streams its events.

    ``run_workflow_live`` is the engine-facing core of the ``run_live`` host tool,
    extracted so it can be tested without a node context (the contextvar rebind is
    covered separately in ``test_ui_bridge``). Driving the offline ``deep_research``
    preset through it must: resolve the workflow by name, run it inline, return a
    non-empty result, and feed a real parallel fan-out plus ordered phases through the
    supplied :class:`UiAdapter`.
    """
    from host_graph import run_workflow_live

    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))

    result = await run_workflow_live("deep_research", {"question": "Q?"}, adapter=adapter)

    assert isinstance(result, str)
    assert result.strip()

    components = [comp for comp, _ in events]
    assert any(comp == "fanout_graph" for comp in components), components

    phase_titles = [
        props["message"]
        for comp, props in events
        if comp == "phase_timeline" and props["kind"] == ProgressKind.PHASE.value
    ]
    assert phase_titles == ["search", "extract", "verify", "synthesize"]


async def test_run_workflow_live_unknown_name_raises() -> None:
    """An unknown workflow name fails loud with the registry's KeyError."""
    from host_graph import run_workflow_live

    adapter = UiAdapter(emit=lambda _comp, _props: None)
    with pytest.raises(KeyError):
        await run_workflow_live("does_not_exist", {}, adapter=adapter)


def _offline_first_tool_call(prompt: str) -> tuple[str, dict[str, Any]]:
    """Drive the offline host one turn on ``prompt``; return the tool name and its args."""
    from _models import OfflineHostModel
    from langchain_core.messages import HumanMessage

    result = OfflineHostModel()._generate([HumanMessage(content=prompt)])
    message = result.generations[0].message
    call = message.tool_calls[0]  # type: ignore[attr-defined]
    return call["name"], call["args"]


def test_offline_host_routes_scenario_to_run_live() -> None:
    """A scenario request drives ``run_live``; a generic ask stays on the hello path.

    Lets a key-free user trigger a preset live run (not only the hello smoke path),
    while keeping ``run_hello_demo`` as the default for any non-scenario message.
    """
    assert _offline_first_tool_call("Do deep, fact-checked research on RAG.")[0] == "run_live"
    assert _offline_first_tool_call("Hi, can you show me the demo?")[0] == "run_hello_demo"


def test_offline_host_routes_named_preset_through_args() -> None:
    """A request that names a preset runs THAT preset, not the default deep_research.

    The offline host must forward the named workflow through the ``run_live`` tool args
    so a key-free "run the capstone scenario" actually reaches capstone; an unnamed
    scenario request leaves args empty so the tool picks its own default.
    """
    name, args = _offline_first_tool_call("Run the capstone scenario end to end.")
    assert name == "run_live"
    assert args.get("workflow") == "capstone"

    # An unnamed scenario request leaves the workflow to the tool default (empty args).
    _name, default_args = _offline_first_tool_call("Do deep, fact-checked research on RAG.")
    assert "workflow" not in default_args


async def test_run_workflow_live_runs_capstone_majority_vote() -> None:
    """The live runner drives the capstone preset to its non-trivial survivor split.

    Closes the gap the offline routing now opens: capstone must be reachable AND
    correct through the same engine-facing path the host tool uses, ending in the
    strict-majority adversarial vote (three survivors, beta voted down).
    """
    from host_graph import run_workflow_live

    adapter = UiAdapter(emit=lambda _comp, _props: None)
    result = await run_workflow_live("capstone", {}, adapter=adapter)
    assert result.startswith("synthesized 3 surviving findings:"), result
    assert "beta:" not in result


async def test_inline_run_emits_ordered_progress() -> None:
    received: list[ProgressEntry] = []

    await run_workflow(
        hello_workflow,
        roster=make_roster(),
        on_progress=received.append,
    )

    # hello_workflow narrates phase("greeting") -> log -> phase("wrap-up") -> log.
    kinds = [e.kind for e in received]
    messages = [e.message for e in received]
    assert kinds == [
        ProgressKind.PHASE,
        ProgressKind.LOG,
        ProgressKind.PHASE,
        ProgressKind.LOG,
    ]
    assert messages == ["greeting", "working...", "wrap-up", "done"]


async def test_raising_progress_sink_does_not_break_orchestration() -> None:
    """The host tool wraps the sink in try/except; mirror that contract here.

    A bare raising sink passed straight to the engine WOULD propagate (the engine
    calls it directly, by design). The host's ``run_hello_demo`` swallows inside the
    sink, so orchestration completes. This asserts the swallow-and-continue contract
    the tool relies on, end to end.
    """

    def raising_then_swallowed(_entry: ProgressEntry) -> None:
        try:
            raise RuntimeError("ui down")
        except Exception:
            pass

    result = await run_workflow(
        hello_workflow,
        roster=make_roster(),
        on_progress=raising_then_swallowed,
    )
    assert result == "ok"

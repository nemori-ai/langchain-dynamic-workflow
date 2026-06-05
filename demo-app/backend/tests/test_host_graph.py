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


def test_run_live_tool_schema_has_no_mangled_args() -> None:
    """The ``run_live`` tool param must not be named ``args``.

    LangChain's tool-schema generation mangles a parameter literally named ``args``
    into ``v__args`` (typed as an array) and then passes that keyword back on
    invocation, raising ``unexpected keyword argument 'v__args'``. The param is named
    ``workflow_args`` to avoid the reserved name; this guards the regression at the
    schema level, where a unit test that calls the ``run_workflow_live`` helper
    directly (bypassing the ``@tool`` schema) would miss it.
    """
    from host_graph import run_live

    fields = set(run_live.get_input_schema().model_fields)
    assert "v__args" not in fields, f"param 'args' got mangled to v__args: {sorted(fields)}"
    assert "workflow_args" in fields


async def test_run_live_executes_through_tool_layer_via_host_graph() -> None:
    """A scenario message drives ``run_live`` through the real tool-invocation layer.

    The other tests call ``run_workflow_live`` directly, bypassing the LangChain
    ``@tool`` schema path — which is exactly where a parameter named ``args`` was
    mangled to ``v__args`` and crashed the tool call. This runs the full offline host
    graph so ``run_live`` is invoked the way ``langgraph dev`` invokes it, proving the
    tool executes end to end and streams the deep-research event vocabulary (phases +
    parallel fan-out) into the ``ui`` channel.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import HumanMessage, ToolMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="please run a deep research workflow on RAG")]},
        config={"configurable": {"thread_id": "test-run-live-tool-layer"}},
    )

    tool_messages = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert any("finished" in m.content for m in tool_messages), "run_live did not execute"

    components = [u.get("name") for u in out.get("ui", [])]
    assert "phase_timeline" in components
    assert "fanout_graph" in components, components


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


async def test_run_workflow_live_emits_running_chip_before_completion() -> None:
    """The live runner opens a running ``agent_span`` before its completed twin.

    Proves the host wires the engine's ``on_span_begin`` sink into ``run_workflow``:
    the engine fires a span-open edge at each leaf's start, which the adapter maps to a
    ``running=True`` ``agent_span``; the matching span-close edge then re-emits the SAME
    ``event_id`` with ``running=False`` (and ``merge=True``) so the chip flips in place.
    A spy :class:`UiAdapter` captures the emit stream and asserts, for at least one
    leaf, that the ``running=True`` open arrives BEFORE the matching ``running=False``
    close on the same ``event_id``. If a future edit drops the ``on_span_begin`` wiring
    no running chip is ever emitted and this fails — it is a behavioral guard, not a
    source-inspection one.
    """
    from host_graph import run_workflow_live

    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))

    await run_workflow_live("deep_research", {"question": "Q?"}, adapter=adapter)

    agent_spans = [props for comp, props in events if comp == "agent_span"]
    running_opens = [p for p in agent_spans if p.get("running") is True]
    assert running_opens, "on_span_begin must surface a running agent_span for each leaf"

    # For each running-open event_id, its open must precede the matching completed close.
    open_indices: dict[str, int] = {}
    close_indices: dict[str, int] = {}
    for index, props in enumerate(agent_spans):
        event_id = props["event_id"]
        if props.get("running") is True:
            open_indices.setdefault(event_id, index)
        elif props.get("running") is False:
            close_indices.setdefault(event_id, index)

    flipped_in_place = [
        event_id
        for event_id, open_at in open_indices.items()
        if event_id in close_indices and open_at < close_indices[event_id]
    ]
    assert flipped_in_place, (
        "at least one leaf must emit running=True before its running=False twin on the "
        "same event_id (the in-place flip) — proving on_span_begin is wired"
    )


async def test_run_workflow_live_resumes_cached_leaves_on_second_run() -> None:
    """A second run on the same resume lane replays leaves as journal hits.

    This is the "pick it back up" headline: the host persists a per-(thread, workflow)
    :class:`_ResumeLane` whose journal / checkpointer / thread_id are threaded into
    ``run_workflow``. The first run executes every leaf fresh (no ``journal_badge``,
    every completed ``agent_span`` ``cached=False``); a second run on the SAME lane must
    replay each recorded leaf from the journal — every completed ``agent_span`` comes
    back ``cached=True`` and a ``journal_badge`` is newly emitted — proving the cached
    ``agent_span`` branch and the ``journal_badge`` component are reachable at runtime,
    not dead paths. The result is identical across runs (the journal replays it).

    The cache flag lives on the COMPLETED (span-close) ``agent_span`` edge; the
    span-open running chip (``running=True``) carries no cache flag (cache state is
    unknown at open), so the assertions filter to the completed (``running is False``)
    edges.
    """
    from host_graph import _ResumeLane, run_workflow_live

    lane = _ResumeLane(thread_id="t-resume::deep_research")

    first: list[tuple[str, dict[str, Any]]] = []
    adapter_first = UiAdapter(emit=lambda comp, props: first.append((comp, dict(props))))
    result_first = await run_workflow_live("deep_research", {}, adapter=adapter_first, lane=lane)

    fresh_spans = [p for c, p in first if c == "agent_span" and p.get("running") is False]
    assert fresh_spans, "first run must emit completed agent spans"
    assert all(p["cached"] is False for p in fresh_spans)
    assert not [c for c, _ in first if c == "journal_badge"], "fresh run emits no journal badge"

    second: list[tuple[str, dict[str, Any]]] = []
    adapter_second = UiAdapter(emit=lambda comp, props: second.append((comp, dict(props))))
    result_second = await run_workflow_live("deep_research", {}, adapter=adapter_second, lane=lane)

    cached_spans = [
        p
        for c, p in second
        if c == "agent_span" and p.get("running") is False and p["cached"] is True
    ]
    badges = [c for c, _ in second if c == "journal_badge"]
    assert len(cached_spans) == len(fresh_spans), "resume must replay every leaf as cached"
    assert len(badges) == len(fresh_spans), "each cached leaf surfaces a journal badge"
    assert result_second == result_first, "the journal replays the recorded result"


async def test_resume_replays_cached_leaves_across_two_host_graph_turns() -> None:
    """Two turns on the SAME chat thread replay the first run's leaves as journal hits.

    This proves the resume story is reachable end to end through the real tool layer —
    not just at the ``run_workflow_live`` helper level (covered above). The first turn
    drives a deep-research run on a host thread; the durable :class:`_ResumeLane` keyed
    on that ``(thread_id, deep_research)`` persists its journal at module scope. A second
    turn on the SAME ``configurable.thread_id`` (a "pick it back up" message that the
    offline host also routes to ``run_live`` for the default preset) reuses that journal,
    so every replayed leaf comes back ``cached=True`` and surfaces a ``journal_badge``.

    Honest scope: this is journal re-run replay (the same persisted journal instance is
    fed back into a second ``run_workflow``), NOT a LangGraph mid-flight interrupt/resume
    — the engine has no mid-flight interrupt. The visible signal is the zero-cost cache
    hit, which is exactly what the demo surfaces.
    """
    from host_graph import _RESUME_LANES, make_host_graph
    from langchain_core.messages import HumanMessage

    # Isolate this test's lane from any other test's module-scope lanes.
    _RESUME_LANES.clear()
    thread = "test-resume-two-turns"
    graph = make_host_graph()

    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="Please run a deep research workflow on RAG.")]},
        config={"configurable": {"thread_id": thread}},
    )
    first_badges = [u for u in first.get("ui", []) if u.get("name") == "journal_badge"]
    assert not first_badges, "the first (fresh) run must not surface any journal badge"

    second = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Earlier you started that research for me — please pick it back up "
                        "where you left off rather than starting over."
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": thread}},
    )
    second_components = [u.get("name") for u in second.get("ui", [])]
    assert "journal_badge" in second_components, second_components

    cached_agent_spans = [
        u
        for u in second.get("ui", [])
        if u.get("name") == "agent_span" and (u.get("props") or {}).get("cached") is True
    ]
    assert cached_agent_spans, "the second turn must replay leaves as cached agent spans"


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


def test_offline_host_routes_resume_message_to_run_live() -> None:
    """A "pick it back up" message routes to ``run_live`` for the default preset.

    The resume scenario is a second live run on the same chat thread: it reuses the
    prior run's durable journal lane and replays its leaves as cache hits. So the
    offline host must route a "pick it back up" / "where you left off" message to the
    SAME ``run_live`` tool with no preset named (default deep_research) — the only way
    a second turn lands on the first turn's lane.
    """
    name, args = _offline_first_tool_call(
        "Earlier you started that research — pick it back up where you left off."
    )
    assert name == "run_live"
    assert "workflow" not in args, "resume must hit the default preset's lane, not a named one"


def test_offline_host_fires_a_new_tool_on_the_second_turn_of_an_accumulated_thread() -> None:
    """A second scenario on the SAME chat thread must still fire its tool.

    Under ``langgraph dev`` the thread state ACCUMULATES messages, so after turn 1
    completes (human -> ai+tool_call -> tool -> ai) the prior ``ToolMessage`` lingers in
    history. The offline host must decide "did a tool run THIS turn" from the messages
    AFTER the latest human turn, not from the whole history — otherwise every turn after
    the first emits a canned reply and never runs a tool, so a second scenario (e.g.
    resume) on the same thread silently does nothing. This builds that accumulated
    history explicitly (the single-message-per-turn unit helper above cannot catch it,
    nor can an in-process two-``ainvoke`` test that starts each turn from fresh state)
    and asserts the host issues a NEW ``run_live`` tool call on the second human turn.
    """
    from _models import OfflineHostModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    accumulated: list[Any] = [
        HumanMessage(content="Do a thorough deep research on RAG trade-offs."),
        AIMessage(
            content="Running the deep-research workflow now.",
            tool_calls=[{"name": "run_live", "args": {}, "id": "live-call-1"}],
        ),
        ToolMessage(
            content="Workflow 'deep_research' finished: <report>",
            tool_call_id="live-call-1",
            name="run_live",
        ),
        AIMessage(content="Done — the workflow streamed its progress into the panel above."),
        HumanMessage(content="Can you pick it back up where you left off, instead of restarting?"),
    ]
    result = OfflineHostModel()._generate(accumulated)
    message = result.generations[0].message
    assert message.tool_calls, "second turn must issue a NEW tool call, not a canned reply"  # type: ignore[attr-defined]
    assert message.tool_calls[0]["name"] == "run_live"  # type: ignore[attr-defined]


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


def test_is_offline_reflects_openrouter_key_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``is_offline`` gates on the OpenRouter key alone (provider is locked to it).

    This is the honest signal the frontend's offline banner renders from. The provider
    is LOCKED to OpenRouter, so it must be ``True`` only when no OpenRouter key is in
    force and flip ``False`` as soon as the ``.env`` ``OPENROUTER_API_KEY`` appears —
    never a hardcoded constant. An ``OPENAI_API_KEY`` is NOT a headline path and must
    NOT flip the demo online (OpenAI was dropped as a real provider).
    """
    from _models import is_offline

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert is_offline() is True

    # OpenAI is no longer a real provider: its key must not flip the demo online.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert is_offline() is True

    # The OpenRouter .env key is the operator/local online source.
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert is_offline() is False


def test_run_status_emitted_once_per_turn_with_real_offline_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host emits a single ``run_status`` event carrying the true offline flag.

    Drives the shared ``_emit_run_status`` helper the tools call right after capturing
    the host-bound emit. The banner the frontend renders must reflect real backend key
    state, so with no key present the event reports ``offline=True``; with a key it
    reports ``offline=False``. The stable ``event_id`` lets a same-turn re-emit dedupe.
    """
    from host_graph import _emit_run_status

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    events: list[tuple[str, dict[str, Any]]] = []
    _emit_run_status(lambda comp, props: events.append((comp, dict(props))))
    assert events == [("run_status", {"offline": True, "event_id": "run-status-1"})]

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    events.clear()
    _emit_run_status(lambda comp, props: events.append((comp, dict(props))))
    assert events == [("run_status", {"offline": False, "event_id": "run-status-1"})]


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

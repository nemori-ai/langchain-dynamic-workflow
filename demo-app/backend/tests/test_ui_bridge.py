"""Red-line coverage for ``ui_bridge.make_host_ui_emit``.

``make_host_ui_emit`` is the single most mechanism-heavy piece of the spike: it
captures the host node's runnable config once, then rebinds
``var_child_runnable_config`` to it around each ``push_ui_message`` so a UI event
fired from deep inside a nested ``run_workflow`` ``@entrypoint`` still targets the
**host** graph's stream writer and ``ui`` channel — not the inner workflow graph,
which has neither. These tests lock down the two behaviors the spike exists to
de-risk, both against the *real* ``ui_bridge.emit`` (not a test-local stand-in):

* the contextvar rebind survives the inner engine's own ``@entrypoint`` context, so
  a progress sink driven from inside ``run_workflow`` routes its UI event onto the
  captured host writer + host ``ui``-channel send, and ``ui_message_reducer`` folds
  the sent event into the channel the frontend renders; and
* when ``push_ui_message`` itself raises, ``emit`` swallows it (never breaking
  orchestration) and the ``finally`` still resets the contextvar token, restoring
  whatever ambient (inner) config was active.

The mechanism faked here mirrors LangGraph exactly: ``push_ui_message`` resolves the
writer via ``get_stream_writer() -> get_config()[CONF][CONFIG_KEY_RUNTIME].stream_writer``
and the state-send via ``get_config()[CONF][CONFIG_KEY_SEND]`` — both off the single
``var_child_runnable_config`` contextvar — so rebinding that one contextvar redirects
both. A bare fake host config with those two slots is therefore sufficient and
honest.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import pytest
import ui_bridge
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SEND,
)
from langgraph.graph.ui import AnyUIMessage, UIMessage, ui_message_reducer
from ui_bridge import make_host_ui_emit
from workflows import hello_workflow, make_roster

from langchain_dynamic_workflow import ProgressEntry, run_workflow


class _FakeRuntime:
    """Minimal stand-in for the LangGraph runtime that exposes ``stream_writer``.

    ``get_stream_writer()`` reads ``get_config()[CONF][CONFIG_KEY_RUNTIME].stream_writer``;
    this object supplies exactly that attribute and nothing more.
    """

    def __init__(self, stream_writer: Any) -> None:
        self.stream_writer = stream_writer


def _make_host_config(
    *, writer_sink: list[UIMessage], send_sink: list[tuple[str, UIMessage]]
) -> RunnableConfig:
    """Build a fake *host* runnable config whose writer/send land in the sinks.

    Mirrors the only two slots ``push_ui_message`` touches: the runtime's
    ``stream_writer`` (where the streamed UI event goes) and ``CONFIG_KEY_SEND``
    (where the ``("ui", evt)`` state update goes). ``push_ui_message`` always emits
    a ``UIMessage`` (never a ``RemoveUIMessage``), so the sinks are typed as such.
    """
    return {  # type: ignore[typeddict-unknown-key]
        CONF: {
            CONFIG_KEY_RUNTIME: _FakeRuntime(lambda evt: writer_sink.append(evt)),
            CONFIG_KEY_SEND: lambda pairs: send_sink.extend(pairs),
        }
    }


async def test_emit_from_inside_nested_run_workflow_routes_to_host_writer() -> None:
    """A sink driven from inside ``run_workflow`` lands on the HOST writer + ui channel.

    This is the load-bearing claim of the spike. We capture a host config (via the
    real ``make_host_ui_emit``) while it is the ambient contextvar, then run a *real*
    ``run_workflow`` whose progress sink calls the captured ``emit``. Inside the
    engine the ambient contextvar is the inner ``@entrypoint``'s own config, so if
    the rebind were broken the events would route there (or fail). Asserting they
    reach the host writer + host send proves the rebind survives the inner graph.
    """
    host_writer_events: list[UIMessage] = []
    host_send_pairs: list[tuple[str, UIMessage]] = []
    host_config = _make_host_config(writer_sink=host_writer_events, send_sink=host_send_pairs)

    # Pretend we are inside the host node: make host_config the ambient context so
    # make_host_ui_emit captures it, then restore (the engine sets its own below).
    token = var_child_runnable_config.set(host_config)
    try:
        emit = make_host_ui_emit(anchor=None)
    finally:
        var_child_runnable_config.reset(token)

    seq = 0

    def on_progress(entry: ProgressEntry) -> None:
        nonlocal seq
        seq += 1
        emit(
            "phase_timeline",
            {"kind": entry.kind.value, "message": entry.message, "event_id": f"p-{seq}"},
        )

    result = await run_workflow(
        hello_workflow,
        roster=make_roster(),
        on_progress=on_progress,
    )
    assert result == "ok"

    # hello_workflow narrates 4 progress entries; every one must have reached the
    # HOST writer and the HOST ui-channel send (proving the rebind across the inner
    # engine @entrypoint), not the inner workflow graph.
    assert len(host_writer_events) == 4
    assert [e["name"] for e in host_writer_events] == ["phase_timeline"] * 4
    assert all(e["type"] == "ui" for e in host_writer_events)

    assert len(host_send_pairs) == 4
    assert all(state_key == "ui" for state_key, _evt in host_send_pairs)

    # The sent ("ui", evt) pairs fold into the host's `ui` channel via the real
    # ui_message_reducer, exactly as HostState's annotated channel does. Each push
    # is a distinct UI message, so the reducer appends all four into the channel.
    channel: list[AnyUIMessage] = []
    for _state_key, evt in host_send_pairs:
        channel = ui_message_reducer(channel, evt)
    assert len(channel) == 4

    # The props the engine streamed survive intact (read off the typed UIMessage
    # send payloads, which always carry `props`, unlike a RemoveUIMessage).
    sent_props = [evt["props"] for _state_key, evt in host_send_pairs]
    assert [p["event_id"] for p in sent_props] == ["p-1", "p-2", "p-3", "p-4"]
    assert sent_props[0]["message"] == "greeting"


def test_emit_threads_event_id_into_ui_message_id_so_reducer_dedupes() -> None:
    """A stable ``event_id`` becomes the SDK ui-message id, so the reducer dedupes.

    The documented same-turn / resume dedup lives at the SDK layer: ``uiMessageReducer``
    keys on the ui-message ``id`` (``ui.id === event.id``) and the frontend's React
    render key is that same id. ``push_ui_message`` mints a fresh ``str(uuid4())`` when
    no ``id`` is passed, so unless ``emit`` threads the adapter's ``event_id`` into
    ``id=`` every re-emit becomes a distinct ui message and renders as a duplicate.

    This drives the *real* ``ui_bridge.emit`` (not a stand-in): two emits carrying the
    SAME ``event_id`` (a same-turn re-emit, second one with a flipped field) and one
    with a distinct id. Folding the sent ``("ui", evt)`` pairs through the real
    ``ui_message_reducer`` — exactly as the frontend does — must collapse the two
    same-id events onto a single ui message (the re-emit replacing the first) while the
    distinct id stays separate. A regression that drops ``id=`` (random UUIDs) leaves
    three messages and fails here.
    """
    host_writer_events: list[UIMessage] = []
    host_send_pairs: list[tuple[str, UIMessage]] = []
    host_config = _make_host_config(writer_sink=host_writer_events, send_sink=host_send_pairs)

    token = var_child_runnable_config.set(host_config)
    try:
        emit = make_host_ui_emit(anchor=None)
    finally:
        var_child_runnable_config.reset(token)

    emit("agent_span", {"name": "r", "cached": False, "event_id": "span-abc"})
    emit("agent_span", {"name": "r", "cached": True, "event_id": "span-abc"})  # re-emit
    emit("agent_span", {"name": "q", "cached": False, "event_id": "span-xyz"})  # distinct

    # The ui-message id is the threaded event_id, not a random UUID.
    assert [evt["id"] for evt in host_writer_events] == ["span-abc", "span-abc", "span-xyz"]

    # Fold the sent pairs through the real reducer, as Stream.tsx's onCustomEvent does.
    channel: list[AnyUIMessage] = []
    for _state_key, evt in host_send_pairs:
        channel = ui_message_reducer(channel, evt)

    # The two same-id events collapsed to one; the re-emit replaced the first (cached
    # now True), and the distinct id stayed a separate message.
    assert [msg["id"] for msg in channel] == ["span-abc", "span-xyz"]
    collapsed = next(msg for msg in channel if msg["id"] == "span-abc")
    # The reducer returns AnyUIMessage (UIMessage | RemoveUIMessage); the collapsed
    # agent_span is a UIMessage — assert that, then read its props type-safely.
    assert collapsed["type"] == "ui", "collapsed message should be a UIMessage, not a removal"
    assert cast(UIMessage, collapsed)["props"]["cached"] is True


async def test_emit_swallows_push_failure_and_resets_contextvar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``push_ui_message`` raises, ``emit`` swallows it and resets the token.

    Asserts ``ui_bridge.emit``'s OWN ``try/except``/``finally`` — not a test-local
    swallow — so removing the swallow (a red-line regression) fails this test. Also
    proves the ``finally`` restores the ambient (inner) config even on the raising
    path, so a failing push cannot leak the host config into the engine's context.
    """
    host_writer_events: list[UIMessage] = []
    host_send_pairs: list[tuple[str, UIMessage]] = []
    host_config = _make_host_config(writer_sink=host_writer_events, send_sink=host_send_pairs)

    token = var_child_runnable_config.set(host_config)
    try:
        emit = make_host_ui_emit(anchor=None)
    finally:
        var_child_runnable_config.reset(token)

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("ui channel down")

    monkeypatch.setattr(ui_bridge, "push_ui_message", _boom)

    # Simulate being inside the inner engine context: a sentinel inner config is
    # ambient. emit() must rebind to host, hit the raising push, swallow, and reset
    # back to this sentinel.
    inner_sentinel: RunnableConfig = {"tags": ["inner-engine-sentinel"]}
    inner_token = var_child_runnable_config.set(inner_sentinel)
    try:
        # Must not raise despite push_ui_message blowing up.
        emit("phase_timeline", {"message": "x", "event_id": "p-1"})
        # finally ran: the ambient contextvar is restored to the inner sentinel,
        # NOT left pinned to the host config.
        assert var_child_runnable_config.get() is inner_sentinel
    finally:
        var_child_runnable_config.reset(inner_token)

    # Nothing was streamed or sent because the push failed before writer()/send().
    assert host_writer_events == []
    assert host_send_pairs == []


async def test_emit_from_a_genuine_to_thread_worker_reaches_the_host_writer() -> None:
    """The host-bound emit works when called from an ``asyncio.to_thread`` WORKER thread.

    This is the load-bearing M5 transport claim. The engine fires a leaf's real
    ``on_command`` edge from inside ``deepagents``' ``execute`` path, which is marshaled
    onto an ``asyncio.to_thread`` worker — so the command-path emit does NOT run on the
    event loop thread. ``push_ui_message`` resolves the host writer + ui-channel send off
    a ``contextvars.ContextVar`` (``var_child_runnable_config``); a worker spawned by
    ``to_thread`` inherits a COPY of the loop thread's context, so the captured host
    config must still be reachable and rebindable from that worker.

    The whole live-terminal-card story rides on this: if a worker-thread emit silently
    failed (the bridge's bare ``except`` would swallow it), every fix-loop terminal card
    would vanish and the headline M5 UI would never appear — with no test catching it.
    This test drives the REAL ``make_host_ui_emit`` from a genuine ``to_thread`` worker
    and asserts (a) the emit ran OFF the loop thread and (b) the event still reached the
    host writer + host ui-channel send. If this ever fails, the bridge must marshal the
    push onto the loop thread (``loop.call_soon_threadsafe``) for the command path.
    """
    host_writer_events: list[UIMessage] = []
    host_send_pairs: list[tuple[str, UIMessage]] = []
    host_config = _make_host_config(writer_sink=host_writer_events, send_sink=host_send_pairs)

    loop_thread_id = threading.get_ident()

    # Capture the emit while the host config is the ambient context (we are "in the host
    # node" on the loop thread), exactly as a host tool does before entering the engine.
    token = var_child_runnable_config.set(host_config)
    try:
        emit = make_host_ui_emit(anchor=None)
    finally:
        var_child_runnable_config.reset(token)

    worker_thread_id: dict[str, int] = {}

    def emit_from_worker() -> None:
        # This body runs on the to_thread worker, NOT the loop thread — exactly where the
        # engine's on_command edge invokes the adapter's sink.
        worker_thread_id["id"] = threading.get_ident()
        emit("execution_command", {"command": "bun test", "status": "running", "event_id": "cmd-1"})

    # asyncio.to_thread copies the current context into the worker, so the captured host
    # config travels with it — this is the real engine marshaling path, not a fake.
    await asyncio.to_thread(emit_from_worker)

    # (a) The emit genuinely ran off the event-loop thread.
    assert worker_thread_id["id"] != loop_thread_id, "emit did not run on a worker thread"

    # (b) The worker-thread emit still reached the HOST writer + ui-channel send. A bare
    # except in the bridge would hide a failure here, so this is the guard that the live
    # terminal card actually streams.
    assert len(host_writer_events) == 1, "the worker-thread emit must reach the host writer"
    assert host_writer_events[0]["name"] == "execution_command"
    assert len(host_send_pairs) == 1
    state_key, evt = host_send_pairs[0]
    assert state_key == "ui"
    assert evt["id"] == "cmd-1"
    assert cast(UIMessage, evt)["props"]["command"] == "bun test"


async def test_merge_true_end_edge_folds_onto_same_id_begin_card() -> None:
    """A ``merge=True`` end edge patches the same-id begin card in place.

    The engine opens a span with a *begin* edge (running, with ``started_at``) and
    closes it with an *end* edge carrying only the completion fields. For the two to
    collapse onto one chip the transport must forward a reserved ``merge`` flag to
    ``push_ui_message(..., merge=)`` so ``ui_message_reducer`` shallow-merges the end
    props onto the begin card (same ui-message id), preserving begin-only fields like
    ``started_at`` while applying the end's ``duration_s``. The reserved transport
    flag must be stripped before it reaches component props.
    """
    host_writer_events: list[UIMessage] = []
    host_send_pairs: list[tuple[str, UIMessage]] = []
    host_config = _make_host_config(writer_sink=host_writer_events, send_sink=host_send_pairs)

    token = var_child_runnable_config.set(host_config)
    try:
        emit = make_host_ui_emit(anchor=None)
        # Begin: creates the card (running, with started_at).
        emit(
            "agent_span",
            {
                "event_id": "span-1",
                "running": True,
                "started_at": 123.0,
                "agent_type": "researcher",
            },
        )
        # End: same id, merge=True, carries the completion fields only.
        emit(
            "agent_span",
            {
                "event_id": "span-1",
                "merge": True,
                "running": False,
                "duration_s": 0.42,
                "cached": False,
            },
        )
    finally:
        var_child_runnable_config.reset(token)

    # Fold the two host-channel sends exactly as the SDK does.
    ui: list[AnyUIMessage] = []
    for _key, evt in host_send_pairs:
        ui = ui_message_reducer(ui, evt)

    assert len(ui) == 1, "the same-id end edge must patch the begin card, not append a new one"
    merged = cast(UIMessage, ui[0])["props"]
    assert merged["started_at"] == 123.0  # begin-only field survives the merge
    assert merged["duration_s"] == 0.42  # end field applied
    assert merged["running"] is False
    # The transport-only 'merge' flag must NOT leak into component props.
    assert "merge" not in merged

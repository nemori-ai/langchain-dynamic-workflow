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

from typing import Any

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

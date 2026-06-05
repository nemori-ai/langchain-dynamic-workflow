"""Bridge engine progress into host-graph Generative-UI events from a node context.

``push_ui_message`` resolves the *current* graph's stream writer and state-send
channel from a contextvar (``var_child_runnable_config``). An inline
``run_workflow`` spins up its own LangGraph ``@entrypoint``/``@task`` substrate,
which replaces that contextvar while the orchestration runs — so a progress sink
that calls ``push_ui_message`` from inside the engine would route its UI events to
the inner workflow graph (which has no ``ui`` channel and is not the stream the
frontend listens to), and they would never reach the chat.

This module captures the **host node's** runnable config once (while still in the
host node context) and, for each engine event, rebinds that contextvar to the
captured host config for the duration of a single ``push_ui_message`` call. The UI
event is then emitted against the host graph's writer and ``ui`` channel, so it
streams to the frontend even though the sink fires deep inside the nested engine
run.

Delivery is **synchronous and inline by necessity**, not by oversight. The engine's
progress/span sinks are plain synchronous callables (``Callable[[Span], None]``), and
``push_ui_message`` resolves and writes the *current* graph's stream writer from a
contextvar — so the emit must run inline, in the rebound host context, on the engine's
own call stack. Deferring it to a background task would both lose the host context and
require an awaitable sink the engine does not offer. The transport itself is a fast,
in-memory stream-writer append (it does no I/O and does not await), so an inline emit
adds bounded, near-constant work per event rather than a blocking call. The one failure
mode the inline path must defend against is a *raising* sink — the engine calls sinks
directly, so an exception would unwind the orchestration — and the emit swallows every
exception to neutralize exactly that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.config import get_config
from langgraph.graph.ui import push_ui_message

UiEmit = Callable[[str, dict[str, Any]], None]
"""Emit a UI component event by ``(component_name, props)`` onto the host graph."""


def make_host_ui_emit(*, anchor: AIMessage | None) -> UiEmit:
    """Capture the host node context and return a host-bound, non-blocking UI emit.

    Must be called from within the host node (before entering ``run_workflow``) so
    the host's runnable config is captured. The returned callable can then be
    invoked from inside the nested engine run — it rebinds the captured host config
    around each ``push_ui_message`` so the event targets the host graph's stream and
    ``ui`` channel, then restores the ambient (inner) config.

    Args:
        anchor: The AI message to attach UI events to (so the frontend renders them
            inline beneath the host's turn), or ``None`` to leave them unanchored.

    Returns:
        A ``(component_name, props) -> None`` emit that is host-bound and swallows
        every exception (never raising, never blocking the orchestration).

    Note:
        The reserved ``merge`` props key is a transport-only flag consumed here and
        stripped before reaching the component: when truthy it forwards
        ``push_ui_message(..., merge=True)`` so a same-``event_id`` event patches the
        existing ui-message card in place (``ui_message_reducer`` shallow-merges its
        props onto the prior card) instead of replacing it wholesale. The end edge of
        a span thus folds its completion fields onto the running begin card without
        clobbering begin-only fields. The flag never appears in component props.
    """
    host_config: RunnableConfig = get_config()
    anchor_kwargs: dict[str, Any] = {"message": anchor} if anchor is not None else {}

    def emit(component_name: str, props: dict[str, Any]) -> None:
        # Pop the reserved transport-only merge flag before building id_kwargs so it
        # never reaches component props. When truthy, the SDK reducer shallow-merges a
        # same-id event onto the existing card (begin -> end folds in place) rather
        # than replacing it; when falsy (the default), the event creates/replaces.
        merge_flag = bool(props.pop("merge", False))
        # Thread the adapter's stable event_id into the SDK ui-message id. The SDK's
        # uiMessageReducer keys on the ui-message id (ui.id === event.id) and the
        # frontend's React render key is that same id — so passing event_id as id is
        # what makes the documented dedupe real: a same-turn re-emit of an identical
        # event collapses onto one ui message instead of minting a fresh random UUID
        # (push_ui_message defaults id to str(uuid4()) when none is given) and
        # rendering a duplicate. Without an event_id we let the SDK mint its own id.
        id_kwargs: dict[str, Any] = {}
        event_id = props.get("event_id")
        if isinstance(event_id, str) and event_id:
            id_kwargs["id"] = event_id
        # Rebind the host config so push_ui_message resolves the HOST writer and
        # ui-channel send, then restore whatever was active (the inner engine
        # graph's config) so the engine's own streaming is untouched.
        token = var_child_runnable_config.set(host_config)
        try:
            push_ui_message(component_name, props, **id_kwargs, **anchor_kwargs, merge=merge_flag)
        except Exception:
            # Red line: the engine calls progress sinks directly; a raising sink
            # would break orchestration. Swallow and continue.
            pass
        finally:
            var_child_runnable_config.reset(token)

    return emit

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
run. The emit is wrapped so a failure is swallowed and never blocks — progress
sinks are called directly inside the engine, where a raising or blocking sink would
break orchestration.
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
    """
    host_config: RunnableConfig = get_config()
    anchor_kwargs: dict[str, Any] = {"message": anchor} if anchor is not None else {}

    def emit(component_name: str, props: dict[str, Any]) -> None:
        # Rebind the host config so push_ui_message resolves the HOST writer and
        # ui-channel send, then restore whatever was active (the inner engine
        # graph's config) so the engine's own streaming is untouched.
        token = var_child_runnable_config.set(host_config)
        try:
            push_ui_message(component_name, props, **anchor_kwargs)
        except Exception:
            # Red line: the engine calls progress sinks directly; a raising sink
            # would break orchestration. Swallow and continue.
            pass
        finally:
            var_child_runnable_config.reset(token)

    return emit

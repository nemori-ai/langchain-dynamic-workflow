"""Host deepagent graph factory for the interactive demo.

``langgraph dev`` loads :func:`make_host_graph` (registered under the ``host`` graph
id in ``langgraph.json``) and serves it on the local API. The host carries a ``ui``
state channel (reduced by ``ui_message_reducer``) so Generative-UI components render
in the chat, and a ``run_hello_demo`` tool that proves two round-trips:

* pushing a trivial ``hello_ui`` component from inside the node context, and
* running a workflow inline with a progress sink that pushes ``phase_timeline``
  components live, from within the same node context.
"""

from __future__ import annotations

from typing import Annotated, Any

from _models import resolve_host_model
from deepagents import DeepAgentState, create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langgraph.prebuilt import InjectedState
from ui_bridge import make_host_ui_emit
from workflows import hello_workflow, make_roster

from langchain_dynamic_workflow import ProgressEntry, run_workflow

HOST_INSTRUCTIONS = (
    "You are a helpful assistant for the dynamic-workflow demo. "
    "When the user wants to see the demo workflow run, call run_hello_demo. "
    "Otherwise answer the user's questions directly and concisely."
)


class HostState(DeepAgentState):
    """Host graph state: the deepagent state plus a Generative-UI message channel.

    Attributes:
        ui: Generative-UI messages pushed via ``push_ui_message`` during a turn.
            Reduced by ``ui_message_reducer`` so pushes append/merge into the
            channel the frontend renders.
    """

    ui: Annotated[list[AnyUIMessage], ui_message_reducer]


def _anchor_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Return the AI message that UI events should attach to.

    The frontend renders a UI message under the chat message whose id matches the
    UI event's ``metadata.message_id``. The right anchor is the most recent AI
    message (the one whose tool call invoked the pushing tool), so pushed
    components appear inline beneath the host's turn.

    Args:
        messages: The conversation messages from graph state.

    Returns:
        The most recent :class:`AIMessage`, or ``None`` if there is none.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


@tool
async def run_hello_demo(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Run the demo workflow, streaming its progress into the UI as it goes.

    Pushes a trivial ``hello_ui`` component to confirm the Generative-UI round-trip,
    then runs ``hello_workflow`` inline with a progress sink that pushes a
    ``phase_timeline`` component for each phase/log entry live (from within this node
    context). The sink swallows every exception and never blocks, so a UI push
    failure can never break orchestration.

    Returns:
        A short human-readable summary of what ran.
    """
    # Capture the host node context now so UI events emitted from inside the nested
    # engine run still target the host graph's stream and ``ui`` channel.
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)

    # Round-trip 1: a trivial local component renders from the node context.
    emit("hello_ui", {"text": "hello from backend", "event_id": "hello-1"})

    # Round-trip 2: inline workflow run with a live, non-blocking progress sink.
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
    return f"Demo workflow finished: {result}"


def make_host_graph() -> Any:
    """Build the host deepagent graph served by ``langgraph dev``.

    Resolves the host model from the environment (a real provider model when a key
    is present, an offline scripted fallback otherwise), extends the deepagent state
    with a ``ui`` channel, and registers the ``run_hello_demo`` tool so the host can
    drive the Generative-UI + inline-run round-trips.

    Returns:
        The compiled deepagent host graph (a runnable LangGraph graph).
    """
    return create_deep_agent(
        model=resolve_host_model(),
        system_prompt=HOST_INSTRUCTIONS,
        tools=[run_hello_demo],
        state_schema=HostState,
    )

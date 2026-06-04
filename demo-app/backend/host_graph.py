"""Host deepagent graph factory for the interactive demo.

``langgraph dev`` loads :func:`make_host_graph` (registered under the ``host`` graph
id in ``langgraph.json``) and serves it on the local API. The host carries a ``ui``
state channel (reduced by ``ui_message_reducer``) so Generative-UI components render
in the chat, and two tools that drive inline workflow runs whose progress/span hooks
flow through a :class:`~ui_adapter.UiAdapter`:

* ``run_hello_demo`` — runs the trivial ``hello_workflow`` and also pushes a
  ``hello_ui`` component, proving the Generative-UI round-trip from a node context.
* ``run_live`` — the generic entry: resolves a named preset workflow
  (``deep_research`` / ``capstone``) from :func:`~workflows.make_workflows`, runs it
  inline against the real roster, and streams its phase / fan-out / span events live.

Both tools capture the host node context up front (``ui_bridge.make_host_ui_emit``)
so events emitted from deep inside the nested engine run still target the host
graph's stream and ``ui`` channel. The shared inline-run mechanics live in
:func:`run_workflow_live`, which is decoupled from the tool decorator so it can be
tested without a node context.
"""

from __future__ import annotations

from typing import Annotated, Any

from _models import is_offline, resolve_host_model
from deepagents import DeepAgentState, create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langgraph.prebuilt import InjectedState
from ui_adapter import UiAdapter
from ui_bridge import UiEmit, make_host_ui_emit
from workflows import hello_workflow, make_roster, make_workflows

from langchain_dynamic_workflow import Ctx, run_workflow

HOST_INSTRUCTIONS = (
    "You are a thorough assistant for the dynamic-workflow demo. A hard, multi-step "
    "task is best tackled by decomposing it into independent sub-work, investigating "
    "those parts in parallel, cross-checking the findings before committing to them, "
    "and only then synthesizing a clear answer — and that heavier orchestration runs "
    "as a workflow rather than being crammed into a single pass. When the user wants "
    "to drive a preset scenario live, run it through the live workflow tool and let "
    "its progress stream into the panel. Otherwise answer directly and concisely."
)

# The preset the offline host (and a key-free user) triggers by default. A richer,
# fan-out-heavy scenario than the hello smoke path, so the demo shows real
# control-flow inversion out of the box.
DEFAULT_LIVE_WORKFLOW = "deep_research"


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


# Component name for the one-time run-status banner the host emits each turn. Carries
# the real offline state (no provider key) so the frontend can surface its offline
# banner from a true backend signal rather than a hardcoded flag.
_RUN_STATUS = "run_status"


def _emit_run_status(emit: UiEmit) -> None:
    """Emit a one-time ``run_status`` UI event carrying the real offline state.

    The offline flag is resolved by :func:`~_models.is_offline` (the same provider-key
    gate :func:`~_models.resolve_host_model` uses), so the frontend's offline banner
    reflects true backend state. The ``event_id`` is fixed so a re-emit on the same
    turn dedupes via the SDK's ui-message reducer.

    Args:
        emit: The host-bound, non-blocking UI emit (rebinds the host node context).
    """
    emit(_RUN_STATUS, {"offline": is_offline(), "event_id": "run-status-1"})


@tool
async def run_hello_demo(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Run the demo workflow, streaming its progress into the UI as it goes.

    Pushes a trivial ``hello_ui`` component to confirm the Generative-UI round-trip,
    then runs ``hello_workflow`` inline, feeding the engine's progress/span hooks
    through a :class:`~ui_adapter.UiAdapter`. The adapter maps each engine event to a
    Gen-UI component (``phase_timeline`` for progress, ``fanout_graph`` /
    ``agent_span`` / ``journal_badge`` for spans), stamps a stable content-based
    ``event_id`` so re-emits dedupe, and swallows transport failures so a UI push can
    never break orchestration. The host-bound ``emit`` rebinds the captured node
    context so events stream live from inside the nested engine run.

    Returns:
        A short human-readable summary of what ran.
    """
    # Capture the host node context now so UI events emitted from inside the nested
    # engine run still target the host graph's stream and ``ui`` channel.
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)
    _emit_run_status(emit)

    # Round-trip 1: a trivial local component renders from the node context.
    emit("hello_ui", {"text": "hello from backend", "event_id": "hello-1"})

    # Round-trip 2: inline workflow run whose progress/span hooks flow through the
    # adapter (event mapping + stable-id dedupe + non-blocking emit).
    adapter = UiAdapter(emit=emit)
    result = await run_workflow(
        hello_workflow,
        roster=make_roster(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
    )
    return f"Demo workflow finished: {result}"


async def run_workflow_live(name: str, args: dict[str, Any], *, adapter: UiAdapter) -> str:
    """Resolve a named preset workflow and run it inline, streaming via ``adapter``.

    This is the engine-facing core of the :func:`run_live` host tool, kept free of
    the node-context machinery so it can be exercised directly in tests. It resolves
    ``name`` from :func:`~workflows.make_workflows`, wraps the two-argument
    ``WorkflowFn`` into the single-argument orchestrator ``run_workflow`` expects
    (binding ``args``), and runs it against the real roster with the adapter's sinks
    wired to ``on_progress`` / ``on_span``. The same registry is also passed as
    ``workflows=`` so a preset that nests ``ctx.workflow(...)`` resolves too.

    Args:
        name: The registered preset workflow name (e.g. ``"deep_research"``).
        args: Arguments forwarded to the workflow (an empty mapping is fine; presets
            fall back to their own defaults).
        adapter: The :class:`~ui_adapter.UiAdapter` whose sinks receive the run's
            progress and span events.

    Returns:
        The workflow's result, coerced to ``str`` for the tool's text reply.

    Raises:
        KeyError: If ``name`` is not a registered preset (the registry lists the
            available names).
    """
    workflows = make_workflows()
    workflow_fn = workflows.resolve(name)  # fail-loud on an unknown name

    async def _orchestrate(ctx: Ctx) -> Any:
        return await workflow_fn(ctx, args)

    result = await run_workflow(
        _orchestrate,
        roster=make_roster(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
        workflows=workflows,
    )
    return str(result)


@tool
async def run_live(
    state: Annotated[dict[str, Any], InjectedState],
    workflow: str = DEFAULT_LIVE_WORKFLOW,
    args: dict[str, Any] | None = None,
) -> str:
    """Run a named preset workflow live, streaming its orchestration into the UI.

    Resolves ``workflow`` (e.g. ``deep_research`` or ``capstone``) from the preset
    registry and runs it inline against the real roster. Each engine progress/span
    event flows through a :class:`~ui_adapter.UiAdapter`, which maps it to a Gen-UI
    component (``phase_timeline`` for phases/logs, ``fanout_graph`` for parallel /
    pipeline fan-out, ``agent_span`` / ``journal_badge`` for leaves) with a stable
    content ``event_id`` for dedupe, and pushes it live from inside the same node
    context — so the chat shows the control-flow inversion as it happens.

    Args:
        state: Injected graph state (used to anchor UI events to the host turn).
        workflow: The preset workflow to run; defaults to ``deep_research``.
        args: Optional workflow arguments (e.g. ``{"question": "..."}``). When
            omitted the preset uses its own defaults.

    Returns:
        A short human-readable summary including the workflow's result.
    """
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)
    _emit_run_status(emit)
    adapter = UiAdapter(emit=emit)
    result = await run_workflow_live(workflow, args or {}, adapter=adapter)
    return f"Workflow {workflow!r} finished: {result}"


def make_host_graph() -> Any:
    """Build the host deepagent graph served by ``langgraph dev``.

    Resolves the host model from the environment (a real provider model when a key
    is present, an offline scripted fallback otherwise), extends the deepagent state
    with a ``ui`` channel, and registers the ``run_hello_demo`` and ``run_live`` tools
    so the host can drive the Generative-UI round-trip and inline preset runs.

    Returns:
        The compiled deepagent host graph (a runnable LangGraph graph).
    """
    return create_deep_agent(
        model=resolve_host_model(),
        system_prompt=HOST_INSTRUCTIONS,
        tools=[run_hello_demo, run_live],
        state_schema=HostState,
    )

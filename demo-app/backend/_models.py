"""Host model resolution for the demo backend (BYO-key with an offline fallback).

The demo follows the project's offline-first discipline: with a real model key in
the environment the host runs against that model, but with no key the graph must
still build and register so ``langgraph dev`` boots and the Gen-UI round-trip is
demonstrable without any credentials. When no key is present this returns a
deterministic offline host model that drives the same tool-call turn logic a real
model would, with the turn logic encoded in code rather than a prompt.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Tools the offline host can call to drive the Gen-UI + inline-run round-trips.
_HELLO_TOOL_NAME = "run_hello_demo"
_LIVE_TOOL_NAME = "run_live"
_META_TOOL_NAME = "run_meta_script"
_BACKGROUND_TOOL_NAME = "run_background"

# Post-tool final replies, branched on which tool ran (the most recent ToolMessage's
# ``name``). The offline demo's whole discipline is honesty, so this final sentence must
# match what each tool actually did: the inline tools (run_live / run_hello_demo) and a
# gate-PASS meta run DO stream live progress into the panel; a run_background detaches and
# streams NOTHING live (lifecycle + result only); a gate-FAIL meta run executes nothing.
# A ToolMessage whose content marks the rejected-gate path is detected by content because
# both meta outcomes share the ``run_meta_script`` tool name.
_REPLY_LIVE_STREAMED = (
    "Done — the workflow ran and streamed its progress into the panel above "
    "(offline demo mode; set a model key for live model runs)."
)
_REPLY_BACKGROUND = (
    "Done — the background run finished off-thread; I reported its lifecycle status and "
    "the final result above. A detached background run can't stream its progress live "
    "into the panel (offline demo mode; set a model key for live model runs)."
)
_REPLY_META_REJECTED = (
    "The AST gate rejected the script I composed, so nothing ran — the rejection reason "
    "is shown above (offline demo mode; set a model key for live model runs)."
)

# Marker in a ``run_meta_script`` ToolMessage that distinguishes the gate-FAIL outcome
# from the gate-PASS one (both share the tool name). Kept in sync with the tool's
# rejection return text in ``host_graph.run_meta_script_live``.
_META_REJECTED_MARKER = "rejected by the ast gate"

# Cue words in the user's message that route the offline host to a live preset run
# instead of the default hello smoke path. The resume cues ("pick it back up" /
# "where you left off" / "resume") deliberately route to the SAME live tool: a second
# live run on the same chat thread reuses the prior run's durable journal lane, so the
# replayed leaves come back cached — the honest "pick it back up" story (journal re-run
# replay, not a mid-flight interrupt the engine does not have).
_LIVE_CUES = (
    "research",
    "deep",
    "capstone",
    "workflow",
    "scenario",
    "fact-check",
    "pick it back up",
    "where you left off",
    "resume",
)

# Cue word -> preset workflow name. A request naming a preset routes the offline host
# to THAT preset, not just the default; absent any named cue the host falls back to the
# tool's own default workflow (so the args stay empty and the preset is chosen there).
_WORKFLOW_CUES: dict[str, str] = {"capstone": "capstone"}

# Cue phrases routing the offline host to the meta layer (run_meta_script): the user has
# no ready-made procedure and wants the host to compose one on the spot. Checked BEFORE
# the live cues so a "no playbook, work out a procedure (research a few topics...)"
# request reaches the meta tool rather than a preset, even though it mentions "research".
_META_CUES = ("no standard playbook", "no playbook", "work out a procedure", "novel task")

# Cue phrases routing the offline host to the meta layer's gate-FAIL path: the user wants
# to SEE the AST gate reject an unsafe script. Routes to the same ``run_meta_script`` tool
# but with ``submit_rejected=True``, so the import-bearing fixture is submitted, the gate
# rejects it, and nothing runs. Checked among the meta cues; a match makes ``_wants_meta_run``
# true (so meta wins over live) and flips the tool's ``submit_rejected`` arg on.
_REJECTED_META_CUES = (
    "rejected script",
    "blocked script",
    "unsafe script",
    "gate reject",
    "show me a rejected",
)

# Cue phrases routing the offline host to a background run (run_background): the user
# wants a heavy job taken off their hands to run while they do other things. Checked
# BEFORE the live cues so a "take it off my hands / run it in the background" request
# reaches the background tool rather than an inline preset run.
_BACKGROUND_CUES = (
    "off my hands",
    "take it off",
    "in the background",
    "don't want to babysit",
    "do not want to babysit",
    "without babysitting",
    "delegate",
)


def _latest_user_text(messages: Sequence[BaseMessage]) -> str | None:
    """Return the most recent human message's lowercased text, if any."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text.lower()
    return None


def _wants_live_run(messages: Sequence[BaseMessage]) -> bool:
    """Return whether the latest user turn asks for a live preset run.

    A key-free user should still be able to trigger ``run_live`` for a preset
    scenario, not only the hello smoke path. The most recent human message is
    inspected for scenario cue words; absent any, the host stays on the default
    hello path.
    """
    text = _latest_user_text(messages)
    return text is not None and any(cue in text for cue in _LIVE_CUES)


def _wants_meta_run(messages: Sequence[BaseMessage]) -> bool:
    """Return whether the latest user turn asks the host to compose a procedure itself.

    A "no ready-made playbook, work out a procedure" request is the meta-layer cue: the
    host should author an orchestration script on the spot rather than launch a preset.
    A "show me a rejected/blocked script" request is also a meta cue (the gate-FAIL
    variant). Checked before the live cue so such a request reaches ``run_meta_script``
    even when it also mentions "research".
    """
    text = _latest_user_text(messages)
    if text is None:
        return False
    return any(cue in text for cue in (*_META_CUES, *_REJECTED_META_CUES))


def _wants_rejected_meta(messages: Sequence[BaseMessage]) -> bool:
    """Return whether the latest user turn asks to SEE the AST gate reject a script.

    This is the meta-layer gate-FAIL cue ("show me a rejected/blocked/unsafe script"):
    the host submits the import-bearing fixture so the gate rejects it and nothing runs.
    Routes to the same ``run_meta_script`` tool as :func:`_wants_meta_run`, but flips its
    ``submit_rejected`` arg on so the rejected fixture (not the clean one) is submitted.
    """
    text = _latest_user_text(messages)
    return text is not None and any(cue in text for cue in _REJECTED_META_CUES)


def _wants_background_run(messages: Sequence[BaseMessage]) -> bool:
    """Return whether the latest user turn asks to hand a heavy job off to the background.

    A "take it off my hands / run it in the background / don't want to babysit" request
    is the background cue: the host should launch the run detached and report its
    lifecycle status rather than block the turn on an inline run. Checked before the
    live cue so such a request reaches ``run_background``.
    """
    text = _latest_user_text(messages)
    return text is not None and any(cue in text for cue in _BACKGROUND_CUES)


def _latest_tool_message(messages: Sequence[BaseMessage]) -> ToolMessage | None:
    """Return the most recent :class:`ToolMessage` in the turn, if any.

    Its presence means a tool has already run this turn, so the host should emit its
    final answer rather than another tool call; its ``name``/content drive which honest
    post-tool reply :func:`_post_tool_reply` returns.
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return message
    return None


def _post_tool_reply(tool_message: ToolMessage) -> str:
    """Return the honest final reply for the tool that just ran.

    The reply must match what the tool actually did — the demo's discipline is offline
    honesty, so a blanket "streamed its progress into the panel" sentence would lie for
    two of the four tools. Branching on the most recent :class:`ToolMessage`:

    * ``run_background`` — the detached run streams NOTHING live, so the reply says the
      lifecycle status and final result were reported (no live stream).
    * ``run_meta_script`` gate-FAIL — detected by the rejection marker in the tool's
      content (both meta outcomes share the tool name); the reply says the gate rejected
      the script and nothing ran.
    * ``run_live`` / ``run_hello_demo`` / ``run_meta_script`` gate-PASS — these do stream
      live, so the reply keeps the "streamed into the panel" wording.

    Args:
        tool_message: The most recent tool result in the conversation.

    Returns:
        The honest final-answer text for the tool that ran.
    """
    if tool_message.name == _BACKGROUND_TOOL_NAME:
        return _REPLY_BACKGROUND
    content = str(tool_message.content).lower()
    if tool_message.name == _META_TOOL_NAME and _META_REJECTED_MARKER in content:
        return _REPLY_META_REJECTED
    return _REPLY_LIVE_STREAMED


def _requested_workflow(messages: Sequence[BaseMessage]) -> str | None:
    """Return the preset workflow the latest user turn names, if any.

    A request that names a preset (e.g. "run the capstone scenario") must actually run
    THAT preset offline, not silently fall through to the default. The most recent human
    message is scanned for a workflow cue; absent any, ``None`` lets the live tool pick
    its own default.
    """
    text = _latest_user_text(messages)
    if text is None:
        return None
    return next((name for cue, name in _WORKFLOW_CUES.items() if cue in text), None)


class OfflineHostModel(BaseChatModel):
    """A deterministic, key-free host model for the offline demo path.

    The turn logic is encoded in code (this is a scripted offline host, exempt from
    the persona rule): on the first turn it calls a demo tool so the Gen-UI and
    inline-run round-trips can be exercised; once a tool result is present it emits a
    short final answer. Which tool it calls depends on the user's intent — a scenario
    request (e.g. "deep research") drives ``run_live`` for a preset, while any other
    ask falls back to the default ``run_hello_demo`` smoke path. This keeps the host
    graph buildable and ``langgraph dev`` bootable with no API key.
    """

    @property
    def _llm_type(self) -> str:
        return "offline-host-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Accept tool bindings as a no-op.

        deepagents binds the host's tools onto the model. The offline host already
        hardcodes which tool to call, so binding is a no-op that returns ``self``.
        """
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        latest_tool = _latest_tool_message(messages)
        if latest_tool is not None:
            message = AIMessage(content=_post_tool_reply(latest_tool))
        elif _wants_meta_run(messages):
            # No ready-made playbook: author a script and submit it through the meta
            # layer. A "show me a rejected/blocked script" cue flips submit_rejected on
            # (gate-FAIL fixture, nothing runs); otherwise the clean gate-PASS fixture is
            # submitted. Checked before the live cue so a "work out a procedure
            # (research...)" request reaches the meta tool, not a preset.
            rejected = _wants_rejected_meta(messages)
            content = (
                "You want to see the gate stop an unsafe script — submitting one now."
                if rejected
                else "No ready-made procedure fits — composing one and running it now."
            )
            message = AIMessage(
                content=content,
                tool_calls=[
                    {
                        "name": _META_TOOL_NAME,
                        "args": {"submit_rejected": rejected},
                        "id": "meta-call-1",
                    }
                ],
            )
        elif _wants_background_run(messages):
            # Heavy job to hand off: launch it in the background and report its
            # lifecycle. Checked before the live cue so a "take it off my hands" request
            # reaches the background tool rather than an inline preset run.
            message = AIMessage(
                content="Taking it off your hands — launching it in the background now.",
                tool_calls=[{"name": _BACKGROUND_TOOL_NAME, "args": {}, "id": "bg-call-1"}],
            )
        elif _wants_live_run(messages):
            # Route to the preset the user named (e.g. capstone); absent a named cue,
            # leave args empty so the live tool runs its own default workflow.
            requested = _requested_workflow(messages)
            live_args: dict[str, Any] = {"workflow": requested} if requested else {}
            message = AIMessage(
                content=f"Running the {requested or 'deep-research'} workflow now.",
                tool_calls=[{"name": _LIVE_TOOL_NAME, "args": live_args, "id": "live-call-1"}],
            )
        else:
            message = AIMessage(
                content="Running the demo workflow now.",
                tool_calls=[{"name": _HELLO_TOOL_NAME, "args": {}, "id": "demo-call-1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def is_offline() -> bool:
    """Return whether the demo is running without a provider key (offline mode).

    The single source of truth for the demo's online/offline state, gating on the same
    keys :func:`resolve_host_model` and :func:`resolve_leaf_model` consult. With neither
    ``OPENAI_API_KEY`` nor ``OPENROUTER_API_KEY`` present the host falls back to the
    scripted :class:`OfflineHostModel` and the roster swaps in fake leaves, so this is
    the honest signal the frontend uses to show its offline banner.

    Returns:
        ``True`` when no provider key is present (offline), ``False`` otherwise.
    """
    return not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))


def resolve_host_model() -> BaseChatModel:
    """Return the host chat model: a real provider model, or the offline fallback.

    Reads the already-loaded environment. With ``OPENAI_API_KEY`` present the host
    runs against ``gpt-4o-mini``; with only ``OPENROUTER_API_KEY`` present it routes
    through OpenRouter; with neither it returns :class:`OfflineHostModel` so the
    graph still builds and ``langgraph dev`` boots without credentials.

    Returns:
        A configured :class:`~langchain_core.language_models.chat_models.BaseChatModel`.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini")
    if os.environ.get("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("LDW_DEMO_HOST_MODEL", "openai/gpt-4o-mini"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],  # type: ignore[arg-type]
        )
    return OfflineHostModel()


def resolve_leaf_model() -> BaseChatModel | None:
    """Return a chat model for workflow leaves, or ``None`` to run them offline.

    Mirrors :func:`resolve_host_model`'s BYO-key gating but signals the offline path
    with ``None`` instead of a scripted model: a leaf is a real ``create_deep_agent``
    only when a provider key is present, otherwise the roster swaps in a deterministic
    fake leaf so an offline run stays reproducible and needs no credentials.

    Returns:
        A configured chat model when a key is present, else ``None``.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("LDW_DEMO_LEAF_MODEL", "gpt-4o-mini"))
    if os.environ.get("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("LDW_DEMO_LEAF_MODEL", "openai/gpt-4o-mini"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],  # type: ignore[arg-type]
        )
    return None


__all__: Sequence[str] = [
    "OfflineHostModel",
    "is_offline",
    "resolve_host_model",
    "resolve_leaf_model",
]

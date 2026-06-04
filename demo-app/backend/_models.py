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

# Cue words in the user's message that route the offline host to a live preset run
# instead of the default hello smoke path.
_LIVE_CUES = ("research", "deep", "capstone", "workflow", "scenario", "fact-check")

# Cue word -> preset workflow name. A request naming a preset routes the offline host
# to THAT preset, not just the default; absent any named cue the host falls back to the
# tool's own default workflow (so the args stay empty and the preset is chosen there).
_WORKFLOW_CUES: dict[str, str] = {"capstone": "capstone"}


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
        already_ran_tool = any(isinstance(m, ToolMessage) for m in messages)
        if already_ran_tool:
            message = AIMessage(
                content=(
                    "Done — the workflow ran and streamed its progress into the panel "
                    "above (offline demo mode; set a model key for live model runs)."
                )
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


__all__: Sequence[str] = ["OfflineHostModel", "resolve_host_model", "resolve_leaf_model"]

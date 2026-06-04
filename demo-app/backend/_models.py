"""Host model resolution for the demo backend (OpenRouter BYO-key, offline fallback).

The provider is LOCKED to OpenRouter and the models are FIXED in code: there is no
user-facing model configuration. A single OpenRouter API key — supplied per session
through the run config (``config.configurable.openrouter_api_key``) or, failing that,
the backend ``.env`` ``OPENROUTER_API_KEY`` — is all that is needed to go online. With
no key anywhere the graph must still build and register so ``langgraph dev`` boots and
the Gen-UI round-trip is demonstrable without any credentials; in that case the host
falls back to a deterministic scripted model that drives the same tool-call turn logic
a real model would, with the turn logic encoded in code rather than a prompt.

Per-session key round-trip. The host model is built once at graph-build time
(``make_host_graph``), but the key arrives per run. :class:`LazyOpenRouterHostModel`
bridges that gap: it is a thin :class:`BaseChatModel` wrapper that, on each
generate call, reads the current run config via ``langgraph.config.get_config`` and
builds the real OpenRouter-backed ``ChatOpenRouter`` from that run's key — so a fresh key
on every session threads through without rebuilding the graph. The leaf models
(:func:`resolve_leaf_model`, used in ``workflows.make_roster``) are baked into their
deepagents at roster-build time; since the roster is built inside the host node where
the run config is visible, the per-run key is captured there and passed in explicitly.

Fixed models. ``HOST_MODEL`` is a strong model that must reliably drive multi-step
tool calls (an M1 real-model finding showed weak models such as ``gpt-4o-mini`` /
``haiku`` cannot sustain the host's multi-turn orchestration); ``LEAF_MODEL`` is an
economical model for the research fan-out. Both are module constants so they are
trivially swappable.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openrouter import ChatOpenRouter

# ── locked provider + fixed models ───────────────────────────────────────────

# The single OpenRouter base URL the demo routes every real call through. The provider
# is LOCKED to OpenRouter; there is no other real provider in the headline path.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The strong host model. The host drives the demo's multi-step tool calls (launch an
# inline preset run, compose + gate a meta script, hand a job to the background) across
# turns, which an M1 real-model finding showed weak models (gpt-4o-mini / haiku) cannot
# do reliably. ``claude-opus-4.8`` is the most capable current OpenRouter id and matches
# the engine examples' ``DEFAULT_OPENROUTER_MODEL``; swap this constant to change it.
HOST_MODEL = "anthropic/claude-opus-4.8"

# The leaf model for the research fan-out. ``claude-sonnet-4.6`` is strong enough to
# drive the native web-search tool reliably (haiku-class models route the search poorly)
# while staying cheaper than the opus host; swap this constant to change the leaf model.
LEAF_MODEL = "anthropic/claude-sonnet-4.6"

# Lock OpenRouter routing to Anthropic's first-party endpoint, no fallback. REQUIRED:
# the native web tools below and Anthropic prompt caching only work on the Anthropic
# provider — a silent fallback to Amazon Bedrock / Google Vertex would drop both.
ANTHROPIC_PROVIDER: dict[str, Any] = {"order": ["Anthropic"], "allow_fallbacks": False}

# OpenRouter's native server-side web search. ``engine="native"`` forces the underlying
# provider's own search; since the demo is provider-locked to Anthropic and runs
# ``anthropic/*`` models, that is Anthropic's built-in web search — reached through
# OpenRouter's unified ``openrouter:web_search`` tool type (the ``openrouter`` SDK that
# backs ``ChatOpenRouter`` validates tool types and rejects the raw Anthropic
# ``web_search_20250305`` spec, but accepts this one). The search runs server-side and
# returns results + citations inline; the model also fetches result pages as part of the
# search. (``openrouter:web_fetch`` is only a *result* type in the current ``openrouter``
# SDK, not a requestable input tool, so it is not bound here — it would fail validation.)
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "openrouter:web_search",
    "parameters": {"engine": "native", "max_results": WEB_SEARCH_MAX_RESULTS},
}
_WEB_TOOLS: tuple[dict[str, Any], ...] = (WEB_SEARCH_TOOL,)

# The run-config field carrying the per-session OpenRouter key. The frontend threads the
# user's one key here on the LangGraph run config; it is read per run (not at graph-build
# time) so each session uses its own key. Named explicitly so it is NOT a secret-logged
# field — the value is a credential and must never be echoed into logs or UI.
RUN_CONFIG_KEY_FIELD = "openrouter_api_key"

# Optional internal escape hatch (NOT the headline path): an operator can pin a different
# model id via these env vars for local experiments. The default — and the only path the
# UI / docs describe — is the fixed HOST_MODEL / LEAF_MODEL constants above.
_HOST_MODEL_ENV_OVERRIDE = "LDW_DEMO_HOST_MODEL"
_LEAF_MODEL_ENV_OVERRIDE = "LDW_DEMO_LEAF_MODEL"

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


def _tool_message_this_turn(messages: Sequence[BaseMessage]) -> ToolMessage | None:
    """Return the :class:`ToolMessage` produced for the CURRENT user turn, if any.

    A tool ran for THIS turn only if a ``ToolMessage`` appears AFTER the most recent
    ``HumanMessage``. Scanning the whole history instead is a multi-turn trap: under
    ``langgraph dev`` the thread state ACCUMULATES messages, so turn 1's ``ToolMessage``
    lingers forever — and a whole-history check would then make the host emit its canned
    final reply on every later turn, so a second scenario on the same thread would never
    fire its tool. (In-process tests that pass a fresh single-message list per turn, or
    call the run helper directly, do not accumulate and so mask this.) Scoping to the
    messages after the last human turn fixes it: each turn independently runs its tool,
    then replies.

    Its presence means a tool already ran this turn, so the host emits its final answer
    rather than another tool call; its ``name`` / content drive which honest post-tool
    reply :func:`_post_tool_reply` returns.
    """
    last_human = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    for message in messages[last_human + 1 :]:
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
        this_turn_tool = _tool_message_this_turn(messages)
        if this_turn_tool is not None:
            message = AIMessage(content=_post_tool_reply(this_turn_tool))
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


# ── per-session OpenRouter key resolution ─────────────────────────────────────


def _run_config_openrouter_key() -> str | None:
    """Return the per-run OpenRouter key from the current run config, if present.

    The frontend threads the user's key onto the LangGraph run config under
    ``configurable.openrouter_api_key`` (:data:`RUN_CONFIG_KEY_FIELD`). This reads it
    via ``langgraph.config.get_config``, which resolves the *current* runnable
    context's config. It is callable only inside a runnable context (a graph node, or
    the host model's generate call, which runs inside the host node); outside one
    ``get_config`` raises ``RuntimeError`` and there is simply no per-run key, so this
    returns ``None`` and the caller falls back to the environment.

    The import is local on purpose: ``langgraph.config`` is only meaningful inside a
    running graph, and keeping the dependency at the call boundary lets the resolvers
    be exercised in a plain unit test without a node context.

    Returns:
        The per-run OpenRouter key when the run config carries one, else ``None``.
    """
    try:
        from langgraph.config import get_config

        config = get_config()
    except (RuntimeError, ImportError):
        # No runnable context (RuntimeError) or langgraph unavailable (ImportError):
        # there is no per-run config to read, so there is no per-run key.
        return None
    configurable = config.get("configurable") or {}
    key = configurable.get(RUN_CONFIG_KEY_FIELD)
    return key if isinstance(key, str) and key else None


def resolve_openrouter_key(*, api_key: str | None = None) -> str | None:
    """Resolve the effective OpenRouter key, honoring the demo's key precedence.

    The single source of truth for "which OpenRouter key (if any) is in force". The
    precedence is: an explicitly supplied ``api_key`` (captured by the caller from the
    run config at the right moment — e.g. the roster builder), then the per-run config
    key (:func:`_run_config_openrouter_key`), then the backend ``.env``
    ``OPENROUTER_API_KEY``. With none of those present the demo is offline.

    Args:
        api_key: An OpenRouter key the caller already resolved (e.g. a leaf builder that
            captured the per-run key inside the host node). Takes precedence over every
            other source so a captured per-session key always wins.

    Returns:
        The effective OpenRouter key, or ``None`` when no key is available (offline).
    """
    if api_key:
        return api_key
    run_key = _run_config_openrouter_key()
    if run_key:
        return run_key
    env_key = os.environ.get("OPENROUTER_API_KEY")
    return env_key if env_key else None


def is_offline() -> bool:
    """Return whether the demo is running without an OpenRouter key (offline mode).

    The single source of truth for the demo's online/offline state, gating on the same
    key sources the model resolvers consult: a per-run ``configurable.openrouter_api_key``
    or the backend ``.env`` ``OPENROUTER_API_KEY``. With neither present the host falls
    back to the scripted :class:`OfflineHostModel` and the roster swaps in fake leaves,
    so this is the honest signal the frontend uses to show its offline banner.

    Returns:
        ``True`` when no OpenRouter key is available (offline), ``False`` otherwise.
    """
    return resolve_openrouter_key() is None


class _WebSearchChatOpenRouter(ChatOpenRouter):
    """A ``ChatOpenRouter`` that appends OpenRouter's native web tools to every binding.

    deepagents binds its own tools onto a leaf model via ``bind_tools``; that call would
    otherwise replace any tools set at construction, dropping web search. This appends the
    web tools (:data:`_WEB_TOOLS`) **raw** — not through ``convert_to_openai_tool``, which
    rewrites a tool into function-call shape and strips the ``openrouter:`` marker the
    server needs — so the search stays available alongside deepagents' tools on every
    request and is executed server-side by OpenRouter.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        bound = super().bind_tools(tools, **kwargs)
        bound_kwargs: dict[str, Any] = dict(getattr(bound, "kwargs", {}))
        bound_kwargs["tools"] = [*bound_kwargs.get("tools", []), *_WEB_TOOLS]
        return self.bind(**bound_kwargs)


def _build_openrouter_model(model: str, api_key: str, *, web_search: bool = False) -> BaseChatModel:
    """Build an OpenRouter-backed ``ChatOpenRouter`` for ``model`` with ``api_key``.

    The locked-provider constructor: every real call routes through OpenRouter pinned to
    Anthropic (:data:`ANTHROPIC_PROVIDER`) with the supplied key. ``ChatOpenRouter`` is
    the same client the engine's runnable examples use, and the one the prompt-caching
    middleware detects to inject Anthropic ``cache_control``.

    Args:
        model: The OpenRouter model id (e.g. :data:`HOST_MODEL`).
        api_key: The OpenRouter key to authenticate with.
        web_search: When ``True``, build a :class:`_WebSearchChatOpenRouter` so the leaf
            carries OpenRouter's native web tools (Anthropic's built-in search, reached
            via the provider lock). Only meaningful for the ``anthropic/*`` leaf model.

    Returns:
        A configured :class:`~langchain_core.language_models.chat_models.BaseChatModel`.
    """
    cls = _WebSearchChatOpenRouter if web_search else ChatOpenRouter
    return cls(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,  # type: ignore[arg-type]  # str coerced to SecretStr by the alias
        openrouter_provider=ANTHROPIC_PROVIDER,
    )


class LazyOpenRouterHostModel(BaseChatModel):
    """A host model that resolves its backend per call from the in-force OpenRouter key.

    The host graph builds its model once at graph-build time, but the OpenRouter key
    arrives per run on the run config. This thin wrapper bridges that gap so the graph
    never has to be rebuilt to go online: it carries no key itself and, on each
    (a)generate call, resolves the effective per-run key (:func:`resolve_openrouter_key`,
    read inside the host node where the run config is visible) and:

    * with a key in force — delegates to a freshly-built OpenRouter-backed ``ChatOpenRouter``
      (:data:`HOST_MODEL`), re-applying any remembered tool binding; and
    * with no key in force — delegates to a :class:`OfflineHostModel`, so a bare
      ``langgraph dev`` boot (no ``.env`` key, no per-session key) still drives the
      scripted demo turn logic and the graph stays bootable.

    This makes online/offline an HONEST per-turn decision: a session that supplies a
    per-run ``configurable.openrouter_api_key`` goes online even when the graph was built
    with no operator key, and a session with no key anywhere stays on the scripted host —
    exactly what :func:`is_offline` reports for that turn.

    Tool bindings are remembered and re-applied to the freshly-built delegate on each
    call, so deepagents' per-request ``bind_tools`` is honored against the real backend.
    """

    bound_tools: tuple[Any, ...] = ()
    bind_kwargs: dict[str, Any] = {}  # noqa: RUF012 — pydantic field default, not mutated in place

    @property
    def _llm_type(self) -> str:
        return "lazy-openrouter-host-model"

    @property
    def _ldw_openrouter_anthropic(self) -> bool:
        """Marker the prompt-caching middleware reads to treat this wrapper as cacheable.

        The wrapper itself is not a ``ChatOpenRouter``, but its per-call delegate is one
        (OpenRouter pinned to Anthropic) and it forwards the middleware's
        ``cache_control``-bearing messages straight through, so the cache breakpoints
        still reach OpenRouter. See ``prompt_caching._is_openrouter_anthropic_model``.
        """
        return True

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Remember the tool binding so it can be re-applied to the per-call delegate.

        deepagents binds the host's tools onto the model per request. Returning a copy
        that records the tools (rather than the SDK's ``RunnableBinding``) keeps this a
        :class:`LazyOpenRouterHostModel`, so the per-call delegate resolution still runs
        and the tools are re-bound onto the freshly-built backend.

        Args:
            tools: The tools to bind on each delegate.
            **kwargs: Extra bind kwargs (e.g. ``tool_choice``) forwarded to the delegate.

        Returns:
            A copy of this wrapper carrying the remembered tools and kwargs.
        """
        return self.model_copy(update={"bound_tools": tuple(tools), "bind_kwargs": dict(kwargs)})

    def _resolve_delegate(self) -> BaseChatModel:
        """Resolve this call's backend: OpenRouter when a key is in force, else offline.

        Re-applies any remembered tool binding to the resolved delegate so deepagents'
        per-request ``bind_tools`` is honored on whichever backend is selected.
        """
        api_key = resolve_openrouter_key()
        if api_key is None:
            # No key in force this turn: drive the scripted offline host so the graph
            # stays bootable and a key-free session still demonstrates the round-trip.
            delegate: BaseChatModel = OfflineHostModel()
        else:
            model = os.environ.get(_HOST_MODEL_ENV_OVERRIDE) or HOST_MODEL
            delegate = _build_openrouter_model(model, api_key)
        if self.bound_tools:
            return delegate.bind_tools(list(self.bound_tools), **self.bind_kwargs)  # type: ignore[return-value]
        return delegate

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self._resolve_delegate().invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=_as_ai_message(result))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await self._resolve_delegate().ainvoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=_as_ai_message(result))])


def _as_ai_message(result: Any) -> AIMessage:
    """Coerce an invoke result into an :class:`AIMessage` for a ``ChatResult``.

    A bound model's ``invoke`` already returns an ``AIMessage`` (the tools were bound
    via :meth:`LazyOpenRouterHostModel.bind_tools`), so this is an identity passthrough
    in practice; the guard keeps the wrapper honest if a delegate ever returns a bare
    ``BaseMessage``.

    Args:
        result: The value returned by the delegate's invoke/ainvoke.

    Returns:
        The result as an :class:`AIMessage`.
    """
    if isinstance(result, AIMessage):
        return result
    return AIMessage(content=getattr(result, "content", str(result)))


def resolve_host_model() -> BaseChatModel:
    """Return the host chat model: always the per-call lazy OpenRouter/offline model.

    Called once at graph-build time, where no run is in flight and the per-session key
    is not yet visible. Rather than freeze the online/offline decision at build time
    (which would strand a per-session-keyed session on the offline model when the graph
    was built with no operator ``.env`` key), this always returns a
    :class:`LazyOpenRouterHostModel`. That wrapper decides online vs. offline PER TURN
    inside the host node, where the run config is visible: with a key in force it drives
    the real :data:`HOST_MODEL` over OpenRouter, with none it drives the scripted
    :class:`OfflineHostModel`. So the same built graph serves a key-free ``langgraph
    dev`` boot and a per-session-keyed session honestly.

    Returns:
        A :class:`LazyOpenRouterHostModel` that self-resolves its backend per call.
    """
    return LazyOpenRouterHostModel()


def resolve_leaf_model(
    *, api_key: str | None = None, web_search: bool = False
) -> BaseChatModel | None:
    """Return a chat model for workflow leaves, or ``None`` to run them offline.

    Mirrors :func:`resolve_host_model`'s key gating but signals the offline path with
    ``None`` instead of a scripted model: a leaf is a real ``create_deep_agent`` only
    when an OpenRouter key is in force, otherwise the roster swaps in a deterministic
    fake leaf so an offline run stays reproducible and needs no credentials.

    Unlike the host model, a leaf model is baked into its deepagent at roster-build time
    (deepagents resolves the leaf model once, not per request). The roster is built
    inside the host node, where the run config is visible, so the caller (``make_roster``)
    captures the per-run key there and passes it as ``api_key`` — and the leaf is built
    eagerly with that exact key.

    Args:
        api_key: The OpenRouter key the caller captured from the run config (or ``None``
            to fall back to the per-run config / ``.env`` resolution). When supplied it
            takes precedence so a captured per-session key threads into the leaf.
        web_search: When ``True``, the leaf model carries Anthropic's native web search
            tool (see :func:`_build_openrouter_model`) so research / verify leaves ground
            their work in live web sources. Has no effect on the offline path.

    Returns:
        A configured OpenRouter leaf model when a key is in force, else ``None``.
    """
    effective_key = resolve_openrouter_key(api_key=api_key)
    if effective_key is None:
        return None
    model = os.environ.get(_LEAF_MODEL_ENV_OVERRIDE) or LEAF_MODEL
    return _build_openrouter_model(model, effective_key, web_search=web_search)


def cache_middleware() -> list[Any]:
    """Return the prompt-caching middleware to register on EVERY real demo agent.

    Anthropic prompt caching via OpenRouter (:class:`~prompt_caching.PromptCachingMiddleware`):
    it injects ``cache_control`` breakpoints so the growing system prompt and tool-call
    history are cached across the host's turns and inside each leaf's tool loop — host and
    leaves alike. ``pin_openrouter_provider`` is off because every model here already pins
    the provider to Anthropic (:data:`ANTHROPIC_PROVIDER`), so cache hits are already
    provider-stable. The import is local so the offline path (fake leaves, no middleware)
    pulls nothing from ``prompt_caching``.

    Returns:
        A one-element middleware list to pass as ``create_deep_agent(middleware=...)``.
    """
    from prompt_caching import PromptCachingMiddleware

    return [
        PromptCachingMiddleware(
            progressive=True,
            cache_last_human_message=True,
            pin_openrouter_provider=False,
        )
    ]


__all__: Sequence[str] = [
    "HOST_MODEL",
    "LEAF_MODEL",
    "OPENROUTER_BASE_URL",
    "RUN_CONFIG_KEY_FIELD",
    "WEB_SEARCH_TOOL",
    "LazyOpenRouterHostModel",
    "OfflineHostModel",
    "cache_middleware",
    "is_offline",
    "resolve_host_model",
    "resolve_leaf_model",
    "resolve_openrouter_key",
]

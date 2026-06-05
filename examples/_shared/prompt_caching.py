"""Prompt caching middleware for LangChain agents.

Extends Anthropic's official ``AnthropicPromptCachingMiddleware`` to also
support Anthropic models accessed through OpenRouter.

The official middleware only works with ``ChatAnthropic`` instances because
it checks ``isinstance(request.model, ChatAnthropic)`` and sets caching
via ``model_settings``.  When using ``ChatOpenRouter`` to access Anthropic
models (e.g. ``anthropic/claude-sonnet-4.5``), the ``isinstance`` check
fails and caching is silently skipped.

This middleware adds an OpenRouter-specific code path that places
``cache_control`` breakpoints directly in message content blocks — the
mechanism OpenRouter expects for Anthropic prompt caching.

Progressive caching
-------------------

When *progressive* is enabled (default), the middleware places an
**advancing breakpoint** (BP3) on the last message whenever the agent is
inside a tool-calling loop (i.e. the last message is a ``ToolMessage`` or
an ``AIMessage`` with ``tool_calls``).  This caches the growing tool-call
history so that each subsequent call only pays the write cost for the
newest round of tokens.

Whether BP3 should fire is determined purely by the **message types**
present in the request — no cross-node state tracking is needed.  This
is critical because LangGraph runs each graph node with an isolated
``copy_context()``, making ``ContextVar``-based counters unreliable
across the agent loop.

Cost analysis (``scripts/prompt_cache_cost_analysis.py``) shows this
saves **70–80 %** vs no-cache for agents with 3+ tool-calling rounds,
with quadratically increasing returns as N grows.

See Also:
    - `Anthropic prompt caching <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>`_
    - `OpenRouter prompt caching <https://openrouter.ai/docs/guides/best-practices/prompt-caching>`_
"""

# This module is a near-verbatim port of an upstream middleware whose docstrings
# use an en dash; RUF002 (ambiguous-unicode-character-docstring) would otherwise
# reject prose we copy byte-for-byte. The upstream project ignores this rule too.
# ruff: noqa: RUF002

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Anthropic model name prefixes on OpenRouter
_ANTHROPIC_MODEL_PREFIX = "anthropic/"

# TTL value in minutes — used for ordering validation
type _TTL = Literal["5m", "1h"]
_TTL_MINUTES: dict[str, int] = {"5m": 5, "1h": 60}

# Cache control dict type for content blocks
type CacheControlDict = dict[str, str]

# Content type expected by LangChain message constructors
type _MessageContent = str | list[str | dict[str, Any]]


def _is_openrouter_anthropic_model(model: Any) -> bool:
    """Check whether *model* is a ChatOpenRouter targeting an Anthropic model.

    Args:
        model: The LLM instance from the agent middleware request.

    Returns:
        ``True`` when the model is ``ChatOpenRouter`` (or a subclass such as
        ``RoutedChatOpenRouter``) **and** the model name starts with
        ``anthropic/`` — OR when the model exposes a truthy
        ``_ldw_openrouter_anthropic`` marker. The marker lets the demo's lazy host
        wrapper (which resolves an OpenRouter Anthropic delegate per call and forwards
        the ``cache_control``-bearing messages straight through) opt into caching even
        though it is not itself a ``ChatOpenRouter`` instance.
    """
    if getattr(model, "_ldw_openrouter_anthropic", False):
        return True

    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError:
        return False

    if not isinstance(model, ChatOpenRouter):
        return False

    model_name: str = getattr(model, "model_name", "") or ""
    return model_name.startswith(_ANTHROPIC_MODEL_PREFIX)


def _is_native_anthropic_model(model: Any) -> bool:
    """Check whether *model* is a native ``ChatAnthropic`` instance.

    Args:
        model: The LLM instance from the agent middleware request.

    Returns:
        ``True`` when the model is ``ChatAnthropic`` (or a subclass).
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return False

    return isinstance(model, ChatAnthropic)


# =========================================================================
# Content block helpers
# =========================================================================


def _add_cache_control_to_content(
    content: str | list[dict[str, Any]],
    cache_control: CacheControlDict,
) -> list[dict[str, Any]]:
    """Append ``cache_control`` to the last text block in *content*.

    If *content* is a plain string it is first converted to a single-element
    content block list.  The ``cache_control`` dict is added to the **last**
    text block to mark it as a cache breakpoint.

    Args:
        content: Message content — either a plain string or a list of
            content block dicts.
        cache_control: The cache control dict, e.g.
            ``{"type": "ephemeral"}``.

    Returns:
        A new list of content block dicts with ``cache_control`` attached to
        the last text block.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": cache_control}]

    # Deep-copy to avoid mutating the original messages
    blocks: list[dict[str, Any]] = copy.deepcopy(content)

    # Find and annotate the last text block
    for block in reversed(blocks):
        if isinstance(block, dict) and block.get("type") == "text":
            block["cache_control"] = cache_control
            break

    return blocks


def _strip_cache_control(
    messages: list[AnyMessage],
) -> list[AnyMessage]:
    """Remove ``cache_control`` from all content blocks in *messages*.

    This prevents stale breakpoints from accumulating across model calls.
    Returns new message objects when changes are needed; unchanged messages
    are returned as-is.

    Args:
        messages: The conversation messages to clean.

    Returns:
        A new list with ``cache_control`` stripped from content blocks.
    """
    result: list[AnyMessage] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            needs_strip = any(
                isinstance(block, dict) and "cache_control" in block for block in content
            )
            if needs_strip:
                cleaned = [
                    {k: v for k, v in block.items() if k != "cache_control"}
                    if isinstance(block, dict)
                    else block
                    for block in content
                ]
                msg = _clone_message(msg, content=cleaned)
        result.append(msg)
    return result


def _clone_message(
    msg: AnyMessage,
    *,
    content: Any = None,
) -> AnyMessage:
    """Create a shallow copy of *msg* with optional content override.

    Preserves type-specific fields (``tool_call_id`` for ``ToolMessage``,
    ``tool_calls`` for ``AIMessage``, etc.).

    Args:
        msg: The original message.
        content: Replacement content.  If ``None``, uses ``msg.content``.

    Returns:
        A new message of the same type.
    """
    new_content = content if content is not None else msg.content

    kwargs: dict[str, Any] = {"content": new_content}
    if msg.id:
        kwargs["id"] = msg.id
    if msg.response_metadata:
        kwargs["response_metadata"] = msg.response_metadata

    if isinstance(msg, ToolMessage):
        kwargs["tool_call_id"] = msg.tool_call_id
        if msg.name:
            kwargs["name"] = msg.name
        if msg.status != "success":
            kwargs["status"] = msg.status
        return ToolMessage(**kwargs)

    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            kwargs["tool_calls"] = msg.tool_calls
        return AIMessage(**kwargs)

    if isinstance(msg, HumanMessage):
        return HumanMessage(**kwargs)

    # Fallback — should not normally be reached
    return type(msg)(**kwargs)


def _clone_message_with_cache_control(
    msg: AnyMessage,
    cache_control: CacheControlDict,
) -> AnyMessage:
    """Create a copy of *msg* with ``cache_control`` on its last text block.

    Works with any message type (``HumanMessage``, ``AIMessage``,
    ``ToolMessage``).

    Args:
        msg: The original message.
        cache_control: The cache control dict to inject.

    Returns:
        A new message with ``cache_control`` attached to the last text block.
    """
    cached_content = _add_cache_control_to_content(
        msg.content,  # pyright: ignore[reportArgumentType]
        cache_control,
    )
    return _clone_message(msg, content=cast("_MessageContent", cached_content))


# =========================================================================
# Breakpoint injection
# =========================================================================


def _inject_cache_breakpoints(
    *,
    system_message: SystemMessage | None,
    messages: list[AnyMessage],
    cache_control: CacheControlDict,
    cache_last_human_message: bool = False,
    cache_last_message: bool = False,
    bp2_cache_control: CacheControlDict | None = None,
    bp3_cache_control: CacheControlDict | None = None,
) -> tuple[SystemMessage | None, list[AnyMessage]]:
    """Inject cache breakpoints into messages.

    Anthropic allows up to 4 cache breakpoints per request.  This function
    places breakpoints on:

    1. **System message** (always) — the most stable and token-heavy content.
    2. **Last HumanMessage** (opt-in via *cache_last_human_message*) — marks
       the boundary of the current turn so that tool-use context from
       previous turns can be cached.
    3. **Last message** (opt-in via *cache_last_message*) — progressive
       breakpoint that advances with the growing tool-call history.  On each
       model call, the prefix up to the previous BP3 is a cache read and
       only the newest round is a cache write.

    BP3 is automatically skipped when the last message is:

    - A ``HumanMessage`` — this is the first model call (or equivalent
      state); BP2 already covers it when enabled, and placing BP3 here
      would be wasteful regardless.
    - An ``AIMessage`` without ``tool_calls`` — this indicates a final
      response and no further model call will follow to read the cache.

    Args:
        system_message: The system message, or ``None``.
        messages: The conversation messages (excluding system).
        cache_control: Default cache control dict used for **BP1** (system
            message) and as fallback for BP2/BP3 when their overrides are
            not provided.
        cache_last_human_message: Place a breakpoint on the last
            ``HumanMessage``.
        cache_last_message: Place a breakpoint on the very last message
            (progressive caching).
        bp2_cache_control: Optional override for BP2.  When ``None``,
            falls back to *cache_control*.
        bp3_cache_control: Optional override for BP3.  When ``None``,
            falls back to *cache_control*.

    Returns:
        A ``(system_message, messages)`` tuple with cache breakpoints
        injected.  The originals are not mutated.
    """
    new_system: SystemMessage | None = system_message
    new_messages: list[AnyMessage] = _strip_cache_control(messages)

    # --- Breakpoint 1: system message ---
    if system_message is not None:
        cached_content = _add_cache_control_to_content(
            system_message.content,  # pyright: ignore[reportArgumentType]
            cache_control,
        )
        new_system = SystemMessage(
            content=cast("_MessageContent", cached_content),
            **(
                {"response_metadata": system_message.response_metadata}
                if system_message.response_metadata
                else {}
            ),
        )

    bp2_cc = bp2_cache_control or cache_control
    bp3_cc = bp3_cache_control or cache_control

    # --- Breakpoint 2 (opt-in): last HumanMessage ---
    if cache_last_human_message:
        for i in range(len(new_messages) - 1, -1, -1):
            if isinstance(new_messages[i], HumanMessage):
                new_messages[i] = _clone_message_with_cache_control(new_messages[i], bp2_cc)
                break

    # --- Breakpoint 3 (opt-in): last message (progressive) ---
    if cache_last_message and new_messages:
        last_idx = len(new_messages) - 1
        last_msg = new_messages[last_idx]
        # Skip when the last message is a HumanMessage (first call or
        # equivalent — BP2 covers it when enabled, and caching here is
        # wasteful regardless), or a final AI response without tool_calls
        # (no future model call will read the cache entry).
        is_human = isinstance(last_msg, HumanMessage)
        is_final_ai = isinstance(last_msg, AIMessage) and not last_msg.tool_calls
        if not is_human and not is_final_ai:
            new_messages[last_idx] = _clone_message_with_cache_control(
                new_messages[last_idx], bp3_cc
            )

    return new_system, new_messages


class PromptCachingMiddleware(AgentMiddleware[Any, Any]):
    """Prompt caching middleware supporting both native Anthropic and OpenRouter.

    This middleware optimises API costs by enabling Anthropic's prompt caching
    feature.  Both ``ChatAnthropic`` and ``ChatOpenRouter`` support
    ``cache_control`` on individual message content blocks, so the middleware
    uses a **unified content-block injection strategy** for both backends.

    For ``ChatOpenRouter``, the middleware can additionally pin the provider to
    Anthropic (via *pin_openrouter_provider*) to prevent routing to
    non-Anthropic backends that would ignore the ``cache_control`` breakpoints.

    For non-Anthropic models the middleware is a no-op (configurable via
    *unsupported_model_behavior*).

    Attributes:
        type: Cache type (only ``"ephemeral"`` is supported).
        ttl: Default time to live for all breakpoints (``"5m"`` or ``"1h"``).
        bp1_ttl: Resolved TTL for BP1 (system message).
        bp2_ttl: Resolved TTL for BP2 (last ``HumanMessage``).
        bp3_ttl: Resolved TTL for BP3 (progressive breakpoint).
        min_messages_to_cache: Minimum message count before caching activates.
        cache_last_human_message: Whether to cache the last HumanMessage.
        progressive: Whether to place an advancing breakpoint on the growing
            tool-call history when inside a tool-calling loop.
        pin_openrouter_provider: Whether to pin OpenRouter requests to the
            Anthropic provider to avoid routing to non-Anthropic backends.
        unsupported_model_behavior: What to do when the model is not an
            Anthropic model.
    """

    def __init__(
        self,
        *,
        type: Literal["ephemeral"] = "ephemeral",
        ttl: _TTL = "5m",
        bp1_ttl: _TTL | None = None,
        bp2_ttl: _TTL | None = None,
        bp3_ttl: _TTL | None = None,
        min_messages_to_cache: int = 0,
        cache_last_human_message: bool = False,
        progressive: bool = True,
        pin_openrouter_provider: bool = True,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> None:
        """Initialise the prompt caching middleware.

        Args:
            type: Cache type.  Only ``"ephemeral"`` is currently supported by
                the Anthropic API.
            ttl: Default cache lifetime for all breakpoints — ``"5m"``
                (default) or ``"1h"``.  Can be overridden per-breakpoint
                with *bp1_ttl*, *bp2_ttl*, *bp3_ttl*.
            bp1_ttl: TTL override for BP1 (system message).  Defaults to
                *ttl* when ``None``.
            bp2_ttl: TTL override for BP2 (last ``HumanMessage``).  Defaults
                to *ttl* when ``None``.
            bp3_ttl: TTL override for BP3 (progressive breakpoint).  Defaults
                to *ttl* when ``None``.
            min_messages_to_cache: Minimum number of messages (including
                system) before caching activates.
            cache_last_human_message: Also place a cache breakpoint on the
                last ``HumanMessage``.  Disabled by default because the
                write cost (1.25x at 5min TTL) may not pay off when human
                messages are small and change every invocation.  Enable when
                human messages are large or the agent performs many
                tool-calling turns per invocation.
            progressive: When ``True`` (default), place an advancing cache
                breakpoint on the last message when the agent is inside a
                tool-calling loop (last message is ``ToolMessage`` or
                ``AIMessage`` with ``tool_calls``).  This caches the
                growing tool-call history so that each subsequent call only
                pays the write cost for the newest round of tokens.
                Requires ``cache_last_human_message=True`` for optimal
                savings (BP2 provides the stable prefix that BP3 extends).
                Has no effect on the native ``ChatAnthropic`` code path.
            pin_openrouter_provider: When ``True`` (default), inject
                ``provider.order=["Anthropic"]`` and
                ``provider.allow_fallbacks=False`` into the OpenRouter
                request via ``model_settings``.  This prevents OpenRouter
                from routing to a non-Anthropic provider which would not
                honour the ``cache_control`` breakpoints.  Has no effect on
                the native ``ChatAnthropic`` code path.
            unsupported_model_behavior: Behaviour when the model is not a
                supported Anthropic model:

                - ``"ignore"`` (default) — silently skip caching.
                - ``"warn"`` — emit a warning and skip caching.
                - ``"raise"`` — raise ``ValueError``.

        Raises:
            ValueError: If the resolved per-breakpoint TTLs are not
                non-increasing (BP1 ≥ BP2 ≥ BP3).  Anthropic requires
                longer-TTL breakpoints to precede shorter-TTL ones within
                the same request.
        """
        super().__init__()
        self.type: Literal["ephemeral"] = type
        self.ttl: _TTL = ttl
        self.bp1_ttl: _TTL = bp1_ttl or ttl
        self.bp2_ttl: _TTL = bp2_ttl or ttl
        self.bp3_ttl: _TTL = bp3_ttl or ttl
        self._validate_ttl_ordering()
        self.min_messages_to_cache = min_messages_to_cache
        self.cache_last_human_message = cache_last_human_message
        self.progressive = progressive
        self.pin_openrouter_provider = pin_openrouter_provider
        self.unsupported_model_behavior = unsupported_model_behavior

    def _validate_ttl_ordering(self) -> None:
        """Ensure BP TTLs are non-increasing (BP1 ≥ BP2 ≥ BP3).

        Anthropic requires that longer-TTL cache breakpoints appear before
        shorter-TTL ones in the same request.

        Raises:
            ValueError: When the ordering constraint is violated.
        """
        pairs = [
            ("bp1_ttl", self.bp1_ttl, "bp2_ttl", self.bp2_ttl),
            ("bp2_ttl", self.bp2_ttl, "bp3_ttl", self.bp3_ttl),
        ]
        for earlier_name, earlier_val, later_name, later_val in pairs:
            if _TTL_MINUTES[earlier_val] < _TTL_MINUTES[later_val]:
                msg = (
                    f"TTL ordering violation: {earlier_name}={earlier_val!r} "
                    f"< {later_name}={later_val!r}. Anthropic requires "
                    f"longer-TTL breakpoints to precede shorter-TTL ones."
                )
                raise ValueError(msg)

    # =====================================================================
    # Strategy selection
    # =====================================================================

    def _should_apply_caching(
        self, request: ModelRequest
    ) -> Literal["native", "openrouter", False]:
        """Determine whether and how caching should be applied.

        Args:
            request: The current model request.

        Returns:
            ``"native"`` for ``ChatAnthropic``, ``"openrouter"`` for
            ``ChatOpenRouter`` with an Anthropic model, or ``False`` when
            caching should not be applied.

        Raises:
            ValueError: If the model is unsupported and
                *unsupported_model_behavior* is ``"raise"``.
        """
        model = request.model

        if _is_native_anthropic_model(model):
            strategy: Literal["native", "openrouter"] = "native"
        elif _is_openrouter_anthropic_model(model):
            strategy = "openrouter"
        else:
            model_type = type(model).__name__
            msg = (
                f"PromptCachingMiddleware only supports Anthropic models "
                f"(ChatAnthropic or ChatOpenRouter with anthropic/* model), "
                f"got {model_type}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                logger.warning(msg)
            return False

        # Check minimum message count
        message_count = len(request.messages) + (1 if request.system_message else 0)
        if message_count < self.min_messages_to_cache:
            return False

        return strategy

    # =====================================================================
    # Caching application
    # =====================================================================

    def _make_cache_control(self, ttl: _TTL) -> CacheControlDict:
        """Build a ``cache_control`` dict for the given *ttl*.

        Args:
            ttl: The TTL value (``"5m"`` or ``"1h"``).

        Returns:
            A ``cache_control`` dict suitable for injection into content
            blocks.  The ``"ttl"`` key is omitted when *ttl* is ``"5m"``
            (the Anthropic default).
        """
        cc: CacheControlDict = {"type": self.type}
        if ttl != "5m":
            cc["ttl"] = ttl
        return cc

    def _apply_caching(
        self,
        request: ModelRequest,
    ) -> ModelRequest:
        """Apply caching by injecting ``cache_control`` into content blocks.

        Both ``ChatAnthropic`` and ``ChatOpenRouter`` support per-message
        content-block-level ``cache_control``.  This method uses the same
        injection strategy for both backends.

        When *pin_openrouter_provider* is enabled and the model is a
        ``ChatOpenRouter`` instance, also injects
        ``provider.order=["Anthropic"]`` and
        ``provider.allow_fallbacks=False`` into ``model_settings`` so that
        OpenRouter does not route the request to a non-Anthropic provider
        (which would ignore the ``cache_control`` breakpoints).

        Args:
            request: The original model request.

        Returns:
            A new request with ``cache_control`` injected into message
            content blocks and optionally provider pinning in
            *model_settings*.
        """
        bp1_cc = self._make_cache_control(self.bp1_ttl)
        bp2_cc = self._make_cache_control(self.bp2_ttl)
        bp3_cc = self._make_cache_control(self.bp3_ttl)

        new_system, new_messages = _inject_cache_breakpoints(
            system_message=request.system_message,
            messages=request.messages,
            cache_control=bp1_cc,
            cache_last_human_message=self.cache_last_human_message,
            cache_last_message=self.progressive,
            bp2_cache_control=bp2_cc if bp2_cc != bp1_cc else None,
            bp3_cache_control=bp3_cc if bp3_cc != bp1_cc else None,
        )

        overrides: dict[str, Any] = {
            "system_message": new_system,
            "messages": new_messages,
        }

        if (
            self.pin_openrouter_provider
            and _is_openrouter_anthropic_model(request.model)
            and "provider" not in request.model_settings
        ):
            overrides["model_settings"] = {
                **request.model_settings,
                "provider": {
                    "order": ["Anthropic"],
                    "allow_fallbacks": False,
                },
            }

        return request.override(**overrides)

    # =====================================================================
    # Model call wrappers
    # =====================================================================

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply prompt caching and delegate to the handler.

        Args:
            request: The model request.
            handler: The next handler in the middleware chain.

        Returns:
            The model response.
        """
        strategy = self._should_apply_caching(request)
        if not strategy:
            return handler(request)

        return handler(self._apply_caching(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply prompt caching and delegate to the async handler.

        Args:
            request: The model request.
            handler: The next async handler in the middleware chain.

        Returns:
            The model response.
        """
        strategy = self._should_apply_caching(request)
        if not strategy:
            return await handler(request)

        return await handler(self._apply_caching(request))


__all__ = ["PromptCachingMiddleware"]

"""Shared model, web-search, caching, and tracing wiring for the runnable examples.

Every example runs fully offline by default: a deterministic fake model, no API
key, no extra dependencies. Opt into real deepagent leaves by setting the
``LDW_DEMO_REAL_MODEL`` environment variable — the agents then run through
OpenRouter (``ChatOpenRouter``) with credentials read from a local ``.env``
(``OPENROUTER_API_KEY``).

The real path needs the demo dependency group, installed with
``uv sync --group example``. Set ``LDW_DEMO_REAL_MODEL`` to a truthy value to use
the default models — ``anthropic/claude-opus-4.8`` for the host
(:func:`real_model`) and ``anthropic/claude-sonnet-4.6`` for the leaves
(:func:`real_leaf_model`) — or set it to an OpenRouter ``provider/model`` slug to
override the host model.

Provider lock, web search, prompt caching. Every real model pins OpenRouter routing
to Anthropic (:data:`ANTHROPIC_PROVIDER`), which is required for the two capabilities
the demos share with the interactive demo app: research leaves carry OpenRouter's
native ``openrouter:web_search`` tool (:func:`real_leaf_model` with ``web_search=True``,
``engine="native"`` so the search is Anthropic's own), and EVERY agent registers the
Anthropic prompt-caching middleware (:func:`demo_cache_middleware`). Both only engage on
the keyed online path; the offline fake-model path stays deterministic and the middleware
is a no-op there.

Loading the local ``.env`` also activates LangSmith tracing when the standard
``LANGSMITH_TRACING`` / ``LANGSMITH_API_KEY`` / ``LANGSMITH_PROJECT`` variables are
present: LangChain reads them at run time, so every example's orchestration shows
up as a trace with no extra code. Each example calls ``load_demo_env`` at the top
of ``main`` so this works on the offline fake path too.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import LangSmithParams
from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_OPENROUTER_MODEL = "anthropic/claude-opus-4.8"
"""Host model slug used when the demo is gated live without an explicit override."""

DEFAULT_LEAF_MODEL = "anthropic/claude-sonnet-4.6"
"""Leaf model slug for the research fan-out — strong enough to drive web search."""

_REAL_MODEL_ENV = "LDW_DEMO_REAL_MODEL"

# Lock OpenRouter routing to Anthropic's first-party endpoint, no fallback. Required for
# the native web search and Anthropic prompt caching, which work only on that provider.
ANTHROPIC_PROVIDER: dict[str, Any] = {"order": ["Anthropic"], "allow_fallbacks": False}

# OpenRouter's native server-side web search. ``engine="native"`` forces the provider's
# own (Anthropic's) search; reached through OpenRouter's unified ``openrouter:web_search``
# type (the ``openrouter`` SDK rejects the raw Anthropic ``web_search_20250305`` spec).
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "openrouter:web_search",
    "parameters": {"engine": "native", "max_results": 5},
}

# LangSmith cost tracking keys on Anthropic's official, hyphenated model ids
# (``claude-opus-4-8``), but OpenRouter slugs are dot-separated and provider-prefixed
# (``anthropic/claude-opus-4.8``), so a traced run would record no cost. Map the
# dot-version (prefix stripped) to the hyphen-version. Ported from omne-next's
# RoutedChatOpenRouter; unknown ids pass through unchanged.
_LANGSMITH_MODEL_NAME_MAP: dict[str, str] = {
    "claude-opus-4.8": "claude-opus-4-8",
    "claude-opus-4.7": "claude-opus-4-7",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4.5": "claude-opus-4-5",
    "claude-opus-4.1": "claude-opus-4-1",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-sonnet-4.5": "claude-sonnet-4-5",
    "claude-haiku-4.5": "claude-haiku-4-5",
    "claude-3.7-sonnet": "claude-3-7-sonnet",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
    "claude-3.5-haiku": "claude-3-5-haiku",
}


def _build_routed_openrouter(model_slug: str, *, web_search: bool) -> BaseChatModel:
    """Build a ChatOpenRouter that normalizes its LangSmith model id, optionally with search.

    The subclasses are defined locally so this module imports no OpenRouter dependency at
    load time (examples run offline by default, without the demo dependency group). The
    routed base rewrites ``ls_provider`` / ``ls_model_name`` so LangSmith cost tracking
    matches Anthropic's pricing table; the web-search variant additionally appends the
    native web tool on every ``bind_tools`` (surviving deepagents' own binding).

    Args:
        model_slug: The OpenRouter model slug (e.g. ``anthropic/claude-opus-4.8``).
        web_search: When ``True``, return the web-search-enabled variant.

    Returns:
        A configured ``ChatOpenRouter`` subclass instance.
    """
    from langchain_openrouter import ChatOpenRouter

    class _RoutedChatOpenRouter(ChatOpenRouter):
        """Reports Anthropic's hyphenated model id to LangSmith for cost matching."""

        def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
            params = super()._get_ls_params(stop=stop, **kwargs)
            raw = params.get("ls_model_name") or self.model_name or ""
            provider, sep, model = raw.partition("/")
            if sep and provider and model:
                params["ls_provider"] = provider
                params["ls_model_name"] = _LANGSMITH_MODEL_NAME_MAP.get(model, model)
            return params

    if not web_search:
        return _RoutedChatOpenRouter(model=model_slug, openrouter_provider=ANTHROPIC_PROVIDER)

    class _WebSearchChatOpenRouter(_RoutedChatOpenRouter):
        """Routed model that also keeps the native web search tool on every binding."""

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            bound = super().bind_tools(tools, **kwargs)
            bound_kwargs: dict[str, Any] = dict(getattr(bound, "kwargs", {}))
            bound_kwargs["tools"] = [*bound_kwargs.get("tools", []), WEB_SEARCH_TOOL]
            return self.bind(**bound_kwargs)

    return _WebSearchChatOpenRouter(model=model_slug, openrouter_provider=ANTHROPIC_PROVIDER)


def load_demo_env() -> None:
    """Populate ``os.environ`` from a local ``.env`` when ``python-dotenv`` is installed.

    Best-effort and idempotent: when the optional demo dependency group is absent
    (the offline default) this is a silent no-op, so the fake-model path keeps
    running with no extra dependencies.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True))


def real_model() -> BaseChatModel | None:
    """Return the OpenRouter HOST chat model when the demo is gated live, else ``None``.

    Reads the already-loaded environment, so call ``load_demo_env`` first (each
    example's ``main`` does). When ``LDW_DEMO_REAL_MODEL`` is unset the demo stays
    offline and this returns ``None``, signalling the caller to fall back to its
    deterministic fake. When set, the value is used as the OpenRouter model slug if it
    looks like one (contains ``/``), otherwise :data:`DEFAULT_OPENROUTER_MODEL`. The
    model is pinned to the Anthropic provider (:data:`ANTHROPIC_PROVIDER`).

    Returns:
        A configured ``ChatOpenRouter`` (the host model), or ``None`` to run offline.
    """
    gate = os.environ.get(_REAL_MODEL_ENV)
    if not gate:
        return None
    model_slug = gate if "/" in gate else DEFAULT_OPENROUTER_MODEL
    return _build_routed_openrouter(model_slug, web_search=False)


def real_leaf_model(*, web_search: bool = False) -> BaseChatModel | None:
    """Return the OpenRouter LEAF chat model when gated live, else ``None``.

    Mirrors :func:`real_model` but always uses :data:`DEFAULT_LEAF_MODEL` (the host's
    ``LDW_DEMO_REAL_MODEL`` slug override does not apply to leaves) and pins the Anthropic
    provider. With ``web_search=True`` the model carries OpenRouter's native web search
    tool, appended on every ``bind_tools`` (so it survives deepagents' tool binding) and
    raw (so the ``openrouter:`` marker reaches OpenRouter for server-side execution).

    Args:
        web_search: When ``True``, bind the native web search tool so the leaf grounds
            its work in live web sources.

    Returns:
        A configured ``ChatOpenRouter`` (the leaf model), or ``None`` to run offline.
    """
    if not os.environ.get(_REAL_MODEL_ENV):
        return None
    return _build_routed_openrouter(DEFAULT_LEAF_MODEL, web_search=web_search)


def demo_cache_middleware() -> list[Any]:
    """Return the Anthropic prompt-caching middleware to register on EVERY example agent.

    Mirrors the interactive demo app: a ``PromptCachingMiddleware`` (see
    ``examples/_prompt_caching.py``) that injects ``cache_control`` breakpoints for
    Anthropic models accessed via ``ChatOpenRouter``. It is a no-op on the offline fake
    path (the scripted model is not an OpenRouter Anthropic model), so it is safe to
    attach unconditionally. ``pin_openrouter_provider`` is off because the models already
    pin the provider (:data:`ANTHROPIC_PROVIDER`).

    Returns:
        A one-element middleware list to pass as ``create_deep_agent(middleware=...)``.
    """
    from _prompt_caching import PromptCachingMiddleware

    return [
        PromptCachingMiddleware(
            progressive=True,
            cache_last_human_message=True,
            pin_openrouter_provider=False,
        )
    ]

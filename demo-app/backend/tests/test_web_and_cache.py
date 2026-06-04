"""Web search + Anthropic prompt caching wiring (unit tier, no network).

Pins the mechanism the demo's real path now relies on:

* leaf/host models are ``ChatOpenRouter`` pinned to the Anthropic provider;
* research / verify leaves additionally carry OpenRouter's native ``openrouter:web_search``
  tool, appended on every ``bind_tools`` so it survives deepagents' own tool binding;
* the prompt-caching middleware recognises both a real ``ChatOpenRouter`` and the lazy
  host wrapper (via its marker), so EVERY agent — host and all leaves — is cached.

All of this is asserted by constructing models / spying the builders; nothing here makes
a network call or builds against a live key.
"""

from __future__ import annotations

from typing import Any

import pytest
from _models import (
    ANTHROPIC_PROVIDER,
    HOST_MODEL,
    LEAF_MODEL,
    WEB_SEARCH_TOOL,
    LazyOpenRouterHostModel,
    OfflineHostModel,
    _build_openrouter_model,
    _WebSearchChatOpenRouter,
    resolve_leaf_model,
)
from langchain_openrouter import ChatOpenRouter
from prompt_caching import PromptCachingMiddleware, _is_openrouter_anthropic_model


def test_web_search_tool_is_the_native_openrouter_type() -> None:
    """The web tool is OpenRouter's native ``openrouter:web_search`` with native engine.

    ``engine="native"`` is what routes the search to the provider's own (Anthropic's)
    web search rather than Exa; the ``openrouter:`` type is the one the ``openrouter`` SDK
    accepts (the raw Anthropic ``web_search_20250305`` spec is rejected by that SDK).
    """
    assert WEB_SEARCH_TOOL["type"] == "openrouter:web_search"
    assert WEB_SEARCH_TOOL["parameters"]["engine"] == "native"


def test_web_search_subclass_appends_the_tool_on_bind_tools() -> None:
    """``_WebSearchChatOpenRouter.bind_tools`` keeps the web tool alongside bound tools.

    deepagents binds its own tools onto the leaf model; a plain assignment would drop the
    web tool. The subclass appends it on every binding so it is always present, and raw
    (not converted) so the ``openrouter:`` marker survives for server-side execution.
    """
    model = _WebSearchChatOpenRouter(model=LEAF_MODEL, api_key="sk-fake")  # type: ignore[arg-type]

    def _a_tool(x: int) -> int:
        """A throwaway client tool to stand in for deepagents' own tools."""
        return x

    bound = model.bind_tools([_a_tool])
    tools = list(bound.kwargs.get("tools", []))  # type: ignore[attr-defined]
    assert WEB_SEARCH_TOOL in tools, "web search tool must be appended to every binding"
    # The original client tool is still there (appended, not replaced).
    assert len(tools) >= 2


def test_build_model_pins_anthropic_provider_and_gates_web_search() -> None:
    """Every built model pins the Anthropic provider; web search is opt-in per leaf.

    The provider lock is REQUIRED (native web search + Anthropic caching only work on the
    Anthropic provider). ``web_search=True`` selects the web-search subclass; the default
    is a plain ``ChatOpenRouter`` (host + non-research leaves do not search).
    """
    plain = _build_openrouter_model(LEAF_MODEL, "sk-fake")
    searcher = _build_openrouter_model(LEAF_MODEL, "sk-fake", web_search=True)

    assert isinstance(plain, ChatOpenRouter) and not isinstance(plain, _WebSearchChatOpenRouter)
    assert isinstance(searcher, _WebSearchChatOpenRouter)
    for model in (plain, searcher):
        assert model.openrouter_provider == ANTHROPIC_PROVIDER  # type: ignore[attr-defined]


def test_langsmith_model_name_normalized_to_anthropic_hyphenated() -> None:
    """Built models report Anthropic's hyphenated id to LangSmith, not the OpenRouter slug.

    LangSmith's pricing table keys on Anthropic's official ``claude-opus-4-8``; the raw
    OpenRouter slug ``anthropic/claude-opus-4.8`` would miss it and a traced run would show
    no cost. The routed subclass rewrites ``ls_provider`` / ``ls_model_name`` in
    ``_get_ls_params`` (dot-version -> hyphen-version), and the web-search leaf inherits it.
    """
    host_params = _build_openrouter_model(HOST_MODEL, "sk-fake")._get_ls_params()
    assert host_params.get("ls_provider") == "anthropic"
    assert host_params.get("ls_model_name") == "claude-opus-4-8"

    leaf_params = _build_openrouter_model(LEAF_MODEL, "sk-fake", web_search=True)._get_ls_params()
    assert leaf_params.get("ls_model_name") == "claude-sonnet-4-6"


def test_resolve_leaf_model_web_search_gated_on_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``resolve_leaf_model(web_search=True)`` builds a searcher online, ``None`` offline."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_leaf_model(web_search=True) is None  # offline: no key, no model

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    online = resolve_leaf_model(web_search=True)
    assert isinstance(online, _WebSearchChatOpenRouter)


def test_cache_middleware_detects_openrouter_anthropic_and_host_wrapper() -> None:
    """The cache middleware fires for a real ChatOpenRouter AND the lazy host wrapper.

    The leaves are real ``ChatOpenRouter`` (detected by isinstance + ``anthropic/`` name);
    the host is a :class:`LazyOpenRouterHostModel` whose per-call delegate is an OpenRouter
    Anthropic model and which forwards cache_control messages through, so it opts in via a
    marker. Both must be detected, or some agents would silently skip caching. The offline
    scripted host must NOT be detected (no caching on the credential-free path).
    """
    leaf = ChatOpenRouter(model=LEAF_MODEL, api_key="sk-fake")  # type: ignore[arg-type]
    assert _is_openrouter_anthropic_model(leaf) is True
    assert _is_openrouter_anthropic_model(LazyOpenRouterHostModel()) is True
    assert _is_openrouter_anthropic_model(OfflineHostModel()) is False


def test_every_real_leaf_is_built_with_cache_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVERY real leaf the roster builds registers the prompt-cache middleware.

    Spies the two builder boundaries the leaves go through (``create_deep_agent`` for the
    text / extractor leaves, ``read_only_leaf`` for the judges) and asserts each receives a
    non-empty ``middleware`` list. A regression that wired caching into only some agents —
    exactly what the user warned against — would surface as a builder called without it.
    """
    import workflows
    from workflows import Claim, Verdict, make_roster

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    middleware_seen: list[Any] = []

    def _spy_cda(*args: Any, **kwargs: Any) -> str:
        middleware_seen.append(kwargs.get("middleware"))
        return "stub-agent"

    def _spy_rol(*args: Any, **kwargs: Any) -> str:
        middleware_seen.append(kwargs.get("middleware"))
        return "stub-judge"

    monkeypatch.setattr(workflows, "create_deep_agent", _spy_cda)
    monkeypatch.setattr(workflows, "read_only_leaf", _spy_rol)

    roster = make_roster()
    # Trigger the schema-aware builders (extractor / skeptic / capstone_skeptic), which the
    # engine would call later via runnable_for, so they too go through the spied boundary.
    roster.runnable_for("extractor", response_format=Claim)
    roster.runnable_for("skeptic", response_format=Verdict)
    roster.runnable_for("capstone_skeptic", response_format=Verdict)

    # researcher + writer + refiner (eager) + extractor + skeptic + capstone_skeptic = 6.
    assert len(middleware_seen) == 6, f"expected 6 real-leaf builds, saw {len(middleware_seen)}"
    assert all(mw for mw in middleware_seen), "every real leaf must get a cache middleware"
    assert all(isinstance(mw[0], PromptCachingMiddleware) for mw in middleware_seen), (
        "the registered middleware must be the prompt-caching one"
    )


def test_host_graph_registers_cache_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host agent is built with the prompt-cache middleware too (not just leaves)."""
    import host_graph

    captured: dict[str, Any] = {}

    def _spy_cda(*args: Any, **kwargs: Any) -> str:
        captured["middleware"] = kwargs.get("middleware")
        return "stub-host"

    monkeypatch.setattr(host_graph, "create_deep_agent", _spy_cda)
    host_graph.make_host_graph()

    mw = captured.get("middleware")
    assert mw and isinstance(mw[0], PromptCachingMiddleware), "host must register cache middleware"


def test_offline_leaf_has_no_web_search_and_no_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offline path stays a deterministic fake leaf: no model, no search, no middleware.

    Builds the roster with NO key and spies the real-leaf boundary; it must never be hit,
    so the credential-free path keeps its reproducible fake-leaf behaviour.
    """
    import workflows
    from workflows import make_roster

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls: list[Any] = []
    monkeypatch.setattr(workflows, "create_deep_agent", lambda *a, **k: calls.append(k) or "x")
    monkeypatch.setattr(workflows, "read_only_leaf", lambda *a, **k: calls.append(k) or "x")

    make_roster()  # eager leaves only; offline → no real builds
    assert calls == [], "offline roster must build no real (cached/search) leaves"
    # And a fake leaf still answers deterministically.
    roster = make_roster()
    leaf = roster.runnable_for("researcher", response_format=None)
    assert leaf is not None

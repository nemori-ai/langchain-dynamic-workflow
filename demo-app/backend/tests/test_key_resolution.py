"""Per-session OpenRouter key resolution checks (the locked-provider contract).

The demo locks the provider to OpenRouter and fixes the models in code; the only
moving part is the single key, which the frontend threads per session on the
LangGraph run config (``config.configurable.openrouter_api_key``). These tests pin
that round-trip through the real surfaces it actually flows through:

* the key precedence — a per-run config key wins over the ``.env`` key, which wins
  over nothing (offline);
* the per-run config key is read via the SAME ``var_child_runnable_config`` contextvar
  ``langgraph.config.get_config`` resolves (so the test exercises the real mechanism,
  not a mock of it); and
* a resolved OpenRouter model is built against the locked base URL with THAT key, and
  ``is_offline`` reflects the in-force key honestly.

The host model picks up the per-run key per call (``LazyOpenRouterHostModel``); the
leaf models pick it up at roster-build time via an explicit ``api_key`` capture. Both
paths funnel through :func:`resolve_openrouter_key`, which is what these tests pin.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config


@pytest.fixture(autouse=True)
def _clean_provider_env() -> Iterator[None]:
    """Start each test with no provider keys in the environment, restoring after."""
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")}
    yield
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@contextmanager
def _run_config(configurable: dict[str, Any]) -> Iterator[None]:
    """Bind a run config onto the contextvar ``langgraph.config.get_config`` reads.

    ``get_config`` resolves ``var_child_runnable_config`` — the exact contextvar the
    host node sets while it runs. Binding it here reproduces the in-node condition the
    per-run key resolver actually runs under, so the test exercises the real read path
    rather than monkeypatching ``get_config``.
    """
    config: RunnableConfig = {"configurable": configurable}
    token = var_child_runnable_config.set(config)
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def test_per_run_config_key_threads_into_built_openrouter_model() -> None:
    """A per-run ``configurable.openrouter_api_key`` builds an OpenRouter model with it.

    The headline round-trip: the frontend's per-session key arrives on the run config;
    the resolver must pick THAT key (not the env) and the built model must point at the
    locked OpenRouter base URL and carry exactly that key. Asserting both base_url and
    the threaded key proves the key actually reaches the model, not just the resolver.
    """
    from _models import (
        LEAF_MODEL,
        OPENROUTER_BASE_URL,
        resolve_leaf_model,
        resolve_openrouter_key,
    )

    session_key = "sk-or-session-1234"
    with _run_config({"openrouter_api_key": session_key}):
        assert resolve_openrouter_key() == session_key
        model = resolve_leaf_model()

    assert model is not None
    # The locked provider: every real call routes through OpenRouter's endpoint.
    assert str(model.openai_api_base) == OPENROUTER_BASE_URL  # type: ignore[attr-defined]
    # The per-session key threaded all the way into the model (SecretStr unwrap).
    assert model.openai_api_key.get_secret_value() == session_key  # type: ignore[attr-defined]
    # The fixed economical leaf model id, not a user-configured one.
    assert model.model_name == LEAF_MODEL  # type: ignore[attr-defined]


def test_explicit_api_key_argument_wins_over_run_config_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly-captured ``api_key`` beats both the run-config key and the env key.

    The leaf path captures the per-run key at roster-build time and passes it as
    ``api_key`` (because the leaf model is baked in before the engine substrate hides the
    run config). That captured key must take precedence so the leaf is built with the
    session's key even if a different run-config / env key is visible at call time.
    """
    from _models import resolve_openrouter_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    with _run_config({"openrouter_api_key": "sk-or-runconfig"}):
        assert resolve_openrouter_key(api_key="sk-or-captured") == "sk-or-captured"


def test_run_config_key_wins_over_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-run config key takes precedence over the backend ``.env`` key.

    A session that brings its own key must use it, not the operator's ``.env`` key —
    the per-session-key contract. (With no run-config key, the env key is the fallback;
    pinned in the next test.)
    """
    from _models import resolve_openrouter_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-fallback")
    with _run_config({"openrouter_api_key": "sk-or-session-wins"}):
        assert resolve_openrouter_key() == "sk-or-session-wins"


def test_env_key_is_the_fallback_when_no_run_config_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no run-config key, the ``.env`` ``OPENROUTER_API_KEY`` is the operator path.

    Both with an empty run config and with no runnable context at all (``get_config``
    raises, the resolver swallows it), the env key is what's in force — the local /
    operator deployment mode.
    """
    from _models import resolve_openrouter_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-operator")

    # No runnable context at all: get_config raises, resolver falls back to env.
    assert resolve_openrouter_key() == "sk-or-operator"

    # An empty run config (a run with no key field) also falls back to env.
    with _run_config({}):
        assert resolve_openrouter_key() == "sk-or-operator"


def test_no_key_anywhere_is_offline_and_models_are_none() -> None:
    """With nothing set anywhere the demo is honestly offline.

    ``resolve_openrouter_key`` returns ``None``, ``is_offline`` is ``True``, and the
    leaf resolver signals the offline path with ``None`` (the roster then swaps in fake
    leaves). This is the bare ``langgraph dev`` boot with no credentials.
    """
    from _models import is_offline, resolve_leaf_model, resolve_openrouter_key

    assert resolve_openrouter_key() is None
    assert is_offline() is True
    assert resolve_leaf_model() is None

    # Even inside a runnable context, an absent key field stays offline.
    with _run_config({"thread_id": "t-1"}):
        assert resolve_openrouter_key() is None
        assert is_offline() is True


def test_is_offline_flips_online_with_a_per_run_key() -> None:
    """``is_offline`` reflects the per-run key honestly: online when a session key is set.

    The frontend's offline banner reads this; a session that supplies its key on the run
    config must read as online for that turn even with no ``.env`` key present.
    """
    from _models import is_offline

    assert is_offline() is True  # no key anywhere
    with _run_config({"openrouter_api_key": "sk-or-live"}):
        assert is_offline() is False


def test_host_model_resolves_offline_delegate_without_a_key() -> None:
    """The lazy host model drives the scripted offline host when no key is in force.

    ``resolve_host_model`` always returns the lazy wrapper (the online/offline decision
    is per-turn, inside the node). With no key in force, an (a)generate call must
    delegate to the scripted ``OfflineHostModel`` — issuing the offline host's first
    tool call — so a key-free ``langgraph dev`` boot still demonstrates the round-trip.
    """
    from _models import LazyOpenRouterHostModel, resolve_host_model
    from langchain_core.messages import HumanMessage

    model = resolve_host_model()
    assert isinstance(model, LazyOpenRouterHostModel)

    # No key in force: the lazy model delegates to the scripted offline host, which on
    # the first turn issues a tool call (here the default hello demo path).
    result = model._generate([HumanMessage(content="show me the demo")])
    message = result.generations[0].message
    assert message.tool_calls, "offline delegate must issue the scripted first tool call"  # type: ignore[attr-defined]
    assert message.tool_calls[0]["name"] == "run_hello_demo"  # type: ignore[attr-defined]


def test_host_model_binds_real_openrouter_backend_with_a_key() -> None:
    """With a per-run key in force, the lazy host model resolves the real OpenRouter backend.

    Asserts the online delegate is built against the locked base URL with the fixed
    strong host model id and the in-force key — without making a network call (the
    delegate is inspected, not invoked). This proves the host picks up the per-session
    key, the headline per-run round-trip for the host side.
    """
    from _models import (
        HOST_MODEL,
        OPENROUTER_BASE_URL,
        LazyOpenRouterHostModel,
    )

    model = LazyOpenRouterHostModel()
    with _run_config({"openrouter_api_key": "sk-or-host-session"}):
        delegate = model._resolve_delegate()

    # No tools bound here, so the delegate is the bare ChatOpenAI (OpenRouter) backend.
    assert str(delegate.openai_api_base) == OPENROUTER_BASE_URL  # type: ignore[attr-defined]
    assert delegate.openai_api_key.get_secret_value() == "sk-or-host-session"  # type: ignore[attr-defined]
    assert delegate.model_name == HOST_MODEL  # type: ignore[attr-defined]


def test_roster_threads_captured_key_into_schema_aware_leaf_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_roster`` captures the per-run key and binds it into the LATER-built leaves.

    The crux of the leaf-key coverage. A leaf model is baked in when the builder runs,
    but the schema-aware builders (``extractor`` / ``skeptic`` / ``capstone_skeptic``)
    are invoked LATER by the engine — ``runnable_for(name, response_format=...)`` — from
    inside the nested workflow substrate, where the host run config is NO LONGER visible.
    A naive ``resolve_leaf_model()`` inside the builder would see no key there and build a
    fake (offline) leaf. The fix captures the per-run key in ``make_roster`` (in the host
    node) and binds it into the builder via ``functools.partial``.

    This spies on the ``api_key`` ``make_roster`` threads into ``resolve_leaf_model``: it
    builds the roster inside a run config carrying the session key, then invokes the
    schema-aware builder OUTSIDE that run config (the substrate condition) and asserts the
    captured session key — not ``None`` — reached the leaf builder. A regression that
    re-resolved inside the builder would record ``None`` here.
    """
    import workflows
    from workflows import Claim, make_roster

    captured_keys: list[str | None] = []
    real_resolve = workflows.resolve_leaf_model

    def _spy_resolve(*, api_key: str | None = None) -> Any:
        captured_keys.append(api_key)
        return real_resolve(api_key=api_key)

    monkeypatch.setattr(workflows, "resolve_leaf_model", _spy_resolve)

    session_key = "sk-or-roster-capture"
    # Build the roster INSIDE the run config (as the host node does), capturing the key.
    with _run_config({"openrouter_api_key": session_key}):
        roster = make_roster()

    # OUTSIDE the run config now: the engine builds schema variants from inside the
    # substrate, where no run-config key is visible. The captured key must still flow in.
    roster.runnable_for("extractor", response_format=Claim)

    assert captured_keys, "the extractor builder must call resolve_leaf_model"
    assert session_key in captured_keys, (
        f"the captured session key must thread into the leaf builder, got {captured_keys}"
    )
    # And none of the leaf builds resolved to the offline path (every captured key is the
    # session key, never None — proving the substrate did not silently fall offline).
    assert all(k == session_key for k in captured_keys), captured_keys


def test_env_model_override_is_an_escape_hatch_not_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed constants are the default; the env override is an internal escape hatch.

    The headline path uses the fixed ``HOST_MODEL`` / ``LEAF_MODEL`` constants with no
    user model config. An operator MAY pin a different id via the env override for local
    experiments; this pins both halves of that contract so a regression that re-wires the
    default to read the env (the old behavior) is caught.
    """
    from _models import LEAF_MODEL, resolve_leaf_model

    # Default: the fixed constant, no env override.
    with _run_config({"openrouter_api_key": "sk-or-x"}):
        default_model = resolve_leaf_model()
    assert default_model is not None
    assert default_model.model_name == LEAF_MODEL  # type: ignore[attr-defined]

    # Escape hatch: the env override pins a different id.
    monkeypatch.setenv("LDW_DEMO_LEAF_MODEL", "anthropic/claude-3.5-haiku")
    with _run_config({"openrouter_api_key": "sk-or-x"}):
        overridden = resolve_leaf_model()
    assert overridden is not None
    assert overridden.model_name == "anthropic/claude-3.5-haiku"  # type: ignore[attr-defined]

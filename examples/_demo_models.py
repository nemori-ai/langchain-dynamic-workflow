"""Shared model and tracing wiring for the runnable examples.

Every example runs fully offline by default: a deterministic fake model, no API
key, no extra dependencies. Opt into real deepagent leaves by setting the
``LDW_DEMO_REAL_MODEL`` environment variable — the leaves then run through
OpenRouter (``ChatOpenRouter``) with credentials read from a local ``.env``
(``OPENROUTER_API_KEY``).

The real path needs the demo dependency group, installed with
``uv sync --group example``. Set ``LDW_DEMO_REAL_MODEL`` to a truthy value to use
the default model (``anthropic/claude-opus-4.8``), or to any OpenRouter
``provider/model`` slug to override it.

Loading the local ``.env`` also activates LangSmith tracing when the standard
``LANGSMITH_TRACING`` / ``LANGSMITH_API_KEY`` / ``LANGSMITH_PROJECT`` variables are
present: LangChain reads them at run time, so every example's orchestration shows
up as a trace with no extra code. Each example calls ``load_demo_env`` at the top
of ``main`` so this works on the offline fake path too.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_OPENROUTER_MODEL = "anthropic/claude-opus-4.8"
"""OpenRouter slug used when the demo is gated live without an explicit override."""

_REAL_MODEL_ENV = "LDW_DEMO_REAL_MODEL"


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
    """Return an OpenRouter chat model when the demo is gated live, else ``None``.

    Reads the already-loaded environment, so call ``load_demo_env`` first (each
    example's ``main`` does) to bring a local ``.env`` into scope. When
    ``LDW_DEMO_REAL_MODEL`` is unset the demo stays offline and this returns
    ``None``, signalling the caller to fall back to its deterministic fake. When
    set, the value is used as the OpenRouter model slug if it looks like one
    (contains ``/``), otherwise ``DEFAULT_OPENROUTER_MODEL`` is used.

    Returns:
        A configured ``ChatOpenRouter``, or ``None`` to run offline with a fake.
    """
    gate = os.environ.get(_REAL_MODEL_ENV)
    if not gate:
        return None
    from langchain_openrouter import ChatOpenRouter

    model_slug = gate if "/" in gate else DEFAULT_OPENROUTER_MODEL
    return ChatOpenRouter(model=model_slug)

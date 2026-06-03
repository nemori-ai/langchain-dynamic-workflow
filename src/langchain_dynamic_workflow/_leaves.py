"""Library-level leaf constructors — host-side conveniences aligned with deepagents.

A read-only leaf can read / grep / glob / ls but cannot write or edit: a deny-write
:class:`FilesystemPermission` blocks every write at the tool boundary. A judge built
from one can only assess, never "fix" — so a hallucinated repair can't land, keeping
the agent that *generates* separate from the one that *judges*.

Read-only is a property of the tool surface, enforced by deepagents — not the engine.
deepagents has no "execute" permission dimension (``FilesystemOperation`` is
read/write only), so a read-only judge is also registered with
``needs_execution=False``: it is handed a ``StateBackend`` and no execute tool, and
that default backend (not a sandbox) honors the deny-write permission.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from deepagents import (
    FilesystemPermission,
    create_deep_agent,  # pyright: ignore[reportUnknownVariableType]  # deepagents typing gap
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

# Deny every write everywhere. Paths must be absolute; "/**" matches all paths.
_DENY_WRITE: tuple[FilesystemPermission, ...] = (
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
)


def read_only_leaf(
    model: BaseChatModel,
    *,
    system_prompt: str | None = None,
    response_format: Any = None,
    **kwargs: Any,
) -> Runnable[Any, Any]:
    """Build a read-only deepagent leaf (deny-write tool surface).

    Args:
        model: The chat model the leaf (e.g. a judge) reasons with.
        system_prompt: Optional system prompt.
        response_format: Optional structured-output binding (e.g. a ``ToolStrategy``);
            forwarded so a judge can return a validated verdict.
        **kwargs: Forwarded to ``create_deep_agent`` (e.g. ``tools``, ``skills``).

    Returns:
        A compiled deepagent whose filesystem writes are denied.
    """
    extra: dict[str, Any] = dict(kwargs)
    if system_prompt is not None:
        extra["system_prompt"] = system_prompt
    if response_format is not None:
        extra["response_format"] = response_format
    # create_deep_agent's return type is only partially typed upstream; the compiled
    # graph is a Runnable, so narrow it explicitly for the engine's leaf contract.
    return cast(
        Runnable[Any, Any], create_deep_agent(model=model, permissions=list(_DENY_WRITE), **extra)
    )


class _Builder(Protocol):
    def __call__(self, *, response_format: Any = None) -> Runnable[Any, Any]: ...


def read_only_builder(
    model: BaseChatModel, *, system_prompt: str | None = None, **kwargs: Any
) -> _Builder:
    """Return a roster ``builder`` that constructs a read-only leaf per response_format.

    Register with ``roster.register("judge", builder=read_only_builder(model, ...))``
    so ``agent(agent_type="judge", schema=Verdict)`` yields a structured, read-only
    judge.

    Args:
        model: The chat model the judge reasons with.
        system_prompt: Optional system prompt baked into every built variant.
        **kwargs: Forwarded to :func:`read_only_leaf`.

    Returns:
        A ``(*, response_format) -> Runnable`` builder for roster registration.
    """

    def _builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        return read_only_leaf(
            model, system_prompt=system_prompt, response_format=response_format, **kwargs
        )

    return _builder

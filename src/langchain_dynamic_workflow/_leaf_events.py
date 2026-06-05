"""Per-leaf runtime-event tap: normalize a leaf's callback subtree out-of-band.

A leaf ``agent()`` call runs as a LangChain runnable whose subtree fires
``on_*_start``/``on_*_end``/``on_*_error`` callbacks, each carrying a ``run_id`` and
a ``parent_run_id`` -- a live run tree. This module's handler is appended to the
leaf's own ``callbacks`` list (the same list deepagents forwards to its subagents),
so it observes the whole leaf subtree, including nested sub-agents.

The handler turns each callback edge into a :class:`LeafEvent` and forwards it to an
out-of-band sink. Correlation to the owning leaf is by the handler instance: it
closes over the leaf's span id at construction (one handler per leaf invocation), so
every edge it receives is, by construction, this leaf's -- no reliance on metadata
inheritance, which the subagent boundary drops. Quarantine is preserved: events go
to the sink, never into the host LLM's message context. ``detail`` is shape-only by
default (node kind/name/timing); raw tool input and model text appear only when the
handler is built with ``include_payloads=True``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _empty_detail() -> dict[str, Any]:
    """Build an empty, fully-typed detail bag (keeps Pyright strict happy)."""
    return {}


@dataclass(frozen=True, slots=True)
class LeafEvent:
    """One normalized runtime edge from a leaf's own callback subtree.

    Delivered out-of-band to an ``on_leaf_event`` sink; never injected into the host
    LLM's context. Correlated to the owning leaf via :attr:`leaf_span_id`; the
    in-leaf tree is reconstructable from :attr:`run_id` / :attr:`parent_run_id`.

    Attributes:
        leaf_span_id: The owning leaf's span id (correlation key to its AGENT span).
        run_id: The runnable's run id (a node in the leaf's run tree).
        parent_run_id: The parent runnable's run id, or ``None`` at the subtree root.
        kind: The runnable type -- ``"chain"`` / ``"chat_model"`` / ``"llm"`` /
            ``"tool"``.
        phase: The edge -- ``"start"`` / ``"end"`` / ``"error"``.
        name: The runnable name (tool name, model name, chain name).
        ts: Wall-clock epoch seconds when the edge fired.
        detail: A bounded, shape-only mapping by default; raw tool input / model
            text appear only when the handler is built with ``include_payloads``.
    """

    leaf_span_id: str
    run_id: str
    parent_run_id: str | None
    kind: str
    phase: str
    name: str
    ts: float
    detail: dict[str, Any] = field(default_factory=_empty_detail)


LeafEventSink = Callable[[LeafEvent], None]
"""Receives each normalized per-leaf runtime event (out-of-band telemetry/UI)."""


class LeafEventHandler(BaseCallbackHandler):
    """A callback handler normalizing a leaf's run subtree into :class:`LeafEvent`s.

    One instance is appended to a single leaf invocation's ``callbacks`` list. It
    closes over that leaf's span id, so every edge it receives is correlated to the
    owning leaf without relying on metadata inheritance (which the subagent boundary
    drops). The handler is a thin normalizer; it does not swallow a raising sink --
    the engine's inline-sink contract makes that the consumer's responsibility.

    Args:
        leaf_span_id: The owning leaf's span id, stamped onto every emitted event.
        sink: The out-of-band callback receiving each :class:`LeafEvent`.
        include_payloads: When ``True``, include bounded raw payloads (tool input,
            model text) in ``detail``. Defaults to ``False`` (shape-only).
    """

    # Surface a raising handler through LangChain's callback manager rather than
    # letting it log-and-swallow, so a raising sink stays the consumer's concern.
    raise_error: bool = True

    def __init__(
        self,
        *,
        leaf_span_id: str,
        sink: LeafEventSink,
        include_payloads: bool = False,
    ) -> None:
        super().__init__()
        self._leaf_span_id = leaf_span_id
        self._sink = sink
        self._include_payloads = include_payloads
        # run_id -> name recorded at the start edge. LangChain's end/error callbacks
        # carry no ``serialized`` payload, so the only way an end/error edge can name
        # its runnable is to remember it from the matching start edge. A run_id is
        # unique per runnable and its start always precedes its end/error, so a plain
        # dict (set on start, popped on end/error) needs no lock.
        self._names: dict[UUID, str] = {}

    # --- name tracking -------------------------------------------------------

    def _remember(self, serialized: dict[str, Any] | None, run_id: UUID) -> str:
        """Record and return the runnable name for ``run_id`` at its start edge."""
        name = self._name_of(serialized)
        self._names[run_id] = name
        return name

    def _recall(self, run_id: UUID) -> str:
        """Return (and forget) the name recorded for ``run_id`` at its start edge."""
        return self._names.pop(run_id, "")

    # --- emit helper ---------------------------------------------------------

    def _emit(
        self,
        *,
        kind: str,
        phase: str,
        name: str,
        run_id: UUID,
        parent_run_id: UUID | None,
        detail: dict[str, Any],
    ) -> None:
        self._sink(
            LeafEvent(
                leaf_span_id=self._leaf_span_id,
                run_id=str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                kind=kind,
                phase=phase,
                name=name,
                ts=time.time(),
                detail=detail,
            )
        )

    @staticmethod
    def _name_of(serialized: dict[str, Any] | None) -> str:
        """Pull a readable runnable name from a serialized payload."""
        if not serialized:
            return ""
        name = serialized.get("name")
        if isinstance(name, str):
            return name
        ident: object = serialized.get("id")
        if isinstance(ident, list):
            elements = cast("list[object]", ident)
            if elements:
                return str(elements[-1])
        return ""

    # --- chain edges ---------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            kind="chain",
            phase="start",
            name=self._remember(serialized, run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail={},
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            kind="chain",
            phase="end",
            name=self._recall(run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail={},
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            kind="chain",
            phase="error",
            name=self._recall(run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail={"error": f"{type(error).__name__}: {error}"},
        )

    # --- chat model / llm edges ---------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            kind="chat_model",
            phase="start",
            name=self._remember(serialized, run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail={},
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        detail: dict[str, Any] = {}
        if self._include_payloads:
            texts = [g.text for gen in response.generations for g in gen if g.text]
            detail["text"] = " ".join(texts)[:500]
        self._emit(
            kind="chat_model",
            phase="end",
            name=self._recall(run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail=detail,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            kind="chat_model",
            phase="error",
            name=self._recall(run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail={"error": f"{type(error).__name__}: {error}"},
        )

    # --- tool edges ----------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        detail: dict[str, Any] = {}
        if self._include_payloads:
            detail["input"] = input_str[:500]
        self._emit(
            kind="tool",
            phase="start",
            name=self._remember(serialized, run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail=detail,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        detail: dict[str, Any] = {}
        if self._include_payloads:
            detail["output"] = str(output)[:500]
        self._emit(
            kind="tool",
            phase="end",
            name=self._recall(run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail=detail,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._emit(
            kind="tool",
            phase="error",
            name=self._recall(run_id),
            run_id=run_id,
            parent_run_id=parent_run_id,
            detail={"error": f"{type(error).__name__}: {error}"},
        )


__all__ = ["LeafEvent", "LeafEventHandler", "LeafEventSink"]

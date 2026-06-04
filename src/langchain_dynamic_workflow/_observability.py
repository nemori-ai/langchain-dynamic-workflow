"""Observability spans for the orchestration primitives (observability-by-default).

Every ``agent()`` / ``parallel()`` / ``pipeline()`` call opens a structured span
that is emitted on completion, so a trace consumer sees what ran without the
orchestration script instrumenting anything: which leaf executed, whether it was a
journal hit, how many tokens it burned, and how a fan-out fared (item count,
survivors). A span that raises is still emitted with its error text, so failures
are observable too.

The v1 implementation is inline and intentionally minimal — a span carries a kind,
a name, a free-form attribute bag, a wall-clock duration, and an optional error.
It maps cleanly onto an external tracer (span kind -> span name, attributes ->
span attributes) without committing the engine to one. When no sink is wired the
recorder is a silent no-op, so observability costs nothing until a consumer opts in.

This module is deliberately substrate-agnostic: it imports nothing from the engine
or LangGraph, so it can be lifted into a standalone tracing subpackage later
without untangling dependencies.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _empty_attributes() -> dict[str, Any]:
    """Build an empty, fully-typed span attribute bag (keeps Pyright strict happy)."""
    return {}


class SpanKind(StrEnum):
    """The orchestration primitive a span describes.

    Attributes:
        AGENT: A single leaf ``agent()`` invocation (or a journal hit).
        PARALLEL: A ``parallel()`` blocking-barrier fan-out.
        PIPELINE: A ``pipeline()`` no-barrier streaming fan-out.
        RACE: A ``race()`` best-of-N early-exit fan-out (or a journaled-decision replay).
    """

    AGENT = "agent"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    RACE = "race"


@dataclass(slots=True)
class ActiveSpan:
    """An open span being populated inside a primitive's body.

    Attributes are set as the primitive learns them (a leaf's cache outcome, its
    token usage, a fan-out's surviving count). The recorder freezes the active
    span into an immutable :class:`Span` when the context block exits.

    Attributes:
        kind: The primitive the span describes.
        name: A short, human-readable span name (e.g. the agent type, or the
            fan-out label).
        attributes: A free-form attribute bag populated via :meth:`set`.
    """

    kind: SpanKind
    name: str
    attributes: dict[str, Any] = field(default_factory=_empty_attributes)

    def set(self, key: str, value: Any) -> None:
        """Record a span attribute (e.g. ``cached`` / ``usage_tokens``)."""
        self.attributes[key] = value


@dataclass(frozen=True, slots=True)
class Span:
    """A completed span emitted to the observability sink.

    Attributes:
        kind: The primitive the span describes.
        name: The span name.
        attributes: The attributes accumulated while the span was open.
        duration_s: Wall-clock seconds the span body took.
        error: The exception's string form if the body raised, else ``None``.
    """

    kind: SpanKind
    name: str
    attributes: dict[str, Any]
    duration_s: float
    error: str | None


SpanSink = Callable[[Span], None]
"""Receives each completed span (e.g. a collector, or a tracer adapter)."""


def _noop_sink(_span: Span) -> None:
    """Discard a span (the default sink keeps observability zero-cost when unused)."""


class SpanRecorder:
    """Opens spans for orchestration primitives and emits them on completion.

    Args:
        sink: Callback invoked with each completed :class:`Span`. Defaults to a
            silent no-op so observability is opt-in and free until a consumer is
            wired in.
    """

    def __init__(self, sink: SpanSink | None = None) -> None:
        self._sink: SpanSink = sink if sink is not None else _noop_sink

    @contextmanager
    def span(self, kind: SpanKind, name: str) -> Generator[ActiveSpan]:
        """Open a span for the duration of the ``with`` block, then emit it.

        The span is emitted whether the body returns or raises: on a raise the
        error text is captured on the emitted span and the exception is re-raised
        (the recorder never swallows). The body sets attributes on the yielded
        :class:`ActiveSpan` as it learns them.

        Args:
            kind: The primitive this span describes.
            name: A short span name.

        Yields:
            The open :class:`ActiveSpan` to populate with attributes.
        """
        active = ActiveSpan(kind=kind, name=name)
        start = time.monotonic()
        error: str | None = None
        try:
            yield active
        except BaseException as exc:
            # Capture the failure so it is observable, then re-raise unchanged.
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._sink(
                Span(
                    kind=active.kind,
                    name=active.name,
                    attributes=dict(active.attributes),
                    duration_s=time.monotonic() - start,
                    error=error,
                )
            )

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

import hashlib
import json
import time
from collections import defaultdict
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
        span_id: The minted resume-stable id, set at open and shared with both the
            opening :class:`SpanBegin` and the matching end :class:`Span`, so the
            primitive's body can correlate the leaf's callback subtree to this span.
        attributes: A free-form attribute bag populated via :meth:`set`.
    """

    kind: SpanKind
    name: str
    span_id: str = ""
    attributes: dict[str, Any] = field(default_factory=_empty_attributes)

    def set(self, key: str, value: Any) -> None:
        """Record a span attribute (e.g. ``cached`` / ``usage_tokens``)."""
        self.attributes[key] = value


@dataclass(frozen=True, slots=True)
class SpanBegin:
    """A span-open edge emitted the instant a primitive's span opens.

    Carries the resume-stable id shared with the matching end :class:`Span`, the
    primitive kind and name, the attributes already known at open, and a wall-clock
    plus monotonic start so a consumer can render a live elapsed timer. End-only
    fields (cache outcome, token usage, duration) are not yet known and are absent.

    Attributes:
        span_id: The span's resume-stable id, shared with the matching end span.
        kind: The primitive the span describes.
        name: The span name.
        attributes: Attributes already set when the span opened.
        started_at: Wall-clock epoch seconds at span open.
        monotonic_start: A monotonic reference for the same open instant.
    """

    span_id: str
    kind: SpanKind
    name: str
    attributes: dict[str, Any]
    started_at: float
    monotonic_start: float


@dataclass(frozen=True, slots=True)
class Span:
    """A completed span emitted to the observability sink.

    Attributes:
        span_id: The span's resume-stable id, shared with its opening
            :class:`SpanBegin`, so a consumer correlates the begin and end edges.
        kind: The primitive the span describes.
        name: The span name.
        attributes: The attributes accumulated while the span was open.
        duration_s: Wall-clock seconds the span body took.
        error: The exception's string form if the body raised, else ``None``.
    """

    span_id: str
    kind: SpanKind
    name: str
    attributes: dict[str, Any]
    duration_s: float
    error: str | None


SpanSink = Callable[[Span], None]
"""Receives each completed span (e.g. a collector, or a tracer adapter)."""

SpanBeginSink = Callable[[SpanBegin], None]
"""Receives a span-open edge for the running state and the elapsed timer."""


def _noop_sink(_span: Span) -> None:
    """Discard a span (the default sink keeps observability zero-cost when unused)."""


def _noop_begin_sink(_begin: SpanBegin) -> None:
    """Discard a span-open edge (the default begin sink is a zero-cost no-op)."""


class SpanRecorder:
    """Opens spans for orchestration primitives, emitting a begin and an end edge.

    Args:
        sink: Callback invoked with each completed :class:`Span`. Defaults to a
            silent no-op so observability is opt-in and free until a consumer is
            wired in.
        begin_sink: Callback invoked with a :class:`SpanBegin` the instant each
            span opens, before its body runs. Defaults to a silent no-op.
    """

    def __init__(
        self,
        sink: SpanSink | None = None,
        *,
        begin_sink: SpanBeginSink | None = None,
    ) -> None:
        self._sink: SpanSink = sink if sink is not None else _noop_sink
        self._begin_sink: SpanBeginSink = begin_sink if begin_sink is not None else _noop_begin_sink
        # Per-(kind, name) occurrence counter. The Nth span sharing a (kind, name)
        # gets ordinal N, so the minted span_id is resume-stable for the sequential
        # path: the script re-executes in the same source order on resume (enforced
        # by the determinism guard), so a fresh run and an honest resume mint the
        # identical id sequence. The counter resets per recorder, i.e. per run.
        self._occurrences: defaultdict[tuple[SpanKind, str], int] = defaultdict(int)

    def _mint_span_id(self, kind: SpanKind, name: str) -> str:
        """Mint a resume-stable span id from ``(kind, name, occurrence-ordinal)``.

        The id is a truncated SHA-256 over the stable triple, so it reproduces
        exactly when the script replays in the same source order. The occurrence
        ordinal distinguishes genuinely-distinct same-(kind, name) spans (the Nth
        gets ordinal N) while keeping honest re-emits on the same id.

        Args:
            kind: The span's primitive kind.
            name: The span's name.

        Returns:
            A 16-char hex id stable across a deterministic resume.
        """
        ordinal = self._occurrences[(kind, name)]
        self._occurrences[(kind, name)] = ordinal + 1
        payload = {"kind": kind.value, "name": name, "ordinal": ordinal}
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @contextmanager
    def span(self, kind: SpanKind, name: str) -> Generator[ActiveSpan]:
        """Open a span for the duration of the ``with`` block, emitting begin+end.

        A :class:`SpanBegin` is emitted the instant the span opens (before the body
        runs), and a completed :class:`Span` is emitted when the block exits —
        whether the body returns or raises (on a raise the error text is captured
        and the exception re-raised; the recorder never swallows). Both edges carry
        the same resume-stable ``span_id``. The body sets attributes on the yielded
        :class:`ActiveSpan` as it learns them.

        Args:
            kind: The primitive this span describes.
            name: A short span name.

        Yields:
            The open :class:`ActiveSpan` to populate with attributes.
        """
        span_id = self._mint_span_id(kind, name)
        active = ActiveSpan(kind=kind, name=name, span_id=span_id)
        started_at = time.time()
        monotonic_start = time.monotonic()
        # Emit the open edge before the body runs so a consumer paints "running"
        # the instant the primitive starts, not when it finishes.
        self._begin_sink(
            SpanBegin(
                span_id=span_id,
                kind=kind,
                name=name,
                attributes=dict(active.attributes),
                started_at=started_at,
                monotonic_start=monotonic_start,
            )
        )
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
                    span_id=span_id,
                    kind=active.kind,
                    name=active.name,
                    attributes=dict(active.attributes),
                    duration_s=time.monotonic() - monotonic_start,
                    error=error,
                )
            )

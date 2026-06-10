"""Progress narration (``phase`` / ``log``) with replay-idempotent delivery.

``phase`` groups work under a title; ``log`` records a free-form narration line.
Both are deterministic side effects of the orchestration script, so a replay
re-emits the same entries in the same order. To keep delivery idempotent the log
is told how many entries the prior run already delivered: on replay the first
that many ``emit`` calls are recorded but *not* re-delivered, while genuinely new
entries (beyond the recorded count) are both recorded and delivered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class ProgressKind(StrEnum):
    """The kind of progress entry.

    Attributes:
        PHASE: A ``phase(title)`` grouping marker.
        LOG: A ``log(message)`` narration line.
        BATCH: A transient ``batch_map`` count/ETA refresh (delivered, never recorded).
    """

    PHASE = "phase"
    LOG = "log"
    BATCH = "batch"


@dataclass(frozen=True, slots=True)
class BatchMetrics:
    """Count/ETA snapshot carried by a transient ``BATCH`` progress entry.

    A live view of a ``batch_map`` fan-out's progress, computed out-of-band from a
    monotonic clock as items settle. The timestamps are non-deterministic and never
    reach a journal key; the snapshot is delivered to the sink and discarded.

    Attributes:
        completed: How many items have settled (returned or failed) so far.
        elapsed_seconds: Wall-clock seconds since the fan-out started.
        rate: Items settled per second (``completed / elapsed_seconds``).
        total: The item count when known (a ``Sized`` input or a ``total=`` hint),
            else ``None``.
        eta_seconds: Estimated seconds to completion (``(total - completed) / rate``)
            when ``total`` is known, else ``None``.
    """

    completed: int
    elapsed_seconds: float
    rate: float
    total: int | None = None
    eta_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ProgressEntry:
    """A single progress narration entry.

    Attributes:
        kind: Whether this entry is a phase marker, a log line, or a batch refresh.
        message: The phase title or log message text.
        metrics: The count/ETA snapshot for a ``BATCH`` entry; ``None`` for
            ``PHASE``/``LOG`` entries.
    """

    kind: ProgressKind
    message: str
    metrics: BatchMetrics | None = None


ProgressSink = Callable[[ProgressEntry], None]
"""Receives each newly-delivered progress entry (e.g. print, or a collector)."""


class ProgressLog:
    """Records progress entries and delivers only the ones not yet delivered.

    Args:
        delivered_count: How many entries a prior run already delivered. ``0`` on
            a fresh run (every entry is delivered); on replay it suppresses that
            many leading entries so already-shown progress is not repeated.
        sink: Callback invoked with each newly-delivered entry.
    """

    def __init__(self, *, delivered_count: int, sink: ProgressSink) -> None:
        self._delivered_count = delivered_count
        self._sink = sink
        self._entries: list[ProgressEntry] = []

    @property
    def entries(self) -> list[ProgressEntry]:
        """All entries recorded so far this run, in emission order."""
        return list(self._entries)

    def emit(self, kind: ProgressKind, message: str) -> None:
        """Record a progress entry and deliver it unless already delivered.

        An entry whose position is below ``delivered_count`` was already shown on
        a prior run and is suppressed; entries at or beyond that position are new
        and flow to the sink.

        Args:
            kind: The entry kind (phase marker or log line).
            message: The phase title or log message.
        """
        entry = ProgressEntry(kind=kind, message=message)
        position = len(self._entries)
        self._entries.append(entry)
        if position >= self._delivered_count:
            self._sink(entry)

    def emit_transient(self, message: str, *, metrics: BatchMetrics) -> None:
        """Deliver a transient ``BATCH`` refresh to the sink without recording it.

        Unlike :meth:`emit`, this is a fire-and-forget live view: the entry flows
        straight to the sink and is never appended to the recorded sequence nor
        counted against ``delivered_count``. It therefore never reaches the
        journal, the determinism guard, or the replay result — the non-deterministic
        timestamps it carries must never key a journal entry — and it is always
        delivered, never suppressed on replay (the consumer overwrites the previous
        refresh, like a progress bar).

        Args:
            message: A human-readable progress line (e.g. ``"batch: 340/1000"``).
            metrics: The count/ETA snapshot for this refresh.
        """
        self._sink(ProgressEntry(kind=ProgressKind.BATCH, message=message, metrics=metrics))

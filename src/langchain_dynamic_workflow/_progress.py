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
    """

    PHASE = "phase"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class ProgressEntry:
    """A single progress narration entry.

    Attributes:
        kind: Whether this entry is a phase marker or a log line.
        message: The phase title or log message text.
    """

    kind: ProgressKind
    message: str


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

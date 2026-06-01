"""Unit tests for progress narration (``phase`` / ``log``) + replay idempotency.

Progress entries are delivered to a sink as the script emits them, and recorded
so a replay knows how many were already delivered. On resume the already-recorded
entries are suppressed (idempotent) while genuinely new entries still flow.
"""

from __future__ import annotations

from langchain_dynamic_workflow._progress import ProgressEntry, ProgressKind, ProgressLog


def test_fresh_run_delivers_and_records_every_entry() -> None:
    delivered: list[ProgressEntry] = []
    log = ProgressLog(delivered_count=0, sink=delivered.append)

    log.emit(ProgressKind.PHASE, "research")
    log.emit(ProgressKind.LOG, "found 3 sources")

    assert [(e.kind, e.message) for e in delivered] == [
        (ProgressKind.PHASE, "research"),
        (ProgressKind.LOG, "found 3 sources"),
    ]
    # The full sequence is recorded for the next replay to count against.
    assert [(e.kind, e.message) for e in log.entries] == [
        (ProgressKind.PHASE, "research"),
        (ProgressKind.LOG, "found 3 sources"),
    ]


def test_replay_suppresses_already_delivered_entries() -> None:
    # Two entries were delivered on the first run. On replay the script re-emits
    # them (same order) plus one new entry; only the new entry reaches the sink.
    delivered: list[ProgressEntry] = []
    log = ProgressLog(delivered_count=2, sink=delivered.append)

    log.emit(ProgressKind.PHASE, "research")  # replayed: already delivered, suppressed
    log.emit(ProgressKind.LOG, "found 3 sources")  # replayed: suppressed
    log.emit(ProgressKind.LOG, "new on resume")  # genuinely new: delivered

    assert [e.message for e in delivered] == ["new on resume"]
    # All three are still recorded so the sequence stays complete for the future.
    assert [e.message for e in log.entries] == [
        "research",
        "found 3 sources",
        "new on resume",
    ]


def test_replay_with_no_new_entries_delivers_nothing() -> None:
    # A faithful replay that re-emits exactly the recorded entries delivers none
    # of them again — the idempotency guarantee at its strictest.
    delivered: list[ProgressEntry] = []
    log = ProgressLog(delivered_count=2, sink=delivered.append)

    log.emit(ProgressKind.PHASE, "a")
    log.emit(ProgressKind.LOG, "b")

    assert delivered == []
    assert len(log.entries) == 2

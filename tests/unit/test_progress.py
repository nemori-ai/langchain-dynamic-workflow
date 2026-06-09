"""Unit tests for progress narration (``phase`` / ``log``) + replay idempotency.

Progress entries are delivered to a sink as the script emits them, and recorded
so a replay knows how many were already delivered. On resume the already-recorded
entries are suppressed (idempotent) while genuinely new entries still flow.
"""

from __future__ import annotations

from langchain_dynamic_workflow._progress import (
    BatchMetrics,
    ProgressEntry,
    ProgressKind,
    ProgressLog,
)


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


def test_batch_metrics_shape() -> None:
    # BatchMetrics is the structured count/ETA payload a BATCH entry carries.
    # Non-default fields first, nullable-with-default last (dataclass ordering).
    metrics = BatchMetrics(
        completed=340,
        elapsed_seconds=12.0,
        rate=28.0,
        total=1000,
        eta_seconds=23.5,
    )
    assert metrics.completed == 340
    assert metrics.elapsed_seconds == 12.0
    assert metrics.rate == 28.0
    assert metrics.total == 1000
    assert metrics.eta_seconds == 23.5
    # Unknown total -> no ETA (graceful degradation): both nullable fields default None.
    open_ended = BatchMetrics(completed=5, elapsed_seconds=2.0, rate=2.5)
    assert open_ended.total is None
    assert open_ended.eta_seconds is None


def test_batch_progress_entry_carries_metrics() -> None:
    metrics = BatchMetrics(
        completed=340,
        elapsed_seconds=12.0,
        rate=28.0,
        total=1000,
        eta_seconds=23.5,
    )
    entry = ProgressEntry(
        kind=ProgressKind.BATCH,
        message="batch: 340/1000 (~24s left)",
        metrics=metrics,
    )
    assert entry.kind == ProgressKind.BATCH
    assert entry.message == "batch: 340/1000 (~24s left)"
    assert entry.metrics is metrics
    # entry.metrics is metrics (asserted above); read through the narrowed local so
    # pyright strict does not flag Optional member access.
    assert metrics.completed == 340


def test_phase_and_log_entries_have_no_metrics() -> None:
    # Backward compat: a PHASE/LOG entry built without metrics leaves the new
    # trailing field None, so existing callers are unaffected.
    phase = ProgressEntry(kind=ProgressKind.PHASE, message="research")
    log = ProgressEntry(kind=ProgressKind.LOG, message="found 3 sources")
    assert phase.metrics is None
    assert log.metrics is None

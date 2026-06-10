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


def test_emit_transient_delivers_a_batch_entry_to_the_sink() -> None:
    # A transient BATCH refresh is delivered to the sink (the live progress bar)
    # carrying its metrics, exactly like a normal emit reaches the consumer.
    delivered: list[ProgressEntry] = []
    log = ProgressLog(delivered_count=0, sink=delivered.append)
    metrics = BatchMetrics(completed=10, elapsed_seconds=1.0, rate=10.0, total=100, eta_seconds=9.0)

    log.emit_transient("batch: 10/100 (~9s left)", metrics=metrics)

    assert len(delivered) == 1
    assert delivered[0].kind == ProgressKind.BATCH
    assert delivered[0].message == "batch: 10/100 (~9s left)"
    assert delivered[0].metrics is metrics


def test_emit_transient_never_records_into_entries() -> None:
    # The transient entry is out-of-band: it must NOT enter the append-only log,
    # so it never reaches the journal / determinism guard / replay result, and a
    # following recorded emit() still lands at position 0 (the transient did not
    # advance the recorded sequence).
    delivered: list[ProgressEntry] = []
    log = ProgressLog(delivered_count=0, sink=delivered.append)
    metrics = BatchMetrics(completed=10, elapsed_seconds=1.0, rate=10.0)

    log.emit_transient("batch: 10/?", metrics=metrics)
    log.emit_transient("batch: 20/?", metrics=metrics)

    # Two transients delivered, but the recorded log did not grow.
    assert len(delivered) == 2
    assert log.entries == []

    # A subsequent recorded emit() is unaffected — it records at position 0.
    log.emit(ProgressKind.PHASE, "research")
    assert [e.kind for e in log.entries] == [ProgressKind.PHASE]
    assert delivered[-1].kind == ProgressKind.PHASE


def test_emit_transient_is_delivered_even_on_replay() -> None:
    # On resume (delivered_count > 0) recorded PHASE/LOG entries are suppressed by
    # position, but a transient BATCH refresh is never suppressed — live progress
    # is a view, not history, so it always reaches the sink.
    delivered: list[ProgressEntry] = []
    log = ProgressLog(delivered_count=5, sink=delivered.append)
    metrics = BatchMetrics(
        completed=42, elapsed_seconds=2.0, rate=21.0, total=100, eta_seconds=2.76
    )

    log.emit_transient("batch: 42/100 (~3s left)", metrics=metrics)

    assert len(delivered) == 1
    assert delivered[0].kind == ProgressKind.BATCH
    assert delivered[0].metrics is metrics
    # The transient never touched the recorded sequence on replay either.
    assert log.entries == []

"""Unit tests for the sqlite-backed persistent workflow store.

These cover ``SqliteWorkflowStore`` and its run-scoped journal view, the durable
counterpart to the in-memory default. They exercise the load-bearing invariants
of the persistence design directly:

* Spec CRUD survives ``aclose`` then a fresh ``open`` of the same db file.
* Journal ``put``/``get`` round-trips a ``JournalRecord``.
* Write-through durability: a ``put`` is durable on return with *no* explicit
  commit, because the store connection is opened in autocommit mode.
* ``get_progress_count`` coalesces a missing row to ``0`` (in-memory parity).
* Concurrent multi-leaf ``put``/``get`` on one connection stays correct without
  any extra lock (the single connection serializes ops).
* ``journal_for`` scopes every query by ``run_id`` so two runs never collide.
* The store is a structural ``WorkflowRunStore``.

The tests are offline (no model, no API keys) and use ``tmp_path`` db files. Each
store is always closed in a ``finally`` so the WAL sidecar files are released.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import aiosqlite
import pytest

from langchain_dynamic_workflow._engine import JournalRecord
from langchain_dynamic_workflow._persistence import CorruptJournalRowError, SqliteWorkflowStore
from langchain_dynamic_workflow._run_store import RunSpec, WorkflowRunStore


def _spec() -> RunSpec:
    """A representative named-workflow spec for round-trip assertions."""
    return RunSpec(
        kind="name",
        name_or_source="incident_triage",
        args={"severity": "high", "alerts": [1, 2, 3]},
        label="Incident triage",
        journal_run_id="origin-run-id",
    )


async def test_spec_crud_persists_across_aclose_and_reopen(tmp_path: Path) -> None:
    """A saved spec survives closing the store and reopening the same db file.

    This is the cross-session promise for the run registry: a fresh process
    pointed at the same db file must rebuild the original launch spec.
    """
    db_path = tmp_path / "workflows.db"
    spec = _spec()

    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.save_spec("run-1", spec)
    finally:
        await store.aclose()

    reopened = await SqliteWorkflowStore.open(db_path)
    try:
        loaded = await reopened.load_spec("run-1")
        assert loaded == spec
        assert await reopened.load_spec("never-saved") is None
    finally:
        await reopened.aclose()


async def test_save_spec_upserts_in_place(tmp_path: Path) -> None:
    """Saving twice under one run id overwrites rather than duplicating."""
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.save_spec("run-1", _spec())
        updated = RunSpec(
            kind="script",
            name_or_source="async def main(ctx): return 'ok'",
            args={"x": 1},
            label="Authored run",
            journal_run_id="another-origin",
        )
        await store.save_spec("run-1", updated)

        loaded = await store.load_spec("run-1")
        assert loaded == updated
    finally:
        await store.aclose()


async def test_save_spec_preserves_null_journal_lineage(tmp_path: Path) -> None:
    """A spec with no journal lineage round-trips with ``journal_run_id`` ``None``.

    A fresh launch persists ``journal_run_id=None`` (the tool stamps the canonical
    origin only when relaunching a resume), so the nullable column must survive the
    db round-trip rather than coercing to a string.
    """
    db_path = tmp_path / "workflows.db"
    fresh = RunSpec(
        kind="name",
        name_or_source="wf",
        args={"x": 1},
        label="wf",
        journal_run_id=None,
    )
    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.save_spec("run-1", fresh)
        loaded = await store.load_spec("run-1")
        assert loaded == fresh
        assert loaded is not None
        assert loaded.journal_run_id is None
    finally:
        await store.aclose()


async def test_delete_spec_removes_the_run_specs_row(tmp_path: Path) -> None:
    """``delete_spec`` deletes the persisted row so a later load misses.

    This backs the orphan-cleanup contract: a launch persists its spec before the
    background manager admits it, so a quota refusal must be able to delete that
    spec and leave no unresumable orphan in the durable registry.
    """
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.save_spec("run-1", _spec())
        assert await store.load_spec("run-1") is not None

        await store.delete_spec("run-1")
        assert await store.load_spec("run-1") is None

        # Deleting an absent row is silent, not an error.
        await store.delete_spec("never-saved")
    finally:
        await store.aclose()


async def test_save_spec_round_trips_unicode_args_faithfully(tmp_path: Path) -> None:
    """Unicode keys and values in ``args`` survive the JSON round-trip exactly.

    ``args`` originate from the model as a JSON object, so JSON-native values are
    the documented contract. This pins the realistic case — non-ASCII keys and
    values — round-tripping through ``json.dumps``/``json.loads`` byte-for-byte.
    """
    db_path = tmp_path / "workflows.db"
    spec = RunSpec(
        kind="name",
        name_or_source="résumé_workflow",
        args={"主题": "电池技术", "note": "naïve café — façade", "emoji": "🚀"},
        label="Unicode launch",
        journal_run_id=None,
    )
    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.save_spec("run-1", spec)
        loaded = await store.load_spec("run-1")
        assert loaded == spec
        assert loaded is not None
        assert loaded.args == {"主题": "电池技术", "note": "naïve café — façade", "emoji": "🚀"}
    finally:
        await store.aclose()


async def test_journal_put_get_round_trips(tmp_path: Path) -> None:
    """A journaled record reads back equal, including its usage."""
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal = store.journal_for("run-1")
        assert await journal.get("key-a") is None

        record = JournalRecord(result="Paris", usage=128)
        await journal.put("key-a", record)

        loaded = await journal.get("key-a")
        assert loaded == record
        assert loaded is not None
        assert loaded.result == "Paris"
        assert loaded.usage == 128
    finally:
        await store.aclose()


async def test_journal_put_is_durable_without_explicit_commit(tmp_path: Path) -> None:
    """A put then close then reopen shows the row with no explicit commit.

    The store connection is opened in autocommit mode, so every ``put`` is
    durable the moment it returns. Were the connection left in the default
    deferred-transaction mode, the uncommitted insert would roll back on close
    and the reopened db would show no row.
    """
    db_path = tmp_path / "workflows.db"
    record = JournalRecord(result="Berlin", usage=64)

    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.journal_for("run-1").put("key-a", record)
    finally:
        await store.aclose()  # no explicit commit anywhere above

    reopened = await SqliteWorkflowStore.open(db_path)
    try:
        loaded = await reopened.journal_for("run-1").get("key-a")
        assert loaded == record
    finally:
        await reopened.aclose()


async def test_journal_sequence_round_trips(tmp_path: Path) -> None:
    """The ordered call-key sequence persists and reads back as a list copy."""
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal = store.journal_for("run-1")
        assert await journal.get_sequence() is None

        sequence = ["key-a", "key-b", "key-c"]
        await journal.put_sequence(sequence)

        loaded = await journal.get_sequence()
        assert loaded == sequence
        # A mutation of the returned list must not corrupt the stored sequence.
        assert loaded is not None
        loaded.append("key-d")
        assert await journal.get_sequence() == sequence
    finally:
        await store.aclose()


async def test_progress_count_missing_coalesces_to_zero(tmp_path: Path) -> None:
    """A run with no progress row reports a count of ``0``, not ``None``.

    This matches ``InMemoryJournalStore`` semantics, where the progress count is
    initialized to ``0``; resume logic relies on a numeric count being present.
    """
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal = store.journal_for("never-progressed")
        assert await journal.get_progress_count() == 0

        await journal.put_progress_count(5)
        assert await journal.get_progress_count() == 5
    finally:
        await store.aclose()


async def test_concurrent_puts_and_gets_are_correct(tmp_path: Path) -> None:
    """Fifty concurrent put/get pairs on one store all resolve correctly.

    The single store connection serializes every op through its worker thread,
    so concurrent multi-leaf access is correct without any extra ``asyncio.Lock``.
    """
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal = store.journal_for("run-1")

        async def put_then_get(index: int) -> tuple[int, JournalRecord | None]:
            record = JournalRecord(result=f"r{index}", usage=index)
            await journal.put(f"key-{index}", record)
            return index, await journal.get(f"key-{index}")

        results = await asyncio.gather(*(put_then_get(i) for i in range(50)))

        for index, loaded in results:
            assert loaded == JournalRecord(result=f"r{index}", usage=index)
    finally:
        await store.aclose()


async def test_journal_for_scopes_by_run_id(tmp_path: Path) -> None:
    """Two runs writing the same key never read each other's record."""
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal_one = store.journal_for("run-1")
        journal_two = store.journal_for("run-2")

        await journal_one.put("shared-key", JournalRecord(result="one", usage=1))
        await journal_two.put("shared-key", JournalRecord(result="two", usage=2))

        assert await journal_one.get("shared-key") == JournalRecord(result="one", usage=1)
        assert await journal_two.get("shared-key") == JournalRecord(result="two", usage=2)

        await journal_one.put_progress_count(7)
        assert await journal_two.get_progress_count() == 0

        await journal_one.put_sequence(["a"])
        assert await journal_two.get_sequence() is None
    finally:
        await store.aclose()


async def test_store_exposes_a_checkpointer(tmp_path: Path) -> None:
    """The store wires a persistent checkpointer over a second connection."""
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        assert store.checkpointer is not None
    finally:
        await store.aclose()


async def test_store_satisfies_the_protocol(tmp_path: Path) -> None:
    """The sqlite store is a structural ``WorkflowRunStore``."""
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        assert isinstance(store, WorkflowRunStore)
    finally:
        await store.aclose()


async def test_open_failure_after_store_conn_closes_the_store_conn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the store connection is opened still closes that connection.

    ``open`` opens the autocommit store connection first, then opens a second
    connection and constructs the saver. If that later work raises, the first
    connection (a live worker thread holding a file lock) must be closed before
    the error propagates, or the host leaks a thread and risks an exit hang.
    """
    db_path = tmp_path / "workflows.db"
    real_connect = aiosqlite.connect
    opened: list[aiosqlite.Connection] = []
    calls = {"n": 0}

    def fake_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        calls["n"] += 1
        if calls["n"] == 1:
            conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
            opened.append(conn)
            return conn
        raise RuntimeError("simulated second-connect failure")

    monkeypatch.setattr("langchain_dynamic_workflow._persistence.aiosqlite.connect", fake_connect)

    with pytest.raises(RuntimeError, match="simulated second-connect failure"):
        await SqliteWorkflowStore.open(db_path)

    assert len(opened) == 1
    store_conn = opened[0]
    # The store connection must have been closed before the error propagated: a
    # closed aiosqlite connection has no active connection and refuses queries.
    assert store_conn._connection is None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="no active connection"):
        await store_conn.execute("SELECT 1")


async def test_open_failure_in_saver_closes_both_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure constructing the saver closes both already-opened connections.

    The saver is built last, after both connections are open. If its constructor
    raises, neither connection has an owner yet, so ``open`` must close both
    rather than leak two worker threads.
    """
    db_path = tmp_path / "workflows.db"
    real_connect = aiosqlite.connect
    opened: list[aiosqlite.Connection] = []

    def tracking_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    def boom(_conn: object) -> object:
        raise RuntimeError("simulated saver construction failure")

    monkeypatch.setattr(
        "langchain_dynamic_workflow._persistence.aiosqlite.connect", tracking_connect
    )
    monkeypatch.setattr("langchain_dynamic_workflow._persistence.AsyncSqliteSaver", boom)

    with pytest.raises(RuntimeError, match="simulated saver construction failure"):
        await SqliteWorkflowStore.open(db_path)

    assert len(opened) == 2
    for conn in opened:
        assert conn._connection is None  # type: ignore[attr-defined]


async def test_aclose_closes_checkpointer_even_if_store_close_raises(
    tmp_path: Path,
) -> None:
    """``aclose`` closes the checkpointer connection even if the store close fails.

    The two closes are independent: a failure tearing down the store connection
    must not strand the checkpointer connection's worker thread and WAL sidecar.
    """
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    store_conn = store._store_conn  # type: ignore[attr-defined]
    checkpointer_conn = store._checkpointer_conn  # type: ignore[attr-defined]
    real_store_close = store_conn.close

    async def failing_close() -> None:
        raise RuntimeError("simulated store close failure")

    # Replace the store connection's close so the first teardown step fails.
    store_conn.close = failing_close  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="simulated store close failure"):
            await store.aclose()

        # The checkpointer connection must still have been closed despite the failure.
        assert checkpointer_conn._connection is None  # type: ignore[attr-defined]
    finally:
        # Restore and run the real close so the store connection's worker thread
        # actually stops (a stubbed-out close would otherwise leak the thread).
        store_conn.close = real_store_close  # type: ignore[method-assign]
        await store_conn.close()


async def test_open_raises_loud_error_on_incompatible_schema_version(
    tmp_path: Path,
) -> None:
    """Opening a db stamped with an unsupported future schema version raises a clear error.

    If the db file carries a ``PRAGMA user_version`` that is neither 0 (fresh/untracked)
    nor the current ``_SCHEMA_VERSION``, ``open`` must raise a descriptive error naming the
    db path, the found version, and the supported version — instead of silently proceeding
    to a cryptic ``OperationalError`` on the first query.
    """
    from langchain_dynamic_workflow._persistence import IncompatibleSchemaError

    db_path = tmp_path / "workflows.db"

    # Stamp the db with a future version that the current engine cannot handle.
    raw = await aiosqlite.connect(str(db_path))
    try:
        await raw.execute("PRAGMA user_version=999")
    finally:
        await raw.close()

    with pytest.raises(IncompatibleSchemaError, match="999"):
        await SqliteWorkflowStore.open(db_path)


async def test_open_stamps_version_on_fresh_db_and_reopens_idempotently(
    tmp_path: Path,
) -> None:
    """A fresh db gets user_version=1 stamped; reopening the same file succeeds.

    This verifies two things in one round-trip:
    * ``open`` on a brand-new (user_version=0) db writes the schema and stamps
      ``user_version = 1`` so a future open can recognise the schema.
    * A subsequent ``open`` on the already-stamped file is idempotent: it neither
      errors nor corrupts the stored data.
    """
    db_path = tmp_path / "workflows.db"
    spec = _spec()

    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.save_spec("run-1", spec)
    finally:
        await store.aclose()

    # Verify the version was stamped.
    raw = await aiosqlite.connect(str(db_path))
    try:
        async with raw.execute("PRAGMA user_version") as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == 1
    finally:
        await raw.close()

    # Idempotent reopen: no error, and data survives.
    reopened = await SqliteWorkflowStore.open(db_path)
    try:
        loaded = await reopened.load_spec("run-1")
        assert loaded == spec
    finally:
        await reopened.aclose()


async def test_open_cleanup_propagates_original_error_even_if_store_close_also_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original open failure propagates even when the cleanup store close also raises.

    Both connections are closed via ``_close_quietly`` on the partial-failure path, so a
    secondary error during cleanup is suppressed and the original exception always propagates.
    This is the symmetric-cleanup invariant for Fix 2.
    """
    db_path = tmp_path / "workflows.db"
    real_connect = aiosqlite.connect
    store_conn_holder: list[aiosqlite.Connection] = []
    real_store_close_holder: list[object] = []

    def tracking_connect(*args: object, **kwargs: object) -> aiosqlite.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        # Capture the store (first) connection and stub its close to raise.
        if not store_conn_holder:
            real_store_close_holder.append(conn.close)

            async def failing_close() -> None:
                raise RuntimeError("secondary cleanup close failure")

            conn.close = failing_close  # type: ignore[method-assign]
            store_conn_holder.append(conn)
        return conn

    def boom(_conn: object) -> object:
        raise RuntimeError("original open failure")

    monkeypatch.setattr(
        "langchain_dynamic_workflow._persistence.aiosqlite.connect", tracking_connect
    )
    monkeypatch.setattr("langchain_dynamic_workflow._persistence.AsyncSqliteSaver", boom)

    # The ORIGINAL error must propagate even though store_conn.close() also raises.
    with pytest.raises(RuntimeError, match="original open failure"):
        await SqliteWorkflowStore.open(db_path)

    # Restore real close and clean up worker thread so the test doesn't leak.
    if store_conn_holder and real_store_close_holder:
        store_conn_holder[0].close = real_store_close_holder[0]  # type: ignore[method-assign]
        with contextlib.suppress(Exception):
            await store_conn_holder[0].close()


async def test_async_context_manager_opens_and_closes(tmp_path: Path) -> None:
    """``async with SqliteWorkflowStore.open(...)`` yields a usable, auto-closed store.

    A host that uses the context-manager form must get a working store inside the
    block and have both connections closed on exit, so it cannot forget the final
    teardown.
    """
    db_path = tmp_path / "workflows.db"
    store_ref: SqliteWorkflowStore
    async with await SqliteWorkflowStore.open(db_path) as store:
        store_ref = store
        await store.save_spec("run-1", _spec())
        assert await store.load_spec("run-1") == _spec()

    # On exit both connections are closed.
    assert store_ref._store_conn._connection is None  # type: ignore[attr-defined]
    assert store_ref._checkpointer_conn._connection is None  # type: ignore[attr-defined]


async def test_concurrent_puts_to_same_key_resolve_without_error(tmp_path: Path) -> None:
    """Concurrent puts to one key settle on a single value with no error (C4).

    The single store connection serializes every write through its worker thread,
    so N racing puts to the *same* ``(run_id, key)`` with distinct values resolve
    last-writer-wins: the final read returns one of the written values and no
    write raises.
    """
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal = store.journal_for("run-1")
        written = {JournalRecord(result=f"v{i}", usage=i) for i in range(50)}

        async def put(index: int) -> None:
            await journal.put("contended-key", JournalRecord(result=f"v{index}", usage=index))

        await asyncio.gather(*(put(i) for i in range(50)))

        final = await journal.get("contended-key")
        assert final in written
    finally:
        await store.aclose()


async def test_corrupt_resume_read_rows_raise_actionable_error(tmp_path: Path) -> None:
    """A torn JSON column on a resume-read path raises a domain error, not a bare decode.

    The resume path decodes the persisted journal sequence and run-spec args from
    JSON columns. A corrupt row (truncated / hand-edited / produced by an
    incompatible writer) must surface as a :class:`CorruptJournalRowError` that
    names the db path and the run id and chains the raw ``json.JSONDecodeError`` as
    its cause — mirroring ``IncompatibleSchemaError``. A bare ``json.JSONDecodeError``
    naming neither would leave the operator no way to locate the offending row.

    Both resume-read decode boundaries are pinned in one test: the journal sequence
    (``get_sequence``) and the run-spec args (``load_spec``).
    """
    db_path = tmp_path / "workflows.db"
    store = await SqliteWorkflowStore.open(db_path)
    try:
        await store.journal_for("run-1").put_sequence(["k1", "k2"])
        await store.save_spec(
            "run-1",
            RunSpec(
                kind="name",
                name_or_source="wf",
                args={"x": 1},
                label="L",
                journal_run_id="run-1",
            ),
        )
    finally:
        await store.aclose()

    # Tear the JSON on both resume-read columns via a raw connection.
    raw = await aiosqlite.connect(str(db_path))
    try:
        await raw.execute("UPDATE journal_sequence SET sequence='{bad' WHERE run_id='run-1'")
        await raw.execute("UPDATE run_specs SET args='{bad' WHERE run_id='run-1'")
        await raw.commit()
    finally:
        await raw.close()

    reopened = await SqliteWorkflowStore.open(db_path)
    try:
        with pytest.raises(CorruptJournalRowError) as seq_ei:
            await reopened.journal_for("run-1").get_sequence()
        seq_message = str(seq_ei.value)
        # Actionable: the message names the db path AND the run id.
        assert str(db_path) in seq_message
        assert "run-1" in seq_message
        # The raw decode error is preserved as the cause, not discarded.
        assert isinstance(seq_ei.value.__cause__, json.JSONDecodeError)

        with pytest.raises(CorruptJournalRowError) as spec_ei:
            await reopened.load_spec("run-1")
        spec_message = str(spec_ei.value)
        assert str(db_path) in spec_message
        assert "run-1" in spec_message
        assert isinstance(spec_ei.value.__cause__, json.JSONDecodeError)
    finally:
        await reopened.aclose()

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
from pathlib import Path

from langchain_dynamic_workflow._engine import JournalRecord
from langchain_dynamic_workflow._persistence import SqliteWorkflowStore
from langchain_dynamic_workflow._run_store import RunSpec, WorkflowRunStore


def _spec() -> RunSpec:
    """A representative named-workflow spec for round-trip assertions."""
    return RunSpec(
        kind="name",
        name_or_source="incident_triage",
        args={"severity": "high", "alerts": [1, 2, 3]},
        label="Incident triage",
        thread_id="thread-abc",
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
            thread_id="thread-xyz",
        )
        await store.save_spec("run-1", updated)

        loaded = await store.load_spec("run-1")
        assert loaded == updated
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

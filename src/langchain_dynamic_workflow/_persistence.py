"""Optional sqlite-backed persistence for cross-session / cross-process resume.

This module hosts the unified sqlite store that backs a workflow run registry,
its per-run journal, and the durable LangGraph checkpointer. It depends on the
optional ``[sqlite]`` extra; the base install stays dependency-free and falls
back to the in-memory store in ``_run_store``.

The store is built around three load-bearing sqlite invariants:

* The registry and journal connection is opened in autocommit mode
  (``isolation_level=None``), so every write is durable the moment it returns
  with no explicit ``commit`` call. The default deferred-transaction mode would
  roll uncommitted writes back on close, losing every journaled leaf.
* The checkpointer uses a *separate* connection to the *same* db file. The store
  connection (autocommit) and the checkpointer (explicit-commit, its own WAL
  regime) have incompatible isolation regimes and must not share a connection;
  under WAL, cross-connection reads still see committed writes.
* A single connection serializes all of its operations through one worker
  thread, so concurrent multi-leaf access on the shared store connection is
  correct without any extra ``asyncio.Lock``. Every query is keyed by the
  primary key and scoped to one ``run_id``.

``SqliteWorkflowStore.open`` is an async factory because the checkpointer binds
to the running event loop at construction; the host must build the store inside
its single persistent loop and reuse the one instance across all runs.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Cross-session persistence requires the optional 'sqlite' extra. "
        "Install: pip install 'langchain-dynamic-workflow[sqlite]' "
        "(or uv sync --extra sqlite)."
    ) from exc

from ._engine import JournalRecord, JournalStore
from ._run_store import RunSpec

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS run_specs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name_or_source TEXT NOT NULL,
    args TEXT NOT NULL,
    label TEXT NOT NULL,
    thread_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_records (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    result TEXT NOT NULL,
    usage INTEGER NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS journal_sequence (
    run_id TEXT PRIMARY KEY,
    sequence TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_progress (
    run_id TEXT PRIMARY KEY,
    count INTEGER NOT NULL
);
"""


class _RunScopedJournal:
    """A ``JournalStore`` view over one ``run_id`` on a shared sqlite connection.

    Every query is scoped to the bound ``run_id`` and keyed by the primary key,
    so distinct runs sharing one db file (and one connection) never collide.
    Writes use ``ON CONFLICT ... DO UPDATE`` upserts so a replay overwrites
    in place rather than failing on the primary key. The connection is opened in
    autocommit mode by the owning :class:`SqliteWorkflowStore`, so each write is
    durable on return with no explicit commit.

    This view holds no resources of its own: the connection is owned by the
    store, which closes it in :meth:`SqliteWorkflowStore.aclose`.
    """

    def __init__(self, conn: aiosqlite.Connection, run_id: str) -> None:
        """Bind the view to a connection and a run id.

        Args:
            conn: The shared, autocommit store connection owned by the store.
            run_id: The run whose journal rows this view reads and writes.
        """
        self._conn = conn
        self._run_id = run_id

    async def get(self, key: str) -> JournalRecord | None:
        """Return the cached record for ``key``, or ``None`` on miss.

        Args:
            key: The content-hash leaf key to look up within this run.

        Returns:
            The stored :class:`JournalRecord`, or ``None`` if absent.
        """
        async with self._conn.execute(
            "SELECT result, usage FROM journal_records WHERE run_id = ? AND key = ?",
            (self._run_id, key),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return JournalRecord(result=row[0], usage=row[1])

    async def put(self, key: str, value: JournalRecord) -> None:
        """Persist ``value`` under ``key`` for this run (upsert in place).

        Args:
            key: The content-hash leaf key to write within this run.
            value: The record to store.
        """
        await self._conn.execute(
            "INSERT INTO journal_records (run_id, key, result, usage) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(run_id, key) DO UPDATE SET "
            "result = excluded.result, usage = excluded.usage",
            (self._run_id, key, value.result, value.usage),
        )

    async def get_sequence(self) -> list[str] | None:
        """Return the recorded ordered call-key sequence, or ``None`` if unset.

        Returns:
            A fresh list copy of the stored sequence, or ``None`` if no sequence
            was recorded for this run.
        """
        async with self._conn.execute(
            "SELECT sequence FROM journal_sequence WHERE run_id = ?",
            (self._run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        decoded: list[str] = json.loads(row[0])
        return decoded

    async def put_sequence(self, sequence: list[str]) -> None:
        """Persist the ordered call-key sequence for this run (upsert in place).

        Args:
            sequence: The ordered leaf call-keys observed on a completed run.
        """
        await self._conn.execute(
            "INSERT INTO journal_sequence (run_id, sequence) VALUES (?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET sequence = excluded.sequence",
            (self._run_id, json.dumps(sequence)),
        )

    async def get_progress_count(self) -> int:
        """Return how many progress entries a prior run delivered for this run.

        A missing row coalesces to ``0`` to match :class:`InMemoryJournalStore`,
        whose progress count is initialized to ``0`` before any progress lands.

        Returns:
            The persisted progress count, or ``0`` if none was recorded.
        """
        async with self._conn.execute(
            "SELECT count FROM journal_progress WHERE run_id = ?",
            (self._run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return 0
        return row[0]

    async def put_progress_count(self, count: int) -> None:
        """Persist the progress-entry count for this run (upsert in place).

        Args:
            count: The number of progress entries delivered on a completed run.
        """
        await self._conn.execute(
            "INSERT INTO journal_progress (run_id, count) VALUES (?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET count = excluded.count",
            (self._run_id, count),
        )


class SqliteWorkflowStore:
    """Durable run registry, per-run journal, and checkpointer over one db file.

    This is the cross-session counterpart to ``InMemoryRunStore``: a fresh
    process pointed at the same db file resumes a run by ``run_id`` and replays
    its completed leaves from the persisted journal at zero new model cost. It
    satisfies the ``WorkflowRunStore`` protocol and additionally exposes a
    persistent LangGraph checkpointer over a second connection to the same file.

    Construct it with the async :meth:`open` factory from inside the host's
    event loop. The checkpointer binds to the running loop at construction, so a
    single instance must be reused across all runs on that loop and never shared
    across event loops. Close it with :meth:`aclose` at host shutdown to release
    both connections and their WAL sidecar files.
    """

    def __init__(
        self,
        store_conn: aiosqlite.Connection,
        checkpointer_conn: aiosqlite.Connection,
        checkpointer: AsyncSqliteSaver,
    ) -> None:
        """Bind the store to its two connections and checkpointer.

        Prefer the :meth:`open` async factory; the connections and checkpointer
        must be created inside the running event loop. This initializer takes
        already-opened resources and performs no I/O.

        Args:
            store_conn: The autocommit connection backing the registry + journal.
            checkpointer_conn: The separate connection wrapped by the saver.
            checkpointer: The persistent saver bound to the running loop.
        """
        self._store_conn = store_conn
        self._checkpointer_conn = checkpointer_conn
        self._checkpointer = checkpointer

    @classmethod
    async def open(cls, db_path: str | os.PathLike[str]) -> SqliteWorkflowStore:
        """Open (or create) the store over ``db_path`` inside the running loop.

        Opens the autocommit store connection, enables WAL and a busy timeout,
        bootstraps the schema, then opens a second connection wrapped in an
        ``AsyncSqliteSaver``. Both connections target the same db file. Must be
        awaited inside the host's persistent event loop: the checkpointer binds
        to that loop at construction.

        Args:
            db_path: Filesystem path to the sqlite db file (created if absent).

        Returns:
            A ready store whose registry, journal, and checkpointer share the db.
        """
        database = os.fspath(db_path)
        # The registry + journal connection runs in autocommit mode so every
        # write is durable on return with no explicit commit (C2). WAL lets the
        # separate checkpointer connection read committed writes, and the busy
        # timeout absorbs brief lock contention between the two connections.
        store_conn = await aiosqlite.connect(database, isolation_level=None)
        await store_conn.execute("PRAGMA journal_mode=WAL")
        await store_conn.execute("PRAGMA busy_timeout=5000")
        await store_conn.executescript(_SCHEMA_DDL)

        # A SECOND, separate connection backs the checkpointer: its explicit-
        # commit + WAL regime is incompatible with the autocommit store
        # connection, so the two must never share a connection (C3). The saver
        # is constructed directly over the connection (never from_conn_string,
        # which would close it on context exit and defeat cross-process resume).
        checkpointer_conn = await aiosqlite.connect(database)
        checkpointer = AsyncSqliteSaver(checkpointer_conn)

        return cls(store_conn, checkpointer_conn, checkpointer)

    @property
    def checkpointer(self) -> AsyncSqliteSaver:
        """The persistent LangGraph checkpointer over the second connection."""
        return self._checkpointer

    async def save_spec(self, run_id: str, spec: RunSpec) -> None:
        """Persist the launch spec for ``run_id`` (upsert in place).

        The spec's ``args`` are stored as a JSON string so a fresh process can
        rebuild the original launch arguments.

        Args:
            run_id: The unique identifier of the launched run.
            spec: The launch description to persist.
        """
        await self._store_conn.execute(
            "INSERT INTO run_specs (run_id, kind, name_or_source, args, label, thread_id) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "kind = excluded.kind, name_or_source = excluded.name_or_source, "
            "args = excluded.args, label = excluded.label, thread_id = excluded.thread_id",
            (
                run_id,
                spec.kind,
                spec.name_or_source,
                json.dumps(spec.args),
                spec.label,
                spec.thread_id,
            ),
        )

    async def load_spec(self, run_id: str) -> RunSpec | None:
        """Return the launch spec for ``run_id``, or ``None`` on miss.

        Args:
            run_id: The identifier of a previously launched run.

        Returns:
            The persisted launch spec rebuilt from its row, or ``None`` if no run
            was saved under ``run_id``.
        """
        async with self._store_conn.execute(
            "SELECT kind, name_or_source, args, label, thread_id FROM run_specs WHERE run_id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        args: dict[str, Any] = json.loads(row[2])
        return RunSpec(
            kind=row[0],
            name_or_source=row[1],
            args=args,
            label=row[3],
            thread_id=row[4],
        )

    def journal_for(self, run_id: str) -> JournalStore:
        """Return the per-run journal view for ``run_id``.

        Args:
            run_id: The identifier of the run whose journal is requested.

        Returns:
            A run-scoped journal view over the shared store connection. The view
            is cheap to recreate and holds no resources of its own, so repeated
            calls return fresh, behaviorally identical views.
        """
        return _RunScopedJournal(self._store_conn, run_id)

    async def aclose(self) -> None:
        """Close both connections, releasing their WAL sidecar files.

        Call this at host shutdown. Closing the store and checkpointer
        connections releases the ``-wal`` / ``-shm`` sidecars; leaving them open
        can also hang the program on exit.
        """
        await self._store_conn.close()
        await self._checkpointer_conn.close()

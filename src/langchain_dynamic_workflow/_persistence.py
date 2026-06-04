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

Schema versioning: ``PRAGMA user_version`` is used to track the schema version.
On open, ``user_version=0`` (fresh or untracked db) triggers DDL bootstrapping
followed by stamping ``user_version=_SCHEMA_VERSION``. A matching version
proceeds without error. Any other non-zero value is an incompatible schema and
raises a :class:`IncompatibleSchemaError` immediately, converting a silent
shape-drift into a loud, actionable failure.

``SqliteWorkflowStore.open`` is an async factory because the checkpointer binds
to the running event loop at construction; the host must build the store inside
its single persistent loop and reuse the one instance across all runs.
"""

from __future__ import annotations

import contextlib
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

_SCHEMA_VERSION: int = 1
"""Current db schema version stamped into ``PRAGMA user_version`` on every fresh open."""


class IncompatibleSchemaError(Exception):
    """Raised when a db file carries a schema version this engine cannot handle.

    Args:
        db_path: Filesystem path to the db file with the incompatible schema.
        found_version: The ``PRAGMA user_version`` value read from the file.
        supported_version: The ``_SCHEMA_VERSION`` this engine expects.
    """

    def __init__(self, db_path: str, found_version: int, supported_version: int) -> None:
        """Build the error with a descriptive message.

        Args:
            db_path: Filesystem path to the db file with the incompatible schema.
            found_version: The ``PRAGMA user_version`` value read from the file.
            supported_version: The ``_SCHEMA_VERSION`` this engine expects.
        """
        super().__init__(
            f"Incompatible db schema at '{db_path}': "
            f"found version {found_version}, supported version {supported_version}. "
            "The db was likely created by a newer version of langchain-dynamic-workflow."
        )
        self.db_path = db_path
        self.found_version = found_version
        self.supported_version = supported_version


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS run_specs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name_or_source TEXT NOT NULL,
    args TEXT NOT NULL,
    label TEXT NOT NULL,
    journal_run_id TEXT
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


async def _close_quietly(conn: aiosqlite.Connection | None) -> None:
    """Close ``conn`` if present, swallowing any close-time error.

    Used only on the partial-failure cleanup path of :meth:`SqliteWorkflowStore.open`,
    where an original exception is already in flight. Suppressing a secondary
    close error keeps that original cause as the one re-raised, while still
    attempting to release the connection's worker thread and file lock.

    Args:
        conn: The connection to close, or ``None`` if it was never opened.
    """
    if conn is None:
        return
    with contextlib.suppress(Exception):
        await conn.close()


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
        # write is durable on return with no explicit commit. WAL lets the
        # separate checkpointer connection read committed writes, and the busy
        # timeout absorbs brief lock contention between the two connections.
        store_conn = await aiosqlite.connect(database, isolation_level=None)
        checkpointer_conn: aiosqlite.Connection | None = None
        try:
            await store_conn.execute("PRAGMA journal_mode=WAL")
            await store_conn.execute("PRAGMA busy_timeout=5000")

            # Schema-version guard: read user_version before touching any tables.
            # user_version=0 means a fresh/untracked db — run DDL and stamp.
            # user_version=_SCHEMA_VERSION — already set up, idempotent re-run is fine.
            # Any other non-zero value — a future/incompatible schema: fail loud.
            async with store_conn.execute("PRAGMA user_version") as _cur:
                _version_row = await _cur.fetchone()
            current_version: int = _version_row[0] if _version_row is not None else 0
            if current_version == 0:
                await store_conn.executescript(_SCHEMA_DDL)
                await store_conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            elif current_version != _SCHEMA_VERSION:
                raise IncompatibleSchemaError(database, current_version, _SCHEMA_VERSION)
            # current_version == _SCHEMA_VERSION: already bootstrapped, proceed.

            # A SECOND, separate connection backs the checkpointer: its explicit-
            # commit + WAL regime is incompatible with the autocommit store
            # connection, so the two must never share a connection. The saver is
            # constructed directly over the connection (never from_conn_string,
            # which would close it on context exit and defeat cross-process
            # resume).
            checkpointer_conn = await aiosqlite.connect(database)
            checkpointer = AsyncSqliteSaver(checkpointer_conn)
        except BaseException:
            # Any failure after the store connection is open leaves a live worker
            # thread holding a file lock. Close BOTH connections defensively so
            # a secondary close error cannot mask the original failure — the
            # original exception always propagates.
            await _close_quietly(checkpointer_conn)
            await _close_quietly(store_conn)
            raise

        return cls(store_conn, checkpointer_conn, checkpointer)

    @property
    def checkpointer(self) -> AsyncSqliteSaver:
        """The persistent LangGraph checkpointer over the second connection."""
        return self._checkpointer

    async def save_spec(self, run_id: str, spec: RunSpec) -> None:
        """Persist the launch spec for ``run_id`` (upsert in place).

        The spec's ``args`` are stored as a JSON string so a fresh process can
        rebuild the original launch arguments; ``args`` must therefore be
        JSON-serializable. The nullable ``journal_run_id`` carries the canonical
        origin (the journal + checkpoint-thread lineage) or ``None`` for a fresh
        launch not yet stamped with its origin.

        Args:
            run_id: The unique identifier of the launched run.
            spec: The launch description to persist.
        """
        await self._store_conn.execute(
            "INSERT INTO run_specs (run_id, kind, name_or_source, args, label, journal_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "kind = excluded.kind, name_or_source = excluded.name_or_source, "
            "args = excluded.args, label = excluded.label, "
            "journal_run_id = excluded.journal_run_id",
            (
                run_id,
                spec.kind,
                spec.name_or_source,
                json.dumps(spec.args),
                spec.label,
                spec.journal_run_id,
            ),
        )

    async def delete_spec(self, run_id: str) -> None:
        """Delete the ``run_specs`` row for ``run_id`` if present.

        Used to roll back a spec persisted before a run was admitted, so a refused
        admission leaves no unresumable orphan. Deleting an absent row affects no
        rows and raises nothing.

        Args:
            run_id: The identifier of the run whose spec should be removed.
        """
        await self._store_conn.execute(
            "DELETE FROM run_specs WHERE run_id = ?",
            (run_id,),
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
            "SELECT kind, name_or_source, args, label, journal_run_id "
            "FROM run_specs WHERE run_id = ?",
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
            journal_run_id=row[4],
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

    async def __aenter__(self) -> SqliteWorkflowStore:
        """Return the already-opened store for ``async with`` use.

        The store is opened by the :meth:`open` async factory; entering the
        context manager performs no further I/O and simply yields ``self`` so the
        host can write ``async with await SqliteWorkflowStore.open(db) as store``.

        Returns:
            This store instance.
        """
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close both connections on exit from an ``async with`` block.

        Always runs :meth:`aclose`, whether the block exited normally or via an
        exception, so a host using the context-manager form cannot forget the
        final teardown.

        Args:
            *exc: The exception type, value, and traceback (all ``None`` on a
                clean exit); unused because teardown is unconditional.
        """
        await self.aclose()

    async def aclose(self) -> None:
        """Close both connections, releasing their WAL sidecar files.

        Call this at host shutdown. Closing the store and checkpointer
        connections releases the ``-wal`` / ``-shm`` sidecars; leaving them open
        can also hang the program on exit. The two closes are independent: if
        closing the store connection raises, the checkpointer connection is still
        closed before the error propagates, so one failure cannot strand the
        other's worker thread.
        """
        try:
            await self._store_conn.close()
        finally:
            await self._checkpointer_conn.close()

"""Per-leaf sandbox isolation, identity derivation, and lifecycle management.

A leaf ``agent()`` that needs execution (shell / code) runs against an isolated
backend rather than sharing one mutable workspace with its siblings. The default
isolation granularity is per-leaf, and a leaf's sandbox identity is derived from
its content-hash journal key — the same key that drives result memoization. That
single source of identity gives four properties at once: retry stability (a leaf
that re-runs maps to the same sandbox), resume stability (a replayed run resolves
the same identity), uniqueness (distinct leaf calls never collide), and self
consistency with journal dedup (identity and cache key cannot drift apart).

Pure-reasoning leaves (``needs_execution=False``) are not allocated a sandbox at
all; they use an ephemeral state-backed store. The number of *active sandboxes*
therefore tracks the number of execution leaves, never the number of logical
agents — a workflow with many reasoning leaves allocates zero of them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileData,
    FileInfo,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend

SANDBOX_ID_PREFIX = "leaf"
"""Prefix applied to every derived sandbox identity for readability in logs."""


def leaf_id_from_key(journal_key: str) -> str:
    """Derive a stable per-leaf sandbox identity from a content-hash journal key.

    The journal key is already a content hash of every input that affects the
    leaf's result (prompt, agent type, effective model, schema, isolation mode),
    so reusing it as the identity source makes the sandbox identity stable across
    retry and resume and unique per distinct leaf call — and keeps it from ever
    drifting apart from the journal's dedup key.

    Args:
        journal_key: The leaf's content-hash journal key.

    Returns:
        A readable, deterministic leaf identity string of the form
        ``"leaf-<journal_key>"``.
    """
    return f"{SANDBOX_ID_PREFIX}-{journal_key}"


class InMemorySandbox(SandboxBackendProtocol):
    """A self-contained, in-process isolated execution backend for one leaf.

    Each instance owns its own file store, so two instances handed to two leaves
    are mutually invisible — writing the same path in one is never observable in
    the other. The backend conforms to
    [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol]
    so it can stand in for a real container-backed sandbox in tests and offline
    runs without any sandbox infrastructure: ``execute`` is a no-op shell that
    reports success, and the file operations operate on the per-instance dict.

    Args:
        identity: The leaf identity that owns this sandbox (the derived
            ``leaf_id``). Surfaced via :attr:`id` so callers can correlate a
            backend with the leaf that holds it.
    """

    def __init__(self, *, identity: str) -> None:
        self._identity = identity
        # Per-instance file store: this dict is the entire reason two sandboxes
        # are isolated — there is no shared state between instances.
        self._files: dict[str, FileData] = {}

    @property
    def id(self) -> str:
        """The owning leaf identity (unique per sandbox instance)."""
        return self._identity

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command in the sandbox (offline no-op echo).

        The offline backend does not spawn a real shell; it echoes the command
        and reports success so an execution leaf has a working ``execute`` tool
        without any sandbox infrastructure.

        Args:
            command: The shell command string.
            timeout: Accepted for protocol compatibility; ignored offline.

        Returns:
            An :class:`ExecuteResponse` echoing ``command`` with exit code ``0``.
        """
        return ExecuteResponse(output=command, exit_code=0, truncated=False)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create ``file_path`` in this sandbox's isolated store.

        Args:
            file_path: Absolute path to create.
            content: File content.

        Returns:
            A :class:`WriteResult` carrying the written path, or an error when the
            file already exists.
        """
        if file_path in self._files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
        self._files[file_path] = FileData(content=content, encoding="utf-8")
        return WriteResult(path=file_path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read ``file_path`` from this sandbox's isolated store.

        Args:
            file_path: Absolute path to read.
            offset: Accepted for protocol compatibility; the full content is
                returned regardless of ``offset``/``limit`` for this backend.
            limit: Accepted for protocol compatibility; see ``offset``.

        Returns:
            A :class:`ReadResult` with the file data, or an error on miss.
        """
        file_data = self._files.get(file_path)
        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")
        return ReadResult(file_data=file_data)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace ``old_string`` with ``new_string`` in ``file_path``.

        Args:
            file_path: Absolute path to edit.
            old_string: Exact substring to replace.
            new_string: Replacement text.
            replace_all: Replace every occurrence when ``True``; otherwise the
                first occurrence only.

        Returns:
            An :class:`EditResult` with the edited path and replacement count, or
            an error on miss.
        """
        file_data = self._files.get(file_path)
        if file_data is None:
            return EditResult(error=f"File '{file_path}' not found")
        content = file_data["content"]
        count = content.count(old_string) if replace_all else (1 if old_string in content else 0)
        updated = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        self._files[file_path] = FileData(content=updated, encoding="utf-8")
        return EditResult(path=file_path, occurrences=count)

    def ls(self, path: str) -> LsResult:
        """List the files held in this sandbox's isolated store.

        Args:
            path: Directory path; only entries under it are returned.

        Returns:
            An :class:`LsResult` with one entry per stored file under ``path``.
        """
        prefix = path if path.endswith("/") else f"{path}/"
        entries: list[FileInfo] = [
            FileInfo(path=stored, is_dir=False, size=len(data["content"]), modified_at="")
            for stored, data in self._files.items()
            if stored == path or stored.startswith(prefix)
        ]
        entries.sort(key=lambda entry: entry["path"])
        return LsResult(entries=entries)


@dataclass(slots=True)
class _SandboxSlot:
    """Bookkeeping for one live sandbox: the backend plus its lifecycle clocks.

    Attributes:
        sandbox: The isolated execution backend instance.
        created_at: Monotonic timestamp when the sandbox was first created;
            drives the hard TTL (total-lifetime cap).
        last_used_at: Monotonic timestamp of the most recent lease release;
            drives the idle TTL (reclaim-after-inactivity).
        in_use: How many concurrent leases currently hold this sandbox; a slot is
            reclaimable only when idle (``in_use == 0``).
    """

    sandbox: InMemorySandbox
    created_at: float
    last_used_at: float
    in_use: int = field(default=0)


class SandboxManager:
    """Owns the lifecycle of per-leaf isolated sandboxes.

    The manager is the single place that decides whether a leaf is allocated an
    isolated execution sandbox and, if so, find-or-creates one keyed by the
    leaf's derived identity. Pure-reasoning leaves are intentionally *not*
    allocated: they are handed an ephemeral :class:`StateBackend` and never
    counted as active sandboxes, so the active-sandbox count tracks execution
    leaves rather than logical agents.

    Acquisition is find-or-create per ``leaf_id``: a leaf that retries within a
    run resolves the same backend instance, which keeps a leaf's workspace stable
    across retries. The manager self-manages the rest of the lifecycle:

    - **Idle / hard TTL**: a sandbox idle past ``idle_ttl`` (or alive past
      ``hard_ttl`` regardless of recent use) is reclaimed on the next
      acquisition, releasing its slot. The hard TTL caps total lifetime so a
      long-lived-but-busy sandbox cannot live forever.
    - **Max-active quota**: at most ``max_active`` sandboxes are live at once.
    - **Backpressure**: when the pool is at the quota and every slot is in use,
      a new :meth:`lease` blocks until a slot frees rather than over-allocating.

    Args:
        max_active: Maximum number of simultaneously live sandboxes, or ``None``
            for an unbounded pool (no backpressure).
        idle_ttl: Seconds a sandbox may sit idle before it is reclaimed, or
            ``None`` to disable idle reclamation.
        hard_ttl: Maximum total seconds a sandbox may live regardless of recent
            use, or ``None`` to disable the hard cap.
        clock: Monotonic time source (seconds); injectable for deterministic
            TTL testing. Defaults to :func:`time.monotonic`.
    """

    def __init__(
        self,
        *,
        max_active: int | None = None,
        idle_ttl: float | None = None,
        hard_ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Live execution sandboxes keyed by leaf identity. Reasoning leaves never
        # enter this map — that is what keeps active_count tied to execution work.
        self._slots: dict[str, _SandboxSlot] = {}
        self._max_active = max_active
        self._idle_ttl = idle_ttl
        self._hard_ttl = hard_ttl
        self._clock = clock
        # Signalled whenever a slot frees, so a lease parked under backpressure
        # can wake and re-check for room rather than busy-spinning.
        self._slot_freed = asyncio.Condition()

    @property
    def active_count(self) -> int:
        """How many isolated execution sandboxes are currently live."""
        return len(self._slots)

    def reclaim_idle(self) -> int:
        """Reclaim every idle sandbox whose idle or hard TTL has elapsed.

        Only idle sandboxes (no in-flight lease) are eligible: a sandbox in use
        is never torn down mid-leaf. A sandbox is reclaimed when it has sat idle
        longer than ``idle_ttl`` or has been alive longer than ``hard_ttl``.

        Returns:
            The number of sandboxes reclaimed.
        """
        now = self._clock()
        expired = [
            leaf_id
            for leaf_id, slot in self._slots.items()
            if slot.in_use == 0 and self._is_expired(slot, now)
        ]
        for leaf_id in expired:
            del self._slots[leaf_id]
        return len(expired)

    def _is_expired(self, slot: _SandboxSlot, now: float) -> bool:
        """Whether ``slot`` has exceeded either its idle or hard TTL."""
        if self._idle_ttl is not None and now - slot.last_used_at >= self._idle_ttl:
            return True
        return self._hard_ttl is not None and now - slot.created_at >= self._hard_ttl

    def acquire(self, *, leaf_id: str, needs_execution: bool) -> BackendProtocol:
        """Return the backend a leaf should run against (tiered admission).

        This is the synchronous find-or-create primitive. It neither waits for a
        slot nor reclaims TTL-expired sandboxes — :meth:`lease` wraps it with the
        full lifecycle (reclamation + backpressure). Use :meth:`lease` from the
        engine; ``acquire`` is exposed for direct, single-leaf use.

        Args:
            leaf_id: The leaf's derived identity (see :func:`leaf_id_from_key`).
            needs_execution: Whether the leaf requires an isolated execution
                sandbox. When ``False`` the leaf is pure reasoning and is handed
                a fresh :class:`StateBackend` without being allocated a sandbox.

        Returns:
            An isolated :class:`InMemorySandbox` for execution leaves (the same
            instance on repeat acquisition of one ``leaf_id``), or a
            :class:`StateBackend` for reasoning leaves.
        """
        if not needs_execution:
            # Tiered admission: reasoning leaves are never allocated a sandbox.
            return StateBackend()
        existing = self._slots.get(leaf_id)
        if existing is not None:
            return existing.sandbox
        now = self._clock()
        sandbox = InMemorySandbox(identity=leaf_id)
        self._slots[leaf_id] = _SandboxSlot(sandbox=sandbox, created_at=now, last_used_at=now)
        return sandbox

    @asynccontextmanager
    async def lease(
        self, *, leaf_id: str, needs_execution: bool
    ) -> AsyncGenerator[BackendProtocol]:
        """Lease a backend for the duration of a leaf invocation.

        For execution leaves this is the full lifecycle path: it reclaims
        TTL-expired idle sandboxes, blocks under backpressure when the pool is at
        its max-active quota with every slot in use, then find-or-creates the
        leaf's sandbox and marks it busy for the body. On exit the sandbox is
        marked idle (its idle clock reset) and kept for find-or-create reuse, and
        a waiter parked under backpressure is woken.

        Reasoning leaves bypass the pool entirely: they yield a fresh
        :class:`StateBackend` without consuming a slot or applying backpressure.

        Args:
            leaf_id: The leaf's derived identity.
            needs_execution: Whether the leaf requires an isolated sandbox.

        Yields:
            The backend the leaf should run against.
        """
        if not needs_execution:
            yield StateBackend()
            return
        async with self._slot_freed:
            # Wait for room: at quota, first reclaim TTL-expired idle sandboxes,
            # then evict the least-recently-used *idle* sandbox to admit new work,
            # and only block when every slot is still in use. A leaf reusing an
            # already-live sandbox (same leaf_id) never waits — it is not new work.
            while self._would_exceed_quota(leaf_id):
                self.reclaim_idle()
                if not self._would_exceed_quota(leaf_id):
                    break
                if self._evict_one_idle():
                    break
                # Every slot is in use: park until a lease releases one.
                await self._slot_freed.wait()
            slot = self._slots.get(leaf_id)
            if slot is None:
                now = self._clock()
                sandbox = InMemorySandbox(identity=leaf_id)
                slot = _SandboxSlot(sandbox=sandbox, created_at=now, last_used_at=now)
                self._slots[leaf_id] = slot
            slot.in_use += 1
        try:
            yield slot.sandbox
        finally:
            async with self._slot_freed:
                slot.in_use -= 1
                slot.last_used_at = self._clock()
                # Wake one parked lease so it can re-check for room.
                self._slot_freed.notify()

    def _would_exceed_quota(self, leaf_id: str) -> bool:
        """Whether admitting a *new* sandbox for ``leaf_id`` would breach the cap.

        Reusing an existing sandbox (``leaf_id`` already live) is always allowed —
        it adds no new sandbox to the pool.
        """
        if self._max_active is None or leaf_id in self._slots:
            return False
        return self.active_count >= self._max_active

    def _evict_one_idle(self) -> bool:
        """Evict the least-recently-used idle sandbox to make room at quota.

        Only idle sandboxes (no in-flight lease) are eligible; an in-use sandbox
        is never torn down mid-leaf. This is what turns the max-active cap into a
        bounded pool: when full but some sandboxes are merely idle, a new leaf
        evicts the stalest one rather than waiting forever.

        Returns:
            ``True`` if an idle sandbox was evicted, ``False`` when every live
            sandbox is currently in use (the caller must then block).
        """
        idle = [
            (slot.last_used_at, leaf_id)
            for leaf_id, slot in self._slots.items()
            if slot.in_use == 0
        ]
        if not idle:
            return False
        idle.sort()
        _, lru_leaf_id = idle[0]
        del self._slots[lru_leaf_id]
        return True

    async def stop(self, leaf_id: str) -> None:
        """Tear down and release the sandbox held by ``leaf_id`` (idempotent).

        Releasing a slot wakes one lease parked under backpressure.

        Args:
            leaf_id: The leaf identity whose sandbox should be released. Stopping
                an unknown or already-released identity is a no-op so cleanup can
                run unconditionally.
        """
        async with self._slot_freed:
            if self._slots.pop(leaf_id, None) is not None:
                self._slot_freed.notify()

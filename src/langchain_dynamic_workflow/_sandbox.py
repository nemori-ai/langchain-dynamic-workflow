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
import fnmatch
import threading
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend

from ._git_worktree import GitWorktreeProvider
from ._local_subprocess import ExecPolicy, LocalSubprocessSandbox
from ._worktree import WorktreeProvider

SANDBOX_ID_PREFIX = "leaf"
"""Prefix applied to every derived sandbox identity for readability in logs."""

SHARED_ROUTE_PREFIX = "/shared/"
"""Route prefix that hands a leaf's files off to the shared artifact store."""

SandboxFactory = Callable[[str], SandboxBackendProtocol]
"""Builds a fresh per-leaf isolated backend from a leaf identity.

The single seam through which the manager constructs a leaf's sandbox. The
default produces an offline :class:`InMemorySandbox`; a host opts into real
execution by passing a factory (for example :func:`local_subprocess_factory`)
to :class:`SandboxManager`.
"""


def local_subprocess_factory(policy: ExecPolicy | None = None) -> SandboxFactory:
    """Build a factory producing real local-subprocess backends sharing one gate.

    Every backend the returned factory creates shares a single
    :class:`threading.BoundedSemaphore`, so the policy's concurrent-execution cap
    is global across all of one run's execution leaves rather than per leaf. The
    semaphore is created once, when this function is called, and captured by the
    returned closure; constructing one factory per :class:`SandboxManager`
    therefore scopes the cap to that manager's run.

    DANGEROUS OPT-IN — the produced backend runs real shell commands on the host
    with the calling user's permissions and is not a security sandbox. See
    :class:`LocalSubprocessSandbox` and the project README before enabling it.

    Args:
        policy: The resilience and admission policy applied to every produced
            backend; ``None`` uses the default :class:`ExecPolicy`.

    Returns:
        A :data:`SandboxFactory` that maps a leaf identity to a fresh
        :class:`LocalSubprocessSandbox` bound to the shared exec gate.
    """
    effective_policy = policy or ExecPolicy()
    exec_gate = threading.BoundedSemaphore(effective_policy.max_concurrent_execs)

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        return LocalSubprocessSandbox(
            identity=leaf_id, policy=effective_policy, exec_gate=exec_gate
        )

    return factory


def normalize_path(path: str) -> str:
    """Canonicalize an absolute path, rejecting any ``..`` escape above root.

    Collapses ``.`` and empty segments and resolves ``..`` segments. A ``..``
    that would climb above the root is a hard error rather than a silent clamp:
    that is the guard that stops a path like ``/shared/../secret`` from escaping
    its route and reaching another backend's namespace.

    Args:
        path: An absolute path (it is treated as rooted at ``/`` regardless of a
            leading slash).

    Returns:
        The canonical absolute path, e.g. ``/shared/a/b``; the bare root is
        returned as ``/``.

    Raises:
        ValueError: If a ``..`` segment would escape above the root.
    """
    parts: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not parts:
                raise ValueError(f"path {path!r} escapes root via '..' traversal")
            parts.pop()
            continue
        parts.append(segment)
    return "/" + "/".join(parts)


def normalize_within_route(path: str, *, route_prefix: str) -> str:
    """Canonicalize ``path`` and reject a ``..`` escape out of ``route_prefix``.

    Generalizes :func:`normalize_path`: in addition to forbidding an escape above
    root, a path that lexically targets ``route_prefix`` (e.g. ``/shared/...``)
    must still resolve under that prefix. This blocks ``/shared/../secret`` from
    climbing out of the shared route into another backend's namespace before the
    composite ever routes it — the independent traversal guard the #2884
    route-isolation leak requires.

    Args:
        path: The requested absolute path.
        route_prefix: The route the path is checked against (e.g. ``/shared/``).

    Returns:
        The canonical absolute path.

    Raises:
        ValueError: If a ``..`` segment escapes above root, or if a path that
            targets ``route_prefix`` resolves outside it.
    """
    canonical = normalize_path(path)
    normalized_route = route_prefix.strip("/")
    bare_prefix = "/" + normalized_route
    # A path "targets" the route only when it is the bare route exactly or sits
    # under it ("shared" / "shared/..."). Matching the bare case by EXACT equality
    # — not endswith — is what keeps a legitimate isolated path whose final segment
    # merely happens to be the route name (e.g. /a/shared) from being misread as an
    # escape: such a path does not target the route and is routed privately.
    stripped = path.strip("/")
    targets_route = stripped == normalized_route or stripped.startswith(normalized_route + "/")
    if targets_route and canonical != bare_prefix and not canonical.startswith(bare_prefix + "/"):
        raise ValueError(
            f"path {path!r} escapes root via '..' traversal out of route {route_prefix!r}"
        )
    return canonical


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

    def close(self) -> None:
        """Release this sandbox's resources (a no-op for the in-memory backend).

        The in-memory backend holds only an in-process dict, so there is nothing
        to release. The method exists so the manager can call ``close`` uniformly
        on teardown and eviction regardless of the concrete backend type, without
        a per-call ``getattr`` probe. It is idempotent.
        """

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

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Search this sandbox's stored files for a literal substring.

        Matching is literal (not regex), mirroring the protocol contract. ``path``
        restricts the search to files at or under that directory; ``glob`` filters
        which files are searched by filename pattern. Matches are returned in
        deterministic (path, line) order.

        Args:
            pattern: Literal substring to search for in each line.
            path: Optional directory to restrict the search to; ``None`` searches
                every stored file.
            glob: Optional filename glob filtering which files are searched.

        Returns:
            A :class:`GrepResult` listing one match per matching line.
        """
        prefix = None if path is None else (path if path.endswith("/") else f"{path}/")
        matches: list[GrepMatch] = []
        for stored, data in sorted(self._files.items()):
            if prefix is not None and stored != path and not stored.startswith(prefix):
                continue
            if glob is not None and not fnmatch.fnmatch(stored, glob):
                continue
            for line_number, line in enumerate(data["content"].splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=stored, line=line_number, text=line))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find stored files matching ``pattern`` under ``path``.

        Args:
            pattern: Glob pattern matched against each stored file's full path.
            path: Base directory the search is rooted at; only files at or under
                it are considered.

        Returns:
            A :class:`GlobResult` of matching files in deterministic path order.
        """
        prefix = path if path.endswith("/") else f"{path}/"
        matches: list[FileInfo] = [
            FileInfo(path=stored, is_dir=False, size=len(data["content"]), modified_at="")
            for stored, data in self._files.items()
            if (stored == path or stored.startswith(prefix)) and fnmatch.fnmatch(stored, pattern)
        ]
        matches.sort(key=lambda entry: entry["path"])
        return GlobResult(matches=matches)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Store each ``(path, content)`` pair as a UTF-8 file (overwriting).

        Upload deliberately overwrites (unlike :meth:`write`, which errors on an
        existing path) so a batch upload is idempotent. Binary content that is not
        valid UTF-8 is reported as that file's ``invalid_path`` error rather than
        aborting the batch.

        Args:
            files: ``(destination_path, content_bytes)`` pairs to store.

        Returns:
            One :class:`FileUploadResponse` per input, in input order.
        """
        responses: list[FileUploadResponse] = []
        for file_path, content in files:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                responses.append(FileUploadResponse(path=file_path, error=INVALID_PATH))
                continue
            self._files[file_path] = FileData(content=text, encoding="utf-8")
            responses.append(FileUploadResponse(path=file_path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Return the bytes of each requested path (partial success per entry).

        Args:
            paths: File paths to download.

        Returns:
            One :class:`FileDownloadResponse` per input path, in input order; a
            missing path lands as that entry's ``file_not_found`` error.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            data = self._files.get(path)
            if data is None:
                responses.append(FileDownloadResponse(path=path, error=FILE_NOT_FOUND))
                continue
            content = data["content"].encode("utf-8")
            responses.append(FileDownloadResponse(path=path, content=content))
        return responses


@runtime_checkable
class _Closeable(Protocol):
    """A backend that releases host-side resources on teardown.

    ``SandboxBackendProtocol`` itself declares no ``close``: an offline,
    in-memory backend has nothing to release. A real backend (for example a
    local-subprocess backend with a private temp directory and a possible
    straggler process) does. This narrow protocol lets the manager release such a
    backend on teardown and eviction without assuming the concrete type.
    """

    def close(self) -> None:
        """Release the backend's resources (must be idempotent)."""
        ...


def _close_backend(backend: SandboxBackendProtocol) -> None:
    """Release a backend's host-side resources if it supports closing.

    A backend that exposes ``close`` (such as the real local-subprocess backend,
    or the in-memory backend's no-op) has it called so a teardown or eviction
    removes its temp directory and terminates any straggler process. A backend
    without ``close`` is left untouched.

    Args:
        backend: The backend a freed slot held.
    """
    if isinstance(backend, _Closeable):
        backend.close()


@dataclass(slots=True)
class _SandboxSlot:
    """Bookkeeping for one live sandbox: the backend plus its lifecycle clocks.

    Attributes:
        sandbox: The isolated execution backend instance. Typed at the protocol
            level so a pluggable factory may produce any full-protocol backend
            (for example a real local-subprocess backend) without the slot
            assuming the in-memory concrete type.
        created_at: Monotonic timestamp when the sandbox was first created;
            drives the hard TTL (total-lifetime cap).
        last_used_at: Monotonic timestamp of the most recent lease release;
            drives the idle TTL (reclaim-after-inactivity).
        in_use: How many concurrent leases currently hold this sandbox; a slot is
            reclaimable only when idle (``in_use == 0``).
    """

    sandbox: SandboxBackendProtocol
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
    run resolves the same backend instance, so an *un-reclaimed* sandbox keeps its
    workspace stable across retries. Identity is unconditionally stable (same
    ``leaf_id`` always maps to the same identity string); workspace persistence is
    the weaker guarantee — it holds until the sandbox is reclaimed by TTL or
    evicted under quota pressure, after which a re-running leaf find-or-creates a
    fresh, empty backend under that same identity. The manager self-manages the
    rest of the lifecycle:

    - **Idle / hard TTL**: a sandbox idle past ``idle_ttl`` (or alive past
      ``hard_ttl`` regardless of recent use) is reclaimed on the next
      acquisition, releasing its slot. The hard TTL caps total lifetime so a
      long-lived-but-busy sandbox cannot live forever.
    - **Max-active quota**: at most ``max_active`` sandboxes are live at once.
    - **Backpressure**: when the pool is at the quota a new :meth:`lease` first
      reclaims TTL-expired idle sandboxes, then evicts the least-recently-used
      *idle* sandbox to admit the new leaf, and only blocks when every slot is
      genuinely in use — never over-allocating past the quota. Eviction is what
      keeps a bounded pool from deadlocking when idle-but-alive sandboxes (kept
      for find-or-create reuse) occupy every slot and no TTL reclaims them.

    Args:
        max_active: Maximum number of simultaneously live sandboxes, or ``None``
            for an unbounded pool (no backpressure).
        idle_ttl: Seconds a sandbox may sit idle before it is reclaimed, or
            ``None`` to disable idle reclamation.
        hard_ttl: Maximum total seconds a sandbox may live regardless of recent
            use, or ``None`` to disable the hard cap.
        clock: Monotonic time source (seconds); injectable for deterministic
            TTL testing. Defaults to :func:`time.monotonic`.
        sandbox_factory: Builds each leaf's isolated backend from its identity.
            ``None`` (the default) keeps the offline, zero-dependency behavior of
            seeding a fresh :class:`InMemorySandbox`; a host opts into real
            execution by passing a factory (for example the product of
            :func:`local_subprocess_factory`).
        git_worktree_provider: When supplied, an ``isolation="worktree"`` leaf is
            leased a backend rooted in a real ``git worktree`` (a real branch per
            leaf) via :meth:`GitWorktreeProvider.open_worktree`, taking precedence
            over the in-memory ``worktree_provider``. The provider's ``on_close``
            hook teardown rides every existing ``_close_backend`` path (no extra
            manager hook). The blocking ``git worktree add`` is thread-offloaded
            outside the slot lock so it never wedges the event loop. ``None`` (the
            default) keeps the in-memory worktree behavior.
    """

    def __init__(
        self,
        *,
        max_active: int | None = None,
        idle_ttl: float | None = None,
        hard_ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        worktree_provider: WorktreeProvider | None = None,
        git_worktree_provider: GitWorktreeProvider | None = None,
        sandbox_factory: SandboxFactory | None = None,
    ) -> None:
        # Live execution sandboxes keyed by leaf identity. Reasoning leaves never
        # enter this map — that is what keeps active_count tied to execution work.
        self._slots: dict[str, _SandboxSlot] = {}
        self._max_active = max_active
        self._idle_ttl = idle_ttl
        self._hard_ttl = hard_ttl
        self._clock = clock
        # Seeds + collects changesets for isolation="worktree" leaves; None keeps
        # worktree leaves as plain empty sandboxes (no seeding source).
        self._worktree_provider = worktree_provider
        # Real-git worktree provider: when set, a worktree leaf is rooted in a real
        # `git worktree add -b leaf/<id>` tree. Takes precedence over the in-memory
        # worktree_provider for worktree leaves.
        self._git_worktree_provider = git_worktree_provider
        # The seam that constructs a leaf's backend. The default reproduces the
        # prior behavior byte-for-byte (a fresh offline InMemorySandbox), so the
        # zero-dependency offline path is unchanged unless a host injects one.
        self._factory: SandboxFactory = sandbox_factory or (
            lambda leaf_id: InMemorySandbox(identity=leaf_id)
        )
        # Signalled whenever a slot frees, so a lease parked under backpressure
        # can wake and re-check for room rather than busy-spinning.
        self._slot_freed = asyncio.Condition()
        # Leaf ids whose backend is being constructed OUTSIDE the slot lock (R8:
        # the blocking git worktree add is thread-offloaded). A second lease of the
        # same leaf id parks on the condition until the in-flight build installs its
        # slot, so concurrent same-leaf creation never builds two worktrees; the
        # rare loser of a race closes its own freshly-built backend.
        self._pending: set[str] = set()

    def _new_sandbox(self, leaf_id: str, isolation: str) -> SandboxBackendProtocol:
        """Create a leaf's sandbox, rooting or seeding a worktree leaf when asked.

        For ``isolation="worktree"`` with a configured ``git_worktree_provider``
        the backend is rooted in a real ``git worktree`` (a real branch per leaf)
        returned already populated by :meth:`GitWorktreeProvider.open_worktree`;
        its ``on_close`` hook tears the worktree down on close. Otherwise the
        configured factory builds the backend (the default produces a fresh offline
        :class:`InMemorySandbox`), and for a worktree leaf with an in-memory
        ``worktree_provider`` the backend is populated with an isolated copy of the
        base snapshot via ``upload_files``; a non-worktree leaf starts empty (the
        prior per-leaf behavior).

        This method may block (a git provider runs ``git worktree add``), so the
        ``lease`` path always calls it via ``asyncio.to_thread`` OUTSIDE the slot
        condition lock (R8) and never on the event loop directly.

        Args:
            leaf_id: The leaf's derived identity.
            isolation: ``"worktree"`` to root/seed a worktree, else ``"shared"``.

        Returns:
            The new, possibly-seeded backend.
        """
        if isolation == "worktree" and self._git_worktree_provider is not None:
            # A real git worktree: open_worktree returns a backend already rooted in
            # the leaf's worktree directory with an on_close -> teardown hook, so the
            # existing _close_backend paths reclaim it with no extra manager hook.
            return self._git_worktree_provider.open_worktree(leaf_id)
        sandbox = self._factory(leaf_id)
        if isolation == "worktree" and self._worktree_provider is not None:
            seed = self._worktree_provider.seed(leaf_id)
            if seed:
                sandbox.upload_files(
                    [(path, content.encode("utf-8")) for path, content in seed.items()]
                )
        return sandbox

    @property
    def active_count(self) -> int:
        """How many isolated execution sandboxes are currently live."""
        return len(self._slots)

    @property
    def git_worktree_provider(self) -> GitWorktreeProvider | None:
        """The real-git worktree provider, or ``None`` when not configured.

        Exposed so the engine can collect a worktree leaf's authoritative real
        ``git diff`` while the lease is still held (before the worktree is torn
        down on ``close``), without the engine reaching into a private field.
        """
        return self._git_worktree_provider

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
            # Release the backend's host-side resources (temp dir, straggler
            # process) before dropping the slot, so a reclaimed real backend
            # leaves nothing behind.
            _close_backend(self._slots[leaf_id].sandbox)
            del self._slots[leaf_id]
        return len(expired)

    def _is_expired(self, slot: _SandboxSlot, now: float) -> bool:
        """Whether ``slot`` has exceeded either its idle or hard TTL."""
        if self._idle_ttl is not None and now - slot.last_used_at >= self._idle_ttl:
            return True
        return self._hard_ttl is not None and now - slot.created_at >= self._hard_ttl

    def acquire(
        self, *, leaf_id: str, needs_execution: bool, isolation: str = "shared"
    ) -> BackendProtocol:
        """Return the backend a leaf should run against (tiered admission).

        This is the synchronous find-or-create primitive that honors the same
        max-active quota and TTL reclamation the manager guarantees, but without
        blocking — that is the one capability reserved for the async :meth:`lease`.
        Admitting a *new* execution sandbox past the quota first reclaims
        TTL-expired idle sandboxes, then evicts the least-recently-used idle one;
        a same-``leaf_id`` reuse always succeeds (it adds no new sandbox). Because
        a sync primitive cannot park for a release, it raises when every live slot
        is genuinely in use and no idle sandbox can be reclaimed or evicted —
        :meth:`lease` is the path that waits instead. Use :meth:`lease` from the
        engine; ``acquire`` is exposed for direct, single-leaf use.

        Args:
            leaf_id: The leaf's derived identity (see :func:`leaf_id_from_key`).
            needs_execution: Whether the leaf requires an isolated execution
                sandbox. When ``False`` the leaf is pure reasoning and is handed
                a fresh :class:`StateBackend` without being allocated a sandbox.
            isolation: ``"worktree"`` seeds the new sandbox from the worktree base
                snapshot (when a provider is configured); ``"shared"`` (the default)
                leaves it empty, preserving the prior per-leaf behavior.

        Returns:
            An isolated :class:`InMemorySandbox` for execution leaves (the same
            instance on repeat acquisition of one ``leaf_id``), or a
            :class:`StateBackend` for reasoning leaves.

        Raises:
            RuntimeError: If admitting a new sandbox would breach ``max_active``
                and no idle sandbox can be reclaimed or evicted to make room (the
                synchronous path cannot wait for an in-use slot to free).
        """
        if not needs_execution:
            # Tiered admission: reasoning leaves are never allocated a sandbox.
            return StateBackend()
        existing = self._slots.get(leaf_id)
        if existing is not None:
            return existing.sandbox
        # Enforce the quota the same way lease() does, minus the blocking step:
        # reclaim TTL-expired idle sandboxes, then evict the LRU idle one. Only if
        # the pool is full of in-use sandboxes (none reclaimable/evictable) do we
        # fail loud rather than over-allocate past the cap — a sync caller cannot
        # park for a release, so unbounded growth here is the bug to prevent.
        if self._would_exceed_quota(leaf_id):
            self.reclaim_idle()
            if self._would_exceed_quota(leaf_id) and not self._evict_one_idle():
                raise RuntimeError(
                    f"sandbox pool exhausted: {self.active_count} active at max_active="
                    f"{self._max_active}, every slot in use and none reclaimable; "
                    "use lease() to wait for a slot to free"
                )
        now = self._clock()
        sandbox = self._new_sandbox(leaf_id, isolation)
        self._slots[leaf_id] = _SandboxSlot(sandbox=sandbox, created_at=now, last_used_at=now)
        return sandbox

    @asynccontextmanager
    async def lease(
        self, *, leaf_id: str, needs_execution: bool, isolation: str = "shared"
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
            isolation: ``"worktree"`` seeds the leased sandbox from the worktree
                base snapshot (when a provider is configured); ``"shared"`` (the
                default) leaves it empty.

        Yields:
            The backend the leaf should run against.
        """
        if not needs_execution:
            yield StateBackend()
            return
        slot = await self._admit_slot(leaf_id, isolation)
        try:
            yield slot.sandbox
        finally:
            async with self._slot_freed:
                slot.in_use -= 1
                slot.last_used_at = self._clock()
                # Wake one parked lease so it can re-check for room.
                self._slot_freed.notify()

    async def _admit_slot(self, leaf_id: str, isolation: str) -> _SandboxSlot:
        """Find-or-create the leaf's slot, building the backend OUTSIDE the lock.

        Honors the same tiered admission as the prior inline body (reclaim idle ->
        evict LRU idle -> park under backpressure), but with one change demanded by
        R8: constructing the backend can block (a git provider runs
        ``git worktree add``), so the construction is thread-offloaded via
        ``asyncio.to_thread`` *outside* the ``self._slot_freed`` condition lock and
        never on the event loop. A ``self._pending`` marker dedups concurrent
        same-leaf creation: a second lease of the same leaf id parks until the
        in-flight build installs its slot, so only one worktree is ever built per
        leaf id; the rare loser of a race closes its own freshly-built backend.

        The lock is held only for the fast decisions (quota wait, slot lookup,
        marking pending, installing the finished slot, bumping ``in_use``). The slow
        build runs unlocked, so a blocking git subprocess can never wedge the event
        loop or stall an unrelated lease.

        Args:
            leaf_id: The leaf's derived identity.
            isolation: ``"worktree"`` to root/seed a worktree, else ``"shared"``.

        Returns:
            The leaf's slot with ``in_use`` already incremented for the caller.
        """
        async with self._slot_freed:
            while True:
                # Wait for room at quota, in three escalating steps:
                #   1. reclaim TTL-expired idle sandboxes (frees slots cleanly);
                #   2. if still full, evict the least-recently-used *idle* sandbox to
                #      admit this new leaf (a slot held only for find-or-create reuse
                #      is yielded so distinct new work is not starved under a bounded
                #      pool with no TTL — without this the pool deadlocks once every
                #      slot holds an idle-but-alive sandbox);
                #   3. only when every slot is genuinely in use, park until a release.
                # A leaf reusing an already-live sandbox (same leaf_id) never waits
                # and is never evicted — it is not new work, so its workspace stays.
                while self._would_exceed_quota(leaf_id):
                    self.reclaim_idle()
                    if not self._would_exceed_quota(leaf_id):
                        break
                    if self._evict_one_idle():
                        break
                    await self._slot_freed.wait()
                existing = self._slots.get(leaf_id)
                if existing is not None:
                    # Find-or-create reuse: another lease (or a prior one) already
                    # built this leaf's backend; reuse it without a second build.
                    existing.in_use += 1
                    return existing
                if leaf_id in self._pending:
                    # Another lease is building THIS leaf's backend outside the lock;
                    # park until it installs the slot, then re-check from the top.
                    await self._slot_freed.wait()
                    continue
                # We are the builder: claim the leaf so a concurrent lease parks, and
                # so the quota counts this in-flight build (see _would_exceed_quota).
                self._pending.add(leaf_id)
                break
        # Build the backend OUTSIDE the lock (R8): a git provider's worktree add is
        # blocking, so thread-offload it and never run it on the event loop.
        try:
            sandbox = await asyncio.to_thread(self._new_sandbox, leaf_id, isolation)
        except BaseException:
            # The build failed: release the pending claim and wake a parked lease so
            # the failure does not strand the leaf id as permanently pending.
            async with self._slot_freed:
                self._pending.discard(leaf_id)
                self._slot_freed.notify_all()
            raise
        async with self._slot_freed:
            self._pending.discard(leaf_id)
            installed = self._slots.get(leaf_id)
            if installed is not None:
                # Lost a race (a concurrent path installed the slot while we built);
                # close our now-redundant backend and reuse the installed one so no
                # worktree leaks.
                _close_backend(sandbox)
                installed.in_use += 1
                self._slot_freed.notify_all()
                return installed
            now = self._clock()
            slot = _SandboxSlot(sandbox=sandbox, created_at=now, last_used_at=now)
            self._slots[leaf_id] = slot
            slot.in_use += 1
            # A build can free a parked same-leaf lease (now it finds the slot) and
            # change the pending count other leases wait on, so wake them to re-check.
            self._slot_freed.notify_all()
            return slot

    def _would_exceed_quota(self, leaf_id: str) -> bool:
        """Whether admitting a *new* sandbox for ``leaf_id`` would breach the cap.

        Reusing an existing sandbox (``leaf_id`` already live) is always allowed —
        it adds no new sandbox to the pool. A backend currently being built outside
        the lock (in ``_pending``) is counted as occupying a slot, so a concurrent
        admission cannot over-allocate past the cap while a thread-offloaded
        ``git worktree add`` is in flight (R8). ``_pending`` is only ever populated
        by the async :meth:`lease` path; the sync :meth:`acquire` path leaves it
        empty, so this term is a no-op there.
        """
        if self._max_active is None or leaf_id in self._slots:
            return False
        # Count installed slots plus in-flight (pending) builds for OTHER leaves.
        pending_others = len(self._pending - {leaf_id})
        return self.active_count + pending_others >= self._max_active

    def _evict_one_idle(self) -> bool:
        """Evict the least-recently-used idle sandbox to make room at quota.

        Only idle sandboxes (no in-flight lease) are eligible; an in-use sandbox
        is never torn down mid-leaf. This is what turns the max-active cap into a
        bounded pool that does not deadlock: a lease keeps its sandbox live for
        find-or-create reuse even after release, so without eviction a pool of
        idle-but-alive sandboxes would block every distinct new leaf forever when
        no TTL reclaims them. Eviction reclaims the stalest idle workspace to admit
        new work, never the caller's own (a same-``leaf_id`` reuse never reaches
        here because it does not exceed the quota).

        Eviction is the one place a leaf's workspace can be discarded while the run
        is still live: an evicted ``leaf_id`` that later re-runs derives the *same*
        identity (so it still maps to one logical sandbox) but find-or-creates a
        fresh, empty backend. Identity stability (same key -> same id) holds
        unconditionally; *workspace* persistence across a reuse holds only while
        the sandbox has not been evicted under quota pressure.

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
        # Eviction is a teardown path too: release the evicted backend's
        # host-side resources before dropping its slot.
        _close_backend(self._slots[lru_leaf_id].sandbox)
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
            removed = self._slots.pop(leaf_id, None)
            if removed is not None:
                # Release the backend's host-side resources (temp dir, straggler
                # process) on teardown so a stopped real backend leaves nothing.
                _close_backend(removed.sandbox)
                self._slot_freed.notify()


class SharedArtifactStore:
    """A process-shared store for explicit ``/shared/`` artifact hand-off.

    Writes are *namespaced by producer* so two leaves writing the same logical
    path never clobber each other, while reads merge across namespaces so a
    consumer can pick up a producer's artifact by its logical path. That split is
    what makes the hand-off both collision-free on write and discoverable on read.

    The store is the single shared object handed to every per-leaf composite
    backend; the per-leaf isolation lives in each leaf's *own* backend, never
    here — so a leaf's non-shared files can never reach this store.

    Every store operation runs under a :class:`threading.Lock`. The hand-off path
    is the one place a single in-memory object is touched concurrently from
    multiple OS threads: each leaf's file op reaches the store through the backend
    protocol's async defaults (``awrite``/``aread`` -> ``asyncio.to_thread``), so
    under ``ctx.parallel`` a producer's ``write_namespaced`` can run on one thread
    while a consumer's ``read_merged`` iterates the same dict on another. The lock
    makes the read snapshot atomic with respect to writes, closing the
    ``dictionary changed size during iteration`` race — the same cross-thread
    safety the :class:`JournalStore` protocol mandates for fan-out, here made
    explicit rather than implicit.
    """

    def __init__(self) -> None:
        # Keyed by (producer namespace, canonical path) so producers are isolated
        # on write; reads scan namespaces in sorted order for a deterministic
        # resolution when two producers wrote the same path.
        self._artifacts: dict[tuple[str, str], str] = {}
        # Guards every access to _artifacts: the dict is mutated and iterated from
        # different to_thread worker threads concurrently under parallel fan-out.
        self._lock = threading.Lock()

    def write_namespaced(self, producer: str, path: str, content: str) -> None:
        """Store ``content`` under ``producer``'s namespace at ``path``."""
        canonical = normalize_path(path)
        with self._lock:
            self._artifacts[(producer, canonical)] = content

    def read_namespaced(self, producer: str, path: str) -> str | None:
        """Read ``producer``'s artifact at ``path``, or ``None`` on miss."""
        canonical = normalize_path(path)
        with self._lock:
            return self._artifacts.get((producer, canonical))

    def read_merged(self, path: str) -> str | None:
        """Read ``path`` across all producer namespaces (deterministic order).

        Args:
            path: The logical shared path.

        Returns:
            The artifact content from the first matching namespace in sorted
            producer order, or ``None`` if no producer wrote that path.
        """
        canonical = normalize_path(path)
        with self._lock:
            # Snapshot under the lock so a concurrent write_namespaced on another
            # to_thread worker can never mutate the dict mid-iteration.
            items = sorted(self._artifacts.items())
        for (_producer, stored_path), content in items:
            if stored_path == canonical:
                return content
        return None

    def stored_paths_under(self, prefix_path: str) -> list[str]:
        """Return the distinct shared paths under ``prefix_path`` (merged view).

        Args:
            prefix_path: The directory path whose contents to list.

        Returns:
            Sorted distinct canonical paths at or under ``prefix_path``.
        """
        canonical = normalize_path(prefix_path)
        prefix = canonical if canonical.endswith("/") else f"{canonical}/"
        with self._lock:
            # Snapshot the keys under the lock for the same reason as read_merged.
            stored_paths = [stored_path for (_producer, stored_path) in self._artifacts]
        seen = {
            stored_path
            for stored_path in stored_paths
            if stored_path == canonical or stored_path.startswith(prefix)
        }
        return sorted(seen)


class _NamespacedSharedView(BackendProtocol):
    """A per-producer view over a :class:`SharedArtifactStore` (the ``/shared/`` route).

    Writes land in the owning producer's namespace; reads merge across all
    namespaces so a consumer leaf can pick up another leaf's artifact. This view
    is the backend the composite routes ``/shared/`` paths to; the composite has
    already stripped the ``/shared/`` prefix, so paths arrive rooted at ``/``.

    Args:
        store: The shared artifact store backing every producer's view.
        producer: The owning leaf's producer namespace for writes.
    """

    def __init__(self, *, store: SharedArtifactStore, producer: str) -> None:
        self._store = store
        self._producer = producer

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write ``content`` to the producer's shared namespace at ``file_path``."""
        self._store.write_namespaced(self._producer, file_path, content)
        return WriteResult(path=file_path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read ``file_path`` merged across shared namespaces."""
        content = self._store.read_merged(file_path)
        if content is None:
            return ReadResult(error=f"File '{file_path}' not found in shared store")
        return ReadResult(file_data=FileData(content=content, encoding="utf-8"))

    def ls(self, path: str) -> LsResult:
        """List shared artifacts under ``path`` (merged across namespaces)."""
        entries: list[FileInfo] = [
            FileInfo(path=stored_path, is_dir=False, size=0, modified_at="")
            for stored_path in self._store.stored_paths_under(path)
        ]
        return LsResult(entries=entries)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Search merged shared artifacts for a literal substring.

        Searches across producer namespaces (merged view), so a consumer leaf can
        grep an artifact a producer leaf wrote. ``path`` scopes the search to a
        directory; ``glob`` filters which shared files are searched by filename.

        Args:
            pattern: Literal substring to match in each line.
            path: Optional directory to scope the search to; ``None`` searches all
                shared artifacts.
            glob: Optional filename glob filtering which shared files are searched.

        Returns:
            A :class:`GrepResult` listing one match per matching line.
        """
        search_root = "/" if path is None else path
        matches: list[GrepMatch] = []
        for stored_path in self._store.stored_paths_under(search_root):
            if glob is not None and not fnmatch.fnmatch(stored_path, glob):
                continue
            content = self._store.read_merged(stored_path)
            if content is None:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=stored_path, line=line_number, text=line))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find merged shared artifacts whose path matches ``pattern`` under ``path``.

        Args:
            pattern: Glob pattern matched against each shared file's full path.
            path: Base directory the search is rooted at.

        Returns:
            A :class:`GlobResult` of matching shared files in path order.
        """
        matches: list[FileInfo] = [
            FileInfo(path=stored_path, is_dir=False, size=0, modified_at="")
            for stored_path in self._store.stored_paths_under(path)
            if fnmatch.fnmatch(stored_path, pattern)
        ]
        return GlobResult(matches=matches)


class _GuardedBackend(SandboxBackendProtocol):
    """Normalizes every path and blocks ``..`` traversal before delegating.

    The wrapper runs *before* the composite routes a path, so a traversal that
    tries to escape the ``/shared/`` route (e.g. ``/shared/../secret``) is
    canonicalized and rejected at the boundary rather than slipping into another
    backend's namespace — the independent guard the #2884 route-isolation leak
    demands. A normalization error is returned as an operation error (rather than
    raised) so a leaf's file tool surfaces it as a recoverable failure.

    Every file operation the composite can route is delegated, not only
    ``write``/``read``/``edit``/``ls``: ``grep``/``glob`` and
    ``upload_files``/``download_files`` are forwarded through the same traversal
    guard so a backend-aware leaf reaches whatever the wrapped composite
    implements rather than the protocol's bare ``NotImplementedError`` default.
    The async ``a*`` variants are inherited from
    [`BackendProtocol`][deepagents.backends.protocol.BackendProtocol], which
    dispatches them to these guarded sync methods via ``asyncio.to_thread``.

    Args:
        inner: The composite backend to delegate normalized paths to.
        isolated: The leaf's isolated sandbox; ``id`` and ``execute`` delegate to
            it so the guarded backend stays a usable execution sandbox.
        route_prefix: The shared route a ``..`` escape must not climb out of.
    """

    def __init__(
        self,
        *,
        inner: BackendProtocol,
        isolated: SandboxBackendProtocol,
        route_prefix: str,
    ) -> None:
        self._inner = inner
        self._isolated = isolated
        self._route_prefix = route_prefix

    @property
    def id(self) -> str:
        """The owning leaf's sandbox identity (delegated to the isolated backend)."""
        return self._isolated.id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command in the isolated sandbox (execution is not routable)."""
        return self._isolated.execute(command, timeout=timeout)

    def _safe(self, path: str) -> str:
        """Canonicalize ``path``, blocking escapes out of the shared route."""
        return normalize_within_route(path, route_prefix=self._route_prefix)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Normalize ``file_path`` then delegate; block traversal escapes."""
        try:
            safe = self._safe(file_path)
        except ValueError as exc:
            return WriteResult(error=str(exc))
        return self._inner.write(safe, content)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Normalize ``file_path`` then delegate; block traversal escapes."""
        try:
            safe = self._safe(file_path)
        except ValueError as exc:
            return ReadResult(error=str(exc))
        return self._inner.read(safe, offset=offset, limit=limit)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Normalize ``file_path`` then delegate; block traversal escapes."""
        try:
            safe = self._safe(file_path)
        except ValueError as exc:
            return EditResult(error=str(exc))
        return self._inner.edit(safe, old_string, new_string, replace_all=replace_all)

    def ls(self, path: str) -> LsResult:
        """Normalize ``path`` then delegate; block traversal escapes."""
        try:
            safe = self._safe(path)
        except ValueError as exc:
            return LsResult(error=str(exc))
        return self._inner.ls(safe)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Normalize ``path`` (when given) then delegate; block traversal escapes.

        A ``None`` ``path`` means search every routed backend, so there is no
        path to guard and the request is forwarded verbatim. The ``glob`` filter
        matches filenames rather than directories, so it is not a traversal
        vector and is forwarded unchanged.
        """
        if path is None:
            return self._inner.grep(pattern, path, glob)
        try:
            safe = self._safe(path)
        except ValueError as exc:
            return GrepResult(error=str(exc))
        return self._inner.grep(pattern, safe, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Normalize the base ``path`` then delegate; block traversal escapes.

        Only the base ``path`` is a directory the request is rooted at; the
        ``pattern`` filters filenames under it and is forwarded unchanged.
        """
        try:
            safe = self._safe(path)
        except ValueError as exc:
            return GlobResult(error=str(exc))
        return self._inner.glob(pattern, safe)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Normalize every destination path then delegate; block traversal escapes.

        A single malformed destination fails only its own entry (the protocol
        allows partial success in a batch), so a guard rejection is reported as
        that file's ``invalid_path`` error rather than aborting the whole upload.
        """
        guarded: list[tuple[str, bytes]] = []
        rejected: list[FileUploadResponse] = []
        for file_path, content in files:
            try:
                guarded.append((self._safe(file_path), content))
            except ValueError:
                rejected.append(FileUploadResponse(path=file_path, error=INVALID_PATH))
        delegated = self._inner.upload_files(guarded) if guarded else []
        return [*delegated, *rejected]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Normalize every source path then delegate; block traversal escapes.

        A single malformed path fails only its own entry (the protocol allows
        partial success in a batch), so a guard rejection is reported as that
        path's ``invalid_path`` error rather than aborting the whole download.
        """
        guarded: list[str] = []
        rejected: list[FileDownloadResponse] = []
        for path in paths:
            try:
                guarded.append(self._safe(path))
            except ValueError:
                rejected.append(FileDownloadResponse(path=path, error=INVALID_PATH))
        delegated = self._inner.download_files(guarded) if guarded else []
        return [*delegated, *rejected]


def build_leaf_backend(
    *,
    isolated: SandboxBackendProtocol,
    shared_store: SharedArtifactStore,
    producer: str,
) -> SandboxBackendProtocol:
    """Wrap a per-leaf isolated sandbox with a guarded ``/shared/`` hand-off route.

    The returned backend routes ``/shared/`` paths to a producer-namespaced view
    of ``shared_store`` (explicit artifact hand-off) and every other path to the
    leaf's own ``isolated`` sandbox (private per-leaf workspace). All paths pass
    through a traversal guard first, so a ``..`` escape from the shared route into
    another namespace is blocked at the boundary — the per-leaf isolation never
    relies on the composite's prefix routing alone. ``id`` and ``execute`` delegate
    to the isolated sandbox, so the wrapped backend remains a usable execution
    sandbox.

    Args:
        isolated: The leaf's private execution sandbox for non-shared paths.
        shared_store: The process-shared artifact store backing ``/shared/``.
        producer: The leaf's producer namespace for shared writes.

    Returns:
        A guarded composite sandbox ready to hand to the leaf.
    """
    shared_view = _NamespacedSharedView(store=shared_store, producer=producer)
    composite = CompositeBackend(default=isolated, routes={SHARED_ROUTE_PREFIX: shared_view})
    return _GuardedBackend(inner=composite, isolated=isolated, route_prefix=SHARED_ROUTE_PREFIX)

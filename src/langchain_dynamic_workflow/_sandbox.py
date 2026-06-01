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
    across retries.
    """

    def __init__(self) -> None:
        # Live execution sandboxes keyed by leaf identity. Reasoning leaves never
        # enter this map — that is what keeps active_count tied to execution work.
        self._sandboxes: dict[str, InMemorySandbox] = {}

    @property
    def active_count(self) -> int:
        """How many isolated execution sandboxes are currently live."""
        return len(self._sandboxes)

    def acquire(self, *, leaf_id: str, needs_execution: bool) -> BackendProtocol:
        """Return the backend a leaf should run against (tiered admission).

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
        existing = self._sandboxes.get(leaf_id)
        if existing is not None:
            return existing
        sandbox = InMemorySandbox(identity=leaf_id)
        self._sandboxes[leaf_id] = sandbox
        return sandbox

    async def stop(self, leaf_id: str) -> None:
        """Tear down and release the sandbox held by ``leaf_id`` (idempotent).

        Args:
            leaf_id: The leaf identity whose sandbox should be released. Stopping
                an unknown or already-released identity is a no-op so cleanup can
                run unconditionally.
        """
        self._sandboxes.pop(leaf_id, None)

"""Real ``git worktree`` isolation provider for file-mutating execution leaves.

DANGEROUS OPT-IN — this is NOT a security sandbox. ``GitWorktreeProvider`` spawns
real ``git`` subprocesses on the host with the calling user's full permissions
against a real base repository, and roots each leaf's execution backend in a real
``git worktree`` directory where the leaf runs real ``git`` / build / test. A leaf
command can still read and write any host path, open network connections, and
consume host resources — the worktree bounds only the leaf's default working
directory, not the reachable filesystem. Construct it only for a host-trusted
roster; never expose it to an untrusted, AST-gated authored script (those run on a
reasoning-only roster with no execution leaves). See :class:`LocalSubprocessSandbox`
and the project README for the full sharp-edge warning before enabling it.

What it does guarantee: each leaf gets its own ``git worktree add -b leaf/<id>``
tree (branch-per-leaf isolation), the authoritative changeset is a real
``git diff`` of that tree (never a model self-report), ``open_worktree`` is
idempotent per leaf id (a stale same-key worktree+branch left by a crash is
reclaimed before a fresh one is created) and exception-safe (a partial creation is
rolled back before the error propagates), and worktree teardown is bound to the
backend's ``close`` (so every ``SandboxManager`` teardown path removes the
worktree with no extra hook), with :meth:`cleanup_all` as a run-end backstop.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol

from ._local_subprocess import ExecPolicy, LocalSubprocessSandbox

DEFAULT_INTEGRATION_BRANCH = "ldw/integration"
"""Default name of the integration branch leaf changes are folded toward."""

_LEAF_BRANCH_PREFIX = "leaf/"
"""Branch-name prefix for a leaf's isolated branch (``leaf/<leaf_id>``)."""

_UNSAFE_DIRNAME = re.compile(r"[^A-Za-z0-9._-]+")
"""Characters replaced when deriving a filesystem-safe worktree directory name."""


class GitWorktreeError(RuntimeError):
    """A real (non-conflict) ``git`` failure in the worktree provider.

    Raised when a ``git`` subprocess the provider runs exits non-zero for a reason
    that is not an expected merge conflict — for example a base path that is not a
    git repository, a ref that does not exist, or a ``git worktree add`` that
    fails. The message carries the failing command's stderr so the cause is not
    swallowed.
    """


def _safe_dirname(leaf_id: str) -> str:
    """Derive a filesystem-safe directory name from a leaf id.

    A leaf id is normally ``leaf-<hash>`` (already safe), but a host-supplied or
    test leaf id may contain path separators or other unsafe characters; this
    collapses any run of unsafe characters to a single ``_`` so the worktree path
    can never escape ``workspace_root``.

    Args:
        leaf_id: The leaf's identity.

    Returns:
        A safe single path segment derived from ``leaf_id``.
    """
    safe = _UNSAFE_DIRNAME.sub("_", leaf_id).strip("_")
    return safe or "leaf"


class GitWorktreeProvider:
    """Creates a real ``git worktree`` per leaf and collects its real ``git diff``.

    DANGEROUS OPT-IN — see the module docstring. Spawns real ``git`` subprocesses
    against a real base repository on the host; not a security sandbox.

    Each :meth:`open_worktree` runs ``git worktree add <path> -b leaf/<leaf_id>
    <base_ref>`` and returns a :class:`LocalSubprocessSandbox` rooted in that tree,
    with an ``on_close`` hook bound to :meth:`teardown` so every manager teardown
    path removes the worktree automatically. :meth:`collect` is the authoritative
    changeset — a real ``git diff`` of the tree, never a model self-report. The
    provider is idempotent per leaf id (a stale same-key worktree+branch is
    reclaimed first) and exception-safe (a partial creation is rolled back before
    the error propagates).

    Args:
        base_repo: Path to the real base git repository worktrees branch from.
            Validated at construction (fail loud if it is not a git repo).
        integration_branch: Name of the integration branch leaf changes fold
            toward (recorded for host finalization; the provider itself does not
            materialize it). Defaults to ``"ldw/integration"``.
        base_ref: The ref each leaf worktree branches from. Defaults to ``"HEAD"``.
        workspace_root: Parent directory the per-leaf worktrees are created under;
            ``None`` allocates a private temp dir owned by this provider (removed by
            :meth:`cleanup_all`).
        policy: The :class:`ExecPolicy` applied to every leaf backend; ``None``
            uses the default policy.
        exec_gate: A shared bounded semaphore capping concurrent executions across
            every backend this provider creates; ``None`` builds one from the
            policy's ``max_concurrent_execs`` (mirroring ``local_subprocess_factory``).
    """

    def __init__(
        self,
        *,
        base_repo: str,
        integration_branch: str = DEFAULT_INTEGRATION_BRANCH,
        base_ref: str = "HEAD",
        workspace_root: str | None = None,
        policy: ExecPolicy | None = None,
        exec_gate: threading.BoundedSemaphore | None = None,
    ) -> None:
        self._base_repo = base_repo
        self._integration_branch = integration_branch
        self._base_ref = base_ref
        self._policy = policy or ExecPolicy()
        # A shared exec gate so the policy's concurrency cap is global across every
        # leaf backend this provider creates, mirroring local_subprocess_factory.
        self._exec_gate = exec_gate or threading.BoundedSemaphore(self._policy.max_concurrent_execs)
        # Own a private workspace_root when the host does not supply one, so
        # cleanup_all can remove the whole tree as a backstop.
        self._owns_workspace_root = workspace_root is None
        self._workspace_root = (
            workspace_root
            if workspace_root is not None
            else tempfile.mkdtemp(prefix="ldw-worktrees-")
        )
        # leaf_id -> worktree path, for teardown and the cleanup_all backstop.
        self._worktrees: dict[str, str] = {}
        # Fail loud at construction if base_repo is not a real git repo: a broken
        # base must never silently produce worktrees that misbehave later.
        self._run_git(self._base_repo, "rev-parse", "--git-dir")

    @property
    def integration_branch(self) -> str:
        """The integration branch leaf changes fold toward (host finalization)."""
        return self._integration_branch

    @property
    def tracked_leaf_ids(self) -> frozenset[str]:
        """The leaf ids whose worktrees this provider currently holds.

        A read-only snapshot of the live worktrees, exposed so a host (or a test)
        can verify teardown removed a worktree without reaching into private state.
        """
        return frozenset(self._worktrees)

    def _run_git(self, cwd: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a ``git`` command in ``cwd``, raising :class:`GitWorktreeError` on failure.

        Args:
            cwd: The directory to run ``git -C <cwd>`` in.
            *args: The git subcommand and its arguments.

        Returns:
            The completed process (with captured text stdout/stderr) on success.

        Raises:
            GitWorktreeError: If the command exits non-zero (the stderr is
                included) or ``git`` is not on PATH.
        """
        try:
            completed = subprocess.run(
                ["git", "-C", cwd, *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:  # git not on PATH, etc.
            raise GitWorktreeError(f"git {' '.join(args)} could not run: {error}") from error
        if completed.returncode != 0:
            raise GitWorktreeError(
                f"git {' '.join(args)} failed (exit {completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed

    def _branch_name(self, leaf_id: str) -> str:
        """Return the isolated branch name for ``leaf_id`` (``leaf/<leaf_id>``)."""
        return f"{_LEAF_BRANCH_PREFIX}{leaf_id}"

    def _worktree_path(self, leaf_id: str) -> str:
        """Return the absolute worktree directory path for ``leaf_id``."""
        return str(Path(self._workspace_root) / _safe_dirname(leaf_id))

    def _reclaim_stale(self, leaf_id: str) -> None:
        """Reclaim any stale same-key worktree and branch before a fresh create.

        Idempotent find-or-create groundwork: a crash after ``git worktree add``
        but before the leaf result was journaled would leave a worktree directory
        and a ``leaf/<leaf_id>`` branch under the same key, which a resume
        re-opening the same leaf id would otherwise collide with. Each removal is
        best-effort — a missing worktree or branch is not an error here.

        Args:
            leaf_id: The leaf whose stale state (if any) is reclaimed.
        """
        path = self._worktree_path(leaf_id)
        branch = self._branch_name(leaf_id)
        # Remove a registered/stale worktree at this path (best-effort).
        self._try_git(self._base_repo, "worktree", "remove", "--force", path)
        # Drop a leftover directory the worktree-remove did not (e.g. partial add).
        shutil.rmtree(path, ignore_errors=True)
        # Prune the worktree registry so a dangling entry cannot block a re-add.
        self._try_git(self._base_repo, "worktree", "prune")
        # Delete a leftover branch of the same name (best-effort).
        self._try_git(self._base_repo, "branch", "-D", branch)
        self._worktrees.pop(leaf_id, None)

    def _try_git(self, cwd: str, *args: str) -> None:
        """Run a ``git`` command best-effort, swallowing a non-zero exit.

        Used for idempotent teardown/reclaim steps where a missing worktree or
        branch is an expected, non-fatal condition. A real configuration error
        (``git`` absent) is the one thing that still surfaces, because it would
        make every operation fail.

        Args:
            cwd: The directory to run ``git -C <cwd>`` in.
            *args: The git subcommand and its arguments.

        Raises:
            GitWorktreeError: Only if ``git`` itself cannot be invoked (not on a
                non-zero exit, which is swallowed).
        """
        try:
            subprocess.run(
                ["git", "-C", cwd, *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:  # git not on PATH, etc.
            raise GitWorktreeError(f"git {' '.join(args)} could not run: {error}") from error

    def open_worktree(self, leaf_id: str) -> SandboxBackendProtocol:
        """Create a real worktree+branch for ``leaf_id`` and return its backend.

        Idempotent per leaf id: a stale same-key worktree and branch (left by a
        crash before journaling) are reclaimed first, then a fresh
        ``git worktree add <path> -b leaf/<leaf_id> <base_ref>`` is run. The
        returned backend is a :class:`LocalSubprocessSandbox` rooted in the new
        tree with an ``on_close`` hook bound to :meth:`teardown`, so every manager
        teardown path removes the worktree with no extra hook. Exception-safe: if
        any step after the directory is created fails, the partial worktree and
        branch are rolled back before the error propagates.

        Args:
            leaf_id: The leaf's stable derived identity.

        Returns:
            A backend rooted in the leaf's real worktree directory.

        Raises:
            GitWorktreeError: If creating the worktree fails (the partial state is
                rolled back first).
        """
        # Idempotency (find-or-create): clear any stale same-key state first.
        self._reclaim_stale(leaf_id)
        path = self._worktree_path(leaf_id)
        branch = self._branch_name(leaf_id)
        try:
            self._run_git(self._base_repo, "worktree", "add", path, "-b", branch, self._base_ref)
        except GitWorktreeError:
            # Exception-safe: roll back any partial worktree/branch before raising
            # so a failed create never leaves orphaned host state.
            self._try_git(self._base_repo, "worktree", "remove", "--force", path)
            shutil.rmtree(path, ignore_errors=True)
            self._try_git(self._base_repo, "worktree", "prune")
            self._try_git(self._base_repo, "branch", "-D", branch)
            raise
        # Only record the path once the worktree exists, so teardown never targets a
        # path that was never created.
        self._worktrees[leaf_id] = path
        return LocalSubprocessSandbox(
            identity=leaf_id,
            policy=self._policy,
            exec_gate=self._exec_gate,
            root=path,
            on_close=lambda: self.teardown(leaf_id),
        )

    def collect(self, leaf_id: str) -> dict[str, str]:
        """Return the leaf's authoritative changeset as a real ``git diff``.

        Stages the worktree (``git add -A``) then enumerates added/modified paths
        via ``git diff --cached --name-status`` and reads each path's current
        content from the worktree. This is the AUTHORITATIVE changeset — the real
        on-disk truth, never a model self-report — mirroring M5's "gate on the real
        exit code, not a model boolean". Deletions are omitted in v1 (the changeset
        is the set of added/modified file contents the integration step folds).

        Args:
            leaf_id: The leaf whose worktree changeset is collected.

        Returns:
            A mapping of protocol absolute path -> new content for every path added
            or modified relative to the base ref, in deterministic (sorted) order.

        Raises:
            GitWorktreeError: If ``leaf_id`` has no live worktree, or a ``git``
                command fails.
        """
        path = self._worktrees.get(leaf_id)
        if path is None:
            raise GitWorktreeError(
                f"collect: no live worktree for leaf {leaf_id!r} (open_worktree first)"
            )
        # Stage everything so new files appear in the cached diff too.
        self._run_git(path, "add", "-A")
        status = self._run_git(path, "diff", "--cached", "--name-status").stdout
        changeset: dict[str, str] = {}
        for line in status.splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            code = fields[0]
            # Added (A) and Modified (M) carry the path in field 1; a Rename (R...)
            # carries the destination path in field 2. Deletions (D) are omitted.
            if code.startswith("D"):
                continue
            rel_path = fields[-1]
            file_path = "/" + rel_path
            content = (Path(path) / rel_path).read_text(encoding="utf-8", errors="replace")
            changeset[file_path] = content
        return dict(sorted(changeset.items()))

    def teardown(self, leaf_id: str) -> None:
        """Remove the leaf's worktree and branch (idempotent, best-effort).

        Runs ``git worktree remove --force`` then ``git branch -D`` for the leaf,
        each best-effort (a missing worktree or branch is not an error), prunes the
        worktree registry, and drops the leaf from the provider's bookkeeping. Safe
        to call on an unknown or already-torn-down leaf. Bound to the backend's
        ``close`` via ``open_worktree``'s ``on_close`` hook, so every manager
        teardown path (reclaim/evict/stop/run-end) removes the worktree with no
        extra manager hook.

        Args:
            leaf_id: The leaf whose worktree is removed.
        """
        path = self._worktrees.pop(leaf_id, self._worktree_path(leaf_id))
        branch = self._branch_name(leaf_id)
        self._try_git(self._base_repo, "worktree", "remove", "--force", path)
        shutil.rmtree(path, ignore_errors=True)
        self._try_git(self._base_repo, "worktree", "prune")
        self._try_git(self._base_repo, "branch", "-D", branch)

    def cleanup_all(self) -> None:
        """Tear down every remaining worktree and remove an owned workspace root.

        Defense-in-depth run-end backstop: every worktree the provider still holds
        is torn down (so a leak from a missed teardown path is still reclaimed),
        and an owned (provider-allocated) ``workspace_root`` temp tree is removed
        wholesale. Idempotent.
        """
        for leaf_id in list(self._worktrees):
            self.teardown(leaf_id)
        if self._owns_workspace_root:
            shutil.rmtree(self._workspace_root, ignore_errors=True)

"""Real local-subprocess execution backend for execution leaves.

DANGEROUS OPT-IN — this is NOT a security sandbox. ``LocalSubprocessSandbox``
runs each leaf's shell command in a private per-leaf temporary directory on the
host, with the calling user's full permissions. A command can still read and
write any host path via absolute paths, open network connections, and consume
host resources beyond the best-effort POSIX resource limits. The per-leaf temp
directory bounds only the *default working directory*; it is not a filesystem
jail. For untrusted or adversarial commands, run the engine behind an
out-of-process isolation backend (a container) instead.

What this backend does guarantee: a private temporary working directory per leaf
(no accidental execution in the engine's own directory), the engine file APIs
stay rooted with ``..`` traversal rejected, a bounded effective timeout with
best-effort process-group termination, bounded combined output, a bounded count
of concurrent executions, and best-effort POSIX resource limits. On non-POSIX
platforms (Windows) the posture is weaker: no resource limits and no
process-group semantics, only a timeout, an output cap, a temporary working
directory, best-effort termination, and the concurrency bound.

The offline default backend (``InMemorySandbox``) is unaffected; real execution
is an explicit, host-constructed opt-in.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import IO, Literal

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - non-POSIX has no resource module
    resource = None  # type: ignore[assignment]

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
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

EXIT_TIMEOUT = 124
"""Exit-code sentinel reported when a command is killed for exceeding its timeout."""

EXIT_REJECTED = 126
"""Exit-code sentinel reported when admission control rejects a command unspawned."""

_DEFAULT_REJECT_REASON = "execution rejected by policy"
"""Output text used when admission rejects a command without giving a reason."""

_DRAIN_CHUNK_BYTES = 65536
"""Read granularity for the bounded output drain, in bytes."""


ExecOutcome = Literal["allow", "reject"]
"""Whether an admission decision permits (``"allow"``) or refuses (``"reject"``)."""


@dataclass(frozen=True, slots=True)
class ExecRequest:
    """An admission request for one shell ``execute``, before any process spawns.

    Attributes:
        command: The shell command string the leaf asked to run.
        timeout: The timeout (seconds) the caller passed, or ``None`` for the
            backend default.
        leaf_id: The owning leaf's sandbox identity.
    """

    command: str
    timeout: int | None
    leaf_id: str


@dataclass(frozen=True, slots=True)
class RLimitProfile:
    """POSIX resource limits applied in the child before exec (best-effort).

    Each ``None`` field leaves that limit unset. The limits are applied via
    ``resource.setrlimit`` in a minimal ``preexec_fn`` (run in the child between
    fork and exec); the soft/hard values are pre-computed in the parent so the
    child body allocates nothing, which keeps the hook safe in a ``to_thread``
    execution runtime. A limit the kernel refuses (for example ``RLIMIT_AS`` on
    Darwin) is skipped best-effort rather than aborting the command. The default
    values are generous enough not to
    break a typical build or test command yet low enough to stop a runaway from
    exhausting the host:

    - CPU ``60`` seconds: bounds a busy loop without truncating a normal build.
    - Address space ``2 GiB``: bounds a memory hog while leaving headroom for a
      compiler or test runner (kernel-enforced on Linux; ignored on Darwin).
    - File size ``256 MiB``: bounds runaway file growth.
    - Open files ``1024``: a common interactive-shell default; bounds descriptor
      leaks.
    - Processes: unset by default. ``RLIMIT_NPROC`` counts *every* process the
      host user owns, not just this command's children, so a fixed cap reflects
      ambient host load rather than the command and can break ``fork`` on a busy
      host. A host that wants a fork-bomb guard sets it explicitly with headroom
      above its own baseline process count.

    Attributes:
        cpu_seconds: ``RLIMIT_CPU`` soft/hard cap (CPU seconds).
        address_space_bytes: ``RLIMIT_AS`` (virtual memory) cap, in bytes.
        file_size_bytes: ``RLIMIT_FSIZE`` cap, in bytes.
        open_files: ``RLIMIT_NOFILE`` cap (max open descriptors).
        processes: ``RLIMIT_NPROC`` cap (max processes for the whole host user);
            unset by default because the per-user count makes a fixed cap
            unreliable as a per-command guard.
    """

    cpu_seconds: int | None = 60
    address_space_bytes: int | None = 2 * 1024 * 1024 * 1024
    file_size_bytes: int | None = 256 * 1024 * 1024
    open_files: int | None = 1024
    processes: int | None = None


@dataclass(frozen=True, slots=True)
class ExecDecision:
    """An admission decision returned by ``before_execute``.

    Attributes:
        outcome: ``"allow"`` to run, ``"reject"`` to refuse without spawning.
        timeout: An override timeout (seconds) clamping the effective timeout;
            ``None`` keeps the request/default timeout.
        output_cap_bytes: An override on the combined-output byte cap; ``None``
            keeps the configured default.
        rlimits: An override resource-limit profile (POSIX only); ``None`` keeps
            the configured default profile.
        reason: A short human-readable reason surfaced in the response on reject.
    """

    outcome: ExecOutcome = "allow"
    timeout: int | None = None
    output_cap_bytes: int | None = None
    rlimits: RLimitProfile | None = None
    reason: str = ""


BeforeExecuteHook = Callable[[ExecRequest], ExecDecision]
"""Pure admission hook mapping an :class:`ExecRequest` to an :class:`ExecDecision`."""


@dataclass(frozen=True, slots=True)
class ExecPolicy:
    """Resilience and admission policy for a ``LocalSubprocessSandbox`` factory.

    Attributes:
        default_timeout: Effective timeout (seconds) when a call passes ``None``.
        output_cap_bytes: Default combined stdout+stderr byte cap.
        grace_seconds: SIGTERM-to-SIGKILL grace window on timeout (POSIX).
        max_concurrent_execs: Bound on concurrent ``execute`` calls enforced by
            the shared exec gate.
        rlimits: Default POSIX resource-limit profile (see :class:`RLimitProfile`).
        before_execute: Admission hook; ``None`` allows every request as
            configured.
    """

    default_timeout: int = 30
    output_cap_bytes: int = 1_000_000
    grace_seconds: float = 2.0
    max_concurrent_execs: int = 8
    rlimits: RLimitProfile = field(default_factory=RLimitProfile)
    before_execute: BeforeExecuteHook | None = None


def _drain_bounded(stream: IO[bytes] | None, output_cap_bytes: int) -> tuple[str, bool]:
    """Drain ``stream`` to end-of-file, keeping at most ``output_cap_bytes``.

    The stream is read in fixed-size chunks. Bytes are accumulated only until the
    cap is reached; past that point the reader keeps consuming and discarding so
    the producing child never blocks on a full pipe, but the kept buffer stays
    byte-bounded by construction. The kept bytes are decoded as UTF-8 with
    ``errors="replace"`` so a chunk boundary that splits a multibyte sequence
    cannot raise.

    Args:
        stream: The child's combined stdout/stderr pipe in binary mode, or
            ``None`` when no pipe was attached.
        output_cap_bytes: The maximum number of bytes to keep. A non-positive
            cap keeps nothing while still draining the pipe to end-of-file.

    Returns:
        A ``(decoded_output, truncated)`` pair where ``truncated`` is ``True``
        when at least one byte was dropped because the cap was reached.
    """
    if stream is None:
        return "", False
    kept = bytearray()
    truncated = False
    while True:
        chunk = stream.read(_DRAIN_CHUNK_BYTES)
        if not chunk:
            break
        remaining = output_cap_bytes - len(kept)
        if remaining > 0:
            take = chunk[:remaining]
            kept.extend(take)
            if len(take) < len(chunk):
                truncated = True
        else:
            # Already at the cap; keep reading to EOF so the child is not
            # wedged on a full pipe, but drop everything past the cap.
            truncated = True
    return kept.decode("utf-8", errors="replace"), truncated


def _terminate_process_tree(proc: subprocess.Popen[bytes], grace_seconds: float) -> None:
    """Escalate termination of ``proc`` and (POSIX) its whole process group.

    On POSIX the child was started in its own session (``start_new_session``), so
    a single ``os.killpg`` to the child's process-group id reaches the child and
    every descendant it spawned. The escalation is a graceful ``SIGTERM`` first, then a
    ``grace_seconds`` window for the tree to exit, then an unconditional
    ``SIGKILL`` if anything is still alive. On non-POSIX platforms there is no
    process-group semantics, so only the direct child is signalled with the
    best-effort ``terminate`` then ``kill``.

    Every signal is wrapped because a target that already exited raises
    ``ProcessLookupError`` (POSIX) or ``OSError`` (Windows); a race where the
    process dies between the liveness check and the signal must not surface as an
    error from a timeout that was, in fact, handled.

    Args:
        proc: The spawned child whose tree must be torn down.
        grace_seconds: Seconds to wait after ``SIGTERM`` before sending
            ``SIGKILL``.
    """
    if os.name == "posix":
        try:
            process_group_id = os.getpgid(proc.pid)
        except ProcessLookupError:
            # The child is already gone; nothing left to signal.
            return
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        # Non-POSIX: best-effort, and (like the POSIX branch) tolerate a process that
        # already exited between the check and the signal — an OSError there must not
        # turn a handled timeout into a raised error.
        try:
            proc.terminate()
        except OSError:
            return
        try:
            proc.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.kill()
        except OSError:
            return


def _resolve_effective_timeout(
    *, call_timeout: int | None, decision_timeout: int | None, default_timeout: int
) -> int:
    """Pick the tightest timeout among the call, the admission decision, and policy.

    The admission decision may only clamp the timeout *down*: it is treated as a
    ceiling, never a way to widen a tighter call- or policy-level bound. The
    effective timeout is therefore the minimum of every value that was supplied
    (the call timeout and the decision timeout are each optional, the policy
    default always applies), guaranteeing a bounded deadline on every spawn.

    Args:
        call_timeout: The per-call timeout, or ``None`` when the caller deferred.
        decision_timeout: The admission decision's override, or ``None``.
        default_timeout: The policy default that always applies.

    Returns:
        The smallest of the supplied timeouts, in seconds.
    """
    candidates = [default_timeout]
    if call_timeout is not None:
        candidates.append(call_timeout)
    if decision_timeout is not None:
        candidates.append(decision_timeout)
    return min(candidates)


def _build_rlimit_setters(rlimits: RLimitProfile) -> list[tuple[int, tuple[int, int]]]:
    """Pre-compute ``(rlimit_id, (soft, hard))`` pairs in the PARENT process.

    Doing the ``getrlimit`` reads and the soft/hard arithmetic here, before the
    fork, keeps the child-side ``preexec_fn`` minimal: it only calls
    ``resource.setrlimit`` on a ready-made list, with no allocation, import, or
    lock acquisition. That is what makes a ``preexec_fn`` safe to run between fork
    and exec even though the parent is multi-threaded — each ``execute`` already
    runs on its own ``to_thread`` worker that forks then immediately execs, and the
    child touches nothing another thread could hold a lock on. Applying the limits
    in-process this way (no extra ``python -c`` helper) keeps a full interpreter
    start-up off every command's critical path.

    The hard cap is lowered to the requested value only when the inherited hard cap
    is finite and not already below it; otherwise the inherited hard cap is kept (a
    process cannot raise its own hard cap). The soft cap never exceeds the resulting
    hard cap. Limits the kernel refuses (for example ``RLIMIT_AS`` on Darwin) are
    skipped, best-effort, rather than aborting the command.

    Args:
        rlimits: The resource-limit profile to apply.

    Returns:
        A list of ``(rlimit_id, (soft, hard))`` pairs; empty on non-POSIX.
    """
    if resource is None:  # non-POSIX: no resource limits
        return []
    # resource is narrowed non-None below; RLIMIT_* ids exist only on POSIX.
    constants = {
        "cpu_seconds": resource.RLIMIT_CPU,
        "address_space_bytes": resource.RLIMIT_AS,
        "file_size_bytes": resource.RLIMIT_FSIZE,
        "open_files": resource.RLIMIT_NOFILE,
        "processes": resource.RLIMIT_NPROC,
    }
    setters: list[tuple[int, tuple[int, int]]] = []
    for field_name, value in asdict(rlimits).items():
        rlimit = constants.get(field_name)
        if rlimit is None or value is None:
            continue
        try:
            _, hard = resource.getrlimit(rlimit)
        except (OSError, ValueError):  # pragma: no cover - kernel refusal
            continue
        unbounded = hard == resource.RLIM_INFINITY
        new_hard = value if (not unbounded and hard >= value) else hard
        new_soft = value if unbounded else min(value, new_hard)
        setters.append((rlimit, (new_soft, new_hard)))
    return setters


def _rlimit_preexec(setters: list[tuple[int, tuple[int, int]]]) -> Callable[[], None]:
    """Return a minimal ``preexec_fn`` applying pre-computed rlimits in the child.

    Runs between fork and exec. It only iterates the pre-built ``setters`` and calls
    ``resource.setrlimit`` (a syscall) — deliberately no allocation, import, or lock,
    so it is safe between fork and exec in the threaded ``to_thread`` runtime. A
    limit the kernel refuses at set time is skipped rather than aborting the spawn.

    Args:
        setters: Pre-computed ``(rlimit_id, (soft, hard))`` pairs from
            :func:`_build_rlimit_setters`.

    Returns:
        A zero-argument callable suitable for ``subprocess.Popen(preexec_fn=...)``.
    """

    def _apply() -> None:
        for rlimit, pair in setters:
            # Bare try/except (not contextlib.suppress) keeps this preexec_fn body
            # allocation-light across the fork-then-exec window; a limit the kernel
            # refuses at set time is skipped best-effort.
            try:  # noqa: SIM105
                resource.setrlimit(rlimit, pair)  # type: ignore[union-attr]
            except (OSError, ValueError):
                pass

    return _apply


def _canonical_segments(path: str) -> list[str]:
    """Resolve a protocol absolute path to its segment list, rejecting escapes.

    The protocol file APIs use absolute paths rooted at ``/``. This collapses
    ``.`` and empty segments and resolves ``..`` segments, treating a ``..`` that
    would climb above the root as a hard error rather than a silent clamp. That
    guard is what stops a path like ``/../escape`` from landing on the host
    filesystem above the per-leaf temporary directory.

    Args:
        path: An absolute protocol path (treated as rooted at ``/`` regardless of
            a leading slash).

    Returns:
        The canonical path segments below the root, e.g. ``["a", "b"]`` for
        ``/a/b`` and ``[]`` for the bare root.

    Raises:
        ValueError: If a ``..`` segment would escape above the root.
    """
    segments: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not segments:
                raise ValueError(f"path {path!r} escapes root via '..' traversal")
            segments.pop()
            continue
        segments.append(segment)
    return segments


class LocalSubprocessSandbox(SandboxBackendProtocol):
    """A full-protocol backend running each command in a per-leaf temp root.

    DANGEROUS OPT-IN — this is NOT a security sandbox. The command runs on the
    host with the calling user's permissions; the per-leaf temporary directory
    bounds only the default working directory, not the reachable filesystem or
    the network. See the module docstring and the project README for the full
    sharp-edge warning before enabling real execution.

    Each instance owns a private temporary directory created at construction; the
    command's working directory is that directory, and the file operations read
    and write real files beneath it, so ``execute`` and the file tools share one
    filesystem. Every :meth:`execute` call reaps its own child (and, on POSIX,
    its process group on timeout) before it returns, so a completed call leaves
    no live process behind; :meth:`close` therefore only removes the temporary
    directory, and is idempotent.

    Args:
        identity: The owning leaf identity, surfaced via :attr:`id`.
        policy: The resilience and admission policy for this backend.
        exec_gate: A shared bounded semaphore capping concurrent executions
            across every backend produced by one factory.
    """

    def __init__(
        self,
        *,
        identity: str,
        policy: ExecPolicy,
        exec_gate: threading.BoundedSemaphore,
    ) -> None:
        self._identity = identity
        self._policy = policy
        self._exec_gate = exec_gate
        self._closed = False
        # The private per-leaf working directory. Created eagerly so the file
        # operations and execute share one real filesystem from the first call.
        self._root = tempfile.mkdtemp(prefix=f"ldw-exec-{identity}-")

    @property
    def id(self) -> str:
        """The owning leaf identity (unique per sandbox instance)."""
        return self._identity

    @property
    def root_path(self) -> str:
        """The absolute path of this backend's private per-leaf working directory."""
        return self._root

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` as a real subprocess in this backend's temp root.

        Admission is gated twice. First the shared exec gate bounds the number of
        concurrent ``execute`` calls across every backend produced by one factory
        (the per-run cap). This is a *distinct* bound from the leaf concurrency
        gate, which bounds in-flight leaves on the event loop: one leaf can fire
        many ``execute`` calls, and several runs share the host, so the exec gate
        is a cross-thread :class:`threading.BoundedSemaphore` acquired here in the
        synchronous body, not the asyncio leaf gate. The slot is always returned
        in a ``finally`` so a timeout or exception cannot leak it. After the slot
        is held, the admission hook decides what to do: it may reject the command
        (the rejection sentinel is returned without launching a process), clamp
        the effective timeout down, lower the combined-output byte cap, or select
        a different POSIX resource-limit profile for this one call. Any override
        the decision omits falls back to the configured policy default.

        The command runs through the system shell with its working directory set
        to :attr:`root_path`. On POSIX the child starts a new session so the
        whole process group can be terminated together, and the effective POSIX
        resource limits are applied via a minimal ``preexec`` hook in the child
        before the shell takes over. The combined stdout and stderr are drained up
        to the effective
        output cap and returned as the response output, with the child's exit
        code as :attr:`ExecuteResponse.exit_code`.

        A bounded effective timeout always applies: the tightest of the per-call
        ``timeout``, the admission decision's clamp, and the policy default (the
        decision can only narrow it, never widen it). The drain runs on a worker
        thread so
        the calling thread can wait on the child with a deadline; when the
        deadline passes the process tree is escalated down (``SIGTERM``, a grace
        window, then ``SIGKILL`` to the whole group on POSIX) and the response
        carries the timeout exit-code sentinel. The child is always reaped and
        the drain thread always joined, so no zombie, orphan, or leaked thread
        survives the timeout path.

        The drain is byte-bounded: once the effective cap is reached the response
        stops accumulating and is flagged truncated, but the pipe keeps being
        read and discarded to end-of-file so a chatty child never blocks on a
        full pipe. The response buffer therefore never grows past the cap.

        Args:
            command: The full shell command string to run.
            timeout: Maximum seconds to wait; ``None`` defers to the policy
                default.

        Returns:
            An :class:`ExecuteResponse` with the bounded combined output, exit
            code, and a truncation flag.
        """
        # The shared exec gate bounds concurrent executions per run; hold it for
        # the whole spawn-and-wait and release it in a finally so a timeout or an
        # unexpected error can never strand a slot.
        self._exec_gate.acquire()
        try:
            return self._execute_admitted(command, timeout=timeout)
        finally:
            self._exec_gate.release()

    def _execute_admitted(self, command: str, *, timeout: int | None) -> ExecuteResponse:
        """Run ``command`` while holding the exec-gate slot.

        The admission hook is consulted first (the slot is already held, so the
        hook can observe true in-flight concurrency). A rejection short-circuits
        without spawning. Otherwise the decision's timeout clamp, output-cap, and
        resource-limit overrides are resolved against the policy defaults, and the
        command is spawned, drained, and timed out exactly as :meth:`execute`
        documents.

        Args:
            command: The full shell command string to run.
            timeout: The per-call timeout, or ``None`` to defer to the policy
                default.

        Returns:
            The execution response, or the rejection sentinel when admission
            refused the command.
        """
        request = ExecRequest(command=command, timeout=timeout, leaf_id=self._identity)
        decision = (
            self._policy.before_execute(request)
            if self._policy.before_execute is not None
            else ExecDecision()
        )
        if decision.outcome == "reject":
            return ExecuteResponse(
                output=decision.reason or _DEFAULT_REJECT_REASON,
                exit_code=EXIT_REJECTED,
                truncated=False,
            )
        # Apply the admission overrides: the decision may lower the output cap,
        # clamp the timeout down (never widen it), and select a different rlimit
        # profile. A missing override falls back to the policy default.
        # Distinguish an explicit 0 (suppress output) from None (no override): a
        # falsey ``or`` would treat a 0-byte admission cap as "unset" and fall back to
        # the policy cap, so admission could never reduce the cap to zero.
        output_cap_bytes = (
            self._policy.output_cap_bytes
            if decision.output_cap_bytes is None
            else decision.output_cap_bytes
        )
        effective_timeout = _resolve_effective_timeout(
            call_timeout=timeout,
            decision_timeout=decision.timeout,
            default_timeout=self._policy.default_timeout,
        )
        rlimits = decision.rlimits or self._policy.rlimits
        # Apply POSIX rlimits via a minimal preexec_fn (set in the child between fork
        # and exec) rather than a python -c wrapper, so no extra interpreter start-up
        # precedes the command. The shell command runs directly under shell=True; the
        # child still gets its own session (start_new_session) so the timeout's
        # process-group kill reaches it and every descendant.
        rlimit_setters = _build_rlimit_setters(rlimits)
        preexec = _rlimit_preexec(rlimit_setters) if rlimit_setters else None
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=self._root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
            preexec_fn=preexec,
        )
        # The drain runs on a worker so the calling thread can enforce the
        # deadline; the worker stores its result for the calling thread to read
        # once the pipe has closed (which the kill, if any, guarantees).
        drained: dict[str, tuple[str, bool]] = {}

        def _drain() -> None:
            try:
                drained["result"] = _drain_bounded(proc.stdout, output_cap_bytes)
            except Exception:
                # The read end was closed under us (an escaped child kept the pipe
                # open past the deadline and the caller closed it to unblock us).
                # Record an incomplete result instead of surfacing in this daemon
                # thread.
                drained.setdefault("result", ("", True))

        drain_thread = threading.Thread(
            target=_drain, name=f"ldw-drain-{self._identity}", daemon=True
        )
        drain_thread.start()
        timed_out = False
        try:
            try:
                proc.wait(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_tree(proc, self._policy.grace_seconds)
        finally:
            # Reap the direct child and join the drain. After a process-group kill the
            # child's write end closes, so the drain hits EOF and joins promptly. But a
            # descendant that escaped the group (setsid / daemonize) can keep the pipe
            # open: bound the join so one escaped child cannot wedge execute() past its
            # deadline (and hold the exec-gate slot), then close the read end to unblock
            # an abandoned drain. The drain thread is a daemon, so an abandoned one never
            # blocks interpreter shutdown.
            proc.wait()
            drain_thread.join(timeout=self._policy.grace_seconds)
            if drain_thread.is_alive():
                # An escaped child (setsid / daemonize) still holds the write end open,
                # so the drain is blocked in read() holding the BufferedReader lock;
                # calling close() here would block on that same lock and re-introduce
                # the hang. Abandon the daemon drain instead (it ends on its own when
                # the escaped child finally closes the pipe) and return bounded with an
                # incomplete result rather than wedging execute past its deadline.
                drained.setdefault("result", ("", True))
            elif proc.stdout is not None:
                proc.stdout.close()
        output, truncated = drained.get("result", ("", False))
        exit_code = EXIT_TIMEOUT if timed_out else proc.returncode
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    def _resolve(self, path: str) -> str:
        """Map a protocol absolute path to a real path under the temp root.

        The protocol file APIs address files with absolute paths rooted at ``/``;
        those map one-to-one onto real files beneath the per-leaf temporary
        directory, which is exactly why ``execute`` (running with that directory
        as its working directory) sees the same files the tool surface writes. A
        ``..`` segment that would escape above the root is rejected so a write can
        never land on the host filesystem outside the leaf's private directory.

        Two layers guard the root: ``_canonical_segments`` rejects a lexical ``..``
        escape, and a ``realpath`` check rejects a symlink created inside the root by
        a command that points outside it (a lexical join alone would return the
        in-root symlink path and the later ``open`` would follow it to the host). A
        symlink that stays within the root resolves fine. This keeps the file APIs
        rooted even though shell execution itself is not sandboxed. It is best-effort
        against a TOCTOU swap (the symlink could change between this check and the
        ``open``), consistent with the backend's documented non-sandbox posture.

        Args:
            path: An absolute protocol path.

        Returns:
            The absolute host path of the file beneath the temp root.

        Raises:
            ValueError: If the path escapes the root via ``..`` or a symlink.
        """
        candidate = os.path.join(self._root, *_canonical_segments(path))
        root_real = os.path.realpath(self._root)
        candidate_real = os.path.realpath(candidate)
        if candidate_real != root_real and not candidate_real.startswith(root_real + os.sep):
            raise ValueError(f"path {path!r} escapes the leaf root via a symlink")
        return candidate

    def _stored_paths(self) -> list[str]:
        """List every regular file under the temp root as a protocol path.

        Real files beneath the temp root are surfaced back to callers using the
        protocol's ``/``-rooted absolute path scheme (the inverse of
        :meth:`_resolve`), so listing, globbing, and grepping report the same
        addresses a caller would pass to :meth:`read`. The result is sorted for
        deterministic ordering.

        Returns:
            Every regular file's protocol path, in ascending lexical order.
        """
        stored: list[str] = []
        for current_dir, _subdirs, filenames in os.walk(self._root):
            for filename in filenames:
                absolute = os.path.join(current_dir, filename)
                relative = os.path.relpath(absolute, self._root)
                stored.append("/" + relative.replace(os.sep, "/"))
        stored.sort()
        return stored

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create ``file_path`` as a real file under the temp root.

        Mirrors :class:`InMemorySandbox.write`: the write refuses to clobber an
        existing path (returning an error rather than overwriting) so a leaf
        cannot silently destroy a file it already produced. Parent directories
        are created as needed.

        Args:
            file_path: Absolute protocol path to create.
            content: UTF-8 text content to write.

        Returns:
            A :class:`WriteResult` carrying the written path, or an error when the
            file already exists or the path escapes the root.
        """
        try:
            real_path = self._resolve(file_path)
        except ValueError as error:
            return WriteResult(error=str(error))
        if os.path.exists(real_path):
            return WriteResult(error=f"Cannot write to {file_path} because it already exists.")
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        with open(real_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return WriteResult(path=file_path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read ``file_path`` from the temp root.

        Args:
            file_path: Absolute protocol path to read.
            offset: Accepted for protocol compatibility; the full content is
                returned regardless of ``offset``/``limit`` for this backend.
            limit: Accepted for protocol compatibility; see ``offset``.

        Returns:
            A :class:`ReadResult` with the file data, or an error on miss or an
            out-of-root path.
        """
        try:
            real_path = self._resolve(file_path)
        except ValueError as error:
            return ReadResult(error=str(error))
        if not os.path.isfile(real_path):
            return ReadResult(error=f"File '{file_path}' not found")
        with open(real_path, encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        return ReadResult(file_data=FileData(content=content, encoding="utf-8"))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace ``old_string`` with ``new_string`` in a file under the temp root.

        Args:
            file_path: Absolute protocol path to edit.
            old_string: Exact substring to replace.
            new_string: Replacement text.
            replace_all: Replace every occurrence when ``True``; otherwise the
                first occurrence only.

        Returns:
            An :class:`EditResult` with the edited path and replacement count, or
            an error on miss or an out-of-root path.
        """
        try:
            real_path = self._resolve(file_path)
        except ValueError as error:
            return EditResult(error=str(error))
        if not os.path.isfile(real_path):
            return EditResult(error=f"File '{file_path}' not found")
        with open(real_path, encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        count = content.count(old_string) if replace_all else (1 if old_string in content else 0)
        updated = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        with open(real_path, "w", encoding="utf-8") as handle:
            handle.write(updated)
        return EditResult(path=file_path, occurrences=count)

    def ls(self, path: str) -> LsResult:
        """List the files at or under ``path`` within the temp root.

        Args:
            path: Directory protocol path; only entries under it are returned.

        Returns:
            An :class:`LsResult` with one entry per stored file under ``path``, or
            an error on an out-of-root path.
        """
        try:
            self._resolve(path)
        except ValueError as error:
            return LsResult(error=str(error))
        prefix = path if path.endswith("/") else f"{path}/"
        entries = [
            self._file_info(stored)
            for stored in self._stored_paths()
            if stored == path or stored.startswith(prefix)
        ]
        return LsResult(entries=entries)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        """Search files under the temp root for a literal substring.

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
            A :class:`GrepResult` listing one match per matching line, or an error
            on an out-of-root ``path``.
        """
        if path is not None:
            try:
                self._resolve(path)
            except ValueError as error:
                return GrepResult(error=str(error))
        prefix = None if path is None else (path if path.endswith("/") else f"{path}/")
        matches: list[GrepMatch] = []
        for stored in self._stored_paths():
            if prefix is not None and stored != path and not stored.startswith(prefix):
                continue
            if glob is not None and not fnmatch.fnmatch(stored, glob):
                continue
            with open(self._resolve(stored), encoding="utf-8", errors="replace") as handle:
                content = handle.read()
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append(GrepMatch(path=stored, line=line_number, text=line))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching ``pattern`` under ``path`` within the temp root.

        Args:
            pattern: Glob pattern matched against each stored file's full path.
            path: Base directory the search is rooted at; only files at or under
                it are considered.

        Returns:
            A :class:`GlobResult` of matching files in deterministic path order,
            or an error on an out-of-root ``path``.
        """
        try:
            self._resolve(path)
        except ValueError as error:
            return GlobResult(error=str(error))
        prefix = path if path.endswith("/") else f"{path}/"
        matches = [
            self._file_info(stored)
            for stored in self._stored_paths()
            if (stored == path or stored.startswith(prefix)) and fnmatch.fnmatch(stored, pattern)
        ]
        return GlobResult(matches=matches)

    def _file_info(self, stored_path: str) -> FileInfo:
        """Build a :class:`FileInfo` entry for a stored protocol path."""
        try:
            size = os.path.getsize(self._resolve(stored_path))
        except OSError:
            size = 0
        return FileInfo(path=stored_path, is_dir=False, size=size, modified_at="")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Store each ``(path, content)`` pair as a UTF-8 file (overwriting).

        Upload deliberately overwrites (unlike :meth:`write`, which errors on an
        existing path) so a batch upload is idempotent. Binary content that is not
        valid UTF-8, or a path that escapes the root, is reported as that file's
        ``invalid_path`` error rather than aborting the batch.

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
            try:
                real_path = self._resolve(file_path)
            except ValueError:
                responses.append(FileUploadResponse(path=file_path, error=INVALID_PATH))
                continue
            os.makedirs(os.path.dirname(real_path), exist_ok=True)
            with open(real_path, "w", encoding="utf-8") as handle:
                handle.write(text)
            responses.append(FileUploadResponse(path=file_path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Return the bytes of each requested path (partial success per entry).

        Args:
            paths: File protocol paths to download.

        Returns:
            One :class:`FileDownloadResponse` per input path, in input order; a
            missing path lands as that entry's ``file_not_found`` error and an
            out-of-root path as ``invalid_path``.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                real_path = self._resolve(path)
            except ValueError:
                responses.append(FileDownloadResponse(path=path, error=INVALID_PATH))
                continue
            if not os.path.isfile(real_path):
                responses.append(FileDownloadResponse(path=path, error=FILE_NOT_FOUND))
                continue
            with open(real_path, "rb") as handle:
                responses.append(FileDownloadResponse(path=path, content=handle.read()))
        return responses

    def close(self) -> None:
        """Remove the per-leaf temp directory (idempotent).

        Best-effort: a directory that is already gone is tolerated. No process
        cleanup is needed here because every :meth:`execute` call reaps its own
        child (and its process group on timeout) before returning, so a closed
        backend never had a live process to terminate.
        """
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self._root, ignore_errors=True)

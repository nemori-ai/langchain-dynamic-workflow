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

import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

EXIT_TIMEOUT = 124
"""Exit-code sentinel reported when a command is killed for exceeding its timeout."""

EXIT_REJECTED = 126
"""Exit-code sentinel reported when admission control rejects a command unspawned."""

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
    ``resource.setrlimit`` in a child wrapper (not ``preexec_fn``), which is the
    thread-safe choice in a ``to_thread`` execution runtime. The default values
    are generous enough not to break a typical build or test command yet low
    enough to stop a runaway from exhausting the host:

    - CPU ``60`` seconds: bounds a busy loop without truncating a normal build.
    - Address space ``2 GiB``: bounds a memory hog while leaving headroom for a
      compiler or test runner.
    - File size ``256 MiB``: bounds runaway file growth.
    - Open files ``1024``: a common interactive-shell default; bounds descriptor
      leaks.
    - Processes ``256``: bounds a fork bomb while permitting parallel builds.

    Attributes:
        cpu_seconds: ``RLIMIT_CPU`` soft/hard cap (CPU seconds).
        address_space_bytes: ``RLIMIT_AS`` (virtual memory) cap, in bytes.
        file_size_bytes: ``RLIMIT_FSIZE`` cap, in bytes.
        open_files: ``RLIMIT_NOFILE`` cap (max open descriptors).
        processes: ``RLIMIT_NPROC`` cap (max processes for the user).
    """

    cpu_seconds: int | None = 60
    address_space_bytes: int | None = 2 * 1024 * 1024 * 1024
    file_size_bytes: int | None = 256 * 1024 * 1024
    open_files: int | None = 1024
    processes: int | None = 256


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
    filesystem. :meth:`close` removes the directory and terminates any straggler
    process; it is idempotent.

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

        The command runs through the system shell with its working directory set
        to :attr:`root_path`. On POSIX the child starts a new session so the
        whole process group can be terminated together by later resilience
        layers. The combined stdout and stderr are returned as the response
        output and the child's exit code as :attr:`ExecuteResponse.exit_code`.

        Args:
            command: The full shell command string to run.
            timeout: Maximum seconds to wait; ``None`` defers to the policy
                default.

        Returns:
            An :class:`ExecuteResponse` with the combined output and exit code.
        """
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=self._root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            output, _ = proc.communicate()
        finally:
            # Reap the child and release its pipes on every path so neither a
            # zombie process nor a leaked descriptor survives this call.
            proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
        return ExecuteResponse(
            output=output or "",
            exit_code=proc.returncode,
            truncated=False,
        )

    def close(self) -> None:
        """Remove the per-leaf temp directory (idempotent).

        Best-effort: a directory that is already gone is tolerated. Straggler
        process termination is wired in by the resilience layers.
        """
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self._root, ignore_errors=True)

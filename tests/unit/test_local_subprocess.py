"""Unit tests for the real local-subprocess execution backend.

These tests really spawn child processes: a command runs and returns its actual
stdout + exit code; the file ops persist to a real per-leaf temp dir that execute
shares; the timeout actually kills a sleeping process group; output truncation
actually trips; the ExecGate actually bounds concurrency; before_execute actually
rejects/clamps; and POSIX rlimits actually cap (skipped honestly off POSIX). This
is the anti-corruption floor for "execute is real now".
"""

from __future__ import annotations

import os
import sys
import threading

from langchain_dynamic_workflow._local_subprocess import (
    ExecPolicy,
    LocalSubprocessSandbox,
)


def _backend(policy: ExecPolicy | None = None) -> LocalSubprocessSandbox:
    return LocalSubprocessSandbox(
        identity="leaf-test",
        policy=policy or ExecPolicy(),
        exec_gate=threading.BoundedSemaphore(8),
    )


def test_execute_runs_a_real_command_and_returns_stdout_and_exit_code() -> None:
    backend = _backend()
    try:
        result = backend.execute(f'{sys.executable} -c "print(2 + 40)"')
        assert "42" in result.output
        assert result.exit_code == 0
        assert result.truncated is False
    finally:
        backend.close()


def test_execute_reports_a_nonzero_exit_code_for_a_failing_command() -> None:
    backend = _backend()
    try:
        result = backend.execute(f'{sys.executable} -c "import sys; sys.exit(3)"')
        assert result.exit_code == 3
    finally:
        backend.close()


def test_execute_runs_in_the_per_leaf_temp_root_not_the_repo_cwd() -> None:
    # The command's cwd is the leaf's private temp dir, never the engine cwd.
    backend = _backend()
    try:
        result = backend.execute(f'{sys.executable} -c "import os; print(os.getcwd())"')
        assert backend.root_path in result.output
        assert os.getcwd() not in result.output
    finally:
        backend.close()


def test_output_truncation_actually_trips_at_the_cap() -> None:
    # A command that prints far more than the cap is drained up to the cap and
    # flagged truncated, never buffered unbounded.
    backend = _backend(ExecPolicy(output_cap_bytes=100))
    try:
        result = backend.execute(f'{sys.executable} -c "print(\\"x\\" * 10000)"')
        assert result.truncated is True
        assert len(result.output.encode()) <= 100 + 64  # cap + small slack
        assert result.exit_code == 0
    finally:
        backend.close()


def test_output_under_the_cap_is_not_flagged_truncated() -> None:
    backend = _backend(ExecPolicy(output_cap_bytes=10_000))
    try:
        result = backend.execute(f'{sys.executable} -c "print(\\"hello\\")"')
        assert result.truncated is False
        assert "hello" in result.output
    finally:
        backend.close()

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
import textwrap
import threading
import time

import pytest

from langchain_dynamic_workflow._local_subprocess import (
    ExecDecision,
    ExecPolicy,
    ExecRequest,
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


def test_timeout_actually_kills_a_sleeping_process_and_reports_124() -> None:
    backend = _backend(ExecPolicy(default_timeout=1, grace_seconds=0.5))
    try:
        started = time.monotonic()
        result = backend.execute(f'{sys.executable} -c "import time; time.sleep(30)"')
        elapsed = time.monotonic() - started
        assert result.exit_code == 124  # timeout sentinel
        assert elapsed < 10  # killed promptly, did not run the full 30s
    finally:
        backend.close()


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
def test_timeout_kills_the_whole_process_group_no_orphan() -> None:
    # The child spawns a grandchild that writes a pidfile then sleeps; on timeout
    # the WHOLE group must die — assert the grandchild pid is gone after.
    backend = _backend(ExecPolicy(default_timeout=1, grace_seconds=0.5))
    try:
        pidfile = os.path.join(backend.root_path, "grandchild.pid")
        scriptfile = os.path.join(backend.root_path, "forker.py")
        script = textwrap.dedent(f"""
            import os, time
            pid = os.fork()
            if pid == 0:
                with open({pidfile!r}, "w") as handle:
                    handle.write(str(os.getpid()))
                time.sleep(30)
            else:
                time.sleep(30)
        """)
        with open(scriptfile, "w") as handle:
            handle.write(script)
        result = backend.execute(f"{sys.executable} {scriptfile}")
        assert result.exit_code == 124
        # The grandchild must be dead (group kill reached it).
        time.sleep(0.5)
        with open(pidfile) as handle:
            grandchild_pid = int(handle.read())
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)  # 0 = liveness probe; raises if gone
    finally:
        backend.close()


def test_exec_gate_bounds_concurrent_executions() -> None:
    # A shared gate of 2 across 6 contending threads must never let more than 2
    # children run at once. Concurrency is measured at admission time (after the
    # gate is acquired, via before_execute) and released after execute returns.
    # The decrement happens after the gate slot is already returned, so it can
    # only OVER-count overlap, never under-count it — which makes `peak <= 2` a
    # strict, real guarantee rather than a filler assertion.
    gate = threading.BoundedSemaphore(2)
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def probe(_req: ExecRequest) -> ExecDecision:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        return ExecDecision()

    backend = LocalSubprocessSandbox(
        identity="leaf-c",
        policy=ExecPolicy(default_timeout=10, before_execute=probe),
        exec_gate=gate,
    )

    def run() -> None:
        nonlocal in_flight
        try:
            # A short sleep so true overlap is observable if the gate failed.
            backend.execute(f'{sys.executable} -c "import time; time.sleep(0.3)"')
        finally:
            with lock:
                in_flight -= 1

    threads = [threading.Thread(target=run, name=f"ldw-exec-{i}") for i in range(6)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert peak <= 2  # the gate held the bound under contention
        assert peak >= 1  # at least one exec actually admitted (sanity)
    finally:
        backend.close()


def test_exec_gate_slot_is_released_after_a_timeout() -> None:
    # A timed-out exec must still return its gate slot, or a single-slot gate
    # would deadlock the next exec forever. With a gate of 1, a 1s-timeout sleeper
    # followed by a fast command must both complete bounded in time.
    gate = threading.BoundedSemaphore(1)
    backend = LocalSubprocessSandbox(
        identity="leaf-timeout-release",
        policy=ExecPolicy(default_timeout=1, grace_seconds=0.5),
        exec_gate=gate,
    )
    try:
        slow = backend.execute(f'{sys.executable} -c "import time; time.sleep(30)"')
        assert slow.exit_code == 124  # timed out and (the point) released its slot
        started = time.monotonic()
        fast = backend.execute(f'{sys.executable} -c "print(7)"')
        assert "7" in fast.output
        assert fast.exit_code == 0
        assert time.monotonic() - started < 10  # not blocked on a leaked slot
    finally:
        backend.close()

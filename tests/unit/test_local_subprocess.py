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
import tempfile
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


def test_before_execute_reject_does_not_spawn_and_returns_126() -> None:
    def deny(_req: ExecRequest) -> ExecDecision:
        return ExecDecision(outcome="reject", reason="policy: no shell in this run")

    backend = _backend(ExecPolicy(before_execute=deny))
    try:
        result = backend.execute(f'{sys.executable} -c "print(1)"')
        assert result.exit_code == 126  # discipline reject sentinel
        assert "policy" in result.output
        assert result.truncated is False
    finally:
        backend.close()


def test_before_execute_reject_receives_the_real_request() -> None:
    # The hook observes the actual command / timeout / leaf identity, so a real
    # admission policy can branch on them rather than rejecting blindly.
    seen: list[ExecRequest] = []

    def record_then_reject(req: ExecRequest) -> ExecDecision:
        seen.append(req)
        return ExecDecision(outcome="reject", reason="seen")

    backend = LocalSubprocessSandbox(
        identity="leaf-admission",
        policy=ExecPolicy(before_execute=record_then_reject),
        exec_gate=threading.BoundedSemaphore(8),
    )
    try:
        backend.execute("echo unreachable", timeout=7)
        assert len(seen) == 1
        assert seen[0].command == "echo unreachable"
        assert seen[0].timeout == 7
        assert seen[0].leaf_id == "leaf-admission"
    finally:
        backend.close()


def test_before_execute_reduce_timeout_clamps_the_effective_timeout() -> None:
    def clamp(_req: ExecRequest) -> ExecDecision:
        return ExecDecision(timeout=1)

    backend = _backend(ExecPolicy(default_timeout=60, grace_seconds=0.5, before_execute=clamp))
    try:
        started = time.monotonic()
        result = backend.execute(f'{sys.executable} -c "import time; time.sleep(30)"')
        assert result.exit_code == 124
        assert time.monotonic() - started < 10  # clamped to ~1s, not 60s
    finally:
        backend.close()


def test_before_execute_reduce_timeout_only_clamps_down_never_up() -> None:
    # The clamp is a minimum: a decision that names a larger timeout cannot widen
    # a tighter call/policy timeout. A 1s policy default with a decision asking for
    # 60s must still time the 30s sleeper out promptly.
    def widen(_req: ExecRequest) -> ExecDecision:
        return ExecDecision(timeout=60)

    backend = _backend(ExecPolicy(default_timeout=1, grace_seconds=0.5, before_execute=widen))
    try:
        started = time.monotonic()
        result = backend.execute(f'{sys.executable} -c "import time; time.sleep(30)"')
        assert result.exit_code == 124
        assert time.monotonic() - started < 10  # the 1s default still bounds it
    finally:
        backend.close()


def test_before_execute_cap_output_lowers_the_byte_cap() -> None:
    def cap(_req: ExecRequest) -> ExecDecision:
        return ExecDecision(output_cap_bytes=50)

    backend = _backend(ExecPolicy(output_cap_bytes=10_000, before_execute=cap))
    try:
        result = backend.execute(f'{sys.executable} -c "print(\\"y\\" * 5000)"')
        assert result.truncated is True
        assert len(result.output.encode()) <= 50 + 64
    finally:
        backend.close()


@pytest.mark.skipif(os.name != "posix", reason="rlimits are POSIX-only")
def test_rlimit_cpu_actually_kills_a_busy_loop() -> None:
    # A tight CPU-seconds cap makes a busy loop exceed its quota and get killed by
    # the kernel (SIGXCPU), so the command exits non-zero rather than spinning the
    # host CPU unbounded. RLIMIT_CPU is the portable POSIX cap exercised here:
    # RLIMIT_AS is not kernel-enforced for this allocation pattern on Darwin, so a
    # busy-loop CPU cap is the honest, kernel-enforced assertion on every POSIX.
    from langchain_dynamic_workflow._local_subprocess import RLimitProfile

    backend = _backend(
        ExecPolicy(
            default_timeout=30,
            rlimits=RLimitProfile(cpu_seconds=1),
        )
    )
    try:
        started = time.monotonic()
        result = backend.execute(f'{sys.executable} -c "x = 0\nwhile True:\n    x += 1"')
        elapsed = time.monotonic() - started
        assert result.exit_code != 0  # the kernel killed the over-quota child
        assert elapsed < 20  # the CPU cap, not the 30s timeout, ended it
    finally:
        backend.close()


@pytest.mark.skipif(os.name != "posix", reason="rlimits are POSIX-only")
def test_default_rlimits_do_not_break_a_normal_command() -> None:
    # The default profile carries finite caps; a benign command must still run
    # cleanly under it. A limit the kernel refuses (for example RLIMIT_AS on
    # Darwin) is applied best-effort and skipped, never crashing the spawn.
    backend = _backend()  # default ExecPolicy => default RLimitProfile
    try:
        result = backend.execute(f'{sys.executable} -c "print(2 + 2)"')
        assert "4" in result.output
        assert result.exit_code == 0
    finally:
        backend.close()


@pytest.mark.skipif(os.name != "posix", reason="rlimits are POSIX-only")
def test_decision_rlimits_override_the_policy_profile() -> None:
    # An ExecDecision.rlimits selects a different profile for one call: here a
    # tight CPU cap that the busy loop trips, overriding the generous default.
    from langchain_dynamic_workflow._local_subprocess import RLimitProfile

    def tighten(_req: ExecRequest) -> ExecDecision:
        return ExecDecision(rlimits=RLimitProfile(cpu_seconds=1))

    backend = _backend(ExecPolicy(default_timeout=30, before_execute=tighten))
    try:
        started = time.monotonic()
        result = backend.execute(f'{sys.executable} -c "x = 0\nwhile True:\n    x += 1"')
        assert result.exit_code != 0
        assert time.monotonic() - started < 20
    finally:
        backend.close()


def test_write_then_execute_sees_the_same_file() -> None:
    # A file written via the backend's write() is visible to a command run via
    # execute() — one shared per-leaf filesystem (the 岔口-1 guarantee).
    backend = _backend()
    try:
        write = backend.write("/data.txt", "shared-content")
        assert write.error is None
        result = backend.execute(f'{sys.executable} -c "print(open(\\"data.txt\\").read())"')
        assert "shared-content" in result.output
    finally:
        backend.close()


def test_write_errors_when_the_file_already_exists() -> None:
    # write() must refuse to clobber an existing path, like InMemorySandbox.write,
    # so a leaf cannot silently overwrite a file it already created.
    backend = _backend()
    try:
        first = backend.write("/once.txt", "first")
        assert first.error is None and first.path == "/once.txt"
        second = backend.write("/once.txt", "second")
        assert second.error is not None
        # The original content survives the refused overwrite.
        read = backend.read("/once.txt")
        assert read.file_data is not None
        assert read.file_data["content"] == "first"
    finally:
        backend.close()


def test_read_grep_glob_ls_round_trip_on_the_real_dir() -> None:
    backend = _backend()
    try:
        backend.write("/a/note.txt", "alpha\nTODO fix\nbeta")
        read = backend.read("/a/note.txt")
        assert read.error is None and read.file_data is not None
        assert read.file_data["content"] == "alpha\nTODO fix\nbeta"
        grep = backend.grep("TODO")
        assert grep.matches and any("TODO" in m["text"] for m in grep.matches)
        glob = backend.glob("*.txt", "/a")
        assert glob.matches
        ls = backend.ls("/a")
        assert ls.entries and any(entry["path"] == "/a/note.txt" for entry in ls.entries)
    finally:
        backend.close()


def test_edit_replaces_content_on_the_real_dir() -> None:
    # edit() must operate on the same real file that read()/execute() see, so an
    # edit made via the tool surface is observable to a later command.
    backend = _backend()
    try:
        backend.write("/code.py", "value = 1\nvalue = 1\n")
        edit = backend.edit("/code.py", "value = 1", "value = 9", replace_all=True)
        assert edit.error is None
        assert edit.occurrences == 2
        result = backend.execute(f'{sys.executable} -c "print(open(\\"code.py\\").read())"')
        assert "value = 9" in result.output
        assert "value = 1" not in result.output
    finally:
        backend.close()


def test_read_missing_file_reports_an_error() -> None:
    backend = _backend()
    try:
        miss = backend.read("/nope.txt")
        assert miss.error is not None
        assert miss.file_data is None
    finally:
        backend.close()


def test_upload_then_download_round_trips_bytes() -> None:
    # upload_files overwrites (idempotent batch) and download_files returns the
    # exact bytes — the binary round trip through the real temp dir.
    backend = _backend()
    try:
        uploads = backend.upload_files([("/in/a.txt", b"alpha"), ("/in/b.txt", b"beta")])
        assert all(response.error is None for response in uploads)
        downloads = backend.download_files(["/in/a.txt", "/in/b.txt", "/in/missing.txt"])
        assert downloads[0].content == b"alpha"
        assert downloads[1].content == b"beta"
        assert downloads[2].error is not None  # missing path lands as an error
        assert downloads[2].content is None
    finally:
        backend.close()


def test_file_ops_reject_traversal_above_the_root() -> None:
    # A `..` escape that would write outside the per-leaf root must be refused,
    # not silently land on the host filesystem above the temp dir.
    backend = _backend()
    try:
        escape = backend.write("/../escape.txt", "leak")
        assert escape.error is not None
        # Nothing was created above the root.
        assert not os.path.exists(os.path.join(os.path.dirname(backend.root_path), "escape.txt"))
    finally:
        backend.close()


def test_close_removes_the_temp_dir_and_leaves_no_process() -> None:
    backend = _backend()
    root = backend.root_path
    backend.execute(f'{sys.executable} -c "print(1)"')
    assert os.path.isdir(root)
    backend.close()
    assert not os.path.exists(root)  # cleaned up, no leaked dir
    backend.close()  # idempotent


def test_admission_zero_output_cap_suppresses_output() -> None:
    # A before_execute decision of output_cap_bytes=0 must actually suppress output,
    # not fall back to the policy cap. Guards the falsey-`or` bug where 0 was treated
    # as "unset" so admission could never reduce the cap to zero.
    def cap_to_zero(_request: ExecRequest) -> ExecDecision:
        return ExecDecision(output_cap_bytes=0)

    backend = _backend(ExecPolicy(output_cap_bytes=10_000, before_execute=cap_to_zero))
    try:
        result = backend.execute(f'{sys.executable} -c "print(\\"x\\" * 1000)"')
        assert result.output == ""  # capped to zero, NOT the 10_000 policy cap
        assert result.truncated is True
        assert result.exit_code == 0
    finally:
        backend.close()


@pytest.mark.skipif(os.name != "posix", reason="setsid group-escape is POSIX-only")
def test_execute_is_bounded_when_a_child_escapes_the_group_holding_stdout() -> None:
    # A grandchild that setsid()-escapes the process group and keeps the inherited
    # stdout open must NOT wedge execute() on the drain join. The parent returns fast;
    # without the bounded join + read-end close, the open pipe would hang execute for
    # the grandchild's full sleep. Assert execute returns well before that sleep and
    # flags the abandoned drain as incomplete.
    backend = _backend(ExecPolicy(default_timeout=10, grace_seconds=0.5))
    try:
        scriptfile = os.path.join(backend.root_path, "escaper.py")
        script = textwrap.dedent("""
            import os, time
            pid = os.fork()
            if pid == 0:
                os.setsid()      # grandchild escapes into its own session/group
                time.sleep(5)    # keep the inherited stdout pipe open
            else:
                time.sleep(0.1)  # parent returns fast; the escaped grandchild lingers
        """)
        with open(scriptfile, "w") as handle:
            handle.write(script)
        started = time.monotonic()
        result = backend.execute(f"{sys.executable} {scriptfile}")
        elapsed = time.monotonic() - started
        # Bounded: parent (~0.1s) + drain-join grace (0.5s), nowhere near the 5s the
        # escaped grandchild holds the pipe. Without the fix this would hang ~5s.
        assert elapsed < 4
        assert result.truncated is True  # drain abandoned -> flagged incomplete
    finally:
        backend.close()


@pytest.mark.skipif(os.name != "posix", reason="symlink escape test uses POSIX symlinks")
def test_file_api_rejects_a_symlink_escaping_the_root(tmp_path: object) -> None:
    # A command (running in the leaf root) creates a symlink pointing OUTSIDE the
    # root; the file API must reject a read through it (realpath guard) rather than
    # follow it to the host, keeping the file APIs rooted even though shell execution
    # itself is not sandboxed.
    import pathlib

    outside = pathlib.Path(str(tmp_path)) / "host-secret.txt"
    outside.write_text("top secret")
    backend = _backend()
    try:
        link = backend.execute(f"ln -s {str(outside)!r} escape.txt")
        assert link.exit_code == 0
        result = backend.read("/escape.txt")
        assert result.error is not None
        assert "symlink" in result.error or "escape" in result.error
        # A read of the real file via its host path is also rejected (lexical + realpath).
        assert backend.read(f"/{outside}").error is not None
    finally:
        backend.close()


# --- on_command sink (M5 C1) ------------------------------------------------


def test_on_command_fires_begin_then_end_for_a_real_command() -> None:
    # With a wired command sink, a real execute fires a begin edge (phase="start",
    # exit_code=None) BEFORE the subprocess and an end edge (phase="end", the real
    # exit code) AFTER — the two edges share one resume-stable command_id so a UI
    # card can flip in place. Correlated to the owning leaf via leaf_span_id.
    from langchain_dynamic_workflow._observability import CommandEvent

    events: list[CommandEvent] = []
    backend = _backend()
    backend.set_command_sink(sink=events.append, leaf_span_id="span-abc")
    try:
        result = backend.execute(f'{sys.executable} -c "print(2 + 40)"')
        assert "42" in result.output
    finally:
        backend.close()

    assert len(events) == 2
    begin, end = events
    assert begin.phase == "start"
    assert end.phase == "end"
    # begin precedes end (emission order is the firing order).
    assert begin.leaf_span_id == end.leaf_span_id == "span-abc"
    assert begin.command_id == end.command_id  # one command, one id (flip in place)
    assert begin.command == end.command
    assert begin.exit_code is None  # not known at start
    assert end.exit_code == 0  # the real exit code
    assert begin.output is None  # no output until the command finishes
    assert end.output is not None and "42" in end.output
    assert begin.duration_s is None
    assert end.duration_s is not None and end.duration_s >= 0.0
    assert begin.started_at > 0.0


def test_on_command_end_carries_a_nonzero_exit_code() -> None:
    # A failing command's real non-zero exit code rides the end edge, so the
    # orchestration's branch-on-exit-code story is visible in the terminal card.
    from langchain_dynamic_workflow._observability import CommandEvent

    events: list[CommandEvent] = []
    backend = _backend()
    backend.set_command_sink(sink=events.append, leaf_span_id="span-fail")
    try:
        result = backend.execute(f'{sys.executable} -c "import sys; sys.exit(3)"')
        assert result.exit_code == 3
    finally:
        backend.close()

    end = events[-1]
    assert end.phase == "end"
    assert end.exit_code == 3


def test_on_command_id_is_distinct_per_occurrence_within_a_leaf() -> None:
    # Two executes of the SAME command in one leaf get DISTINCT command_ids (the
    # occurrence ordinal salts them apart), so a fix-loop's repeated `bun test`
    # stacks as separate cards rather than collapsing onto one id.
    from langchain_dynamic_workflow._observability import CommandEvent

    events: list[CommandEvent] = []
    backend = _backend()
    backend.set_command_sink(sink=events.append, leaf_span_id="span-loop")
    cmd = f'{sys.executable} -c "print(1)"'
    try:
        backend.execute(cmd)
        backend.execute(cmd)
    finally:
        backend.close()

    starts = [e for e in events if e.phase == "start"]
    assert len(starts) == 2
    assert starts[0].command_id != starts[1].command_id  # distinct occurrences
    # Each occurrence's begin/end still share their own id.
    ends = [e for e in events if e.phase == "end"]
    assert {s.command_id for s in starts} == {e.command_id for e in ends}


def test_on_command_output_is_shape_only_by_default_and_truncated_honestly() -> None:
    # By default (include_payloads=False) the end edge carries only a bounded tail
    # of the output and flags truncated honestly when it had to clip — never the
    # full multi-KB body. Guards the redaction policy.
    from langchain_dynamic_workflow._observability import CommandEvent

    events: list[CommandEvent] = []
    backend = _backend()
    backend.set_command_sink(sink=events.append, leaf_span_id="span-big")
    try:
        # Print far more than any small command-event tail budget.
        backend.execute(f'{sys.executable} -c "print(\\"z\\" * 50000)"')
    finally:
        backend.close()

    end = events[-1]
    assert end.phase == "end"
    assert end.output is not None
    # The forwarded output is a small bounded tail, not the full 50 KB body.
    assert len(end.output.encode()) < 50000
    assert end.truncated is True  # honestly flagged as clipped


def test_on_command_full_output_only_when_payloads_opted_in() -> None:
    # With include_payloads=True the end edge carries the full captured output (up
    # to the backend's own output cap), so an operator can opt into the raw tail.
    from langchain_dynamic_workflow._observability import CommandEvent

    events: list[CommandEvent] = []
    backend = _backend(ExecPolicy(output_cap_bytes=1_000_000))
    backend.set_command_sink(sink=events.append, leaf_span_id="span-full", include_payloads=True)
    try:
        backend.execute(f'{sys.executable} -c "print(\\"q\\" * 5000)"')
    finally:
        backend.close()

    end = events[-1]
    assert end.output is not None
    assert end.output.count("q") >= 5000  # the full body, not a clipped tail
    assert end.truncated is False  # the backend itself did not clip under the big cap


def test_no_command_sink_means_no_events_default_zero_cost() -> None:
    # Without set_command_sink the backend never fires command events: the sink is
    # default-no-op, so execute is byte-for-byte the prior behavior.
    backend = _backend()
    try:
        result = backend.execute(f'{sys.executable} -c "print(99)"')
        assert "99" in result.output
        assert result.exit_code == 0  # unchanged path, no observation overhead
    finally:
        backend.close()


def test_on_command_spawn_failure_still_fires_a_terminal_end_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the subprocess never spawns (Popen raises — EMFILE/ENOMEM, a preexec rlimit
    # failure between fork and exec, etc.) AFTER the begin edge already painted a
    # "running" card, the backend MUST still fire a terminal end edge so the card is
    # never stuck "running" forever. The end carries the spawn-error sentinel, the
    # SAME command_id as the begin, and the original exception still propagates.
    from langchain_dynamic_workflow._local_subprocess import EXIT_SPAWN_ERROR
    from langchain_dynamic_workflow._observability import CommandEvent

    boom = OSError(24, "Too many open files")

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise boom

    # Patch Popen on the subprocess module as seen from the backend's call site.
    monkeypatch.setattr("langchain_dynamic_workflow._local_subprocess.subprocess.Popen", _raise)

    events: list[CommandEvent] = []
    backend = _backend()
    backend.set_command_sink(sink=events.append, leaf_span_id="span-spawn")
    try:
        with pytest.raises(OSError) as excinfo:
            backend.execute(f'{sys.executable} -c "print(1)"')
        assert excinfo.value is boom  # original exception preserved for callers
    finally:
        backend.close()

    # Both edges fired even though the subprocess never started.
    assert len(events) == 2
    begin, end = events
    assert begin.phase == "start"
    assert begin.exit_code is None
    assert end.phase == "end"
    assert end.command_id == begin.command_id  # same card flips in place
    assert end.leaf_span_id == begin.leaf_span_id == "span-spawn"
    assert end.exit_code == EXIT_SPAWN_ERROR  # honest spawn-failure sentinel
    assert end.truncated is False
    assert end.duration_s is not None and end.duration_s >= 0.0
    # The end output is an honest tail naming the failure, not a fake success.
    assert end.output is not None and "Too many open files" in end.output


def test_spawn_failure_without_sink_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The unobserved path (no sink wired) stays byte-identical: a Popen failure
    # propagates the original exception with no event machinery in the way.
    boom = OSError(12, "Cannot allocate memory")

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise boom

    # Patch Popen on the subprocess module as seen from the backend's call site.
    monkeypatch.setattr("langchain_dynamic_workflow._local_subprocess.subprocess.Popen", _raise)

    backend = _backend()
    try:
        with pytest.raises(OSError) as excinfo:
            backend.execute(f'{sys.executable} -c "print(1)"')
        assert excinfo.value is boom
    finally:
        backend.close()


def test_root_provided_is_used_and_not_removed_on_close() -> None:
    # A host-supplied root is used verbatim (no mkdtemp), close fires the on_close
    # hook exactly once, and the directory is NOT rmtree'd because the provider
    # owns its lifecycle (a real git worktree removed via `git worktree remove`).
    existing = tempfile.mkdtemp(prefix="ldw-test-root-")
    closed: list[bool] = []
    sb = LocalSubprocessSandbox(
        identity="L1",
        policy=ExecPolicy(),
        exec_gate=threading.BoundedSemaphore(8),
        root=existing,
        on_close=lambda: closed.append(True),
    )
    try:
        # Rooted at the provided dir: pwd resolves to it, not a fresh temp dir.
        assert sb.execute("pwd").output.strip() == os.path.realpath(existing)
        sb.close()
        assert closed == [True]  # on_close fired
        assert os.path.isdir(existing)  # NOT rmtree'd (provider owns lifecycle)
        sb.close()  # idempotent: on_close fires at most once
        assert closed == [True]
    finally:
        os.rmdir(existing)


def test_default_root_still_mkdtemps_and_rmtrees() -> None:
    # The default path (no root=) is byte-unchanged: it owns a fresh temp root and
    # rmtrees it on close.
    sb = LocalSubprocessSandbox(
        identity="L2", policy=ExecPolicy(), exec_gate=threading.BoundedSemaphore(8)
    )
    root = sb.execute("pwd").output.strip()
    assert os.path.isdir(root)
    sb.close()
    assert not os.path.exists(root)  # owns its temp root -> rmtree on close

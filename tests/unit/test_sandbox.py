"""Unit tests for the sandbox layer — identity derivation + SandboxManager.

These tests pin the locked Phase 4 mechanics without any real sandbox
infrastructure: a leaf identity derived from the content-hash journal key (so it
is stable across retry/resume), find-or-create acquisition, tiered admission,
TTL/quota/backpressure lifecycle, the offline ``InMemorySandbox`` file surface
(grep/glob/upload/download), and the ``/shared/`` hand-off backend with ``..``
traversal blocked.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.state import StateBackend

from langchain_dynamic_workflow._journal import journal_key
from langchain_dynamic_workflow._sandbox import (
    InMemorySandbox,
    SandboxManager,
    _SandboxSlot,  # pyright: ignore[reportPrivateUsage]
    leaf_id_from_key,
)


class _FakeClock:
    """A manually-advanced monotonic clock for deterministic TTL tests."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _is_live(manager: SandboxManager, leaf_id: str) -> bool:
    """Whether ``leaf_id`` currently holds a live slot in ``manager``.

    Inspects the manager's slot map directly: there is no public per-leaf liveness
    query, and these tests need to assert *which* sandbox eviction reclaimed (not
    merely the count), to anti-corrupt the LRU-eviction policy.
    """
    return leaf_id in manager._slots  # pyright: ignore[reportPrivateUsage]


def test_leaf_id_is_derived_from_journal_key_and_is_stable() -> None:
    # The same leaf call (same content-hash key) must yield the same leaf_id on
    # every derivation, so retry/resume route to the same sandbox identity.
    key = journal_key(
        prompt="research X", agent_type="worker", model=None, schema=None, isolation="shared"
    )
    first = leaf_id_from_key(key)
    second = leaf_id_from_key(key)
    assert first == second


def test_leaf_id_differs_for_different_journal_keys() -> None:
    # Two distinct leaf calls (different keys) must get distinct identities so
    # their sandboxes never collide.
    key_a = journal_key(
        prompt="a", agent_type="worker", model=None, schema=None, isolation="shared"
    )
    key_b = journal_key(
        prompt="b", agent_type="worker", model=None, schema=None, isolation="shared"
    )
    assert leaf_id_from_key(key_a) != leaf_id_from_key(key_b)


def test_leaf_id_tracks_isolation_mode() -> None:
    # isolation participates in the journal key, so the same prompt under a
    # different isolation mode is a different leaf identity — the key-vs-execution
    # gap closes here: isolation actually partitions sandbox identity.
    shared = journal_key(
        prompt="a", agent_type="worker", model=None, schema=None, isolation="shared"
    )
    isolated = journal_key(
        prompt="a", agent_type="worker", model=None, schema=None, isolation="isolated"
    )
    assert leaf_id_from_key(shared) != leaf_id_from_key(isolated)


async def test_acquire_execution_leaf_returns_isolated_sandbox() -> None:
    # An execution leaf is allocated an isolated sandbox backend (one that can
    # run shell commands), not a shared state store.
    manager = SandboxManager()
    backend = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    assert isinstance(backend, SandboxBackendProtocol)
    await manager.stop("leaf-a")


async def test_acquire_reuses_same_backend_for_same_leaf_id() -> None:
    # find-or-create: the second acquire for an already-live leaf_id must return
    # the very same instance — stable identity across retry within a run.
    manager = SandboxManager()
    first = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    second = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    assert first is second
    await manager.stop("leaf-a")


async def test_acquire_lazy_creates_distinct_backends_per_leaf_id() -> None:
    manager = SandboxManager()
    a = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    b = manager.acquire(leaf_id="leaf-b", needs_execution=True)
    assert a is not b
    assert manager.active_count == 2
    await manager.stop("leaf-a")
    await manager.stop("leaf-b")


async def test_acquire_honors_max_active_by_evicting_idle_not_over_allocating() -> None:
    # The public acquire() primitive must carry the same max-active quota the
    # manager guarantees, minus blocking: at quota, admitting a NEW leaf evicts the
    # LRU idle sandbox rather than over-allocating past the cap. This pins that a
    # caller following the documented public interface gets bounded allocation, not
    # unbounded growth.
    clock = _FakeClock()
    manager = SandboxManager(max_active=2, clock=clock)
    manager.acquire(leaf_id="leaf-a", needs_execution=True)
    clock.advance(1.0)  # leaf-a now strictly less-recently-used than leaf-b
    manager.acquire(leaf_id="leaf-b", needs_execution=True)
    assert manager.active_count == 2
    # leaf-c is new work at quota: it must evict the LRU idle one (leaf-a), staying
    # at the cap rather than growing to 3.
    manager.acquire(leaf_id="leaf-c", needs_execution=True)
    assert manager.active_count == 2
    live = {leaf_id for leaf_id in ("leaf-a", "leaf-b", "leaf-c") if _is_live(manager, leaf_id)}
    assert live == {"leaf-b", "leaf-c"}
    await manager.stop("leaf-b")
    await manager.stop("leaf-c")


async def test_acquire_same_leaf_id_reuse_never_breaches_quota() -> None:
    # Reacquiring an already-live leaf_id at quota adds no new sandbox, so it must
    # always succeed and return the same instance — never trip the cap.
    manager = SandboxManager(max_active=1)
    first = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    second = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    assert first is second
    assert manager.active_count == 1
    await manager.stop("leaf-a")


async def test_acquire_reclaims_ttl_expired_idle_before_admitting_new_work() -> None:
    # At quota, acquire() first reclaims TTL-expired idle sandboxes (the clean path)
    # before resorting to eviction — so a new leaf reuses a freed slot rather than
    # evicting a still-valid one.
    clock = _FakeClock()
    manager = SandboxManager(max_active=1, idle_ttl=10.0, clock=clock)
    manager.acquire(leaf_id="leaf-a", needs_execution=True)
    clock.advance(11.0)  # leaf-a is now idle past its TTL
    manager.acquire(leaf_id="leaf-b", needs_execution=True)
    # leaf-a was reclaimed (TTL), leaf-b took the freed slot; cap held at 1.
    assert manager.active_count == 1
    assert not _is_live(manager, "leaf-a")
    assert _is_live(manager, "leaf-b")
    await manager.stop("leaf-b")


async def test_acquire_fails_loud_when_pool_full_of_in_use_sandboxes() -> None:
    # The synchronous primitive cannot park for a release, so when the pool is at
    # quota with every slot genuinely IN USE (an in-flight lease), acquire() must
    # fail loud rather than over-allocate — directing the caller to lease(), which
    # can wait. Mixing acquire with an in-flight lease is the only way to reach a
    # non-evictable full pool.
    manager = SandboxManager(max_active=1)

    async with manager.lease(leaf_id="leaf-busy", needs_execution=True):
        assert manager.active_count == 1
        with pytest.raises(RuntimeError, match="sandbox pool exhausted"):
            manager.acquire(leaf_id="leaf-new", needs_execution=True)
    # The in-use slot was never over-allocated past; after the lease exits the pool
    # holds only the original (now idle) sandbox.
    assert manager.active_count == 1
    await manager.stop("leaf-busy")


async def test_reasoning_leaf_uses_state_backend_and_is_not_allocated() -> None:
    # Tiered admission: a pure-reasoning leaf gets a StateBackend and is never
    # registered as an active sandbox — N logical agents != N active sandboxes.
    manager = SandboxManager()
    backend = manager.acquire(leaf_id="leaf-r", needs_execution=False)
    assert isinstance(backend, StateBackend)
    assert manager.active_count == 0


async def test_stop_releases_the_sandbox_slot() -> None:
    manager = SandboxManager()
    manager.acquire(leaf_id="leaf-a", needs_execution=True)
    assert manager.active_count == 1
    await manager.stop("leaf-a")
    assert manager.active_count == 0


async def test_stop_unknown_leaf_id_is_a_noop() -> None:
    # Stopping a leaf that was never allocated (e.g. a reasoning leaf) must not
    # raise — cleanup is idempotent.
    manager = SandboxManager()
    await manager.stop("never-allocated")
    assert manager.active_count == 0


async def test_lease_acquires_and_releases_a_slot() -> None:
    # The async lease is the engine-facing path: it acquires a backend for the
    # leaf and, once it exits, the sandbox remains live but idle (reusable).
    manager = SandboxManager(max_active=2)
    async with manager.lease(leaf_id="leaf-a", needs_execution=True) as backend:
        assert isinstance(backend, SandboxBackendProtocol)
        assert manager.active_count == 1
    # The slot is released for reuse but the sandbox persists for find-or-create.
    assert manager.active_count == 1
    await manager.stop("leaf-a")


async def test_idle_ttl_reclaims_expired_sandboxes() -> None:
    # A sandbox idle past its idle TTL is reclaimed on the next acquisition,
    # freeing its slot. The fake clock makes expiry deterministic.
    clock = _FakeClock()
    manager = SandboxManager(idle_ttl=10.0, clock=clock)
    async with manager.lease(leaf_id="leaf-a", needs_execution=True):
        pass
    assert manager.active_count == 1
    clock.advance(11.0)  # leaf-a is now idle past its TTL
    reclaimed = manager.reclaim_idle()
    assert reclaimed == 1
    assert manager.active_count == 0


async def test_hard_ttl_reclaims_even_recently_used_sandboxes() -> None:
    # The hard TTL caps a sandbox's total lifetime regardless of recent use, so a
    # long-lived-but-busy sandbox cannot live forever.
    clock = _FakeClock()
    manager = SandboxManager(idle_ttl=100.0, hard_ttl=10.0, clock=clock)
    async with manager.lease(leaf_id="leaf-a", needs_execution=True):
        pass
    clock.advance(11.0)  # past the hard TTL even though idle TTL has not elapsed
    reclaimed = manager.reclaim_idle()
    assert reclaimed == 1
    assert manager.active_count == 0


async def test_lease_evicts_lru_idle_sandbox_to_admit_new_work_at_quota() -> None:
    # At quota with NO in-flight lease (every slot held only for find-or-create
    # reuse), a distinct new leaf must be admitted by evicting the stalest idle
    # sandbox — not block forever. Without eviction the pool deadlocks once every
    # slot holds an idle-but-alive sandbox and no TTL reclaims it. The fake clock
    # makes "least-recently-used" deterministic: leaf-a is released first, so it is
    # the eviction target when leaf-c needs a slot.
    clock = _FakeClock()
    manager = SandboxManager(max_active=2, clock=clock)
    async with manager.lease(leaf_id="leaf-a", needs_execution=True):
        pass
    clock.advance(1.0)  # leaf-a now strictly less-recently-used than leaf-b
    async with manager.lease(leaf_id="leaf-b", needs_execution=True):
        pass
    assert manager.active_count == 2  # both idle but kept alive for reuse
    # leaf-c is new work; the pool is full of idle sandboxes. It must evict the
    # LRU idle one (leaf-a) and be admitted without blocking — never exceeding the
    # quota.
    async with manager.lease(leaf_id="leaf-c", needs_execution=True):
        assert manager.active_count == 2  # quota never breached
    # leaf-a was the eviction target (released earliest); leaf-b and leaf-c remain.
    live = {leaf_id for leaf_id in ("leaf-a", "leaf-b", "leaf-c") if _is_live(manager, leaf_id)}
    assert live == {"leaf-b", "leaf-c"}
    await manager.stop("leaf-b")
    await manager.stop("leaf-c")


async def test_evicted_then_reacquired_leaf_gets_a_fresh_workspace() -> None:
    # Eviction's contract limit: identity is stable across an evict+reacquire (same
    # leaf_id -> same id), but the *workspace* is not — a re-running evicted leaf
    # find-or-creates a brand-new, empty backend. This pins the qualified guarantee
    # so the identity-stability docs are not mistaken for workspace persistence.
    clock = _FakeClock()
    manager = SandboxManager(max_active=1, clock=clock)
    async with manager.lease(leaf_id="leaf-a", needs_execution=True) as first:
        assert isinstance(first, SandboxBackendProtocol)
        first.write("/state.txt", "v1")
    clock.advance(1.0)
    # leaf-b is new work at quota=1: it evicts the only idle sandbox (leaf-a).
    async with manager.lease(leaf_id="leaf-b", needs_execution=True):
        pass
    assert not _is_live(manager, "leaf-a")  # leaf-a was evicted
    await manager.stop("leaf-b")
    # leaf-a re-runs: SAME derived identity, but a FRESH instance with no prior
    # file state (the evicted workspace is gone).
    async with manager.lease(leaf_id="leaf-a", needs_execution=True) as reacquired:
        assert isinstance(reacquired, SandboxBackendProtocol)
        assert reacquired.id == "leaf-a"  # identity stable
        read = reacquired.read("/state.txt")
        assert read.file_data is None  # workspace did NOT survive eviction
    await manager.stop("leaf-a")


async def test_max_active_quota_applies_backpressure_until_a_slot_frees() -> None:
    # With the pool at its max-active quota and both slots held by in-flight
    # leases, a third lease must block (backpressure) until one of them exits —
    # never over-allocate past the cap.
    manager = SandboxManager(max_active=2)
    started = asyncio.Event()
    release_first = asyncio.Event()

    async def hold(leaf: str, *, gate: asyncio.Event | None = None) -> None:
        async with manager.lease(leaf_id=leaf, needs_execution=True):
            if gate is not None:
                await gate.wait()

    holder_a = asyncio.create_task(hold("leaf-a", gate=release_first))
    holder_b = asyncio.create_task(hold("leaf-b", gate=release_first))
    # Let both holders take their slots.
    while manager.active_count < 2:
        await asyncio.sleep(0)

    third_entered = asyncio.Event()

    async def third() -> None:
        async with manager.lease(leaf_id="leaf-c", needs_execution=True):
            third_entered.set()

    third_task = asyncio.create_task(third())
    # Give the third lease a chance to run; it must be parked on the full pool.
    await asyncio.sleep(0.02)
    assert not third_entered.is_set()
    assert manager.active_count == 2  # never exceeded the quota

    # Free the two holders; the third now gets a slot.
    release_first.set()
    await asyncio.wait_for(asyncio.gather(holder_a, holder_b), timeout=2.0)
    started.set()
    await asyncio.wait_for(third_task, timeout=2.0)
    assert third_entered.is_set()
    await manager.stop("leaf-c")


def test_inmemory_sandbox_grep_finds_literal_matches_scoped_and_globbed() -> None:
    # The offline sandbox must implement grep so a backend-aware leaf can search
    # its isolated workspace — not hit the protocol's NotImplementedError. Matching
    # is literal, scoped by path, and filtered by glob, returning (path, line, text).
    sandbox = InMemorySandbox(identity="s")
    sandbox.write("/src/a.py", "import os\nTODO: x\n")
    sandbox.write("/src/b.txt", "TODO: y\n")
    sandbox.write("/other/c.py", "TODO: z\n")
    # Scoped to /src and filtered to *.py: only /src/a.py's TODO line matches.
    result = sandbox.grep("TODO", "/src", "*.py")
    assert result.error is None
    assert result.matches is not None
    found = [(m["path"], m["line"], m["text"]) for m in result.matches]
    assert found == [("/src/a.py", 2, "TODO: x")]


def test_inmemory_sandbox_glob_matches_paths_under_base() -> None:
    sandbox = InMemorySandbox(identity="s")
    sandbox.write("/a.py", "x")
    sandbox.write("/sub/b.py", "y")
    sandbox.write("/sub/c.txt", "z")
    result = sandbox.glob("/sub/*.py", "/sub")
    assert result.error is None
    assert result.matches is not None
    assert [m["path"] for m in result.matches] == ["/sub/b.py"]


def test_inmemory_sandbox_upload_overwrites_and_download_round_trips() -> None:
    # upload_files overwrites (idempotent batch) and download_files round-trips
    # bytes; a missing download path is a per-entry file_not_found, not a raise.
    sandbox = InMemorySandbox(identity="s")
    first = sandbox.upload_files([("/data.txt", b"v1")])
    assert [r.error for r in first] == [None]
    # Re-upload overwrites without the write()-style "already exists" error.
    second = sandbox.upload_files([("/data.txt", b"v2")])
    assert [r.error for r in second] == [None]
    downloaded = sandbox.download_files(["/data.txt", "/missing.txt"])
    assert downloaded[0].error is None and downloaded[0].content == b"v2"
    assert downloaded[1].error == "file_not_found" and downloaded[1].content is None


def test_default_factory_still_yields_an_in_memory_sandbox() -> None:
    # The zero-dep default is unchanged: no factory ⇒ InMemorySandbox.
    manager = SandboxManager()
    backend = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    assert isinstance(backend, InMemorySandbox)


async def test_injected_factory_product_is_leased() -> None:
    # A custom factory's backend is what a leaf is handed (the pluggable seam).
    from langchain_dynamic_workflow._local_subprocess import (
        ExecPolicy,
        LocalSubprocessSandbox,
    )
    from langchain_dynamic_workflow._sandbox import local_subprocess_factory

    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))
    backend = manager.acquire(leaf_id="leaf-real", needs_execution=True)
    try:
        assert isinstance(backend, LocalSubprocessSandbox)
        assert isinstance(backend, SandboxBackendProtocol)
    finally:
        await manager.stop("leaf-real")


async def test_stop_closes_a_real_backend_temp_dir() -> None:
    # Teardown releases the real backend's resources (temp dir removed).
    import os

    from langchain_dynamic_workflow._local_subprocess import (
        ExecPolicy,
        LocalSubprocessSandbox,
    )
    from langchain_dynamic_workflow._sandbox import local_subprocess_factory

    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))
    backend = manager.acquire(leaf_id="leaf-real", needs_execution=True)
    assert isinstance(backend, LocalSubprocessSandbox)
    root = backend.root_path
    await manager.stop("leaf-real")
    assert not os.path.exists(root)  # stop() called close()


async def test_evicting_an_idle_real_backend_closes_its_temp_dir() -> None:
    # Quota-pressure eviction of an idle real backend must also release its temp
    # dir — eviction is a teardown path, not only stop().
    import os

    from langchain_dynamic_workflow._local_subprocess import (
        ExecPolicy,
        LocalSubprocessSandbox,
    )
    from langchain_dynamic_workflow._sandbox import local_subprocess_factory

    manager = SandboxManager(max_active=1, sandbox_factory=local_subprocess_factory(ExecPolicy()))
    first = manager.acquire(leaf_id="leaf-one", needs_execution=True)
    assert isinstance(first, LocalSubprocessSandbox)
    first_root = first.root_path
    # leaf-one is idle (acquire does not hold a lease), so admitting leaf-two at
    # max_active=1 evicts the LRU idle backend — which must close it.
    second = manager.acquire(leaf_id="leaf-two", needs_execution=True)
    assert isinstance(second, LocalSubprocessSandbox)
    assert not os.path.exists(first_root)  # evicted ⇒ closed
    await manager.stop("leaf-two")


def test_inmemory_sandbox_close_is_a_no_op() -> None:
    # InMemorySandbox gains a uniform close() so teardown need not special-case
    # the backend type; it must be a harmless no-op (idempotent).
    sandbox = InMemorySandbox(identity="s")
    sandbox.write("/keep.txt", "v")
    sandbox.close()
    sandbox.close()
    assert sandbox.read("/keep.txt").error is None


def test_real_execution_public_surface_is_exported() -> None:
    # A host opts into real execution from the package root, so every type it
    # needs to wire and tune a LocalSubprocessSandbox must be a top-level export.
    import langchain_dynamic_workflow as ldw

    for name in (
        "LocalSubprocessSandbox",
        "SandboxFactory",
        "local_subprocess_factory",
        "ExecPolicy",
        "ExecRequest",
        "ExecDecision",
        "RLimitProfile",
    ):
        assert hasattr(ldw, name), name
        assert name in ldw.__all__


class _BlockingCloseSandbox(InMemorySandbox):
    """An offline sandbox whose ``close`` blocks until a real threading event fires.

    Stands in for a real-git worktree backend whose ``close`` (``on_close`` ->
    ``GitWorktreeProvider.teardown``) runs blocking ``git`` subprocesses. It lets a
    test prove the manager offloads teardown off the event loop: ``close`` blocks a
    worker thread, not the loop.
    """

    def __init__(
        self, *, identity: str, started: threading.Event, release: threading.Event
    ) -> None:
        super().__init__(identity=identity)
        self._started = started
        self._release = release

    def close(self) -> None:
        # Signal we entered close, then block on a real (cross-thread) event — only a
        # to_thread offload can let the event loop keep running while this blocks.
        self._started.set()
        self._release.wait(timeout=10.0)


async def test_blocking_teardown_on_eviction_is_offloaded_off_the_event_loop() -> None:
    # FIX-2 (H3): a git-worktree backend's blocking teardown (close -> on_close ->
    # git subprocesses) runs during reclaim/evict inside the async admit path. It
    # MUST be thread-offloaded so the event loop keeps running; otherwise a single
    # blocking teardown wedges every other coroutine. With max_active=1, leasing a
    # second distinct leaf forces eviction of the first idle slot, whose close
    # blocks — a concurrent coroutine must still advance while close is in flight.
    #
    # Detection is wedge-proof: a separate OS thread watches the async-incremented
    # counter while close blocks. If the loop is offloaded, the counter advances
    # during the watch window (-> offloaded flag set); if the loop is wedged in a
    # synchronous close, the counter is frozen and the flag stays clear. The watcher
    # always releases the block, so a wedged run fails fast (assert) rather than
    # hanging on close's own timeout.
    started = threading.Event()
    release = threading.Event()
    offloaded = threading.Event()
    progress = 0

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        return _BlockingCloseSandbox(identity=leaf_id, started=started, release=release)

    def _watch() -> None:
        # Wait for the eviction close to begin blocking, sample the counter, then
        # give the loop a window. If it advanced, teardown was offloaded.
        if not started.wait(timeout=10.0):
            release.set()
            return
        before = progress
        threading.Event().wait(0.2)  # real sleep; independent of the event loop
        if progress > before:
            offloaded.set()
        release.set()

    manager = SandboxManager(max_active=1, sandbox_factory=factory)
    # Lease + release L1 so it becomes an idle slot eligible for eviction.
    async with manager.lease(leaf_id="L1", needs_execution=True):
        pass

    async def keep_advancing() -> None:
        nonlocal progress
        while not release.is_set():
            progress += 1
            await asyncio.sleep(0)

    async def lease_l2() -> str:
        async with manager.lease(leaf_id="L2", needs_execution=True) as backend:
            assert isinstance(backend, SandboxBackendProtocol)
            return backend.id

    watcher = threading.Thread(target=_watch, name="ldw-test-watch", daemon=True)
    watcher.start()
    advancer = asyncio.create_task(keep_advancing())
    leased_id = await asyncio.wait_for(lease_l2(), timeout=10.0)
    await asyncio.wait_for(advancer, timeout=10.0)
    watcher.join(timeout=10.0)
    assert leased_id == "L2"
    # The async counter advanced WHILE the eviction teardown was blocking in close
    # => teardown was thread-offloaded, not run on (and wedging) the event loop.
    assert offloaded.is_set(), "event loop was wedged by a blocking teardown under the slot lock"
    await manager.stop("L2")


class _TrackingSandbox(InMemorySandbox):
    """An offline sandbox that records its ``close`` calls into a shared list.

    Lets a test assert that a built backend was (or was not) closed by leaf
    identity, so a cancellation/leak guard can prove no backend is orphaned and no
    backend is double-closed on the happy / lost-race paths.
    """

    def __init__(self, *, identity: str, closed: list[str]) -> None:
        super().__init__(identity=identity)
        self._closed = closed

    def close(self) -> None:
        self._closed.append(self.id)


async def test_admit_slot_cancel_after_build_closes_backend_and_clears_pending() -> None:
    # A CancelledError delivered at the POST-build lock re-acquire (after the
    # backend is already built, before/at the slot install) must not leak the
    # built backend and must not strand the leaf id in _pending. Otherwise a real
    # git-worktree backend's on-disk worktree + branch orphan, and the stranded
    # pending permanently consumes a max_active slot — every future lease of the
    # same leaf id parks forever on the condition.
    built: list[str] = []
    closed: list[str] = []
    build_done = threading.Event()

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        # Runs inside asyncio.to_thread (off-loop). Sleep so the build is in flight
        # while the test grabs the post-build lock, then signal completion.
        built.append(leaf_id)
        threading.Event().wait(0.08)
        build_done.set()
        return _TrackingSandbox(identity=leaf_id, closed=closed)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def leaser() -> None:
        async with manager.lease(leaf_id="leafA", needs_execution=True):
            pass  # pragma: no cover - cancelled before the body runs

    task = asyncio.create_task(leaser())
    # Let the leaser reach the fast-path, mark pending, and enter to_thread.
    while "leafA" not in manager._pending:  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
    # Hold the condition lock so the coroutine's post-build re-acquire suspends.
    await manager._slot_freed.acquire()  # pyright: ignore[reportPrivateUsage]
    try:
        # Wait until the off-loop build actually produced the backend.
        while not build_done.is_set():
            await asyncio.sleep(0)
        # Cancel now: the coroutine is parked waiting for the post-build lock, so
        # the CancelledError lands at the post-build section — not in the build
        # try/except that only guards the to_thread call.
        task.cancel()
    finally:
        manager._slot_freed.release()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(asyncio.CancelledError):
        await task

    # No strand: the leaf id was discarded from _pending on the cancelled exit.
    assert "leafA" not in manager._pending  # pyright: ignore[reportPrivateUsage]
    # No leak: every built backend was either installed into a slot or closed.
    assert len(built) == len(closed) + len(manager._slots)  # pyright: ignore[reportPrivateUsage]
    assert closed == ["leafA"]  # the orphaned backend was closed exactly once
    # A future same-leaf lease must not park forever on the stranded pending; it
    # builds fresh and completes.
    async with manager.lease(leaf_id="leafA", needs_execution=True) as backend:
        assert isinstance(backend, SandboxBackendProtocol)
    await manager.stop("leafA")


async def test_admit_slot_second_cancel_during_orphan_close_still_cleans_up() -> None:
    # Edge case: a SECOND cancellation delivered while the orphan-close cleanup is
    # in flight (the off-loop close is blocking) must not re-strand the leaf id in
    # _pending nor leak the backend. The cleanup is driven to completion under a
    # cancellation shield, so the close finishes and _pending is cleared even though
    # the leaser is being cancelled twice over.
    built: list[str] = []
    closed: list[str] = []
    build_done = threading.Event()
    close_started = threading.Event()
    close_release = threading.Event()

    class _BlockingTrackingSandbox(InMemorySandbox):
        def close(self) -> None:
            close_started.set()
            # Block in the worker thread so a second cancel can land during close.
            close_release.wait(timeout=10.0)
            closed.append(self.id)

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        threading.Event().wait(0.05)
        build_done.set()
        return _BlockingTrackingSandbox(identity=leaf_id)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def leaser() -> None:
        async with manager.lease(leaf_id="leafA", needs_execution=True):
            pass  # pragma: no cover - cancelled before the body runs

    task = asyncio.create_task(leaser())
    while "leafA" not in manager._pending:  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
    await manager._slot_freed.acquire()  # pyright: ignore[reportPrivateUsage]
    try:
        while not build_done.is_set():
            await asyncio.sleep(0)
        task.cancel()  # first cancel: lands at the post-build lock re-acquire
    finally:
        manager._slot_freed.release()  # pyright: ignore[reportPrivateUsage]

    # Wait until the orphan-close has begun (and is blocking) off-loop.
    await asyncio.wait_for(asyncio.to_thread(close_started.wait, 5.0), timeout=5.0)
    task.cancel()  # second cancel: delivered DURING the off-loop close
    await asyncio.sleep(0)
    close_release.set()  # let the off-loop close finish

    with pytest.raises(asyncio.CancelledError):
        await task

    assert "leafA" not in manager._pending  # pyright: ignore[reportPrivateUsage]
    assert closed == ["leafA"]  # close ran to completion despite the second cancel
    assert not _is_live(manager, "leafA")  # no slot leaked


async def test_admit_slot_happy_path_installs_once_without_spurious_close() -> None:
    # The uncancelled happy path must install the built backend exactly once and
    # never close it on the way in — the cancellation guard must not regress a
    # clean lease into a spurious teardown.
    built: list[str] = []
    closed: list[str] = []

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        return _TrackingSandbox(identity=leaf_id, closed=closed)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)
    async with manager.lease(leaf_id="leafA", needs_execution=True) as backend:
        assert isinstance(backend, SandboxBackendProtocol)
        assert backend.id == "leafA"
    assert built == ["leafA"]
    assert closed == []  # installed, not closed
    assert _is_live(manager, "leafA")  # slot installed exactly once
    await manager.stop("leafA")
    assert closed == ["leafA"]  # closed once, only on teardown


async def test_admit_slot_lost_race_closes_redundant_backend_once_and_reuses_winner() -> None:
    # When a concurrent lease installs the slot while THIS lease's backend is still
    # building, the loser must close its now-redundant backend exactly once and
    # reuse the installed winner — never install a second slot, never leak. The
    # cancellation guard must not double-close the loser's backend on this path.
    built: list[str] = []
    closed: list[str] = []
    first_in_thread = threading.Event()
    release_build = threading.Event()

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        # The FIRST builder blocks until the test releases it; this widens the race
        # window so a second same-leaf lease can install the slot meanwhile.
        if len(built) == 1:
            first_in_thread.set()
            release_build.wait(timeout=10.0)
        return _TrackingSandbox(identity=leaf_id, closed=closed)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def lease_once() -> str:
        async with manager.lease(leaf_id="leafA", needs_execution=True) as backend:
            assert isinstance(backend, SandboxBackendProtocol)
            return backend.id

    first = asyncio.create_task(lease_once())
    # Let the first lease mark pending and enter its (blocked) build.
    await asyncio.wait_for(asyncio.to_thread(first_in_thread.wait, 10.0), timeout=10.0)
    # Force the first builder to "lose": drop its pending marker and install a
    # winning slot directly, exactly as a concurrent admit would have.
    winner = _TrackingSandbox(identity="leafA", closed=closed)
    async with manager._slot_freed:  # pyright: ignore[reportPrivateUsage]
        manager._pending.discard("leafA")  # pyright: ignore[reportPrivateUsage]
        manager._slots["leafA"] = _SandboxSlot(  # pyright: ignore[reportPrivateUsage]
            sandbox=winner, created_at=0.0, last_used_at=0.0
        )
    # Now let the first builder finish; it must detect the installed winner.
    release_build.set()
    leased_id = await asyncio.wait_for(first, timeout=10.0)

    assert leased_id == "leafA"
    assert built == ["leafA"]  # only the first builder ran its factory
    assert closed == ["leafA"]  # its redundant backend closed once (not the winner)
    assert manager._slots["leafA"].sandbox is winner  # pyright: ignore[reportPrivateUsage]
    assert manager._slots["leafA"].in_use == 0  # pyright: ignore[reportPrivateUsage]
    await manager.stop("leafA")


async def test_admit_slot_lost_race_redundant_close_runs_off_the_event_loop() -> None:
    # R8: the lost-race redundant-backend close must NOT run synchronously on the
    # event loop / under the admission lock — a real git-worktree backend's close
    # runs blocking git subprocesses. Prove it runs on a worker thread (the asyncio
    # to_thread offload), not the loop thread.
    loop_thread = threading.get_ident()
    closed_on: list[int] = []
    built: list[str] = []
    first_in_thread = threading.Event()
    release_build = threading.Event()

    class _ThreadRecordingSandbox(InMemorySandbox):
        def close(self) -> None:
            closed_on.append(threading.get_ident())

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        if len(built) == 1:
            first_in_thread.set()
            release_build.wait(timeout=10.0)
        return _ThreadRecordingSandbox(identity=leaf_id)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def lease_once() -> str:
        async with manager.lease(leaf_id="leafA", needs_execution=True) as backend:
            assert isinstance(backend, SandboxBackendProtocol)
            return backend.id

    first = asyncio.create_task(lease_once())
    await asyncio.wait_for(asyncio.to_thread(first_in_thread.wait, 10.0), timeout=10.0)
    winner = _ThreadRecordingSandbox(identity="leafA")
    async with manager._slot_freed:  # pyright: ignore[reportPrivateUsage]
        manager._pending.discard("leafA")  # pyright: ignore[reportPrivateUsage]
        manager._slots["leafA"] = _SandboxSlot(  # pyright: ignore[reportPrivateUsage]
            sandbox=winner, created_at=0.0, last_used_at=0.0
        )
    release_build.set()
    await asyncio.wait_for(first, timeout=10.0)

    assert len(closed_on) == 1  # the redundant backend closed exactly once
    assert closed_on[0] != loop_thread  # ... and OFF the event loop thread (R8)
    await manager.stop("leafA")


async def test_admit_slot_cancel_during_prebuild_victim_close_is_safe() -> None:
    # A CancelledError landing during the PRE-build off-loop victim close (the third
    # cancellation window) must not strand leaf_id in _pending and must not leave the
    # popped victim unclosed. The victim was already removed from _slots, so an
    # unclosed victim is a pure leak; the stranded pending would permanently consume
    # a max_active slot and park every future same-leaf lease forever.
    closed: list[str] = []
    victim_close_started = threading.Event()
    victim_close_release = threading.Event()
    build_started = threading.Event()

    class _BlockingVictimSandbox(InMemorySandbox):
        def close(self) -> None:
            # Block in the worker thread so a cancel can land while the victim close
            # is in flight (the pre-build _close_backends_off_loop await).
            victim_close_started.set()
            victim_close_release.wait(timeout=10.0)
            closed.append(self.id)

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        if leaf_id == "leafA":
            build_started.set()  # pragma: no cover - the leaser is cancelled first
        return _BlockingVictimSandbox(identity=leaf_id)

    # max_active=1 so leasing leafA must evict the idle victim leafV.
    manager = SandboxManager(max_active=1, sandbox_factory=factory)
    # Seed an idle victim slot (leased + released -> idle, eligible for eviction).
    async with manager.lease(leaf_id="leafV", needs_execution=True):
        pass
    assert _is_live(manager, "leafV")

    async def leaser() -> None:
        async with manager.lease(leaf_id="leafA", needs_execution=True):
            pass  # pragma: no cover - cancelled during the pre-build victim close

    task = asyncio.create_task(leaser())
    # Wait until leafA has claimed pending and the victim close is in flight.
    await asyncio.wait_for(asyncio.to_thread(victim_close_started.wait, 10.0), timeout=10.0)
    assert "leafA" in manager._pending  # pyright: ignore[reportPrivateUsage]
    # Cancel now: the CancelledError lands at the pre-build victim-close await.
    task.cancel()
    await asyncio.sleep(0)
    victim_close_release.set()  # let the (drained) victim close finish

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not build_started.is_set()  # cancelled before the leafA build ran
    assert "leafA" not in manager._pending  # pyright: ignore[reportPrivateUsage]
    assert "leafV" in closed  # the popped victim was closed (no leak)
    assert not _is_live(manager, "leafA")  # no slot installed for the cancelled leaf
    # No permanent strand: a fresh same-leaf lease proceeds (does not park forever).
    async with manager.lease(leaf_id="leafA", needs_execution=True) as backend:
        assert isinstance(backend, SandboxBackendProtocol)
    await manager.stop("leafA")


async def test_admit_slot_lost_race_close_failure_rebalances_winner_in_use() -> None:
    # [P1 #2] If the lost-race redundant-backend close RAISES, _admit_slot raises
    # before lease() reaches its finally, so the finally cannot release the winner's
    # optimistic in_use bump. The handler must release that bump itself before
    # propagating — otherwise the winner (someone else's live slot) keeps a phantom
    # in_use forever, blocking its eviction/reclaim and parking future leases.
    built: list[str] = []
    first_in_thread = threading.Event()
    release_build = threading.Event()

    class _RaisingCloseSandbox(InMemorySandbox):
        def close(self) -> None:
            raise RuntimeError("redundant close failed (e.g. git teardown error)")

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        if len(built) == 1:
            first_in_thread.set()
            release_build.wait(timeout=10.0)
        # The first builder's backend (the redundant one) raises on close.
        return _RaisingCloseSandbox(identity=leaf_id)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def lease_once() -> None:
        async with manager.lease(leaf_id="leafA", needs_execution=True):
            pass  # pragma: no cover - the redundant close raises before the body

    first = asyncio.create_task(lease_once())
    await asyncio.wait_for(asyncio.to_thread(first_in_thread.wait, 10.0), timeout=10.0)
    # Install a winning slot directly (the concurrent winner), with in_use=0.
    winner = InMemorySandbox(identity="leafA")
    async with manager._slot_freed:  # pyright: ignore[reportPrivateUsage]
        manager._pending.discard("leafA")  # pyright: ignore[reportPrivateUsage]
        manager._slots["leafA"] = _SandboxSlot(  # pyright: ignore[reportPrivateUsage]
            sandbox=winner, created_at=0.0, last_used_at=0.0
        )
    release_build.set()

    # The failing redundant close propagates as the lease's failure.
    with pytest.raises(RuntimeError, match="redundant close failed"):
        await asyncio.wait_for(first, timeout=10.0)

    # The winner's in_use bump was rebalanced: it is idle again (not stranded), so a
    # quota-pressure eviction can reclaim it and a fresh distinct-leaf lease proceeds.
    assert manager._slots["leafA"].in_use == 0  # pyright: ignore[reportPrivateUsage]
    await manager.stop("leafA")


async def test_admit_slot_lost_race_cancel_during_redundant_close_propagates_and_rebalances() -> (
    None
):
    # [P2 #3] A CancelledError landing during the lost-race redundant close must
    # PROPAGATE (a race() loser / cancelled background run must abort, not keep
    # running its leaf body) AND the winner's optimistic in_use bump must be released
    # (the caller never reaches lease()'s finally). Otherwise the cancel is swallowed
    # and/or the winner is stranded, parking future leases under quota.
    built: list[str] = []
    first_in_thread = threading.Event()
    release_build = threading.Event()
    redundant_close_started = threading.Event()
    redundant_close_release = threading.Event()

    class _BlockingCloseSandbox(InMemorySandbox):
        def close(self) -> None:
            redundant_close_started.set()
            redundant_close_release.wait(timeout=10.0)

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        if len(built) == 1:
            first_in_thread.set()
            release_build.wait(timeout=10.0)
        return _BlockingCloseSandbox(identity=leaf_id)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def lease_once() -> None:
        async with manager.lease(leaf_id="leafA", needs_execution=True):
            pass  # pragma: no cover - cancelled during the redundant close

    first = asyncio.create_task(lease_once())
    await asyncio.wait_for(asyncio.to_thread(first_in_thread.wait, 10.0), timeout=10.0)
    winner = InMemorySandbox(identity="leafA")
    async with manager._slot_freed:  # pyright: ignore[reportPrivateUsage]
        manager._pending.discard("leafA")  # pyright: ignore[reportPrivateUsage]
        manager._slots["leafA"] = _SandboxSlot(  # pyright: ignore[reportPrivateUsage]
            sandbox=winner, created_at=0.0, last_used_at=0.0
        )
    release_build.set()
    # Wait until the redundant close is in flight (off-loop), then cancel.
    await asyncio.wait_for(asyncio.to_thread(redundant_close_started.wait, 10.0), timeout=10.0)
    first.cancel()
    await asyncio.sleep(0)
    redundant_close_release.set()  # let the drained close finish

    # The observed cancellation propagates (not swallowed by the lost-race return).
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=10.0)

    # The winner's in_use bump was rebalanced despite the cancel.
    assert manager._slots["leafA"].in_use == 0  # pyright: ignore[reportPrivateUsage]
    await manager.stop("leafA")


async def test_admit_slot_reclaim_clears_pending_even_if_built_close_raises() -> None:
    # [P1 #1] A cancellation at the post-build install window triggers the generic
    # reclaim, which closes the built-but-uninstalled backend. If THAT close raises,
    # the reclaim must still discard _pending + notify (every close is best-effort) —
    # otherwise the leaf strands in _pending permanently (the original bug, resurrected
    # for the close-itself-fails branch).
    built: list[str] = []
    build_done = threading.Event()

    class _RaisingCloseSandbox(InMemorySandbox):
        def close(self) -> None:
            raise RuntimeError("built-backend close failed during reclaim")

    def factory(leaf_id: str) -> SandboxBackendProtocol:
        built.append(leaf_id)
        threading.Event().wait(0.08)
        build_done.set()
        return _RaisingCloseSandbox(identity=leaf_id)

    manager = SandboxManager(max_active=4, sandbox_factory=factory)

    async def leaser() -> None:
        async with manager.lease(leaf_id="leafA", needs_execution=True):
            pass  # pragma: no cover - cancelled at the post-build install window

    task = asyncio.create_task(leaser())
    while "leafA" not in manager._pending:  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
    await manager._slot_freed.acquire()  # pyright: ignore[reportPrivateUsage]
    try:
        while not build_done.is_set():
            await asyncio.sleep(0)
        task.cancel()  # lands at the post-build lock re-acquire -> generic reclaim
    finally:
        manager._slot_freed.release()  # pyright: ignore[reportPrivateUsage]

    # The original cancellation still propagates (the failing close does not mask it).
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10.0)

    # No strand despite the close failure: _pending was discarded, so a fresh same-leaf
    # lease proceeds rather than parking forever on the condition. (No stop() here: the
    # backend's close raises, and stop() closes synchronously; lease's own exit only
    # releases the lease, it does not close — so this exercises the no-strand fix
    # without tripping the raising teardown.)
    assert "leafA" not in manager._pending  # pyright: ignore[reportPrivateUsage]
    assert not _is_live(manager, "leafA")
    async with manager.lease(leaf_id="leafA", needs_execution=True) as backend:
        assert isinstance(backend, SandboxBackendProtocol)

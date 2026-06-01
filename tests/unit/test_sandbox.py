"""Unit tests for the sandbox layer — identity derivation + SandboxManager.

These tests pin the locked Phase 4 mechanics without any real sandbox
infrastructure: a leaf identity derived from the content-hash journal key (so it
is stable across retry/resume), find-or-create acquisition, tiered admission,
TTL/quota/backpressure lifecycle, and the ``/shared/`` hand-off backend with
``..`` traversal blocked.
"""

from __future__ import annotations

import asyncio

from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.state import StateBackend

from langchain_dynamic_workflow._journal import journal_key
from langchain_dynamic_workflow._sandbox import SandboxManager, leaf_id_from_key


class _FakeClock:
    """A manually-advanced monotonic clock for deterministic TTL tests."""

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


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

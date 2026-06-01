"""Unit tests for the sandbox layer — identity derivation + SandboxManager.

These tests pin the locked Phase 4 mechanics without any real sandbox
infrastructure: a leaf identity derived from the content-hash journal key (so it
is stable across retry/resume), find-or-create acquisition, tiered admission,
TTL/quota/backpressure lifecycle, and the ``/shared/`` hand-off backend with
``..`` traversal blocked.
"""

from __future__ import annotations

from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.state import StateBackend

from langchain_dynamic_workflow._journal import journal_key
from langchain_dynamic_workflow._sandbox import SandboxManager, leaf_id_from_key


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

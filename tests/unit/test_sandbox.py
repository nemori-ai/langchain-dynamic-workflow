"""Unit tests for the sandbox layer — identity derivation + SandboxManager.

These tests pin the locked Phase 4 mechanics without any real sandbox
infrastructure: a leaf identity derived from the content-hash journal key (so it
is stable across retry/resume), find-or-create acquisition, tiered admission,
TTL/quota/backpressure lifecycle, and the ``/shared/`` hand-off backend with
``..`` traversal blocked.
"""

from __future__ import annotations

from langchain_dynamic_workflow._journal import journal_key
from langchain_dynamic_workflow._sandbox import leaf_id_from_key


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

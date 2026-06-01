"""Unit tests for the ``/shared/`` artifact hand-off and path-traversal guard.

These pin the locked Phase 4 hand-off semantics: a producer leaf writes an
artifact under ``/shared/`` (into its own producer namespace), a consumer leaf
reads it back, per-leaf isolated paths never leak between sibling backends sharing
one shared store (the #2884 risk verified independently), and a ``..`` traversal
that tries to escape the shared route is normalized and blocked before routing.
"""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow._sandbox import (
    InMemorySandbox,
    SharedArtifactStore,
    build_leaf_backend,
    normalize_path,
)


def test_normalize_path_collapses_dot_segments() -> None:
    assert normalize_path("/shared/./a/b") == "/shared/a/b"
    assert normalize_path("/shared/a/../b") == "/shared/b"
    assert normalize_path("/a//b/") == "/a/b"


def test_normalize_path_rejects_escape_above_root() -> None:
    # A traversal that would climb above root must be rejected, not silently
    # clamped — this is the guard that stops "/shared/../secret" from escaping
    # the shared route into another backend's namespace.
    with pytest.raises(ValueError, match="escapes root"):
        normalize_path("/shared/../../etc/passwd")
    with pytest.raises(ValueError, match="escapes root"):
        normalize_path("/../escape")


async def test_shared_handoff_producer_writes_consumer_reads() -> None:
    # The hand-off contract: a producer writes under /shared/, a consumer with a
    # DIFFERENT isolated backend but the SAME shared store reads it back.
    store = SharedArtifactStore()
    producer = build_leaf_backend(
        isolated=InMemorySandbox(identity="prod"), shared_store=store, producer="prod"
    )
    consumer = build_leaf_backend(
        isolated=InMemorySandbox(identity="cons"), shared_store=store, producer="cons"
    )
    await producer.awrite("/shared/report.txt", "findings")
    read = await consumer.aread("/shared/report.txt")
    assert read.file_data is not None
    assert read.file_data["content"] == "findings"


async def test_isolated_paths_do_not_leak_across_shared_store() -> None:
    # #2884 verified independently: two leaves sharing one shared store must keep
    # their NON-shared (isolated) writes private. A file written outside /shared/
    # by one leaf must be invisible to the other, even though both composite
    # backends point at the same shared store.
    store = SharedArtifactStore()
    a = build_leaf_backend(isolated=InMemorySandbox(identity="a"), shared_store=store, producer="a")
    b = build_leaf_backend(isolated=InMemorySandbox(identity="b"), shared_store=store, producer="b")
    await a.awrite("/work/private.txt", "secret-a")
    # b cannot see a's isolated file.
    read = await b.aread("/work/private.txt")
    assert read.file_data is None
    assert read.error is not None


async def test_shared_writes_are_producer_namespaced() -> None:
    # Two producers writing the same /shared/ path must not clobber each other:
    # each lands in its own producer namespace under the shared store.
    store = SharedArtifactStore()
    a = build_leaf_backend(isolated=InMemorySandbox(identity="a"), shared_store=store, producer="a")
    b = build_leaf_backend(isolated=InMemorySandbox(identity="b"), shared_store=store, producer="b")
    await a.awrite("/shared/out.txt", "from-a")
    await b.awrite("/shared/out.txt", "from-b")
    # Distinct namespaces => both artifacts coexist in the shared store.
    assert store.read_namespaced("a", "/out.txt") == "from-a"
    assert store.read_namespaced("b", "/out.txt") == "from-b"


async def test_traversal_through_shared_route_is_blocked() -> None:
    # A leaf that tries to escape the shared route with ".." must be blocked at
    # the backend boundary rather than reaching another namespace.
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="x"), shared_store=store, producer="x"
    )
    res = await leaf.awrite("/shared/../escape.txt", "payload")
    assert res.error is not None
    assert "escapes root" in res.error

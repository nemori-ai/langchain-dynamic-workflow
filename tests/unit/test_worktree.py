"""Unit tests for the in-memory worktree provider (seed + diff collection).

``isolation="worktree"`` seeds a leaf's sandbox from a base snapshot and collects
the leaf's changeset relative to that seed. These tests pin both halves: a seed is
an isolated copy (mutating it never touches the base), and ``collect`` returns only
the paths a leaf added or modified.
"""

from __future__ import annotations

from langchain_dynamic_workflow import InMemoryWorktreeProvider


def test_seed_returns_isolated_base_snapshot_copy() -> None:
    base = {"/a.py": "print(1)\n", "/b.py": "x = 2\n"}
    provider = InMemoryWorktreeProvider(base)

    seeded = provider.seed("leaf-1")
    assert seeded == base

    # A leaf mutating its seeded copy must never leak back into the shared base.
    dict(seeded)["/a.py"] = "mutated"
    assert provider.seed("leaf-1")["/a.py"] == "print(1)\n"


def test_collect_returns_only_added_or_modified_paths() -> None:
    base = {"/a.py": "print(1)\n", "/b.py": "x = 2\n"}
    provider = InMemoryWorktreeProvider(base)

    after = {"/a.py": "print(2)\n", "/b.py": "x = 2\n", "/c.py": "new\n"}
    changes = provider.collect("leaf-1", after)

    # /a.py modified, /c.py added; unchanged /b.py is excluded.
    assert changes == {"/a.py": "print(2)\n", "/c.py": "new\n"}


def test_collect_is_empty_when_nothing_changed() -> None:
    base = {"/a.py": "print(1)\n"}
    provider = InMemoryWorktreeProvider(base)
    assert provider.collect("leaf-1", dict(base)) == {}


def test_base_is_isolated_from_constructor_argument() -> None:
    # Mutating the dict passed in must not change the provider's base snapshot.
    source = {"/a.py": "print(1)\n"}
    provider = InMemoryWorktreeProvider(source)
    source["/a.py"] = "tampered"
    assert provider.seed("leaf-1")["/a.py"] == "print(1)\n"

"""Unit tests for the content-hash journal."""

from __future__ import annotations

from pydantic import BaseModel

from langchain_dynamic_workflow import InMemoryJournalStore, journal_key


class _Schema(BaseModel):
    answer: str


def test_journal_key_is_stable_for_identical_inputs() -> None:
    a = journal_key(prompt="hi", agent_type="x", model=None, schema=None, isolation="shared")
    b = journal_key(prompt="hi", agent_type="x", model=None, schema=None, isolation="shared")
    assert a == b


def test_journal_key_changes_with_each_keyed_input() -> None:
    base = journal_key(prompt="hi", agent_type="x", model=None, schema=None, isolation="shared")
    assert (
        journal_key(prompt="bye", agent_type="x", model=None, schema=None, isolation="shared")
        != base
    )
    assert (
        journal_key(prompt="hi", agent_type="y", model=None, schema=None, isolation="shared")
        != base
    )
    assert (
        journal_key(prompt="hi", agent_type="x", model="opus", schema=None, isolation="shared")
        != base
    )
    assert (
        journal_key(prompt="hi", agent_type="x", model=None, schema=None, isolation="worktree")
        != base
    )
    assert (
        journal_key(prompt="hi", agent_type="x", model=None, schema=_Schema, isolation="shared")
        != base
    )


async def test_in_memory_store_roundtrip() -> None:
    store = InMemoryJournalStore()
    assert await store.get("k") is None
    await store.put("k", "value")
    assert await store.get("k") == "value"

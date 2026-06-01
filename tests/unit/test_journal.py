"""Unit tests for the content-hash journal."""

from __future__ import annotations

from pydantic import BaseModel

from langchain_dynamic_workflow import InMemoryJournalStore, journal_key
from langchain_dynamic_workflow._journal import JournalRecord


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
    await store.put("k", JournalRecord(result="value", usage=0))
    record = await store.get("k")
    assert record is not None
    assert record.result == "value"
    assert record.usage == 0


async def test_journal_record_roundtrips_result_and_usage() -> None:
    # The journal value carries both the folded result and the leaf's token
    # usage, so a resumed run can rebuild ``budget.spent()`` from the journal
    # alone (no model calls) and get the same cumulative total as the first run.
    store = InMemoryJournalStore()
    await store.put("k", JournalRecord(result="answer", usage=42))
    record = await store.get("k")
    assert record is not None
    assert record.result == "answer"
    assert record.usage == 42

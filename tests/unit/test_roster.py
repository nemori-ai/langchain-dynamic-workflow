"""Unit tests for the roster registry."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda

from langchain_dynamic_workflow import Roster


def _noop() -> RunnableLambda[dict[str, object], dict[str, object]]:
    return RunnableLambda(lambda x: x)


def test_register_returns_self_for_chaining() -> None:
    roster = Roster()
    assert roster.register("a", _noop()) is roster


def test_resolve_returns_entry_with_metadata() -> None:
    roster = Roster().register("researcher", _noop(), description="d", needs_execution=True)
    entry = roster.resolve("researcher")
    assert entry.name == "researcher"
    assert entry.description == "d"
    assert entry.needs_execution is True


def test_contains() -> None:
    roster = Roster().register("a", _noop())
    assert "a" in roster
    assert "b" not in roster


def test_resolve_unknown_raises_with_available_list() -> None:
    roster = Roster().register("alpha", _noop()).register("beta", _noop())
    with pytest.raises(KeyError, match="alpha"):
        roster.resolve("missing")

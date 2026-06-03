"""Unit tests for the roster registry."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

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


def test_register_runnable_and_builder_are_mutually_exclusive() -> None:
    def builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        del response_format
        return _noop()

    with pytest.raises(ValueError, match="exactly one"):
        Roster().register("x", _noop(), builder=builder)
    with pytest.raises(ValueError, match="exactly one"):
        Roster().register("x")


def test_runnable_for_no_schema_uses_runnable_entry() -> None:
    runnable = _noop()
    roster = Roster().register("plain", runnable)
    assert roster.runnable_for("plain", response_format=None) is runnable


def test_runnable_for_schema_on_runnable_only_entry_fails_loud() -> None:
    roster = Roster().register("plain", _noop())
    with pytest.raises(ValueError, match="builder"):
        roster.runnable_for("plain", response_format={"any": "fmt"})


def test_runnable_for_builds_and_caches_per_response_format() -> None:
    built: list[Any] = []

    def builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        leaf = _noop()
        built.append(response_format)
        return leaf

    roster = Roster().register("skeptic", builder=builder)
    fmt = {"k": "v"}
    first = roster.runnable_for("skeptic", response_format=fmt)
    second = roster.runnable_for("skeptic", response_format=fmt)
    assert first is second  # cached: built once for one response_format identity
    assert len(built) == 1


def test_runnable_for_builder_none_format_builds_once() -> None:
    calls = 0

    def builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        nonlocal calls
        calls += 1
        return _noop()

    roster = Roster().register("skeptic", builder=builder)
    roster.runnable_for("skeptic", response_format=None)
    roster.runnable_for("skeptic", response_format=None)
    assert calls == 1

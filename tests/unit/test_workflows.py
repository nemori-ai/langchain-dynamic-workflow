"""Unit tests for the named-workflow registry's discoverability surface.

The registry mirrors the leaf :class:`Roster` at the script level. These tests
cover the catalog enumeration a host LLM uses to discover registered workflows
without their names being hard-coded in its prompt: an explicit per-registration
``description``, a docstring-first-line fallback when none is given, an empty
summary when the callable has no docstring, sorted-by-name enumeration, and the
unchanged ``resolve`` / ``__contains__`` behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from langchain_dynamic_workflow import Ctx
from langchain_dynamic_workflow._workflows import WorkflowEntry, WorkflowRegistry


async def _noop(ctx: Ctx, args: dict[str, Any]) -> str:
    return "ok"


def test_register_with_explicit_description_is_listed() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        return "x"

    registry = WorkflowRegistry().register("alpha", wf, description="Does alpha things.")
    entries = registry.list_workflows()
    assert entries == [WorkflowEntry(name="alpha", description="Does alpha things.")]


def test_register_without_description_falls_back_to_docstring_first_line() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        """Summarize a topic into three bullets.

        A longer paragraph that must NOT bleed into the one-line summary.
        """
        return "x"

    registry = WorkflowRegistry().register("beta", wf)
    entries = registry.list_workflows()
    assert entries == [
        WorkflowEntry(name="beta", description="Summarize a topic into three bullets.")
    ]


def test_register_without_docstring_yields_empty_description() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        return "x"

    # The local coroutine above carries no docstring, so the fallback is "".
    registry = WorkflowRegistry().register("gamma", wf)
    entries = registry.list_workflows()
    assert entries == [WorkflowEntry(name="gamma", description="")]


def test_docstring_fallback_skips_leading_blank_lines() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        """

        First real line after blanks.
        """
        return "x"

    registry = WorkflowRegistry().register("delta", wf)
    entries = registry.list_workflows()
    assert entries == [WorkflowEntry(name="delta", description="First real line after blanks.")]


def test_explicit_description_overrides_docstring() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        """Docstring line that must be ignored when an explicit one is given."""
        return "x"

    registry = WorkflowRegistry().register("epsilon", wf, description="Explicit wins.")
    entries = registry.list_workflows()
    assert entries == [WorkflowEntry(name="epsilon", description="Explicit wins.")]


def test_list_workflows_is_sorted_by_name() -> None:
    registry = (
        WorkflowRegistry()
        .register("charlie", _noop, description="c")
        .register("alpha", _noop, description="a")
        .register("bravo", _noop, description="b")
    )
    names = [entry.name for entry in registry.list_workflows()]
    assert names == ["alpha", "bravo", "charlie"]


def test_register_returns_self_for_chaining() -> None:
    registry = WorkflowRegistry()
    assert registry.register("alpha", _noop) is registry


def test_resolve_and_contains_unchanged() -> None:
    registry = WorkflowRegistry().register("alpha", _noop, description="a")
    assert "alpha" in registry
    assert "missing" not in registry
    assert registry.resolve("alpha") is _noop
    with pytest.raises(KeyError):
        registry.resolve("missing")


def test_positional_register_signature_preserved() -> None:
    # Existing callers `register("name", fn)` must keep working unchanged.
    registry = WorkflowRegistry().register("alpha", _noop)
    assert "alpha" in registry
    assert registry.resolve("alpha") is _noop


def test_empty_registry_lists_nothing() -> None:
    assert WorkflowRegistry().list_workflows() == []


def test_whitespace_only_description_falls_back_to_docstring() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        """Real summary from the docstring."""
        return "x"

    # A blank / whitespace-only explicit description must not shadow the docstring.
    registry = WorkflowRegistry().register("alpha", wf, description="   \n\t ")
    entries = registry.list_workflows()
    assert entries == [WorkflowEntry(name="alpha", description="Real summary from the docstring.")]


def test_multiline_description_collapses_to_one_catalog_line() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        return "x"

    # A multi-line explicit description would otherwise inject fake catalog lines.
    registry = WorkflowRegistry().register(
        "alpha", wf, description="line one\nline two\n\tindented three"
    )
    (entry,) = registry.list_workflows()
    assert "\n" not in entry.description
    assert entry.description == "line one line two indented three"


def test_overlong_description_is_truncated() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        return "x"

    overlong = "z" * 5000
    registry = WorkflowRegistry().register("alpha", wf, description=overlong)
    (entry,) = registry.list_workflows()
    # A pathologically long description is bounded to a compact, ellipsized one-liner.
    assert entry.description.endswith("…")
    assert len(entry.description) < len(overlong)
    assert len(entry.description) <= 256


def test_non_string_docstring_degrades_to_empty() -> None:
    async def wf(ctx: Ctx, args: dict[str, Any]) -> str:
        return "x"

    # An exotic callable can carry a non-str __doc__; the summary must degrade to ""
    # rather than crashing on .splitlines().
    wf.__doc__ = 12345  # type: ignore[assignment]
    registry = WorkflowRegistry().register("alpha", wf)
    assert registry.list_workflows() == [WorkflowEntry(name="alpha", description="")]

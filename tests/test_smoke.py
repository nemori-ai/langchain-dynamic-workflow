"""Smoke tests validating the package installs and imports cleanly."""

from __future__ import annotations

import subprocess
import sys

import langchain_dynamic_workflow


def test_package_imports() -> None:
    """The top-level package imports without error."""
    assert langchain_dynamic_workflow is not None


def test_version_is_exposed() -> None:
    """``__version__`` is resolvable from installed package metadata."""
    assert isinstance(langchain_dynamic_workflow.__version__, str)
    assert langchain_dynamic_workflow.__version__


def test_reduce_helpers_exported_from_package_root() -> None:
    import langchain_dynamic_workflow as ldw

    for name in (
        "survives",
        "dedup",
        "reconcile",
        "corroborate",
        "ReviewItem",
        "Reconciled",
        "Consensus",
    ):
        assert name in ldw.__all__, f"{name} missing from __all__"
        assert hasattr(ldw, name), f"{name} not importable from the package root"


def test_race_surface_exported_from_package_root() -> None:
    import langchain_dynamic_workflow as ldw

    for name in ("RaceCandidate", "RaceResult", "race_key"):
        assert name in ldw.__all__, f"{name} missing from __all__"
        assert hasattr(ldw, name), f"{name} not importable from the package root"


def test_run_store_surface_exported_eagerly() -> None:
    """The dependency-free run-store surface is importable from the root.

    ``WorkflowRunStore``, ``RunSpec``, and ``InMemoryRunStore`` carry no optional
    dependency, so they are eagerly exported and must be present without touching
    the sqlite extra.
    """
    import langchain_dynamic_workflow as ldw

    for name in ("WorkflowRunStore", "RunSpec", "InMemoryRunStore"):
        assert name in ldw.__all__, f"{name} missing from __all__"
        assert hasattr(ldw, name), f"{name} not importable from the package root"


def test_sqlite_store_is_lazily_resolved_not_eagerly_imported() -> None:
    """``SqliteWorkflowStore`` is in ``__all__`` but resolves only on access.

    A bare ``import langchain_dynamic_workflow`` must not eagerly import the
    optional-dependency ``_persistence`` module, so the base install stays
    dependency-free. The symbol is declared in ``__all__`` for discoverability and
    materialized lazily through the package ``__getattr__`` on first access.
    """
    import langchain_dynamic_workflow as ldw

    assert "SqliteWorkflowStore" in ldw.__all__

    # In-process, the lazy attribute resolves to the concrete sqlite store class
    # (the [sqlite] extra is installed in dev/CI), exercising the __getattr__ hop.
    store_cls = ldw.SqliteWorkflowStore
    assert store_cls.__name__ == "SqliteWorkflowStore"

    # A fresh subprocess proves the eager import path never pulls _persistence in:
    # importing the package alone leaves _persistence absent from sys.modules,
    # while accessing the attribute then materializes it.
    probe = (
        "import sys, langchain_dynamic_workflow as ldw; "
        "assert 'langchain_dynamic_workflow._persistence' not in sys.modules, "
        "'_persistence imported eagerly'; "
        "store_cls = ldw.SqliteWorkflowStore; "
        "assert 'langchain_dynamic_workflow._persistence' in sys.modules, "
        "'attribute access did not import _persistence'; "
        "assert store_cls.__name__ == 'SqliteWorkflowStore'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_unknown_top_level_attribute_raises_attribute_error() -> None:
    """The lazy ``__getattr__`` still raises ``AttributeError`` for junk names."""
    import langchain_dynamic_workflow as ldw

    try:
        _ = ldw.NoSuchSymbol  # type: ignore[attr-defined]
    except AttributeError:
        pass
    else:  # pragma: no cover - the assertion below fails the test
        raise AssertionError("expected AttributeError for an unknown attribute")

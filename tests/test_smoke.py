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


def test_real_git_surface_exported_from_package_root() -> None:
    """The real-git worktree + PR seam is exported eagerly with no NEW dependency.

    ``GitWorktreeProvider`` / ``PullRequestProvider`` / ``PullRequestRef`` /
    ``LocalPullRequestProvider`` add no dependency beyond the existing base/eager
    stack the package already imports (``deepagents`` + the M5
    ``LocalSubprocessSandbox`` the git provider roots its backends in); the new code
    itself is plain ``subprocess`` + stdlib. So they are eagerly exported and
    importable from the package root on a base install.
    """
    import langchain_dynamic_workflow as ldw

    for name in (
        "GitWorktreeProvider",
        "PullRequestProvider",
        "PullRequestRef",
        "LocalPullRequestProvider",
    ):
        assert name in ldw.__all__, f"{name} missing from __all__"
        assert hasattr(ldw, name), f"{name} not importable from the package root"


def test_sqlite_store_is_lazily_resolved_not_eagerly_imported() -> None:
    """``SqliteWorkflowStore`` resolves only on access, via the package getattr.

    A bare ``import langchain_dynamic_workflow`` must not eagerly import the
    optional-dependency ``_persistence`` module, so the base install stays
    dependency-free. The symbol is materialized lazily through the package
    ``__getattr__`` on first access.
    """
    import langchain_dynamic_workflow as ldw

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


def test_sqlite_store_excluded_from_all_so_import_star_stays_base_safe() -> None:
    """``SqliteWorkflowStore`` is absent from ``__all__`` so ``import *`` is safe.

    A base install without the ``[sqlite]`` extra resolves ``import *`` through the
    package ``__getattr__``; were ``SqliteWorkflowStore`` listed in ``__all__``,
    the star import would trigger its lazy resolution and raise the optional-dep
    ``ImportError``, breaking a dependency-free install. Keeping it out of
    ``__all__`` (while leaving the ``__getattr__`` alias for explicit access and
    IDE discoverability) makes ``import *`` base-safe.
    """
    import langchain_dynamic_workflow as ldw

    assert "SqliteWorkflowStore" not in ldw.__all__
    # The lazy alias still resolves on explicit attribute access (the [sqlite]
    # extra is present in dev/CI), so discoverability is preserved.
    assert ldw.SqliteWorkflowStore.__name__ == "SqliteWorkflowStore"


def test_import_star_does_not_pull_persistence_eagerly() -> None:
    """``from langchain_dynamic_workflow import *`` leaves ``_persistence`` unloaded.

    Standing in for a base (no ``[sqlite]``) install: a star import must bind only
    the eager, dependency-free surface and must NOT materialize the optional
    sqlite module. A fresh subprocess proves ``_persistence`` is absent from
    ``sys.modules`` after the star import, so the same import on a base install
    would not hit the optional-dep ``ImportError``.
    """
    probe = (
        "import sys; "
        "ns: dict = {}; "
        "exec('from langchain_dynamic_workflow import *', ns); "
        "assert 'langchain_dynamic_workflow._persistence' not in sys.modules, "
        "'import * pulled _persistence eagerly'; "
        "assert 'SqliteWorkflowStore' not in ns, 'SqliteWorkflowStore leaked via import *'; "
        "assert 'RunSpec' in ns and 'InMemoryRunStore' in ns, 'base surface missing'"
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

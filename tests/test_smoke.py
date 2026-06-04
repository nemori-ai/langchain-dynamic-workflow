"""Smoke tests validating the package installs and imports cleanly."""

from __future__ import annotations

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

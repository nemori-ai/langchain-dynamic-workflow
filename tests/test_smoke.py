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

"""Architecture boundary guard — the import-linter contracts must hold.

The locked three-layer architecture (Layer 2 host-facing -> Layer 1 orchestration
runtime -> Layer 0 substrate binding) is mechanically enforced by import-linter
contracts declared in ``pyproject.toml``. Running ``lint-imports`` is a separate
quality gate, but pinning the contracts here means the boundary is also checked by
the test suite: a regression that couples the host-facing surface to engine
internals (or leaks LangGraph into the substrate-agnostic modules) fails a test,
not just an out-of-band lint someone might forget to run.
"""

from __future__ import annotations

from pathlib import Path

from importlinter.application.use_cases import lint_imports
from importlinter.configuration import configure

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_layer_import_contracts_hold() -> None:
    # Drive import-linter against the project's own pyproject.toml so the exact
    # contracts the project ships are the ones this test asserts — no duplicated
    # boundary definitions that could drift from the real config.
    configure()
    kept = lint_imports(config_filename=str(_PYPROJECT))
    assert kept is True, (
        "import-linter architecture contracts are broken — a layer boundary was "
        "violated. Run `uv run lint-imports` for the full per-contract report."
    )

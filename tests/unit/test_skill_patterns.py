"""Anti-corruption: every python code block in SKILL.md must parse and pass the AST gate.

The skill ships runnable orchestration patterns. If a documented pattern stops
parsing, or reaches for a construct the security gate forbids (an import, a dunder,
a banned builtin), a host that copies it would author a script the engine rejects.
These tests keep the documented patterns honest against the real gate.
"""

from __future__ import annotations

import ast
import re
import textwrap

from langchain_dynamic_workflow import skills_path
from langchain_dynamic_workflow._ast_gate import validate_workflow_source

_PY_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_SKILL_NAME = "dynamic-workflow"


def _skill_text() -> str:
    return (skills_path() / _SKILL_NAME / "SKILL.md").read_text(encoding="utf-8")


def _python_blocks() -> list[str]:
    return [textwrap.dedent(block) for block in _PY_BLOCK.findall(_skill_text())]


def test_skill_has_quality_patterns_section() -> None:
    # The quality-pattern library is the author wisdom that makes CC-grade
    # orchestration; its absence means the skill only teaches the mechanical basics.
    assert "## Quality patterns" in _skill_text()


def test_every_skill_python_block_parses_and_passes_gate() -> None:
    blocks = _python_blocks()
    assert blocks, "SKILL.md has no python code blocks to validate"
    for block in blocks:
        # Must be syntactically valid Python.
        ast.parse(block)
        # And must survive the same AST gate a run_script submission faces. Blocks
        # that are already a full orchestrate definition are checked as-is; bare
        # snippets are wrapped in the orchestrate skeleton the gate expects.
        source = (
            block
            if "def orchestrate" in block
            else "async def orchestrate(ctx, args):\n" + textwrap.indent(block, "    ")
        )
        validate_workflow_source(source)

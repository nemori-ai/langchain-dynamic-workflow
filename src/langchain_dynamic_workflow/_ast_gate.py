"""The AST security gate — static validation of untrusted orchestration source.

The meta layer turns an LLM-authored source string into a runnable orchestration
callable. Before that source is ever compiled or executed, it passes through this
gate: a single static walk of its abstract syntax tree that rejects the
constructs an escape attempt needs. The gate is the *security dimension* of the
meta layer and is applied only to untrusted (LLM-authored) source; a hand-written
callable handed straight to the engine bypasses it.

What the gate rejects:

- ``import`` / ``from ... import`` — no module may be pulled in (which, as a side
  effect, also denies ``time`` / ``random`` / ``os``).
- Dunder attribute access and dunder name references — closing the
  ``().__class__.__mro__.__subclasses__()`` escape chain and direct reach for
  ``__builtins__`` / ``__import__``.
- A fixed set of dangerous builtins (``eval`` / ``exec`` / ``open`` / ``getattr``
  / ``globals`` / ...) referenced by name.
- ``str.format`` / ``str.format_map`` — a format spec can traverse attributes
  (``"{0.__class__}".format(obj)``), a vector a source scan of the string literal
  cannot see; f-strings carry no such risk and are allowed.
- ``ctx.checkpoint`` — in-run human sign-off is a registered-workflow capability,
  not an authored-script one (an authored run has no resume lane and could park a
  quota slot indefinitely).

What the gate deliberately ignores: determinism. Non-determinism that changes the
observable ``agent()`` call pattern is caught at runtime by the journal-divergence
backstop, which applies to every source (hand-written or LLM-authored), so the
gate does not duplicate that check.

Security boundary: this gate plus a restricted-builtins ``exec`` namespace blocks
an honest model's slip, not a determined adversary — an in-process restricted
``exec`` is not a security sandbox. Only validate source you authored yourself.
"""

from __future__ import annotations

import ast
import re

from ._errors import WorkflowScriptError

_DUNDER_PATTERN = re.compile(r"^__\w+__$")
"""Matches a dunder identifier (e.g. ``__class__``, ``__builtins__``)."""

_BANNED_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "id",
        "hash",
        "input",
    }
)
"""Builtin names that are never permitted in an orchestration script."""

_BANNED_ATTRS = frozenset({"format", "format_map"})
"""String-formatting methods that can traverse attributes via a format spec."""

_BANNED_CTX_METHODS = frozenset({"checkpoint"})
"""``ctx`` methods an authored script may not call.

``ctx.checkpoint`` pauses a run for a human sign-off (parking it indefinitely). That
is a deliberate capability of a *registered* (trusted, hand-written) workflow, not of
an LLM-authored script: an authored run has no resume lane to be approved against and
could park a quota slot forever, so the gate denies it (defense-in-depth alongside the
reasoning-only roster, which already restricts which ``agent_type`` an authored script
can reach)."""


def validate_workflow_source(source: str) -> None:
    """Reject an untrusted orchestration script that violates the security gate.

    Parses ``source`` and walks its AST, collecting every security violation in a
    single pass. A syntax error or any violation raises
    :class:`~langchain_dynamic_workflow._errors.WorkflowScriptError`; the message
    enumerates all violations (with line numbers) so the author can fix them all
    at once. A clean script returns ``None``.

    Args:
        source: The orchestration script source to validate.

    Raises:
        WorkflowScriptError: If the source has a syntax error or contains any
            gate violation (an import, dunder access, banned builtin, or
            ``str.format`` attribute access).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkflowScriptError(
            f"the orchestration script has a syntax error: {exc.msg} (line {exc.lineno})"
        ) from exc

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.append((node.lineno, "import statement (imports are not allowed)"))
        elif isinstance(node, ast.ImportFrom):
            violations.append((node.lineno, "from-import statement (imports are not allowed)"))
        elif isinstance(node, ast.Attribute):
            if _DUNDER_PATTERN.match(node.attr):
                violations.append(
                    (node.lineno, f"dunder attribute access '.{node.attr}' (escape vector)")
                )
            elif node.attr in _BANNED_ATTRS:
                violations.append(
                    (node.lineno, f"str.{node.attr}() attribute access (use an f-string instead)")
                )
            elif node.attr in _BANNED_CTX_METHODS:
                violations.append(
                    (
                        node.lineno,
                        f"ctx.{node.attr}() is not allowed in an authored script "
                        "(in-run human sign-off is a registered-workflow capability)",
                    )
                )
        elif isinstance(node, ast.Name):
            if _DUNDER_PATTERN.match(node.id):
                violations.append(
                    (node.lineno, f"dunder name reference '{node.id}' (escape vector)")
                )
            elif node.id in _BANNED_NAMES:
                violations.append((node.lineno, f"banned builtin '{node.id}'"))

    if violations:
        violations.sort()
        listing = "\n".join(f"  - line {lineno}: {detail}" for lineno, detail in violations)
        raise WorkflowScriptError(
            "the orchestration script was rejected by the AST security gate; "
            f"fix every item and resubmit:\n{listing}"
        )

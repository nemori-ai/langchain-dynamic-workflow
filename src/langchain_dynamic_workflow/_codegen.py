"""The meta-layer codegen — the single, gated ``exec`` point of the library.

This module is where an untrusted source string becomes a runnable orchestration
callable, and it is the *only* place the library compiles or executes a string.
The flow is deliberately narrow: AST security gate, then ``compile``, then a
single ``exec`` into a namespace whose ``__builtins__`` is a curated safe
whitelist (never the real, escape-capable builtins), then the ``orchestrate``
coroutine is extracted and structurally validated.

The engine itself (``run_workflow``) never sees a source string — it only ever
runs a callable. That keeps the trusted execution core source-unaware and
confines ``exec`` to this one auditable seam: a hand-written callable handed
straight to the engine bypasses this module entirely (and with it the security
gate), while an LLM-authored string must pass through here first.

Security boundary: the AST gate plus the restricted-builtins namespace blocks an
honest model's slip, not a determined adversary. An in-process restricted ``exec``
is not a security sandbox — a sufficiently clever script can still escape. Only
compile source you authored yourself; for adversarial input run the engine behind
an out-of-process isolation backend.
"""

from __future__ import annotations

import ast
import builtins
import inspect
from typing import Any, cast

from ._ast_gate import validate_workflow_source
from ._engine import run_workflow
from ._errors import WorkflowScriptError
from ._reduce import (
    Consensus,
    Reconciled,
    ReviewItem,
    corroborate,
    dedup,
    reconcile,
    survives,
)
from ._roster import Roster
from ._workflows import WorkflowFn

_SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)
"""Builtin names a script may use — data/iteration helpers with no escape vector.

Deliberately excludes ``print`` (use ``ctx.log`` / ``ctx.phase``), ``type`` /
``object`` / ``super`` / ``getattr`` (type-graph and attribute escape vectors),
``eval`` / ``exec`` / ``compile`` / ``open`` / ``__import__`` / ``globals`` /
``locals`` / ``vars`` / ``input``, and ``__build_class__`` (which also denies
``class`` definitions in a script — an orchestration script needs none).
"""

_SAFE_BUILTINS: dict[str, Any] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
"""The exact ``__builtins__`` mapping bound into a compiled script's namespace."""

_SCRIPT_REDUCE_API: dict[str, Any] = {
    "survives": survives,
    "dedup": dedup,
    "reconcile": reconcile,
    "corroborate": corroborate,
    "ReviewItem": ReviewItem,
    "Reconciled": Reconciled,
    "Consensus": Consensus,
}
"""Cross-leaf reduce helpers injected as script globals so a host-authored script
calls them by name without an import (the AST gate forbids imports)."""


def compile_workflow_source(source: str) -> WorkflowFn:
    """Validate, compile, and execute an untrusted script into an orchestration callable.

    Runs ``source`` through the AST security gate, compiles it, executes it once in
    a namespace whose ``__builtins__`` is :data:`_SAFE_BUILTINS`, and returns the
    script's ``orchestrate`` coroutine. The returned callable has the same
    ``(ctx, args)`` signature as a registered workflow, so it can be launched
    through the same paths.

    Args:
        source: The orchestration script source. It must define a top-level
            ``async def orchestrate(ctx, args)`` coroutine.

    Returns:
        The script's ``orchestrate`` coroutine, ready to run under the engine.

    Raises:
        WorkflowScriptError: If the source fails the AST gate, has a syntax error,
            does not define ``orchestrate``, or defines it with the wrong shape
            (not a coroutine, or not exactly two positional parameters).
    """
    validate_workflow_source(source)
    code = compile(source, "<workflow-script>", "exec")
    # The single exec point of the library, reachable only after the gate passes.
    # The namespace's __builtins__ is the safe whitelist (Python leaves a provided
    # __builtins__ untouched), so the executed module — and the orchestrate closure
    # it defines, whose __globals__ is this namespace — can never reach the real
    # builtins.
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **_SCRIPT_REDUCE_API}
    exec(code, namespace)

    orchestrate = namespace.get("orchestrate")
    if orchestrate is None:
        raise WorkflowScriptError(
            "the script must define a top-level 'async def orchestrate(ctx, args)' "
            "coroutine, but no 'orchestrate' was defined"
        )
    if not inspect.iscoroutinefunction(orchestrate):
        raise WorkflowScriptError(
            "'orchestrate' must be declared with 'async def' (it is not a coroutine function)"
        )
    positional = [
        parameter
        for parameter in inspect.signature(orchestrate).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    ]
    if len(positional) != 2:
        raise WorkflowScriptError(
            "'orchestrate' must accept exactly two positional arguments (ctx, args); "
            f"got {len(positional)}"
        )
    return cast(WorkflowFn, orchestrate)


def extract_meta(source: str) -> dict[str, Any] | None:
    """Statically extract a top-level ``meta`` literal from a script, if present.

    Scans the module body for a top-level ``meta = {...}`` assignment and returns
    its value via :func:`ast.literal_eval`, so only a pure literal is accepted (no
    names, calls, or computed values). Returns ``None`` when the script has no
    top-level ``meta``. This is metadata-only — useful for labeling a run — and is
    independent of :func:`compile_workflow_source`.

    Args:
        source: The orchestration script source.

    Returns:
        The ``meta`` mapping if a pure-literal ``meta`` dict is present, else
        ``None``.

    Raises:
        WorkflowScriptError: If the source has a syntax error, or if ``meta`` is
            present but is not a pure literal dict.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkflowScriptError(
            f"the orchestration script has a syntax error: {exc.msg} (line {exc.lineno})"
        ) from exc
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not (isinstance(target, ast.Name) and target.id == "meta") or value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise WorkflowScriptError(
                "top-level 'meta' must be a pure literal (no names, calls, or computed values)"
            ) from exc
        if not isinstance(literal, dict):
            raise WorkflowScriptError(
                f"top-level 'meta' must be a dict literal; got {type(literal).__name__}"
            )
        return cast(dict[str, Any], literal)
    return None


async def run_workflow_from_source(
    source: str,
    *,
    roster: Roster,
    args: dict[str, Any] | None = None,
    **run_workflow_kwargs: Any,
) -> Any:
    """Compile an untrusted script and run it to completion through the engine.

    A thin convenience over :func:`compile_workflow_source` plus
    :func:`~langchain_dynamic_workflow.run_workflow`: it compiles ``source`` (AST
    gate + restricted ``exec``), binds ``args``, and runs the resulting
    orchestration callable. Any additional keyword arguments are forwarded to
    ``run_workflow`` (``journal`` / ``checkpointer`` / ``thread_id`` /
    ``max_concurrency`` / ``budget`` / ``on_progress`` / ``on_span`` /
    ``sandbox_manager`` / ``workflows``).

    Args:
        source: The orchestration script source defining ``orchestrate(ctx, args)``.
        roster: The leaf registry resolved by the script's ``agent()`` calls.
        args: Arguments passed to the script's ``orchestrate``; an empty mapping is
            used when omitted.
        **run_workflow_kwargs: Forwarded verbatim to ``run_workflow``.

    Returns:
        Whatever the compiled orchestration callable returns.

    Raises:
        WorkflowScriptError: If the source cannot be compiled (see
            :func:`compile_workflow_source`).
    """
    workflow_fn = compile_workflow_source(source)
    bound_args = args if args is not None else {}

    async def _orchestrate(ctx: Any) -> Any:
        return await workflow_fn(ctx, bound_args)

    return await run_workflow(_orchestrate, roster=roster, **run_workflow_kwargs)

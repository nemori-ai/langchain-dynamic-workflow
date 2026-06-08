"""Unit tests for the Layer 2 AST gate — the security wall over untrusted scripts.

The gate is the *security dimension* of the meta layer: it statically rejects an
LLM-authored orchestration script that reaches for an escape hatch (imports,
dunder attribute traversal, dangerous builtins, ``str.format`` attribute
injection) before the source is ever compiled or executed. These tests pin each
rejection and confirm that a realistic, well-behaved orchestration script passes
untouched. The gate deliberately does *not* judge determinism — that is the
runtime journal-divergence backstop's job — so these tests only assert security
behavior.
"""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow._ast_gate import validate_workflow_source
from langchain_dynamic_workflow._errors import WorkflowScriptError


def _wrap(body: str) -> str:
    """Indent ``body`` into a realistic ``async def orchestrate(ctx, args)`` script."""
    indented = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
    return f"async def orchestrate(ctx, args):\n{indented}\n"


_CLEAN_SCRIPT = """\
async def orchestrate(ctx, args):
    ctx.phase("research")
    topics = sorted(args["topics"])
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    surviving = [f for f in findings if f is not None]
    joined = "\\n".join(s.lower() for s in surviving)
    draft = await ctx.agent(f"Synthesize:\\n{joined}", agent_type="writer")
    while ctx.budget.remaining() > 500:
        critique = await ctx.agent(f"Critique: {draft}", agent_type="critic")
        if "looks good" in critique.lower():
            break
        draft = await ctx.agent(f"Revise per: {critique}", agent_type="writer")
    return draft
"""


def test_clean_realistic_script_passes() -> None:
    # A full parallel/pipeline/while-budget orchestration with f-strings,
    # comprehensions, sorted(), and str methods must pass the gate untouched.
    validate_workflow_source(_CLEAN_SCRIPT)  # no raise


def test_rejects_import_statement() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source("import os\n")
    assert "import" in str(exc.value).lower()


def test_rejects_from_import_statement() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source("from os import path\n")
    assert "import" in str(exc.value).lower()


def test_rejects_dunder_attribute_access() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source(_wrap("x = ().__class__\n"))
    assert "__class__" in str(exc.value)


def test_rejects_ctx_checkpoint_in_authored_script() -> None:
    # Review M3: ctx.checkpoint (in-run human sign-off) is a registered-workflow
    # capability, not an authored-script one — an authored run has no resume lane and
    # could park a quota slot forever. The gate denies it (defense-in-depth).
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source(_wrap("d = await ctx.checkpoint('approve?')\n"))
    assert "checkpoint" in str(exc.value)


def test_rejects_subclasses_escape_chain() -> None:
    # The classic sandbox escape: walk the type tree to reach arbitrary classes.
    with pytest.raises(WorkflowScriptError):
        validate_workflow_source(_wrap("evil = ().__class__.__mro__[1].__subclasses__()\n"))


def test_rejects_dunder_name_reference() -> None:
    # Referencing the real builtins table by its dunder name is an escape vector.
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source(_wrap("b = __builtins__\n"))
    assert "__builtins__" in str(exc.value)


@pytest.mark.parametrize(
    "snippet",
    [
        'x = eval("1")',
        'exec("x = 1")',
        'c = compile("1", "<s>", "eval")',
        'f = open("/etc/passwd")',
        'm = __import__("os")',
        'g = getattr(ctx, "agent")',
        'setattr(ctx, "x", 1)',
        'delattr(ctx, "x")',
        "d = globals()",
        "lo = locals()",
        "v = vars(ctx)",
        "i = id(ctx)",
        "h = hash(ctx)",
        "s = input()",
    ],
)
def test_rejects_each_banned_name(snippet: str) -> None:
    with pytest.raises(WorkflowScriptError):
        validate_workflow_source(_wrap(snippet + "\n"))


def test_rejects_str_format_attribute() -> None:
    # str.format with a format spec can traverse attributes ("{0.__class__}"),
    # a vector invisible to a source scan of the string literal — ban the method.
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source(_wrap('s = "{}".format(args)\n'))
    assert "format" in str(exc.value)


def test_rejects_format_map_attribute() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source(_wrap('s = "{x}".format_map(args)\n'))
    assert "format" in str(exc.value)


def test_allows_fstrings_and_str_methods() -> None:
    # f-strings compile to FORMAT_VALUE, not str.format, and ordinary str methods
    # other than format/format_map carry no attribute-traversal risk.
    validate_workflow_source(
        _wrap('s = f"{args}".lower().strip()\nj = "-".join(["a", "b"])\n')
    )  # no raise


def test_collects_all_violations_in_one_message() -> None:
    # A single pass enumerates every violation so the author can fix them all at
    # once rather than discovering them one rejection at a time.
    source = "import os\n" + _wrap('x = eval("1")\ny = ().__class__\n')
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source(source)
    message = str(exc.value)
    assert "import" in message.lower()
    assert "eval" in message
    assert "__class__" in message


def test_syntax_error_becomes_script_error() -> None:
    with pytest.raises(WorkflowScriptError) as exc:
        validate_workflow_source("async def orchestrate(:\n    pass\n")
    assert "syntax" in str(exc.value).lower()

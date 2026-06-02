"""Phase 8 integration: an LLM-authored script drives the real engine end to end.

These tests cross the full meta-layer seam — a source *string* compiled through the
AST gate + restricted ``exec`` into an orchestration callable, then run by the real
``run_workflow`` engine with fake leaves (no host agent, no API key). They pin that
a compiled ad-hoc script drives the genuine fan-out primitive, that its leaves
journal so a resume replays at zero model cost, and that the determinism backstop
covers the compiled path just as it covers a hand-written one.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import (
    InMemoryJournalStore,
    Roster,
    WorkflowDeterminismError,
    WorkflowScriptError,
    compile_workflow_source,
    run_workflow_from_source,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]
DeepLeafFactory = Callable[[str], tuple[Runnable[Any, Any], Any]]


def _load_example() -> ModuleType:
    """Import ``examples/08_meta_layer_run_script.py`` as a module (sibling import safe)."""
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "08_meta_layer_run_script.py"
    spec = importlib.util.spec_from_file_location("_ldw_meta_layer_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PARALLEL_SCRIPT = """\
async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    surviving = [f for f in findings if f is not None]
    return " | ".join(surviving)
"""

_SINGLE_SCRIPT = """\
async def orchestrate(ctx, args):
    return await ctx.agent("Capital of France?", agent_type="geographer")
"""


async def test_compiled_script_drives_parallel_fanout(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, state = make_fake_leaf("finding")
    roster = Roster().register("researcher", leaf)

    result = await run_workflow_from_source(
        _PARALLEL_SCRIPT, roster=roster, args={"topics": ["wind", "solar", "batteries"]}
    )

    # The compiled script fanned out one real leaf per topic through ctx.parallel.
    assert state.calls == 3
    assert result == "finding | finding | finding"


async def test_compiled_script_resume_replays_journal_zero_model_calls(
    make_deep_leaf: DeepLeafFactory,
) -> None:
    leaf, model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)
    journal = InMemoryJournalStore()

    first = await run_workflow_from_source(_SINGLE_SCRIPT, roster=roster, journal=journal)
    assert first == "Paris"
    calls_after_first = model.calls

    # Re-run the same source against the same journal: the leaf replays from cache.
    second = await run_workflow_from_source(_SINGLE_SCRIPT, roster=roster, journal=journal)
    assert second == "Paris"
    assert model.calls == calls_after_first  # zero additional model calls on resume


async def test_determinism_backstop_covers_compiled_scripts(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # The determinism oracle is not bypassed by the meta layer: a compiled script
    # that diverges from the recorded call sequence on replay fails loud rather
    # than serving a positionally misaligned cache entry. Script B issues the same
    # two leaf calls as script A but in the opposite order, so replaying it against
    # A's journal trips the backstop at the first call.
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("w", leaf)
    journal = InMemoryJournalStore()

    script_a = (
        "async def orchestrate(ctx, args):\n"
        '    a = await ctx.agent("first", agent_type="w")\n'
        '    b = await ctx.agent("second", agent_type="w")\n'
        "    return a + b\n"
    )
    script_b = (
        "async def orchestrate(ctx, args):\n"
        '    b = await ctx.agent("second", agent_type="w")\n'
        '    a = await ctx.agent("first", agent_type="w")\n'
        "    return b + a\n"
    )

    await run_workflow_from_source(script_a, roster=roster, journal=journal)
    with pytest.raises(WorkflowDeterminismError):
        await run_workflow_from_source(script_b, roster=roster, journal=journal)


async def test_example_authored_script_is_gate_valid_and_runs(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    # Guard the shipped demo: the script the example host eventually submits must
    # stay gate-valid and actually fan out + synthesize through the real engine.
    module = _load_example()
    researcher, researcher_state = make_fake_leaf("a finding")
    writer, _writer_state = make_fake_leaf("the recommendation")
    roster = Roster().register("researcher", researcher).register("writer", writer)

    result = await run_workflow_from_source(
        module.AUTHORED_SCRIPT, roster=roster, args={"topics": module.TOPICS}
    )

    assert researcher_state.calls == len(module.TOPICS)  # one researcher per topic
    assert result == "the recommendation"


def test_example_rejected_script_actually_trips_the_gate() -> None:
    # The example's "teachable" first script must really be rejected (for its
    # import), so the demo's feed-back-and-retry loop is not fiction.
    module = _load_example()
    with pytest.raises(WorkflowScriptError) as exc:
        compile_workflow_source(module.REJECTED_SCRIPT)
    assert "import" in str(exc.value).lower()

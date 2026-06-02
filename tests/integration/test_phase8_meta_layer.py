"""Phase 8 integration: an LLM-authored script drives the real engine end to end.

These tests cross the full meta-layer seam — a source *string* compiled through the
AST gate + restricted ``exec`` into an orchestration callable, then run by the real
``run_workflow`` engine with fake leaves (no host agent, no API key). They pin that
a compiled ad-hoc script drives the genuine fan-out primitive, that its leaves
journal so a resume replays at zero model cost, and that the determinism backstop
covers the compiled path just as it covers a hand-written one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import (
    InMemoryJournalStore,
    Roster,
    WorkflowDeterminismError,
    run_workflow_from_source,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]
DeepLeafFactory = Callable[[str], tuple[Runnable[Any, Any], Any]]

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

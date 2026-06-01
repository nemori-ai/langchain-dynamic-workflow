"""Phase 1 integration: a single ``agent()`` leaf, end-to-end, journaled & resumable.

These tests exercise the full ``run_workflow`` → ``@entrypoint`` → ``@task`` →
deepagent leaf → fold → journal path with a fake model (no API keys), so they
run green and deterministically in CI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow

DeepLeafFactory = Callable[[str], tuple[Runnable[Any, Any], Any]]
FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_single_deepagent_returns_folded_result(make_deep_leaf: DeepLeafFactory) -> None:
    leaf, _model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    result = await run_workflow(orchestrate, roster=roster, thread_id="t1")
    assert result == "Paris"


async def test_resume_hits_journal_zero_model_calls(make_deep_leaf: DeepLeafFactory) -> None:
    leaf, model = make_deep_leaf("Paris")
    roster = Roster().register("geographer", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    r1 = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    calls_after_first = model.calls
    # Fresh thread, SAME journal → content-hash cache hit, no new model call.
    r2 = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert r1 == r2 == "Paris"
    assert calls_after_first == 1
    assert model.calls == calls_after_first  # zero additional model calls on resume


async def test_journal_is_success_only(make_fake_leaf: FakeLeafFactory) -> None:
    # Leaf raises on its first invocation, succeeds on the second.
    leaf, state = make_fake_leaf("recovered", fail_times=1)
    roster = Roster().register("flaky", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("do it", agent_type="flaky")

    with pytest.raises(Exception):  # noqa: B017 - leaf failure propagates
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")

    # Failure was NOT cached: a retry re-runs the leaf and succeeds.
    result = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert result == "recovered"
    assert state.calls == 2


async def test_unknown_agent_type_raises(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("known", leaf)

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("q", agent_type="unknown")

    with pytest.raises(KeyError, match="unknown"):
        await run_workflow(orchestrate, roster=roster, thread_id="t1")

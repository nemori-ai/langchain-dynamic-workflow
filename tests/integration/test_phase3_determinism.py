"""Phase 3 integration: the journal-divergence determinism backstop.

These tests drive the full ``run_workflow`` path with a fake leaf (no API keys)
and prove that a replay whose ``agent()`` call sequence diverges from the first
run's recorded sequence fails loud with :class:`WorkflowDeterminismError`,
instead of silently serving a positionally misaligned cache entry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    WorkflowDeterminismError,
    run_workflow,
)

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_replay_same_sequence_passes(make_fake_leaf: FakeLeafFactory) -> None:
    # A deterministic script: the recorded call sequence is reproduced exactly on
    # replay, so the backstop stays silent and the journal serves the cache hits.
    leaf, state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[str]:
        return [
            await ctx.agent("a", agent_type="worker"),
            await ctx.agent("b", agent_type="worker"),
        ]

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert first == second == ["answer", "answer"]
    # Both leaves journaled on the first run; the replay served both from cache.
    assert state.calls == 2


async def test_replay_divergent_sequence_fails_loud(make_fake_leaf: FakeLeafFactory) -> None:
    # The script branches on external (non-journaled) state: the first run issues
    # one call sequence, the resume a different one. The backstop must raise
    # rather than feed a misaligned cache entry back into the orchestration.
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()
    branch = {"first": "x"}

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent(branch["first"], agent_type="worker")

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert first == "answer"

    # Flip the non-deterministic input so the replay produces a different call-key.
    branch["first"] = "y"
    with pytest.raises(WorkflowDeterminismError, match="diverged"):
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

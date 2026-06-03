"""Unit tests for ``Ctx.pipeline`` — delegation + journal coordination."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._roster import Roster


class _CountingLeaf:
    """A leaf runner that records prompts and counts invocations."""

    def __init__(self, *, prefix: str) -> None:
        self.calls = 0
        self.prefix = prefix

    async def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
    ) -> LeafOutcome:
        self.calls += 1
        return LeafOutcome(
            state={"messages": [AIMessage(content=f"{self.prefix}:{prompt}")]}, usage=0
        )


def _ctx(leaf: _CountingLeaf, journal: InMemoryJournalStore) -> Ctx:
    roster = Roster()
    roster.register("worker", object())  # type: ignore[arg-type]
    return Ctx(
        roster=roster,
        journal=journal,
        leaf_runner=leaf,
        gate=ConcurrencyGate(limit=8),
    )


async def test_pipeline_threads_agent_results_through_stages() -> None:
    leaf = _CountingLeaf(prefix="R")
    ctx = _ctx(leaf, InMemoryJournalStore())

    async def stage_one(prev: str, item: str, index: int) -> str:
        return await ctx.agent(f"study {item}", agent_type="worker")

    async def stage_two(prev: str, item: str, index: int) -> str:
        return await ctx.agent(f"summarize {prev}", agent_type="worker")

    results = await ctx.pipeline(["alpha", "beta"], stage_one, stage_two)
    assert results == [
        "R:summarize R:study alpha",
        "R:summarize R:study beta",
    ]
    # Two items x two stages = four leaf calls (no journal reuse, all distinct).
    assert leaf.calls == 4


async def test_pipeline_reuses_journal_for_repeated_leaf_calls() -> None:
    # Same journal => a repeated identical agent() call inside a stage is cached.
    journal = InMemoryJournalStore()
    leaf = _CountingLeaf(prefix="R")
    ctx = _ctx(leaf, journal)

    async def stage(prev: str, item: str, index: int) -> str:
        # Identical prompt for every item -> one miss, rest are journal hits.
        return await ctx.agent("constant prompt", agent_type="worker")

    results = await ctx.pipeline(["a", "b", "c"], stage)
    assert results == ["R:constant prompt"] * 3
    # Only the first item produced a real leaf call; the rest hit the journal.
    assert leaf.calls == 1


async def test_pipeline_stage_error_drops_item() -> None:
    leaf = _CountingLeaf(prefix="R")
    ctx = _ctx(leaf, InMemoryJournalStore())

    async def stage(prev: str, item: str, index: int) -> str:
        if item == "bad":
            raise RuntimeError("stage blew up")
        return await ctx.agent(f"ok {item}", agent_type="worker")

    results = await ctx.pipeline(["good", "bad", "fine"], stage)
    assert results == ["R:ok good", None, "R:ok fine"]

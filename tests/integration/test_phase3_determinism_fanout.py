"""Phase 3 integration: the determinism backstop under fan-out (parallel/pipeline).

The backstop records the ordered ``agent()`` call sequence on the first run and
replays it to catch divergence. Inside ``parallel()`` / ``pipeline()`` the observe
order is wall-clock-dependent — it follows per-leaf completion timing, not the
orchestration's source order — so recording fan-out calls would trip the backstop
spuriously on a perfectly deterministic resume.

These tests drive the full ``run_workflow`` path with fake, variable-latency
leaves (no API keys) and pin two things:

1. A fan-out workflow whose leaf *completion order* differs between the recording
   run and the resume still finalizes cleanly — no spurious
   :class:`WorkflowDeterminismError`. This is the regression these tests guard.
2. The persisted call sequence excludes fan-out leaves, so a sequential
   ``agent()`` outside the fan-out is still positionally guarded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    run_workflow,
)


class _OrderFlippingLeaf:
    """A fake leaf whose per-prompt latency flips between runs.

    On the first run a prompt at order index ``i`` sleeps proportionally to ``i``;
    on the resume it sleeps proportionally to ``prompt_count - 1 - i`` (reversed).
    This forces the leaf-completion order — and therefore the wall-clock
    ``agent()`` observe order inside a fan-out — to differ between the recording
    run and the resume, exactly the condition that made the finding's spurious
    failure possible. The leaf itself is content-deterministic: identical prompt
    in, identical reply out, so every leaf is a journal hit on resume.
    """

    def __init__(self, *, prompt_count: int) -> None:
        self.calls = 0
        self._prompt_count = prompt_count
        self._run_index = 0

    def as_runnable(self) -> Runnable[Any, Any]:
        async def _call(inp: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            messages = inp["messages"]
            prompt = messages[-1].content
            order_index = int(str(prompt).rsplit("#", 1)[-1])
            # Flip the latency profile between the first run and the resume so the
            # completion order is reversed run-to-run.
            position = order_index if self._run_index == 0 else self._prompt_count - 1 - order_index
            await asyncio.sleep(0.005 * position)
            return {"messages": [*messages, AIMessage(content=f"done:{prompt}")]}

        return RunnableLambda(_call)

    def begin_run(self) -> None:
        """Advance the run counter so the next pass uses the reversed latency."""
        self._run_index += 1


async def test_parallel_resume_with_flipped_completion_order_does_not_raise() -> None:
    # parallel() fans out leaves whose completion order is reversed on resume. If
    # the backstop recorded fan-out observe order, the resume's order would not
    # match the record and it would raise spuriously. The fix excludes fan-out
    # calls from the record, so the resume must finalize cleanly.
    item_count = 5
    leaf = _OrderFlippingLeaf(prompt_count=item_count)
    roster = Roster().register("worker", leaf.as_runnable())
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [lambda i=i: ctx.agent(f"task#{i}", agent_type="worker") for i in range(item_count)]
        )

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert first == [f"done:task#{i}" for i in range(item_count)]
    assert leaf.calls == item_count

    # Nothing sequential ran, so the recorded sequence is empty — fan-out leaves
    # are guarded by content hash, not by observe order.
    recorded = await journal.get_sequence()
    assert recorded == []

    # Resume with the latency profile flipped: every leaf is a journal hit, so the
    # completion order during replay differs but no model call happens. The
    # backstop must stay silent (no spurious WorkflowDeterminismError).
    leaf.begin_run()
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == first
    assert leaf.calls == item_count  # all served from the journal on resume


async def test_pipeline_resume_with_flipped_completion_order_does_not_raise() -> None:
    # pipeline() interleaves stage-1 and stage-2 agent() calls by per-item leaf
    # completion timing; that interleaving is exactly what differs run-to-run under
    # real latency. With completion order flipped on resume, the recorded-vs-replay
    # observe order would diverge if fan-out calls were recorded. The fix must keep
    # the resume green.
    item_count = 4
    leaf = _OrderFlippingLeaf(prompt_count=item_count)
    roster = Roster().register("worker", leaf.as_runnable())
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[Any | None]:
        async def stage_one(prev: str, item: str, index: int) -> str:
            # Encode the order index in the prompt so the leaf can flip its latency.
            return await ctx.agent(f"study {item}#{index}", agent_type="worker")

        async def stage_two(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"refine {prev}#{index}", agent_type="worker")

        return await ctx.pipeline(list(range(item_count)), stage_one, stage_two)

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    # Two stages x four items = eight leaf calls on the recording run.
    assert leaf.calls == item_count * 2
    assert len(first) == item_count
    assert all(r is not None for r in first)

    # Pipeline-internal calls are excluded from the recorded sequence.
    recorded = await journal.get_sequence()
    assert recorded == []

    # Resume with flipped latency: all eight leaves are journal hits, the
    # interleaving differs, and the backstop must not raise.
    leaf.begin_run()
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == first
    assert leaf.calls == item_count * 2  # no new model calls on resume


async def test_sequential_call_around_fanout_is_still_guarded(
    make_fake_leaf: Callable[..., tuple[Runnable[Any, Any], Any]],
) -> None:
    # The fan-out exclusion must not blind the backstop to the sequential path: an
    # agent() outside parallel()/pipeline() is still recorded and positionally
    # guarded. A deterministic replay of the sequential prefix passes; only the
    # fan-out internals are omitted from the record.
    leaf, _state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        lead = await ctx.agent("lead", agent_type="worker")  # sequential -> recorded
        fanned = await ctx.parallel(
            [lambda i=i: ctx.agent(f"fan{i}", agent_type="worker") for i in range(3)]
        )
        return {"lead": lead, "fanned": fanned}

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert first["lead"] == "answer"

    # Exactly one key recorded — the sequential lead — not the three fan-out leaves.
    recorded = await journal.get_sequence()
    assert recorded is not None
    assert len(recorded) == 1

    # A faithful replay passes (sequential prefix reproduced, fan-out unrecorded).
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == first

"""Phase 3 integration: concurrent depth-0 observe sites fail loud.

The determinism backstop records the ordered call sequence on the sequential
(depth-0) path and validates that order on resume. Three primitives observe into
that same ordered sequence at fan-out depth 0 — ``agent()``, ``race()`` and
``checkpoint()`` — and a hand-written orchestration that fans any of them out
concurrently at depth 0 via a raw ``await asyncio.gather(branch_a(ctx),
branch_b(ctx))`` observes its keys in wall-clock order, which flips run to run.
That would trip a *spurious* :class:`WorkflowDeterminismError` on a
logically-deterministic resume, leaving a correct workflow permanently
un-resumable.

The engine refuses this the moment two depth-0 observes overlap, on the *first*
run, with a :class:`WorkflowConcurrencyError` that steers the author to the
supported concurrency primitives (``ctx.parallel`` / ``ctx.dag`` / ``ctx.race``),
which mark their fan-out so leaf observe order is excluded from the positional
guard. A single depth-0 choke point (``Ctx._observe_depth0``) is shared by all
three sites, so guarding ``agent()`` alone is not enough — ``race()`` and
``checkpoint()`` would otherwise still race the shared sequence. These tests pin:

1. Raw concurrent depth-0 ``agent()`` fails loud on the first run with guidance.
2. The supported ``ctx.parallel`` pattern over the same work runs and resumes on
   the same journal with no spurious failure.
3. Two *sequential* depth-0 ``agent()`` calls record and resume cleanly — the
   in-flight counter introduces no false positive on the normal path.
4. Concurrent depth-0 ``race()`` (two races, and a race alongside an ``agent``)
   fails loud — the choke point covers ``race``'s observe, not just ``agent``'s.
5. ``checkpoint()`` shares the choke point: its observe + counter management run
   through ``_observe_depth0`` so a concurrent checkpoint would be caught too.
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
    RaceCandidate,
    Roster,
    WorkflowConcurrencyError,
    run_workflow,
)
from langchain_dynamic_workflow._context import LeafOutcome
from langchain_dynamic_workflow._errors import WorkflowSignoffRequired


class _OrderFlippingLeaf:
    """A fake leaf whose per-prompt latency flips between runs.

    On the first run a prompt at order index ``i`` sleeps proportionally to ``i``;
    on the resume it sleeps proportionally to ``prompt_count - 1 - i`` (reversed),
    forcing the leaf-completion order to differ run to run. The leaf is otherwise
    content-deterministic: identical prompt in, identical reply out, so every leaf
    is a journal hit on resume.
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
            position = order_index if self._run_index == 0 else self._prompt_count - 1 - order_index
            await asyncio.sleep(0.005 * position)
            return {"messages": [*messages, AIMessage(content=f"done:{prompt}")]}

        return RunnableLambda(_call)

    def begin_run(self) -> None:
        """Advance the run counter so the next pass uses the reversed latency."""
        self._run_index += 1


async def test_concurrent_depth0_agent_fails_loud_with_guidance() -> None:
    # Raw concurrent fan-out at depth 0: two branches each call ctx.agent inside a
    # bare asyncio.gather. The first branch increments the in-flight counter before
    # its first await (the journal lookup); the second branch then sees the counter
    # >= 1 and the engine fails loud on the FIRST run — converting what would be a
    # spurious resume-time WorkflowDeterminismError into a clear, actionable error.
    leaf = _OrderFlippingLeaf(prompt_count=2)
    roster = Roster().register("worker", leaf.as_runnable())
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> tuple[str, str]:
        async def branch(label: str) -> str:
            return await ctx.agent(f"task#{label}", agent_type="worker")

        # "0"/"1" so the order-flipping leaf can parse the trailing index.
        return await asyncio.gather(branch("0"), branch("1"))

    raised: WorkflowConcurrencyError | None = None
    try:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    except WorkflowConcurrencyError as exc:
        raised = exc

    assert raised is not None, "concurrent depth-0 agent() must fail loud on the first run"
    message = str(raised)
    # The guidance must name the supported concurrency primitives.
    assert "ctx.parallel" in message
    assert "fan-out depth 0" in message or "depth 0" in message


async def test_ctx_parallel_concurrent_agents_resume_without_false_positive() -> None:
    # The SUPPORTED pattern: the same concurrent work via ctx.parallel(), whose
    # frame marks the fan-out so the leaves are excluded from the positional guard.
    # The leaf-completion order is flipped on resume; the run must finalize on the
    # first run AND resume on the same journal without any spurious failure.
    item_count = 4
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

    # Fan-out leaves are excluded from the recorded sequence (guarded by content
    # hash, not observe order), so the persisted sequence is empty.
    recorded = await journal.get_sequence()
    assert recorded == []

    # Resume with the latency profile flipped: every leaf is a journal hit, the
    # completion order during replay differs, and no WorkflowConcurrencyError nor
    # spurious WorkflowDeterminismError is raised.
    leaf.begin_run()
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == first
    assert leaf.calls == item_count  # all served from the journal on resume


async def test_sequential_depth0_agents_resume_fine(
    make_fake_leaf: Callable[..., tuple[Runnable[Any, Any], Any]],
) -> None:
    # Regression guard for the in-flight counter: two SEQUENTIAL depth-0 agent()
    # calls record and resume cleanly with zero false detection. Each call
    # increments the counter, awaits its leaf, then decrements in finally BEFORE
    # the next call enters — so the counter is never >= 1 when the next call's
    # synchronous check runs. This must pass both before and after the fix.
    leaf, state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> list[str]:
        first = await ctx.agent("a", agent_type="worker")
        second = await ctx.agent("b", agent_type="worker")
        return [first, second]

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    assert first == ["answer", "answer"]
    assert state.calls == 2

    # Both keys recorded on the sequential path, in source order.
    recorded = await journal.get_sequence()
    assert recorded is not None
    assert len(recorded) == 2

    # A faithful replay reproduces the recorded sequence and serves both from cache
    # with no new model calls and no spurious detection.
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")
    assert second == first


async def test_concurrent_depth0_races_fail_loud() -> None:
    # Two depth-0 race() calls in a raw gather: each observes its race-key into the
    # SAME ordered determinism sequence at depth 0. Guarding agent() alone would
    # leave this racing the shared sequence; the shared _observe_depth0 choke point
    # catches it. RED before this fix (race() observed unguarded), GREEN after.
    leaf = _OrderFlippingLeaf(prompt_count=2)
    roster = Roster().register("worker", leaf.as_runnable())
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> tuple[Any, Any]:
        async def branch(label: str, tag: str) -> Any:
            return await ctx.race(
                [RaceCandidate(prompt=f"task#{label}", agent_type="worker")],
                win=lambda text: True,
                win_tag=tag,
            )

        return await asyncio.gather(branch("0", "a"), branch("1", "b"))

    raised: WorkflowConcurrencyError | None = None
    try:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    except WorkflowConcurrencyError as exc:
        raised = exc

    assert raised is not None, "two concurrent depth-0 race() calls must fail loud"
    assert "ctx.parallel" in str(raised)


async def test_concurrent_depth0_race_and_agent_fail_loud() -> None:
    # A depth-0 race() concurrent with a depth-0 agent(): both observe into the same
    # depth-0 sequence, so the overlap must be caught regardless of which site wins
    # the race to observe first. RED before (only agent() guarded), GREEN after.
    leaf = _OrderFlippingLeaf(prompt_count=2)
    roster = Roster().register("worker", leaf.as_runnable())
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> tuple[Any, Any]:
        async def race_branch() -> Any:
            return await ctx.race(
                [RaceCandidate(prompt="task#0", agent_type="worker")],
                win=lambda text: True,
                win_tag="r",
            )

        async def agent_branch() -> str:
            return await ctx.agent("task#1", agent_type="worker")

        return await asyncio.gather(race_branch(), agent_branch())

    raised: WorkflowConcurrencyError | None = None
    try:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    except WorkflowConcurrencyError as exc:
        raised = exc

    assert raised is not None, "a depth-0 race() concurrent with a depth-0 agent() must fail loud"
    assert "ctx.parallel" in str(raised)


async def test_checkpoint_observe_routes_through_depth0_guard() -> None:
    # checkpoint() shares the depth-0 choke point: its observe + counter management
    # run through _observe_depth0. A truly-concurrent checkpoint test is awkward
    # (checkpoint parks via WorkflowSignoffRequired, which propagates out of
    # run_workflow), so this drives Ctx directly: with a depth-0 observe already in
    # flight (counter == 1), checkpoint() must fail loud BEFORE parking — proving its
    # observe is guarded, not just agent()'s. RED before (checkpoint observed
    # unguarded and would have parked), GREEN after.
    async def _never_run_leaf(*_args: Any, **_kwargs: Any) -> LeafOutcome:
        raise AssertionError("checkpoint() must not dispatch a leaf")

    ctx = Ctx(
        roster=Roster(),
        journal=InMemoryJournalStore(),
        leaf_runner=_never_run_leaf,
    )

    # Simulate a sibling depth-0 observe already in flight (as a concurrent agent /
    # race branch would leave the shared counter mid-flight).
    ctx._depth0_inflight = 1  # pyright: ignore[reportPrivateUsage]

    raised: WorkflowConcurrencyError | None = None
    try:
        await ctx.checkpoint("approve?", tag="gate")
    except WorkflowConcurrencyError as exc:
        raised = exc

    assert raised is not None, (
        "checkpoint() with a sibling depth-0 observe in flight must fail loud"
    )
    assert "ctx.parallel" in str(raised)
    # The guard fired before observe/increment, so the counter is untouched (still 1,
    # the simulated sibling) — checkpoint did not leak an increment on the raise path.
    assert ctx._depth0_inflight == 1  # pyright: ignore[reportPrivateUsage]


async def test_checkpoint_park_does_not_let_a_gathered_agent_run_past_it(
    make_fake_leaf: Callable[..., tuple[Runnable[Any, Any], Any]],
) -> None:
    # Regression guard (GREEN, no fix needed — see below). A depth-0 checkpoint()
    # concurrent with a depth-0 agent() in a raw gather, gate un-approved (first run,
    # InMemory journal). A cross-model review worried the checkpoint park path would
    # decrement the in-flight counter (its finally fires) and let the sibling agent
    # then observe cleanly and run its leaf — work PAST an un-approved sign-off. This
    # was empirically REFUTED, but NOT for the reason a naive read suggests: a bare
    # asyncio.gather with the default return_exceptions=False does NOT cancel its
    # siblings when one branch raises (per the CPython docs; verified directly — the
    # sibling runs to completion). The real guarantee comes from the engine path:
    # orchestrate runs inside a LangGraph @entrypoint (pregel executor), and when that
    # node raises the checkpoint park (WorkflowSignoffRequired), the durable executor
    # tears the node down and cancels the run's still-pending child tasks during
    # run_workflow's own unwind — so the orphaned gathered agent() task is cancelled at
    # its next await (its journal.get / leaf dispatch) BEFORE its leaf runs.
    #
    # The keep-alive loop below is the load-bearing part of this assertion: it simulates
    # a PERSISTENT host loop (which does not close after a park). Without it, a green
    # state.calls == 0 would be a false positive — the agent might merely not have been
    # scheduled yet, only to be cancelled later at pytest's loop teardown. Keeping the
    # loop alive proves the agent is cancelled DURING the run's unwind, not incidentally
    # at loop close.
    leaf, state = make_fake_leaf("answer")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> Any:
        async def agent_branch() -> str:
            return await ctx.agent("past-the-gate", agent_type="worker")

        return await asyncio.gather(
            ctx.checkpoint({"ask": "go?"}, tag="gate"),
            agent_branch(),
        )

    # The run must NOT settle normally: it either parks (WorkflowSignoffRequired) or
    # fails loud (WorkflowConcurrencyError); whichever surfaces, no leaf must have run.
    raised: BaseException | None = None
    try:
        await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    except (WorkflowSignoffRequired, WorkflowConcurrencyError) as exc:
        raised = exc

    assert raised is not None, "the gather must not settle normally (park or fail loud)"

    # Keep a PERSISTENT host loop alive after the park (~100ms): if the agent leaf were
    # going to run at all, it would run in this window. It does not — the durable
    # executor cancelled the orphaned agent task during run_workflow's unwind.
    for _ in range(50):
        await asyncio.sleep(0.002)

    # The smoking gun: the agent leaf must NOT have executed, even with the loop kept
    # alive. If state.calls > 0, the agent ran past the un-approved park (the bug).
    assert state.calls == 0, "no agent leaf may run past an un-approved sign-off park"

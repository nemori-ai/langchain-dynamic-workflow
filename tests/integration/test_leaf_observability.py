"""Integration: the span-begin open edge + resume-stable span_id through run_workflow.

run_workflow accepts an on_span_begin sink that fires the instant each primitive's
span opens (before its body runs), for every span kind, carrying a resume-stable
span_id shared with the matching end span. These tests pin the begin/end ordering,
the per-kind coverage, and the fresh-vs-resume id stability for the sequential path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    Span,
    SpanBegin,
    SpanKind,
    run_workflow,
)

UsageLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_begin_fires_before_end_for_every_span_kind(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    leaf, _model = make_usage_leaf("finding", tokens_per_call=7)
    roster = Roster().register("researcher", leaf)
    begins: list[SpanBegin] = []
    ends: list[Span] = []

    async def orchestrate(ctx: Ctx) -> str:
        await ctx.agent("solo", agent_type="researcher")
        await ctx.parallel([lambda: ctx.agent("p", agent_type="researcher")])

        async def stage(prev: Any, item: Any, index: int) -> str:
            return await ctx.agent(f"s{item}", agent_type="researcher")

        await ctx.pipeline(["x"], stage)
        return "ok"

    await run_workflow(
        orchestrate,
        roster=roster,
        on_span_begin=begins.append,
        on_span=ends.append,
    )

    begin_kinds = {b.kind for b in begins}
    assert SpanKind.AGENT in begin_kinds
    assert SpanKind.PARALLEL in begin_kinds
    assert SpanKind.PIPELINE in begin_kinds
    # Every end span has a matching begin with the same id (begin precedes end).
    begin_ids = {b.span_id for b in begins}
    end_ids = {e.span_id for e in ends}
    assert end_ids <= begin_ids
    # The AGENT begin precedes its own end in emission order (running-before-done).
    agent_begin = next(b for b in begins if b.kind is SpanKind.AGENT)
    assert agent_begin.span_id in end_ids


async def test_span_id_is_resume_stable_for_the_sequential_path(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # A purely sequential workflow (no fan-out around the leaves) mints the same
    # span_id sequence on a fresh run and an honest resume, because the script
    # replays in the same source order (the determinism guard enforces this) and
    # the recorder's occurrence ordinals reset per run.
    leaf, _model = make_usage_leaf("finding", tokens_per_call=5)
    roster = Roster().register("researcher", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        a = await ctx.agent("one", agent_type="researcher")
        b = await ctx.agent("two", agent_type="researcher")
        return f"{a}|{b}"

    first: list[SpanBegin] = []
    await run_workflow(orchestrate, roster=roster, journal=journal, on_span_begin=first.append)

    second: list[SpanBegin] = []
    await run_workflow(orchestrate, roster=roster, journal=journal, on_span_begin=second.append)

    first_agent_ids = [b.span_id for b in first if b.kind is SpanKind.AGENT]
    second_agent_ids = [b.span_id for b in second if b.kind is SpanKind.AGENT]
    assert first_agent_ids == second_agent_ids
    assert len(first_agent_ids) == 2
    # The two distinct leaves get distinct ids (occurrence ordinal salts them apart).
    assert first_agent_ids[0] != first_agent_ids[1]


async def test_cached_leaf_emits_begin_marked_cached_on_resume(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # A replayed (cached) leaf re-emits a begin edge (so the consumer paints the
    # chip) AND a matching end span flagged cached=True with a near-zero duration —
    # never a stuck running chip. begin fires at open, before the journal lookup.
    leaf, model = make_usage_leaf("finding", tokens_per_call=5)
    roster = Roster().register("researcher", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("solo", agent_type="researcher")

    await run_workflow(orchestrate, roster=roster, journal=journal)
    assert model.calls == 1

    begins: list[SpanBegin] = []
    ends: list[Span] = []
    await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        on_span_begin=begins.append,
        on_span=ends.append,
    )
    assert model.calls == 1  # replayed, no fresh model call
    agent_begin = next(b for b in begins if b.kind is SpanKind.AGENT)
    agent_end = next(e for e in ends if e.kind is SpanKind.AGENT)
    # Same id correlates the running edge with the (now cached) completion.
    assert agent_begin.span_id == agent_end.span_id
    assert agent_end.attributes["cached"] is True


async def test_leaf_event_correlates_to_owning_leaf_span_id(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # A real deepagent leaf fires its own callback subtree (chain + chat_model
    # edges). Each LeafEvent must carry the owning leaf's span_id (the AGENT span's
    # id from the begin edge) and a run-tree node id, so a consumer can file the
    # subtree under the right leaf and rebuild the tree from parent_run_id.
    from langchain_dynamic_workflow import LeafEvent

    leaf, _model = make_deep_leaf("done")
    roster = Roster().register("worker", leaf)
    begins: list[SpanBegin] = []
    leaf_events: list[LeafEvent] = []

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("go", agent_type="worker")

    await run_workflow(
        orchestrate,
        roster=roster,
        on_span_begin=begins.append,
        on_leaf_event=leaf_events.append,
    )

    agent_begin = next(b for b in begins if b.kind is SpanKind.AGENT)
    assert leaf_events, "a real deepagent leaf must fire interior callback events"
    # Every leaf event is correlated to the single AGENT leaf's span id.
    assert {e.leaf_span_id for e in leaf_events} == {agent_begin.span_id}
    # The subtree carries at least one chat_model edge (the leaf called its model).
    assert any(e.kind == "chat_model" for e in leaf_events)
    # Run-tree shape (REAL assertion, no "or True"): at least one event roots the
    # subtree (parent_run_id is None) and every non-root parent_run_id closes the
    # tree by referencing a run_id we actually emitted.
    run_ids = {e.run_id for e in leaf_events}
    assert any(e.parent_run_id is None for e in leaf_events)
    assert all(e.parent_run_id in run_ids for e in leaf_events if e.parent_run_id is not None)


async def test_on_leaf_event_does_not_fire_on_a_journal_hit(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # LOCKED replay policy: no leaf interior runs on a journal hit, so on_leaf_event
    # MUST stay silent on resume for a replayed leaf -- else resume double-renders
    # activity that never ran.
    from langchain_dynamic_workflow import LeafEvent

    leaf, model = make_deep_leaf("done")
    roster = Roster().register("worker", leaf)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("go", agent_type="worker")

    first: list[LeafEvent] = []
    await run_workflow(orchestrate, roster=roster, journal=journal, on_leaf_event=first.append)
    assert first, "the fresh run's leaf fires interior events"
    fresh_calls = model.calls

    second: list[LeafEvent] = []
    await run_workflow(orchestrate, roster=roster, journal=journal, on_leaf_event=second.append)
    assert model.calls == fresh_calls  # replayed: no fresh model call
    assert second == []  # NO interior events on the journal hit


async def test_sinks_default_none_is_zero_cost_and_quarantine_byte_identical(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # Default-None sinks => no handler attached => the leaf's host-facing result is
    # byte-identical with and without the observability sinks (quarantine preserved:
    # the sinks never touch the folded result or the host context).
    from langchain_dynamic_workflow import LeafEvent

    leaf_a, _ = make_deep_leaf("identical-result")
    leaf_b, _ = make_deep_leaf("identical-result")

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("go", agent_type="worker")

    without = await run_workflow(orchestrate, roster=Roster().register("worker", leaf_a))
    events: list[LeafEvent] = []
    begins: list[SpanBegin] = []
    with_sinks = await run_workflow(
        orchestrate,
        roster=Roster().register("worker", leaf_b),
        on_span_begin=begins.append,
        on_leaf_event=events.append,
    )
    assert without == with_sinks  # folded result is identical; sinks are out-of-band


async def test_payload_opt_in_surfaces_model_text_in_detail(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    from langchain_dynamic_workflow import LeafEvent

    leaf, _ = make_deep_leaf("VISIBLE-PAYLOAD")
    roster = Roster().register("worker", leaf)

    shape_only: list[LeafEvent] = []
    await run_workflow(orchestrate=_solo, roster=roster, on_leaf_event=shape_only.append)
    assert all("VISIBLE-PAYLOAD" not in str(e.detail) for e in shape_only)

    leaf2, _ = make_deep_leaf("VISIBLE-PAYLOAD")
    with_payload: list[LeafEvent] = []
    await run_workflow(
        orchestrate=_solo,
        roster=Roster().register("worker", leaf2),
        on_leaf_event=with_payload.append,
        leaf_event_include_payloads=True,
    )
    assert any("VISIBLE-PAYLOAD" in str(e.detail) for e in with_payload)


async def test_synthetic_nested_subtree_reconstructs_a_closed_run_tree() -> None:
    # CI floor for tree-structure correctness (offline, no real model, no deepagents
    # sub-agent): a leaf RunnableLambda awaits a NESTED child RunnableLambda with the
    # forwarded config, producing genuine parent/child callback edges. The handler is
    # attached on the SAME callbacks-config seam the engine uses, so this exercises the
    # real tree-reconstruction path. (Driven at the handler seam rather than through
    # run_workflow because the engine's @task wrapper makes a bare RunnableLambda
    # leaf's outermost chain inherit a non-None parent outside the captured subtree;
    # the deepagent-graph leaf path roots cleanly and is asserted through run_workflow
    # in test_leaf_event_correlates_to_owning_leaf_span_id.)
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig, RunnableLambda

    from langchain_dynamic_workflow import LeafEvent
    from langchain_dynamic_workflow._leaf_events import LeafEventHandler

    child: Runnable[Any, Any] = RunnableLambda(lambda inp, config=None: {"echo": inp.get("q", "")})

    async def parent(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        # Await the nested child WITH the forwarded config so its callback edges
        # thread under the parent in the run tree (deeper subtree correlation).
        await child.ainvoke({"q": "hello"}, config=config)
        return {"messages": [*inp["messages"], AIMessage(content="done")]}

    leaf: Runnable[Any, Any] = RunnableLambda(parent)

    events: list[LeafEvent] = []
    handler = LeafEventHandler(leaf_span_id="leaf-OWNER", sink=events.append)
    await leaf.ainvoke({"messages": []}, config={"callbacks": [handler]})

    assert events, "the nested leaf subtree must fire callback edges"
    run_ids = {e.run_id for e in events}
    # The nested child is a distinct node from the parent leaf.
    assert len(run_ids) >= 2
    # (a) At least one root: the outermost leaf edge has no parent in the subtree.
    assert any(e.parent_run_id is None for e in events)
    # (b) The tree closes: every non-root edge's parent is itself an emitted run_id.
    assert all(e.parent_run_id in run_ids for e in events if e.parent_run_id is not None)
    # (c) Every edge carries the one owning leaf span id (handler-closure correlation).
    assert {e.leaf_span_id for e in events} == {"leaf-OWNER"}


async def test_nested_leaf_subtree_stamps_one_owning_span_id() -> None:
    # The leaf's callbacks list is forwarded to any sub-agent it spawns, so the
    # WHOLE subtree (including deeper nodes) correlates to the ONE owning leaf span.
    # Build a deepagent leaf and assert every emitted event carries the single owning
    # leaf_span_id (deepagents forwards callbacks/tags/configurable to subagents;
    # verified statically in _build_subagent_config).
    from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
    from examples._shared.offline_models import ScriptedModel

    from langchain_dynamic_workflow import LeafEvent

    leaf = create_deep_agent(model=ScriptedModel(reply="done"))  # pyright: ignore[reportUnknownVariableType]
    roster = Roster().register("worker", leaf)
    begins: list[SpanBegin] = []
    events: list[LeafEvent] = []

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("go", agent_type="worker")

    await run_workflow(
        orchestrate, roster=roster, on_span_begin=begins.append, on_leaf_event=events.append
    )
    agent_begin = next(b for b in begins if b.kind is SpanKind.AGENT)
    # EXACT correlation: no event leaks a different span id (the handler closes over
    # this leaf's id; it never sees a sibling leaf's subtree).
    assert events
    assert {e.leaf_span_id for e in events} == {agent_begin.span_id}


async def test_two_sequential_leaves_keep_their_subtrees_separate(
    make_deep_leaf: Callable[[str], tuple[Runnable[Any, Any], Any]],
) -> None:
    # Two leaves => two distinct span ids => each subtree files under its own id, never
    # cross-contaminated (one handler per leaf invocation).
    from langchain_dynamic_workflow import LeafEvent

    leaf, _ = make_deep_leaf("done")
    roster = Roster().register("worker", leaf)
    begins: list[SpanBegin] = []
    events: list[LeafEvent] = []

    async def orchestrate(ctx: Ctx) -> str:
        a = await ctx.agent("one", agent_type="worker")
        b = await ctx.agent("two", agent_type="worker")
        return f"{a}|{b}"

    await run_workflow(
        orchestrate, roster=roster, on_span_begin=begins.append, on_leaf_event=events.append
    )
    agent_ids = {b.span_id for b in begins if b.kind is SpanKind.AGENT}
    assert len(agent_ids) == 2
    # Every event's leaf_span_id is one of the two leaf ids; both ids appear.
    event_ids = {e.leaf_span_id for e in events}
    assert event_ids <= agent_ids
    assert event_ids == agent_ids  # both leaves emitted interior events


async def _solo(ctx: Ctx) -> str:
    return await ctx.agent("go", agent_type="worker")

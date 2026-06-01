"""Phase 6 capstone integration: a multi-stage workflow, host-driven, all offline.

The capstone exercises the engine end-to-end with every major feature stacked:

1. ``parallel`` fan-out research over N sources (blocking barrier).
2. ``pipeline`` refinement of each finding (no barrier between stages).
3. adversarial verification: each refined finding is challenged by N skeptic
   leaves in ``parallel``; a finding survives only if a majority vote it valid.
4. synthesis of the survivors into a final conclusion.

The whole thing is budgeted (a shared token pool metered through the leaves),
sandbox-admitted (a ``needs_execution`` leaf is leased an isolated backend), and
driven by a host deepagent that launches it in the background through the workflow
tool. Everything runs on fake models with no API key, and observability spans are
collected so the test asserts the full primitive trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import PrivateAttr

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    SandboxManager,
    Span,
    SpanKind,
    WorkflowRegistry,
    run_workflow,
)
from langchain_dynamic_workflow._background import BgRunManager, BgStatus
from langchain_dynamic_workflow.middleware import (
    WORKFLOW_NOTIFICATION_TAG,
    create_workflow_middleware,
)

UsageLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]

TOPICS = ["alpha", "beta", "gamma", "delta"]
SKEPTICS_PER_FINDING = 3


def _verdict_leaf() -> Runnable[Any, Any]:
    """A deterministic skeptic leaf: votes ``valid`` / ``invalid`` by content.

    The verdict is a pure function of the prompt so the majority outcome is
    reproducible: a finding mentioning an *even-length* topic is challenged
    successfully by every skeptic (all vote ``invalid``), while an odd-length
    topic survives (all vote ``valid``). That gives a clean, asserted split
    between surviving and rejected findings without any randomness.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        # The topic is the last whitespace-separated token of the challenge prompt.
        topic = prompt.split()[-1] if prompt.split() else ""
        verdict = "valid" if len(topic) % 2 == 1 else "invalid"
        return {"messages": [*inp["messages"], AIMessage(content=verdict)]}

    return RunnableLambda(_call)


def _bind(args: dict[str, Any]) -> Callable[[Ctx], Any]:
    """Adapt the two-arg workflow callable to the one-arg run_workflow orchestrator."""

    async def _orchestrate(ctx: Ctx) -> str:
        return await _capstone(ctx, args)

    return _orchestrate


async def _capstone(ctx: Ctx, args: dict[str, Any]) -> str:
    """The capstone orchestration: research -> refine -> verify -> synthesize."""
    topics: list[str] = args.get("topics", TOPICS)

    ctx.phase("research")
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    researched = [(t, f) for t, f in zip(topics, findings, strict=True) if f is not None]

    ctx.phase("refine")

    async def refine_stage(prev: Any, item: Any, index: int) -> tuple[str, str]:
        topic, _finding = item
        refined = await ctx.agent(f"Refine {topic}", agent_type="refiner")
        return (topic, refined)

    refined_pairs = await ctx.pipeline(researched, refine_stage)
    refined = [p for p in refined_pairs if p is not None]

    ctx.phase("verify")
    survivors: list[str] = []
    for topic, refined_text in refined:
        # Adversarial verification: N skeptics challenge each finding in parallel.
        verdicts = await ctx.parallel(
            [
                lambda t=topic: ctx.agent(f"Challenge {t}", agent_type="skeptic")
                for _ in range(SKEPTICS_PER_FINDING)
            ]
        )
        valid_votes = sum(1 for v in verdicts if v == "valid")
        if valid_votes * 2 > SKEPTICS_PER_FINDING:  # strict majority survives
            survivors.append(f"{topic}:{refined_text}")

    ctx.phase("synthesize")
    return f"synthesized {len(survivors)} surviving findings: " + " | ".join(sorted(survivors))


async def test_capstone_multi_stage_runs_green_with_budget_and_sandbox(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # Three roster entries: a sandbox-admitted researcher (needs_execution), a
    # pure-reasoning refiner, and the deterministic skeptic.
    researcher_leaf, _rmodel = make_usage_leaf("finding", tokens_per_call=3)
    refiner_leaf, _fmodel = make_usage_leaf("refined", tokens_per_call=2)
    roster = (
        Roster()
        .register("researcher", researcher_leaf, needs_execution=True)
        .register("refiner", refiner_leaf)
        .register("skeptic", _verdict_leaf())
    )

    spans: list[Span] = []
    sandbox_manager = SandboxManager()
    result = await run_workflow(
        _bind({}),
        roster=roster,
        budget=10_000,
        sandbox_manager=sandbox_manager,
        on_span=spans.append,
    )

    # Odd-length topics survive (alpha=5, gamma=5, delta=5 are odd; beta=4 is even
    # and is voted down by every skeptic), so three findings survive.
    assert "synthesized 3 surviving findings" in result
    assert "alpha:refined" in result
    assert "gamma:refined" in result
    assert "delta:refined" in result
    assert "beta:" not in result  # the even-length topic was rejected by the majority

    # The trace shows the full primitive stack: research+verify parallels, a
    # refine pipeline, and one agent span per leaf invocation.
    kinds = [s.kind for s in spans]
    assert kinds.count(SpanKind.PIPELINE) == 1
    # One research parallel + one verify parallel per refined finding (4 refined).
    assert kinds.count(SpanKind.PARALLEL) == 1 + len(TOPICS)
    # 4 research + 4 refine + 4*3 skeptic = 20 agent leaves.
    assert kinds.count(SpanKind.AGENT) == 4 + 4 + len(TOPICS) * SKEPTICS_PER_FINDING

    # The sandbox-admitted researcher leased and the engine tore everything down:
    # no live sandbox remains after the run settles.
    assert sandbox_manager.active_count == 0


async def test_capstone_resume_replays_completed_leaves_at_zero_cost(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # The capstone is resumable: a second run against the same journal replays every
    # completed leaf from cache with zero new model calls.
    researcher_leaf, rmodel = make_usage_leaf("finding", tokens_per_call=3)
    refiner_leaf, fmodel = make_usage_leaf("refined", tokens_per_call=2)
    roster = (
        Roster()
        .register("researcher", researcher_leaf)
        .register("refiner", refiner_leaf)
        .register("skeptic", _verdict_leaf())
    )
    journal = InMemoryJournalStore()

    first = await run_workflow(_bind({}), roster=roster, journal=journal, thread_id="cap-1")
    calls_after_first = (rmodel.calls, fmodel.calls)
    assert rmodel.calls > 0 and fmodel.calls > 0

    second_spans: list[Span] = []
    second = await run_workflow(
        _bind({}), roster=roster, journal=journal, thread_id="cap-2", on_span=second_spans.append
    )

    assert second == first  # identical conclusion on replay
    assert (rmodel.calls, fmodel.calls) == calls_after_first  # zero new model calls
    # Every agent span on the resumed run is a journal hit.
    agent_spans = [s for s in second_spans if s.kind == SpanKind.AGENT]
    assert agent_spans
    assert all(s.attributes["cached"] is True for s in agent_spans)


class _CapstoneHost(BaseChatModel):
    """A scripted host that launches the capstone in the background, then folds it.

    Mirrors the Phase 5 scripted-host pattern: first turn launches the workflow
    (non-blocking), and once a completion notification is injected the host calls
    ``status`` and folds the conclusion into a final answer.
    """

    _run_id_box: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def run_id_box(self) -> dict[str, str]:
        """Holds the launched run_id so a later turn can target status by it."""
        return self._run_id_box

    @property
    def _llm_type(self) -> str:
        return "capstone-host"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        notification_seen = any(WORKFLOW_NOTIFICATION_TAG in m.text for m in messages)
        status_done = any("done." in m.text for m in tool_messages)
        launched = any("Launched workflow" in m.text for m in tool_messages)

        if status_done:
            result = next(m.text for m in reversed(tool_messages) if "done." in m.text)
            return _say(f"FINAL: {result}")
        if notification_seen:
            return _call("status", run_id=self._run_id_box.get("run_id", ""))
        if launched:
            return _say("Capstone launched; awaiting completion.")
        return _call("run", workflow="capstone")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _say(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _call(command: str, **args: str) -> ChatResult:
    call = AIMessage(
        content="",
        tool_calls=[{"name": "workflow", "args": {"command": command, **args}, "id": command}],
    )
    return ChatResult(generations=[ChatGeneration(message=call)])


async def test_capstone_driven_by_host_agent_in_background(
    make_usage_leaf: UsageLeafFactory,
) -> None:
    # The full outward form: a host deepagent launches the capstone through the
    # workflow tool in the background, gets a completion notification, fetches the
    # result, and folds the conclusion — all offline.
    researcher_leaf, _r = make_usage_leaf("finding", tokens_per_call=1)
    refiner_leaf, _f = make_usage_leaf("refined", tokens_per_call=1)
    roster = (
        Roster()
        .register("researcher", researcher_leaf)
        .register("refiner", refiner_leaf)
        .register("skeptic", _verdict_leaf())
    )
    workflows = WorkflowRegistry().register("capstone", _capstone)
    manager = BgRunManager()
    middleware = create_workflow_middleware(roster, workflows=workflows, manager=manager)

    host_model = _CapstoneHost()
    host = create_deep_agent(model=host_model, middleware=[middleware])  # pyright: ignore[reportUnknownVariableType, reportArgumentType]
    config: RunnableConfig = {"configurable": {"thread_id": "cap-host"}}

    state1 = await host.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        {"messages": [{"role": "user", "content": "Run the capstone"}]}, config=config
    )
    runs: list[dict[str, Any]] = state1["workflow_runs"]
    run_id = runs[-1]["run_id"]
    assert isinstance(run_id, str)
    host_model.run_id_box["run_id"] = run_id

    await manager.wait(run_id, thread_id="cap-host")
    assert manager.poll(run_id, thread_id="cap-host") == BgStatus.DONE

    state2 = await host.ainvoke(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        {"messages": [{"role": "user", "content": "Any update?"}]}, config=config
    )
    final_messages: list[BaseMessage] = state2["messages"]
    transcript = "\n".join(m.text for m in final_messages)
    assert WORKFLOW_NOTIFICATION_TAG in transcript
    final_ai = [m for m in final_messages if isinstance(m, AIMessage) and m.text]
    assert final_ai
    assert "FINAL:" in final_ai[-1].text
    assert "synthesized 3 surviving findings" in final_ai[-1].text


def test_real_model_variant_is_env_gated() -> None:
    # The capstone has a real-model variant in examples/06_capstone.py gated behind
    # LDW_DEMO_REAL_MODEL; the offline tests above never need a key. This asserts the
    # gating contract is documented in the example so CI never hits a real provider.
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "examples" / "06_capstone.py"
    source = example.read_text(encoding="utf-8")
    assert "LDW_DEMO_REAL_MODEL" in source

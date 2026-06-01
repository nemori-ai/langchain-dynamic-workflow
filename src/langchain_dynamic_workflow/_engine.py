"""The workflow engine — Layer 0 substrate binding.

``run_workflow`` builds a LangGraph ``@entrypoint`` whose body constructs a
:class:`Ctx` and runs the user's orchestration callable. Each leaf invocation is
a ``@task`` (durable execution), with the content-hash journal layered on top so
completed leaves return cached results on resume.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

from ._budget import Budget, total_tokens_from_handler
from ._concurrency import (
    HARD_CEILING,
    ConcurrencyGate,
    resolve_max_concurrency,
    with_max_concurrency,
)
from ._context import Ctx, LeafOutcome
from ._determinism import CallSequenceGuard
from ._journal import InMemoryJournalStore, JournalStore
from ._progress import ProgressEntry, ProgressLog, ProgressSink
from ._roster import Roster


def _default_progress_sink(entry: ProgressEntry) -> None:
    """Print a progress entry to stdout (the default narration sink)."""
    print(f"[{entry.kind.value}] {entry.message}")


Orchestrator = Callable[[Ctx], Awaitable[Any]]
"""A workflow script: ``async def orchestrate(ctx) -> result``."""


async def run_workflow(
    orchestrate: Orchestrator,
    *,
    roster: Roster,
    journal: JournalStore | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    thread_id: str = "default",
    max_concurrency: int | None = None,
    budget: int | None = None,
    on_progress: ProgressSink | None = None,
) -> Any:
    """Run an orchestration script to completion and return its final result.

    Args:
        orchestrate: The async orchestration callable, called with a ``Ctx``.
        roster: The leaf registry resolved by ``ctx.agent(agent_type=...)``.
        journal: Content-hash journal store; defaults to an in-memory store.
            Pass the *same* instance across calls to get cached-result resume.
        checkpointer: LangGraph checkpointer; defaults to an in-memory saver.
        thread_id: Durable-execution thread id.
        budget: Optional shared token ceiling for all leaves in this run. When a
            leaf would push spend past the cap the next ``agent()`` raises
            :class:`~langchain_dynamic_workflow.WorkflowBudgetExceededError`;
            ``None`` (the default) leaves the budget unbounded so the
            loop-until-budget idiom (``while ctx.budget.remaining() > T``) is a
            no-op cap. The spend is rebuilt from journal usage on resume.
        on_progress: Optional sink receiving each newly-delivered ``phase``/``log``
            entry; defaults to printing to stdout. Delivery is replay-idempotent —
            entries already delivered on a prior run are not re-emitted on resume.
        max_concurrency: Optional explicit concurrency cap on in-flight leaves.
            LangGraph has no default (``None`` means unbounded at the substrate),
            so the engine always sets both layers explicitly. The shared
            :class:`ConcurrencyGate` is the *authoritative* bound (resolved to
            ``min(16, cores - 2)`` when omitted). The substrate ``max_concurrency``
            is pinned to the hard ceiling so it is never unbounded yet never
            throttles below the gate — setting both to the same value makes the two
            semaphores interleave and the effective cap fall one below the target.

    Returns:
        Whatever the orchestration callable returns.
    """
    journal_store: JournalStore = journal if journal is not None else InMemoryJournalStore()
    saver: BaseCheckpointSaver[Any] = checkpointer if checkpointer is not None else InMemorySaver()
    limit = resolve_max_concurrency(max_concurrency)
    gate = ConcurrencyGate(limit=limit)

    @task
    async def leaf_task(agent_type: str, prompt: str, model: str | None) -> LeafOutcome:
        entry = roster.resolve(agent_type)
        # Forward a usage callback into the leaf's config — the same callback
        # forwarding deepagents performs for its own subagents — so token usage
        # aggregates across every (possibly nested) model call the leaf makes,
        # even though we bypass the LLM-driven task tool and invoke directly.
        usage_handler = UsageMetadataCallbackHandler()
        configurable: dict[str, Any] = {}
        if model is not None:
            # Thread the model override into the leaf config so it reaches
            # execution, not just the journal key (closes the key-vs-execution gap).
            configurable["model"] = model
        leaf_config: RunnableConfig = {"callbacks": [usage_handler], "configurable": configurable}
        state: dict[str, Any] = await entry.runnable.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}, config=leaf_config
        )
        return LeafOutcome(state=state, usage=total_tokens_from_handler(usage_handler))

    async def leaf_runner(agent_type: str, prompt: str, model: str | None) -> LeafOutcome:
        # The single durable leaf path, shared by agent / parallel / pipeline.
        # The shared gate (applied inside Ctx) bounds how many of these run at
        # once across every fan-out path in this run.
        return await leaf_task(agent_type, prompt, model)

    recorded_sequence = await journal_store.get_sequence()
    delivered_progress = await journal_store.get_progress_count()
    progress_sink: ProgressSink = on_progress if on_progress is not None else _default_progress_sink

    @entrypoint(checkpointer=saver)
    async def _run(_input: Any) -> Any:
        sequence_guard = CallSequenceGuard(recorded=recorded_sequence)
        # A fresh budget per run: spend is rebuilt as leaves resolve (cache hits
        # re-count journaled usage, new leaves count their metered usage), so a
        # resumed run reaches the first run's cumulative total.
        run_budget = Budget(total=budget)
        # On resume, suppress progress entries already delivered on the prior run
        # so phase/log narration is not repeated; new entries still flow.
        progress = ProgressLog(delivered_count=delivered_progress, sink=progress_sink)
        ctx = Ctx(
            roster=roster,
            journal=journal_store,
            leaf_runner=leaf_runner,
            gate=gate,
            sequence_guard=sequence_guard,
            budget=run_budget,
            progress=progress,
        )
        result = await orchestrate(ctx)
        # Persist run-level state only after completion, so the determinism
        # backstop and progress idempotency have a record to replay against on the
        # next resume.
        await journal_store.put_sequence(ctx.observed_call_sequence)
        await journal_store.put_progress_count(ctx.progress_entry_count)
        return result

    base_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # The gate is the precise cap; the substrate semaphore is pinned high (never
    # None/unbounded) so it stays explicit without fighting the gate.
    config = with_max_concurrency(base_config, HARD_CEILING)
    result: Any = await _run.ainvoke({}, config=config)  # pyright: ignore[reportUnknownMemberType]
    return result

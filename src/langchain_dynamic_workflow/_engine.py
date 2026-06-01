"""The workflow engine — Layer 0 substrate binding.

``run_workflow`` builds a LangGraph ``@entrypoint`` whose body constructs a
:class:`Ctx` and runs the user's orchestration callable. Each leaf invocation is
a ``@task`` (durable execution), with the content-hash journal layered on top so
completed leaves return cached results on resume.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

from ._concurrency import (
    HARD_CEILING,
    ConcurrencyGate,
    resolve_max_concurrency,
    with_max_concurrency,
)
from ._context import Ctx
from ._determinism import CallSequenceGuard
from ._journal import InMemoryJournalStore, JournalStore
from ._roster import Roster

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
) -> Any:
    """Run an orchestration script to completion and return its final result.

    Args:
        orchestrate: The async orchestration callable, called with a ``Ctx``.
        roster: The leaf registry resolved by ``ctx.agent(agent_type=...)``.
        journal: Content-hash journal store; defaults to an in-memory store.
            Pass the *same* instance across calls to get cached-result resume.
        checkpointer: LangGraph checkpointer; defaults to an in-memory saver.
        thread_id: Durable-execution thread id.
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
    async def leaf_task(agent_type: str, prompt: str) -> dict[str, Any]:
        entry = roster.resolve(agent_type)
        result: dict[str, Any] = await entry.runnable.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        return result

    async def leaf_runner(agent_type: str, prompt: str) -> dict[str, Any]:
        # The single durable leaf path, shared by agent / parallel / pipeline.
        # The shared gate (applied inside Ctx) bounds how many of these run at
        # once across every fan-out path in this run.
        return await leaf_task(agent_type, prompt)

    recorded_sequence = await journal_store.get_sequence()

    @entrypoint(checkpointer=saver)
    async def _run(_input: Any) -> Any:
        sequence_guard = CallSequenceGuard(recorded=recorded_sequence)
        ctx = Ctx(
            roster=roster,
            journal=journal_store,
            leaf_runner=leaf_runner,
            gate=gate,
            sequence_guard=sequence_guard,
        )
        result = await orchestrate(ctx)
        # Persist the observed call sequence only after the run completes, so the
        # determinism backstop has a record to replay against on the next resume.
        await journal_store.put_sequence(ctx.observed_call_sequence)
        return result

    base_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # The gate is the precise cap; the substrate semaphore is pinned high (never
    # None/unbounded) so it stays explicit without fighting the gate.
    config = with_max_concurrency(base_config, HARD_CEILING)
    result: Any = await _run.ainvoke({}, config=config)  # pyright: ignore[reportUnknownMemberType]
    return result

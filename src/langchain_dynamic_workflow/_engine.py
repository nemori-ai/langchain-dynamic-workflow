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

from ._context import Ctx
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
        max_concurrency: Optional explicit concurrency cap (LangGraph has no
            default — leaving this ``None`` means unbounded at the substrate).

    Returns:
        Whatever the orchestration callable returns.
    """
    journal_store: JournalStore = journal if journal is not None else InMemoryJournalStore()
    saver: BaseCheckpointSaver[Any] = checkpointer if checkpointer is not None else InMemorySaver()

    @task
    async def leaf_task(agent_type: str, prompt: str) -> dict[str, Any]:
        entry = roster.resolve(agent_type)
        result: dict[str, Any] = await entry.runnable.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )
        return result

    async def leaf_runner(agent_type: str, prompt: str) -> dict[str, Any]:
        return await leaf_task(agent_type, prompt)

    @entrypoint(checkpointer=saver)
    async def _run(_input: Any) -> Any:
        ctx = Ctx(roster=roster, journal=journal_store, leaf_runner=leaf_runner)
        return await orchestrate(ctx)

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    result: Any = await _run.ainvoke({}, config=config)  # pyright: ignore[reportUnknownMemberType]
    return result

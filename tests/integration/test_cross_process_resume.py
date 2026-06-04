"""Cross-process persistence integration: AsyncSqliteSaver durability spike.

These tests front-load the highest external risk of the persistent checkpointer:
``AsyncSqliteSaver`` msgpack-serializes *every* ``@task`` return, which for this
engine is a ``LeafOutcome`` wrapping the leaf's raw output state — including
LangChain message objects. This file proves that a realistic deepagent-shaped
leaf state round-trips through the sqlite checkpointer with no msgpack failure,
and that a fresh saver over a new connection to the same db file sees the prior
thread's checkpoint (cross-process resume-by-``thread_id``).

The checkpointer is constructed via ``AsyncSqliteSaver(conn)`` over a host-owned
``aiosqlite.Connection`` *inside the test's running event loop* — never via
``from_conn_string`` (which closes the connection on context exit, defeating
cross-process resume) and never outside a running loop (the saver binds to the
loop at construction; reuse across loops hangs).

The spike is deliberately offline: no real model, no API keys. It tests
serialization and durability, not a model run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_dynamic_workflow import Ctx, Roster, run_workflow
from langchain_dynamic_workflow._context import LeafOutcome


def _make_deepagent_shaped_leaf(reply: str) -> Runnable[Any, Any]:
    """Return a fake leaf whose output state mirrors a real deepagent.

    A ``create_deep_agent`` leaf returns a state dict whose ``messages`` list
    holds LangChain message objects (the input ``HumanMessage`` plus the model's
    ``AIMessage``) alongside other typical keys a deepagent threads through its
    graph (``todos``, ``files``). That message-bearing dict is exactly what the
    engine wraps in a ``LeafOutcome`` and what the checkpointer must msgpack-
    serialize on every ``@task`` return, so the fake reproduces that shape.

    Args:
        reply: The text the leaf's terminal ``AIMessage`` carries.

    Returns:
        A runnable appending an ``AIMessage(reply)`` to the input messages and
        attaching deepagent-typical state keys.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        messages = [*inp["messages"], AIMessage(content=reply)]
        return {
            "messages": messages,
            "todos": [{"content": "answer the question", "status": "completed"}],
            "files": {"notes.txt": "scratch reasoning"},
        }

    return RunnableLambda(_call)


async def test_async_sqlite_saver_round_trips_a_real_run(tmp_path: Path) -> None:
    """A run over a deepagent-shaped leaf persists through AsyncSqliteSaver.

    This is the C6 serialization spike: the leaf returns a state dict carrying
    LangChain ``HumanMessage`` / ``AIMessage`` objects, the engine wraps it in a
    ``LeafOutcome`` as the ``@task`` return, and ``AsyncSqliteSaver`` must
    msgpack-serialize it. The run must complete with no
    ``Type is not msgpack serializable: LeafOutcome`` error, and the checkpoint
    must be readable back from the same saver.
    """
    db_path = tmp_path / "workflows.db"
    leaf = _make_deepagent_shaped_leaf("Paris")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("Capital of France?", agent_type="geographer")

    # Construct the saver INSIDE the running loop over a host-owned connection
    # (C1): the saver binds to the loop at __init__, and from_conn_string would
    # close the connection on exit and defeat cross-process resume.
    conn = await aiosqlite.connect(db_path)
    try:
        saver = AsyncSqliteSaver(conn)
        result = await run_workflow(
            orchestrate,
            roster=roster,
            checkpointer=saver,
            thread_id="run-1",
        )
        assert result == "Paris"

        # The checkpoint persisted (no msgpack failure aborted the run): the same
        # saver reads back a non-None tuple for the run's thread.
        config: RunnableConfig = {"configurable": {"thread_id": "run-1"}}
        tuple_ = await saver.aget_tuple(config)
        assert tuple_ is not None
    finally:
        await conn.close()


async def test_fresh_saver_new_connection_sees_prior_checkpoint(tmp_path: Path) -> None:
    """A fresh AsyncSqliteSaver over a NEW connection sees the prior checkpoint.

    This proves cross-process resume-by-``thread_id``: the first saver/connection
    stands in for the process that ran the workflow; a second saver over a brand-
    new connection to the *same* db file (standing in for a fresh process) reads
    back the persisted checkpoint for that ``thread_id``. The deepagent-shaped
    ``LeafOutcome`` state survives the serialize-then-reopen round trip.
    """
    db_path = tmp_path / "workflows.db"
    leaf = _make_deepagent_shaped_leaf("Berlin")
    roster = Roster().register("geographer", leaf)

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("Capital of Germany?", agent_type="geographer")

    # Process A: run the workflow, persist, close the connection.
    conn_a = await aiosqlite.connect(db_path)
    try:
        saver_a = AsyncSqliteSaver(conn_a)
        result = await run_workflow(
            orchestrate,
            roster=roster,
            checkpointer=saver_a,
            thread_id="run-2",
        )
        assert result == "Berlin"
    finally:
        await conn_a.close()

    # Process B: a fresh connection + fresh saver to the same db file sees the
    # checkpoint persisted by process A.
    conn_b = await aiosqlite.connect(db_path)
    try:
        saver_b = AsyncSqliteSaver(conn_b)
        config: RunnableConfig = {"configurable": {"thread_id": "run-2"}}
        tuple_ = await saver_b.aget_tuple(config)
        assert tuple_ is not None
        # An unrelated thread_id has no checkpoint, confirming the read is scoped.
        empty: RunnableConfig = {"configurable": {"thread_id": "never-ran"}}
        assert await saver_b.aget_tuple(empty) is None
    finally:
        await conn_b.close()


def test_leaf_outcome_payload_is_strict_msgpack_serializable() -> None:
    """The leaf ``@task`` payload survives strict-msgpack serialization.

    The checkpointer serializer revives a custom class instance only through a
    deprecated unregistered-type path; the strict serde (``allowed_msgpack_modules
    =None``, the future default once ``LANGGRAPH_STRICT_MSGPACK`` flips) blocks
    that path, so persisting a bare ``LeafOutcome`` dataclass under strict mode
    does not round-trip. The engine instead hands the checkpointer the outcome's
    msgpack-native payload mapping (``to_payload``), which round-trips cleanly even
    under strict mode while preserving the nested LangChain messages and usage.

    This is the regression pin for the C6 serialization adaptation: were the
    ``@task`` to return the dataclass directly again, the strict round-trip below
    would fail.
    """
    # Build the serde in its strict (future-default) mode explicitly, so the test
    # does not depend on an import-time environment variable.
    strict_serde = JsonPlusSerializer(allowed_msgpack_modules=None)

    outcome = LeafOutcome(
        state={
            "messages": [HumanMessage(content="Capital of France?"), AIMessage(content="Paris")],
            "todos": [{"content": "answer", "status": "completed"}],
            "files": {"notes.txt": "scratch"},
        },
        usage=42,
    )

    # The bare dataclass is blocked under strict mode: the serializer revives an
    # unregistered class instance only via the deprecated path strict mode forbids,
    # so the round trip does not reproduce the original outcome.
    type_, blob = strict_serde.dumps_typed(outcome)
    assert strict_serde.loads_typed((type_, blob)) != outcome

    # The msgpack-native payload the engine actually persists round-trips exactly,
    # including the nested LangChain message objects and usage.
    payload_type, payload_blob = strict_serde.dumps_typed(outcome.to_payload())
    revived = strict_serde.loads_typed((payload_type, payload_blob))
    restored = LeafOutcome.from_payload(revived)
    assert restored == outcome
    assert [type(m).__name__ for m in restored.state["messages"]] == [
        "HumanMessage",
        "AIMessage",
    ]
    assert all(isinstance(m, BaseMessage) for m in restored.state["messages"])
    assert restored.usage == 42

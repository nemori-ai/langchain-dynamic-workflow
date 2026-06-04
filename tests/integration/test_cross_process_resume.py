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

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_dynamic_workflow import Ctx, Roster, run_workflow
from langchain_dynamic_workflow._context import LeafOutcome

_WORKER = Path(__file__).resolve().parent / "_cross_process_worker.py"
_RESULT_MARKER = "WORKER_RESULT "


def _run_worker(mode: str, db_path: Path, counter_path: Path) -> dict[str, Any]:
    """Launch the cross-process worker as a real OS process and parse its result.

    Each call spawns a *separate* Python interpreter (via ``subprocess``), so the
    run and resume halves genuinely run in different processes that share only the
    on-disk db and counter files — there is no shared in-process state that could
    mask a persistence gap.

    Args:
        mode: The worker mode (``run`` / ``run-fail`` / ``resume``).
        db_path: The shared sqlite db file both processes target.
        counter_path: The shared leaf-invocation counter file.

    Returns:
        The decoded ``WORKER_RESULT`` JSON payload the worker printed.
    """
    completed = subprocess.run(
        [sys.executable, str(_WORKER), mode, str(db_path), str(counter_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"worker mode {mode!r} exited {completed.returncode}\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    for line in completed.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            decoded: dict[str, Any] = json.loads(line[len(_RESULT_MARKER) :])
            return decoded
    raise AssertionError(
        f"worker mode {mode!r} printed no result line\nSTDOUT:\n{completed.stdout}"
    )


def _count_invocations(counter_path: Path) -> int:
    """Return how many live leaf invocations the counter file has recorded."""
    if not counter_path.exists():
        return 0
    return sum(1 for line in counter_path.read_text(encoding="utf-8").splitlines() if line)


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


def test_true_cross_process_resume_replays_leaves_at_zero_new_cost(tmp_path: Path) -> None:
    """Two real OS processes share one db; the resume replays every leaf for free.

    This is the M3 headline. Process A (a separate Python interpreter) opens a
    ``SqliteWorkflowStore`` on a temp db, runs a two-leaf workflow whose offline
    fakes record each live invocation to a shared counter file, persists the
    journal and the launch spec, and exits. Process B — a brand-new interpreter —
    reopens the store from the *same* db file, resumes the run by its ``run_id``,
    and must add **zero** new leaf invocations (both completed leaves replay from
    the persisted journal) while returning an identical result.

    The resume side runs with ``checkpointer=None`` on purpose: the persisted
    content-hash journal alone delivers the zero-cost replay, independently of the
    LangGraph task cache.
    """
    db_path = tmp_path / "workflows.db"
    counter_path = tmp_path / "invocations.txt"

    # Process A: run to completion in its own interpreter.
    run_result = _run_worker("run", db_path, counter_path)
    assert run_result["result"] == "plan+draft"
    assert run_result["error"] is None
    # Both leaves ran live exactly once on the first run.
    assert _count_invocations(counter_path) == 2

    # Process B: a fresh interpreter resumes from the shared db file.
    resume_result = _run_worker("resume", db_path, counter_path)
    assert resume_result["result"] == "plan+draft"
    assert resume_result["error"] is None
    # Smoking gun: the resume added NO live invocations — every completed leaf was
    # served from the persisted journal at zero new model cost.
    assert _count_invocations(counter_path) == 2


def test_cross_process_failed_then_retried_replays_completed_leaf_free(tmp_path: Path) -> None:
    """A run that fails mid-flight retries with its completed leaf replayed free.

    Process A runs a workflow that journals its first leaf, then raises a
    mid-flight budget breach before the second — so only the first leaf is
    journaled and the call sequence is never persisted (run-level state is written
    only on a clean return). Process B reopens the same db and retries: the first
    leaf replays from the journal at zero cost (no new invocation), and the
    determinism guard behaves as a fresh recording run because the prior run never
    persisted a sequence, so the same breach surfaces again rather than a spurious
    divergence error.
    """
    db_path = tmp_path / "workflows.db"
    counter_path = tmp_path / "invocations.txt"

    # Process A: fail mid-flight after the first leaf journals.
    run_result = _run_worker("run-fail", db_path, counter_path)
    assert run_result["result"] is None
    assert run_result["error"] == "WorkflowBudgetExceededError"
    # Only the first leaf ran live before the breach.
    assert _count_invocations(counter_path) == 1

    # Process B: retry. The completed leaf replays free; the breach recurs.
    retry_result = _run_worker("resume", db_path, counter_path)
    assert retry_result["error"] == "WorkflowBudgetExceededError"
    # The first leaf was replayed from the journal — no new live invocation — and
    # the run failed again at the same point rather than tripping the determinism
    # guard (the failed run never persisted a sequence to diverge from).
    assert _count_invocations(counter_path) == 1

"""A standalone worker driving a workflow run/resume in a real OS process.

The cross-process headline of the persistence layer is "a fresh process pointed
at the same db file replays completed leaves at zero new model cost". Proving it
needs two *separate* Python processes that share one sqlite db file, so this
module is written to be launched with ``python <this file> <mode> <db> <counter>``
from the integration test, never imported into the test process.

Each fake leaf is offline and deterministic, but it records every live invocation
by appending a line to a counter file on disk. That on-disk count is the
cross-process observable: the run process records its invocations, exits, and the
resume process — reading the same journal from the shared db — must add *zero* new
lines, because every completed leaf replays from the journal.

Modes:
    ``run``     Open the store, run the workflow to completion against the run's
                journal (and the persistent checkpointer), persist the launch
                spec, and exit. Prints the run id and the workflow result.
    ``resume``  Reopen the store from the same db file, load the spec, and re-run
                the *same* orchestration against the run's journal with
                ``checkpointer=None`` — isolating the journal as the sole source
                of zero-cost replay (the checkpointer is deliberately not reused).
                Prints the result.
    ``run-fail``    Like ``run`` but the orchestration raises mid-flight after one
                    leaf completes (a budget breach), so only the first leaf is
                    journaled and the call sequence is never persisted.

The worker prints a single JSON line on stdout so the parent test can parse the
run id and result without scraping logs.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    Roster,
    RunSpec,
    SqliteWorkflowStore,
    WorkflowBudgetExceededError,
    run_workflow,
)

# A fixed run id shared by the run and resume modes: the resume process must
# address the exact run the run process journaled, and both processes derive the
# db / counter paths from argv, so a constant id keeps the two halves aligned
# without threading the id through a file.
_RUN_ID = "cross-process-run"
_THREAD_ID = "cross-process-thread"


def _make_counting_leaf(reply: str, counter_path: Path) -> RunnableLambda[Any, Any]:
    """Build an offline leaf that records each live invocation to ``counter_path``.

    The leaf appends one line per invocation to the counter file before returning
    a deepagent-shaped state dict. Because the file is on disk, the invocation
    count survives the process exit and is readable by the resume process — which
    is exactly how the test proves a replayed leaf never ran live again.

    Args:
        reply: The text the leaf's terminal ``AIMessage`` carries.
        counter_path: The file each invocation appends a line to.

    Returns:
        A runnable that counts its invocation and returns a leaf state dict.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        # Append-then-flush so a crash mid-run still leaves a durable record of the
        # invocations that did happen — the resume side asserts on the delta.
        with counter_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{reply}\n")
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call)


def _roster(counter_path: Path) -> Roster:
    """Assemble the two-leaf roster both modes share."""
    return (
        Roster()
        .register("planner", _make_counting_leaf("plan", counter_path))
        .register("writer", _make_counting_leaf("draft", counter_path))
    )


async def _orchestrate(ctx: Ctx) -> str:
    """Run two sequential leaves and join their folded results.

    The two leaves are journaled by content hash, so a resume replays both for
    free. The join string is the workflow result the test compares across the run
    and resume processes.
    """
    plan = await ctx.agent("Outline the report", agent_type="planner")
    draft = await ctx.agent("Write the report", agent_type="writer")
    return f"{plan}+{draft}"


async def _orchestrate_failing(ctx: Ctx) -> str:
    """Run one leaf, then raise a budget breach before the second.

    Only the first leaf is journaled (success-only), and because the script raises
    before a clean return the call sequence is never persisted. A subsequent run
    therefore replays the first leaf for free while the determinism guard behaves
    as a fresh recording run.
    """
    plan = await ctx.agent("Outline the report", agent_type="planner")
    # Fail loud mid-flight, after the first leaf has completed and journaled.
    raise WorkflowBudgetExceededError(
        f"simulated mid-flight budget breach after producing {plan!r}"
    )


async def _run(db_path: Path, counter_path: Path, *, failing: bool) -> dict[str, Any]:
    """Process A: launch the workflow, persist, and report the run id + result."""
    store = await SqliteWorkflowStore.open(db_path)
    try:
        journal = store.journal_for(_RUN_ID)
        orchestrate = _orchestrate_failing if failing else _orchestrate
        result: str | None
        error: str | None
        try:
            result = await run_workflow(
                orchestrate,
                roster=_roster(counter_path),
                journal=journal,
                checkpointer=store.checkpointer,
                thread_id=_THREAD_ID,
            )
            error = None
        except WorkflowBudgetExceededError as exc:
            # The failing variant raises by design: record the failure so the
            # parent test can assert the run did not complete, then persist the
            # spec anyway so the resume process can rebuild the launch.
            result = None
            error = type(exc).__name__
        await store.save_spec(
            _RUN_ID,
            RunSpec(
                kind="name",
                name_or_source="cross_process_demo",
                args={"failing": failing},
                label="Cross-process demo",
                journal_run_id=_RUN_ID,
            ),
        )
        return {"run_id": _RUN_ID, "result": result, "error": error}
    finally:
        await store.aclose()


async def _resume(db_path: Path, counter_path: Path) -> dict[str, Any]:
    """Process B: reopen the store, replay from the journal, report the result.

    The checkpointer is deliberately ``None`` here so the journal is the sole
    source of zero-cost replay (C5): a fresh process resumes purely from the
    persisted content-hash journal, not the LangGraph task cache.
    """
    store = await SqliteWorkflowStore.open(db_path)
    try:
        spec = await store.load_spec(_RUN_ID)
        if spec is None:
            raise AssertionError(f"resume found no spec for run id {_RUN_ID!r}")
        journal = store.journal_for(_RUN_ID)
        failing = bool(spec.args.get("failing", False))
        orchestrate = _orchestrate_failing if failing else _orchestrate
        try:
            result = await run_workflow(
                orchestrate,
                roster=_roster(counter_path),
                journal=journal,
                checkpointer=None,
                thread_id=spec.journal_run_id or _RUN_ID,
            )
            return {"run_id": _RUN_ID, "result": result, "error": None}
        except WorkflowBudgetExceededError as exc:
            # A retried failing run replays its completed leaf free, then raises
            # the same mid-flight breach again — the determinism guard re-records
            # from scratch because the prior run never persisted its sequence.
            return {"run_id": _RUN_ID, "result": None, "error": type(exc).__name__}
    finally:
        await store.aclose()


def main(argv: list[str]) -> int:
    """Dispatch a worker mode from argv and print a single JSON result line.

    Args:
        argv: ``[mode, db_path, counter_path]``.

    Returns:
        Process exit code: ``0`` on a clean dispatch (including the expected
        mid-flight failure of ``run-fail``, which reports its error in JSON).
    """
    mode, db_arg, counter_arg = argv[0], argv[1], argv[2]
    db_path = Path(db_arg)
    counter_path = Path(counter_arg)
    if mode == "run":
        payload = asyncio.run(_run(db_path, counter_path, failing=False))
    elif mode == "run-fail":
        payload = asyncio.run(_run(db_path, counter_path, failing=True))
    elif mode == "resume":
        payload = asyncio.run(_resume(db_path, counter_path))
    else:
        raise SystemExit(f"unknown worker mode {mode!r}")
    # A unique marker prefix lets the parent test isolate the result line from any
    # incidental stdout (e.g. the engine's default progress sink).
    print("WORKER_RESULT " + json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

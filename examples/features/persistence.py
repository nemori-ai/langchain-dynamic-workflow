"""A durable background workflow survives a process restart at zero replay cost.

The headline of cross-session persistence: a run launched in one process can be
*resumed* in a fresh one — pointed at the same sqlite db file — and every leaf the
first process already completed replays from the persisted content-hash journal at
**zero new model cost**. The checkpointer is the durable add-on; the journal is
what makes the replay free.

This example stages a process restart inside a single ``main`` so it is
self-contained and runnable offline:

1. **Process 1 (launch + persist).** Open a :class:`SqliteWorkflowStore` over a
   temp db file, build the host surface (manager + middleware + host agent), and
   let the scripted host launch a two-leaf background workflow. Both leaves run
   live exactly once and journal their results into the db. Capture the
   ``run_id``.
2. **Simulated restart.** Close the store and drop every in-memory object — the
   manager, the middleware, the tool, the host. Nothing survives but the db file
   on disk, exactly as if the process exited.
3. **Process 2 (reopen + resume).** Reopen ``SqliteWorkflowStore`` from the SAME
   db file, rebuild a brand-new host surface (fresh manager, fresh middleware,
   fresh host), and let the scripted host resume the run by its ``run_id``. The
   workflow re-runs against the persisted journal: both completed leaves replay
   for free and the result is reproduced.

The smoking gun is a per-leaf live-invocation counter that persists across the
"restart" (a real restart only spends money once; the journal in the db is what
spares the second process from re-paying). After the resume the counters are
unchanged — the resumed run added nothing.

Runs fully offline with a deterministic fake; the optional ``sqlite`` extra (in
the dev group, installed by ``uv sync``) backs the durable store.

    uv run python -m examples.features.persistence
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    BgRunManager,
    Ctx,
    Roster,
    SqliteWorkflowStore,
    WorkflowRegistry,
    create_workflow_middleware,
)
from langchain_dynamic_workflow.middleware import WORKFLOW_NOTIFICATION_TAG


class _LeafCounter:
    """A per-leaf live-invocation counter that outlives the simulated restart.

    Each leaf increments this on every *live* call. The journal in the db is what
    makes a resumed leaf replay without a live call, so a flat counter across the
    restart is the observable proof that the resume re-paid for nothing.
    """

    def __init__(self) -> None:
        """Start the live-call tally at zero."""
        self.live_calls = 0


def _build_leaf(reply: str, counter: _LeafCounter) -> Runnable[Any, Any]:
    """Build a leaf that counts each live invocation.

    Args:
        reply: The text the offline fake's terminal ``AIMessage`` carries.
        counter: The counter incremented once per live invocation, shared across
            the restart so the zero-cost claim is checkable.

    Returns:
        A runnable leaf. Every live invocation bumps ``counter`` — and a journal
        replay never invokes it.
    """

    async def _fake(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        counter.live_calls += 1
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_fake)


async def report_workflow(ctx: Ctx, args: dict[str, Any]) -> str:
    """A two-leaf background workflow: outline, then draft, then join.

    Each ``ctx.agent`` call is journaled by content hash, so a resumed run replays
    both leaves from the persisted journal rather than re-invoking them.
    """
    topic: str = args["topic"]
    outline = await ctx.agent(
        f"Outline a short report on {topic} in one line.", agent_type="planner"
    )
    draft = await ctx.agent(f"Write a one-line draft of the {topic} report.", agent_type="writer")
    return f"outline=({outline}) draft=({draft})"


class _ScriptedRunHost(BaseChatModel):
    """Process 1's host: launch the background workflow, then end the turn.

    The turn logic lives in code (the offline-host exemption), but the user-facing
    request still reads like a real person handing off durable work.
    """

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-run-host"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if any("Launched workflow" in m.text for m in tool_messages):
            return _say("Kicked off the report in the background; I'll have it shortly.")
        return _call("run", workflow="report", args={"topic": "tidal energy"})

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


class _ScriptedResumeHost(BaseChatModel):
    """Process 2's host: resume the prior run by id, then confirm it landed.

    It is handed the ``run_id`` recovered from the persisted store; on the first
    turn it resumes that run, and once a completion notification arrives it fetches
    the (replayed) result and folds it into the reply.
    """

    run_id: str
    """The origin ``run_id`` recovered from the persisted store, to be resumed."""

    @property
    def _llm_type(self) -> str:
        return "demo-scripted-resume-host"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        status_done = next((m.text for m in reversed(tool_messages) if "done." in m.text), None)
        notification_seen = any(WORKFLOW_NOTIFICATION_TAG in m.text for m in messages)
        resumed = next((m.text for m in reversed(tool_messages) if "Resumed" in m.text), None)

        if status_done is not None:
            return _say(f"Picked the report back up where it left off — {status_done}")
        if notification_seen and resumed is not None:
            new_run_id = resumed.split("New run_id:")[1].split(".")[0].strip()
            return _call("status", run_id=new_run_id)
        return _call("resume", run_id=self.run_id)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _say(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _call(command: str, **args: Any) -> ChatResult:
    call = AIMessage(
        content="",
        tool_calls=[{"name": "workflow", "args": {"command": command, **args}, "id": command}],
    )
    return ChatResult(generations=[ChatGeneration(message=call)])


def _make_roster(planner_counter: _LeafCounter, writer_counter: _LeafCounter) -> Roster:
    """Assemble the two-leaf roster, reusing the cross-restart counters.

    The roster object is rebuilt for the second process (fresh in-memory state),
    but the leaf runnables wrap the SAME counters, so the live-call tally is
    continuous across the restart — that continuity is the whole point of the
    proof.

    Args:
        planner_counter: The shared counter for the planner leaf's live calls.
        writer_counter: The shared counter for the writer leaf's live calls.

    Returns:
        A roster with a counting ``planner`` and ``writer`` leaf registered.
    """
    return (
        Roster()
        .register("planner", _build_leaf("plan", planner_counter), description="Outlines a report")
        .register("writer", _build_leaf("draft", writer_counter), description="Drafts a report")
    )


async def main() -> None:
    # One temp db file the two processes share; cleaned up on exit.
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "workflows.db"

        # Counters live above both processes: they model the live model cost
        # already spent, which a restart cannot refund — only the journal can spare
        # the second process from re-spending it.
        planner_counter = _LeafCounter()
        writer_counter = _LeafCounter()
        workflows = WorkflowRegistry().register("report", report_workflow)

        # ── Process 1: launch the background workflow and persist it ──────────
        # ``open`` is an async factory (it returns the store after binding the
        # checkpointer to this loop), so ``await`` it before entering the context.
        async with await SqliteWorkflowStore.open(db_path) as store:
            roster = _make_roster(planner_counter, writer_counter)
            manager = BgRunManager()
            middleware = create_workflow_middleware(
                roster,
                workflows=workflows,
                manager=manager,
                store=store,
                checkpointer=store.checkpointer,
            )
            host = create_deep_agent(model=_ScriptedRunHost(), middleware=[middleware])
            config: RunnableConfig = {"configurable": {"thread_id": "session-1"}}

            state1 = await host.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Put together a short report on tidal energy when you get a "
                                "chance — no rush, I'll check back later."
                            ),
                        }
                    ]
                },
                config=config,
            )
            run_id = state1["workflow_runs"][-1]["run_id"]
            print(f"[process 1] host launched background run_id={run_id}")
            print(f"[process 1] host reply: {state1['messages'][-1].text}")

            # Let the background run settle, then confirm both leaves ran live once.
            await manager.wait(run_id, thread_id="session-1")
            print(
                f"[process 1] both leaves ran live once: "
                f"planner={planner_counter.live_calls}, writer={writer_counter.live_calls}"
            )

        # ── Simulated process restart ────────────────────────────────────────
        # The store is closed (context-manager exit) and every in-memory object
        # built above goes out of scope. Only the db file on disk remains.
        del host, middleware, manager, roster
        print(f"[restart] store closed; only the db file survives: {db_path.name}")

        # ── Process 2: reopen the SAME db and resume the run by run_id ────────
        async with await SqliteWorkflowStore.open(db_path) as store:
            roster = _make_roster(planner_counter, writer_counter)
            manager = BgRunManager()
            middleware = create_workflow_middleware(
                roster,
                workflows=workflows,
                manager=manager,
                store=store,
                checkpointer=store.checkpointer,
            )
            host = create_deep_agent(
                model=_ScriptedResumeHost(run_id=run_id), middleware=[middleware]
            )
            config = {"configurable": {"thread_id": "session-2"}}

            # Turn 1: the fresh host resumes the prior run by id (non-blocking).
            state2 = await host.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "I'm back — can you carry on with that tidal energy report?",
                        }
                    ]
                },
                config=config,
            )
            resumed_id = state2["workflow_runs"][-1]["run_id"]
            print(f"[process 2] resumed prior run as new run_id={resumed_id}")

            # Let the resumed run settle; it replays the journal rather than re-running.
            await manager.wait(resumed_id, thread_id="session-2")

            # Turn 2: notification injected; host fetches the (replayed) result.
            state3 = await host.ainvoke(
                {"messages": [{"role": "user", "content": "How did it turn out?"}]},
                config=config,
            )
            print(f"[process 2] host final answer: {state3['messages'][-1].text}")

        # ── The smoking gun: the resume re-paid for nothing ──────────────────
        print(
            f"[proof] live invocations after resume: "
            f"planner={planner_counter.live_calls}, writer={writer_counter.live_calls} "
            f"(unchanged — the resume replayed both leaves from the journal for free)"
        )
        assert planner_counter.live_calls == 1, "planner leaf re-ran on resume (not free)"
        assert writer_counter.live_calls == 1, "writer leaf re-ran on resume (not free)"


if __name__ == "__main__":
    asyncio.run(main())

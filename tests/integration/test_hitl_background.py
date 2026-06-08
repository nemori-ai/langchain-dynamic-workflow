"""HITL sign-off at the background-run layer: park, observe, approve, continue.

These drive a real ``run_workflow`` sign-off through the ``BgRunManager`` (the
host-turn background substrate) with offline fakes. A parked run settles at the
non-terminal ``AWAITING_SIGNOFF`` status, still counts as in-flight, surfaces its
ask, and enqueues a notice; an approve continues the *same* run_id with the human
value and the run completes. Multiple gates park one at a time under one run_id.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow
from langchain_dynamic_workflow._background import BgRunManager, BgRunStateError, BgStatus
from langchain_dynamic_workflow._run_store import InMemoryRunStore
from langchain_dynamic_workflow._workflows import WorkflowRegistry
from langchain_dynamic_workflow.tool import create_workflow_tool


def _runtime(*, thread_id: str, tool_call_id: str = "call-1") -> ToolRuntime[Any, Any]:
    """Build a minimal ToolRuntime carrying the host thread id and call id."""
    return ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"thread_id": thread_id}},
        stream_writer=lambda _chunk: None,
        tool_call_id=tool_call_id,
        store=None,
    )


async def _invoke(tool: Any, args: dict[str, Any], runtime: ToolRuntime[Any, Any]) -> Any:
    """Invoke the tool's async implementation with an injected runtime."""
    return await tool.coroutine(runtime=runtime, **args)


def _run_id_of(command_out: Any) -> str:
    """Extract the launched/continued run_id from a Command update."""
    assert isinstance(command_out, Command)
    update: dict[str, Any] = command_out.update or {}
    run_id = update["workflow_runs"][-1]["run_id"]
    assert isinstance(run_id, str)
    return run_id


def _signoff_coro(
    *,
    journal: InMemoryJournalStore,
    roster: Roster,
    seen: list[Any],
    resume: Any = ...,
) -> Any:
    """Build a coro that runs a one-gate sign-off workflow (optionally resuming)."""

    async def orchestrate(ctx: Any) -> str:
        decision = await ctx.checkpoint({"ask": "approve report?"}, tag="g1")
        seen.append(decision)
        return f"done:{decision}"

    async def _coro() -> str:
        kwargs: dict[str, Any] = {} if resume is ... else {"resume": resume}
        result = await run_workflow(
            orchestrate, roster=roster, journal=journal, thread_id="canon", **kwargs
        )
        return result if isinstance(result, str) else str(result)

    return _coro()


async def test_manager_parks_then_approve_continues_same_run_id() -> None:
    manager = BgRunManager()
    roster = Roster()
    journal = InMemoryJournalStore()
    seen: list[Any] = []

    manager.start(
        _signoff_coro(journal=journal, roster=roster, seen=seen),
        run_id="r1",
        thread_id="t",
        label="signoff-wf",
    )
    await manager.wait("r1", thread_id="t")

    # Parked at the gate: non-terminal, still in-flight, ask is readable.
    assert manager.poll("r1", thread_id="t") == BgStatus.AWAITING_SIGNOFF
    assert manager.get_signoff("r1", thread_id="t") == {"ask": "approve report?"}
    assert manager.active_run_count() == 1

    # The park enqueues a notice so the host learns a sign-off is needed.
    notices = manager.drain_notifications("t")
    assert [n.status for n in notices] == [BgStatus.AWAITING_SIGNOFF]

    # Approve: continue the SAME run_id with the human value.
    manager.approve(
        _signoff_coro(journal=journal, roster=roster, seen=seen, resume="YES"),
        run_id="r1",
        thread_id="t",
    )
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.DONE
    assert manager.get_result("r1", thread_id="t").value == "done:YES"
    assert seen == ["YES"]


async def test_manager_two_gates_park_one_at_a_time_under_one_run_id() -> None:
    manager = BgRunManager()
    roster = Roster()
    journal = InMemoryJournalStore()
    order: list[Any] = []

    async def orchestrate(ctx: Any) -> list[Any]:
        d1 = await ctx.checkpoint({"ask": "gate 1?"}, tag="g1")
        order.append(d1)
        d2 = await ctx.checkpoint({"ask": "gate 2?"}, tag="g2")
        order.append(d2)
        return [d1, d2]

    def coro(resume: Any = ...) -> Any:
        async def _coro() -> str:
            kwargs: dict[str, Any] = {} if resume is ... else {"resume": resume}
            return str(
                await run_workflow(
                    orchestrate, roster=roster, journal=journal, thread_id="canon", **kwargs
                )
            )

        return _coro()

    manager.start(coro(), run_id="r1", thread_id="t")
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.AWAITING_SIGNOFF
    assert manager.get_signoff("r1", thread_id="t") == {"ask": "gate 1?"}

    # Approve gate 1 -> parks again at gate 2, same run_id.
    manager.approve(coro(resume="D1"), run_id="r1", thread_id="t")
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.AWAITING_SIGNOFF
    assert manager.get_signoff("r1", thread_id="t") == {"ask": "gate 2?"}

    # Approve gate 2 -> completes.
    manager.approve(coro(resume="D2"), run_id="r1", thread_id="t")
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.DONE
    assert manager.get_result("r1", thread_id="t").value == "['D1', 'D2']"
    # The script BODY re-executes from the top on each approve (the journal replays
    # gate DECISIONS at zero cost, but `order.append` is a script side effect): the
    # third run replays g1's decision "D1" (appending it again) then consumes "D2".
    # The final RESULT is correct; non-idempotent script side effects repeat. This
    # is the documented re-execution-on-resume boundary (cf. plan D4).
    assert order == ["D1", "D1", "D2"]


async def test_tool_run_status_approve_endtoend() -> None:
    # Drive the whole HITL loop through the host tool surface (not the bare engine):
    # run -> status shows awaiting_signoff + ask -> approve with a decision -> done.
    roster = Roster()

    async def gated(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
        decision = await ctx.checkpoint({"ask": "merge to main?"}, tag="merge")
        return {"merged": bool(decision.get("approved"))}

    workflows = WorkflowRegistry().register("gated", gated)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="h")

    launched = await _invoke(tool, {"command": "run", "workflow": "gated"}, runtime)
    run_id = _run_id_of(launched)
    await manager.wait(run_id, thread_id="h")

    status = await _invoke(tool, {"command": "status", "run_id": run_id}, runtime)
    assert isinstance(status, str)
    assert "awaiting sign-off" in status.lower()
    assert "merge to main?" in status

    runs_listing = await _invoke(tool, {"command": "runs"}, runtime)
    assert isinstance(runs_listing, str)
    assert "awaiting_signoff" in runs_listing

    # Approve with the human decision carried in `args`; the run continues same id.
    approved = await _invoke(
        tool,
        {"command": "approve", "run_id": run_id, "args": {"approved": True}},
        runtime,
    )
    assert _run_id_of(approved) == run_id
    await manager.wait(run_id, thread_id="h")

    final = await _invoke(tool, {"command": "status", "run_id": run_id}, runtime)
    assert isinstance(final, str)
    assert "done" in final.lower()
    assert "'merged': True" in final


async def test_tool_approve_rejects_non_parked_run() -> None:
    # Approving a run that is not awaiting sign-off is refused loud (no relaunch).
    roster = Roster()

    async def quick(ctx: Ctx, args: dict[str, Any]) -> str:
        return "immediate"

    workflows = WorkflowRegistry().register("quick", quick)
    manager = BgRunManager()
    tool = create_workflow_tool(roster, manager=manager, workflows=workflows)
    runtime = _runtime(thread_id="h")

    launched = await _invoke(tool, {"command": "run", "workflow": "quick"}, runtime)
    run_id = _run_id_of(launched)
    await manager.wait(run_id, thread_id="h")

    out = await _invoke(tool, {"command": "approve", "run_id": run_id, "args": {}}, runtime)
    assert isinstance(out, str)
    assert "not awaiting sign-off" in out


async def test_cross_process_approve_is_refused_without_a_live_parked_slot() -> None:
    # Security (review M1/Codex#3): a sign-off can be approved ONLY where its parked run
    # is live. The parked state (which gate, the ask) lives in the in-memory manager, not
    # the run store, so a fresh process (or a swept slot) cannot verify the run is
    # genuinely awaiting a gate — relaunching it could advance a NON-parked run past a
    # sign-off no human saw, or silently drop the decision. So a no-local-slot approve is
    # refused, and the original parked run is left untouched.
    store = InMemoryRunStore()
    roster = Roster()

    async def gated(ctx: Ctx, args: dict[str, Any]) -> str:
        decision = await ctx.checkpoint({"ask": "ship it?"}, tag="ship")
        return f"shipped:{decision.get('ok')}"

    workflows = WorkflowRegistry().register("gated", gated)
    runtime = _runtime(thread_id="h")

    # Process A: launch and park.
    manager_a = BgRunManager()
    tool_a = create_workflow_tool(roster, manager=manager_a, workflows=workflows, store=store)
    launched = await _invoke(tool_a, {"command": "run", "workflow": "gated"}, runtime)
    run_id = _run_id_of(launched)
    await manager_a.wait(run_id, thread_id="h")
    assert manager_a.poll(run_id, thread_id="h") == BgStatus.AWAITING_SIGNOFF

    # Process B: fresh manager, SAME store, no live slot for run_id → refuse loud.
    manager_b = BgRunManager()
    tool_b = create_workflow_tool(roster, manager=manager_b, workflows=workflows, store=store)
    out = await _invoke(
        tool_b, {"command": "approve", "run_id": run_id, "args": {"ok": "YES"}}, runtime
    )
    assert isinstance(out, str)
    assert "not awaiting sign-off on this process" in out
    # The original parked run is untouched (no relaunch advanced it past the gate).
    assert manager_a.poll(run_id, thread_id="h") == BgStatus.AWAITING_SIGNOFF


async def test_double_approve_is_refused_no_orphaned_continuation() -> None:
    # Review H1/Codex#2: approve() flips status to RUNNING synchronously, so a second
    # approve racing the loop is refused (no orphaned second continuation against one
    # journal). The first continuation runs to completion normally.
    manager = BgRunManager()
    roster = Roster()
    journal = InMemoryJournalStore()
    seen: list[Any] = []

    manager.start(
        _signoff_coro(journal=journal, roster=roster, seen=seen),
        run_id="r1",
        thread_id="t",
    )
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.AWAITING_SIGNOFF

    # First approve flips the slot to RUNNING synchronously...
    manager.approve(
        _signoff_coro(journal=journal, roster=roster, seen=seen, resume="YES"),
        run_id="r1",
        thread_id="t",
    )
    assert manager.poll("r1", thread_id="t") == BgStatus.RUNNING
    # ...so a second approve before the loop yields is refused (its coro is closed).
    with pytest.raises(BgRunStateError):
        manager.approve(
            _signoff_coro(journal=journal, roster=roster, seen=seen, resume="NO"),
            run_id="r1",
            thread_id="t",
        )
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.DONE
    assert manager.get_result("r1", thread_id="t").value == "done:YES"
    assert seen == ["YES"]  # only the first approve's decision was applied


async def test_park_ttl_expires_an_abandoned_signoff() -> None:
    # Review M2: an abandoned parked run does not hold a quota slot forever. sweep past
    # park_ttl expires it to CANCELLED (a defended bound), freeing the active count.
    manager = BgRunManager(park_ttl_seconds=100.0)
    roster = Roster()
    journal = InMemoryJournalStore()
    seen: list[Any] = []

    manager.start(
        _signoff_coro(journal=journal, roster=roster, seen=seen),
        run_id="r1",
        thread_id="t",
    )
    await manager.wait("r1", thread_id="t")
    assert manager.poll("r1", thread_id="t") == BgStatus.AWAITING_SIGNOFF
    assert manager.active_run_count() == 1

    # Not yet past the TTL: still parked.
    manager.sweep(now=time.monotonic() + 50.0)
    assert manager.poll("r1", thread_id="t") == BgStatus.AWAITING_SIGNOFF

    # Past the TTL: expired to CANCELLED, no longer holding a quota slot.
    manager.sweep(now=time.monotonic() + 1000.0)
    assert manager.poll("r1", thread_id="t") == BgStatus.CANCELLED
    assert manager.active_run_count() == 0

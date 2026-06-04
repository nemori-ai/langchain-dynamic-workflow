"""Preset roster and workflows for the demo host.

The spike ships one minimal workflow, :func:`hello_workflow`, that emits a couple of
progress entries via ``ctx.phase`` / ``ctx.log`` (no leaf ``agent()`` calls) so the
inline-run progress-streaming path can be proven end to end with no model. Richer
preset workflows (deep research, capstone) are layered on in later phases.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda

from langchain_dynamic_workflow import Ctx, Roster


def _echo_leaf(state: dict[str, Any]) -> dict[str, Any]:
    """Return the incoming messages unchanged (a no-op placeholder leaf)."""
    return {"messages": state["messages"]}


def make_roster() -> Roster:
    """Build the demo leaf roster.

    ``run_workflow`` requires a roster even for a workflow that makes no leaf
    ``agent()`` calls. The spike registers a single trivial echo leaf as a
    placeholder so the contract is satisfied; preset scenarios register real
    deepagent leaves here later.

    Returns:
        A :class:`~langchain_dynamic_workflow.Roster` with one placeholder leaf.
    """
    echo: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(_echo_leaf)
    return Roster().register(
        "echo",
        echo,
        description="Trivial echo leaf (placeholder for the spike).",
    )


async def hello_workflow(ctx: Ctx) -> str:
    """A minimal workflow that narrates two phases and returns a result.

    Uses ``ctx.phase`` / ``ctx.log`` (both synchronous side effects) to drive the
    engine's progress sink, so an inline run streams ``phase_timeline`` events live.

    Args:
        ctx: The orchestration context supplied by ``run_workflow``.

    Returns:
        A short completion string.
    """
    ctx.phase("greeting")
    ctx.log("working...")
    ctx.phase("wrap-up")
    ctx.log("done")
    return "ok"


__all__: list[str] = ["hello_workflow", "make_roster"]

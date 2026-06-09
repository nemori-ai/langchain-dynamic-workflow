"""``workflow()`` named nesting — multiple levels deep, with a cycle guard.

A parent orchestration calls a named child workflow via ``ctx.workflow()``, which
in turn calls a grandchild workflow, which calls a leaf — three levels of nesting,
well under the default cap of 8. The leaf result folds all the way back through
every level so the parent sees it. Named workflows share the parent's journal,
budget, and concurrency gate; each level is just function composition with no
extra overhead.

The demo also proves the cycle guard: a workflow that calls itself is refused with
``WorkflowCycleError`` before any leaf runs. Runs fully offline with a
deterministic fake.

    uv run python -m examples.features.nesting
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    Roster,
    WorkflowCycleError,
    WorkflowRegistry,
    run_workflow,
)

# ── leaf (deterministic, offline fake) ───────────────────────────────────────


def _build_writer(*, response_format: Any = None) -> Any:
    """A deterministic fake that echoes the prompt behind a 'leaf' prefix."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content) if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"leaf({prompt})")]}

    return RunnableLambda(_leaf)


# ── three-level nesting chain ────────────────────────────────────────────────


async def paragraph(ctx: Ctx, args: dict[str, Any]) -> str:
    """Innermost (depth 3): calls a leaf and returns its result."""
    topic: str = args.get("topic", "default")
    return await ctx.agent(f"Write paragraph about {topic}", agent_type="writer")


async def section(ctx: Ctx, args: dict[str, Any]) -> str:
    """Mid level (depth 2): inlines paragraph and wraps the result."""
    topic: str = args.get("topic", "default")
    para = await ctx.workflow("paragraph", {"topic": topic})
    return f"section[{para}]"


async def chapter(ctx: Ctx, args: dict[str, Any]) -> str:
    """Outer level (depth 1): inlines section and wraps the result."""
    topic: str = args.get("topic", "default")
    sec = await ctx.workflow("section", {"topic": topic})
    return f"chapter[{sec}]"


# ── cycle-guard demo ─────────────────────────────────────────────────────────


async def self_referencing(ctx: Ctx, args: dict[str, Any]) -> str:
    """A workflow that calls itself — the cycle guard must refuse it."""
    return await ctx.workflow("self_referencing", args)


# ── main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    roster = Roster().register(
        "writer",
        builder=_build_writer,
        description="Writes a short paragraph on a given topic",
    )
    workflows = (
        WorkflowRegistry()
        .register("paragraph", paragraph)
        .register("section", section)
        .register("chapter", chapter)
        .register("self_referencing", self_referencing)
    )

    # Three-level nesting: orchestrate -> chapter -> section -> paragraph -> leaf.
    async def orchestrate(ctx: Ctx) -> str:
        result = await ctx.workflow("chapter", {"topic": "photosynthesis"})
        return f"document[{result}]"

    document = await run_workflow(
        orchestrate,
        roster=roster,
        workflows=workflows,
        thread_id="nesting-3-levels",
    )
    print(f"three-level nesting:\n  {document}")
    # The leaf result is wrapped by paragraph, section, chapter, and orchestrate in order:
    # document[chapter[section[leaf(Write paragraph about photosynthesis)]]]
    assert document.startswith("document[chapter[section[")
    assert document.endswith(")]]]")
    print("OK: ctx.workflow() inlined 3 levels of named nesting, result folded all the way back.")

    # Cycle guard: a workflow that calls itself must be refused.
    async def trigger_cycle(ctx: Ctx) -> str:
        return await ctx.workflow("self_referencing", {})

    cycle_error: WorkflowCycleError | None = None
    try:
        await run_workflow(
            trigger_cycle,
            roster=roster,
            workflows=workflows,
            thread_id="nesting-cycle",
        )
    except WorkflowCycleError as exc:
        cycle_error = exc
    print(f"cycle refused: {type(cycle_error).__name__}: {cycle_error}")
    assert cycle_error is not None, "a self-referencing workflow must raise WorkflowCycleError"
    print("OK: ctx.workflow() refused the self-cycle with WorkflowCycleError.")


if __name__ == "__main__":
    asyncio.run(main())

"""``ctx.dag`` — dependency-order (topological) fan-out with transitive skip.

A documentation generator whose work has a strict dependency order: the package doc
must exist before each module doc, and a module doc before its symbol docs. The script
declares the graph; the engine runs each node only after its predecessors settle and
feeds their results in. Independent branches run concurrently with no level barrier.
When one node fails, only the nodes that (transitively) depend on it are skipped — the
rest of the graph still completes.

    package mypkg
      |-- module io      -> symbol open, symbol read
      |-- module net     -> symbol fetch        (module net FAILS -> fetch skipped)

Run it:

    uv run python -m examples.features.dag
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import Ctx, DagNode, Roster, run_workflow

# The one module whose documenter fails, to show transitive skip of its symbols.
_FAILING_MODULE = "module net"


def _build_documenter(*, response_format: Any = None) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content)
        if _FAILING_MODULE in prompt:
            raise RuntimeError("documenter crashed on module net")
        return {"messages": [*inp["messages"], AIMessage(content=f"DOC[{prompt}]")]}

    return RunnableLambda(_leaf)


async def generate_docs(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any | None]:
    """package -> modules -> symbols, in dependency order."""
    ctx.phase("generate docs")
    return await ctx.dag(
        [
            DagNode("pkg", deps=[], run=lambda d: ctx.agent("package mypkg", agent_type="doc")),
            DagNode(
                "mod_io",
                deps=["pkg"],
                run=lambda d: ctx.agent(f"module io | {d['pkg']}", agent_type="doc"),
            ),
            DagNode(
                "mod_net",
                deps=["pkg"],
                run=lambda d: ctx.agent(f"module net | {d['pkg']}", agent_type="doc"),
            ),
            DagNode(
                "sym_open",
                deps=["mod_io"],
                run=lambda d: ctx.agent(f"symbol open | {d['mod_io']}", agent_type="doc"),
            ),
            DagNode(
                "sym_read",
                deps=["mod_io"],
                run=lambda d: ctx.agent(f"symbol read | {d['mod_io']}", agent_type="doc"),
            ),
            DagNode(
                "sym_fetch",
                deps=["mod_net"],
                run=lambda d: ctx.agent(f"symbol fetch | {d['mod_net']}", agent_type="doc"),
            ),
        ]
    )


async def main() -> None:
    roster = Roster().register(
        "doc",
        builder=_build_documenter,
        description="Documents one package / module / symbol given its parent's doc",
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any | None]:
        return await generate_docs(ctx, {})

    result = await run_workflow(orchestrate, roster=roster)
    for node_id in sorted(result):
        print(f"{node_id}: {result[node_id]}")

    # Topological data-flow: a symbol doc embeds its module doc, which embeds the package doc.
    assert result["sym_read"] == "DOC[symbol read | DOC[module io | DOC[package mypkg]]]"
    # Transitive skip: module net failed, so its symbol fetch was skipped — but the io
    # branch is independent and completed.
    assert result["mod_net"] is None
    assert result["sym_fetch"] is None
    assert result["sym_open"] is not None
    print("OK: dag ran in topological order and skipped only the failed branch.")


if __name__ == "__main__":
    asyncio.run(main())

"""Integration: ctx.dag topological doc-generation + a named sub-workflow nested >1 level.

The #10 acceptance shape, offline and deterministic: a package doc feeds its module
docs, which feed their symbol docs (package -> module -> symbol topological order), and
the per-symbol step is delegated to a registered sub-workflow inlined two levels deep —
closing requirement ① (dependency-order fan-out + named nesting beyond one level) end
to end. Fake leaves echo their prompt so the assertion can prove the data actually
flowed along the dependency edges and that the nested workflow ran.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_dynamic_workflow import Ctx, DagNode, Roster, run_workflow
from langchain_dynamic_workflow._workflows import WorkflowRegistry


def _echo_leaf() -> RunnableLambda[dict[str, Any], dict[str, Any]]:
    async def _leaf(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content)
        return {"messages": [*inp["messages"], AIMessage(content=f"DOC[{prompt}]")]}

    return RunnableLambda(_leaf)


async def test_dag_doc_generation_topological_with_nested_workflow() -> None:
    roster = Roster().register("documenter", _echo_leaf())

    # A registered sub-workflow that documents one symbol; inlined two levels deep
    # (orchestrate -> document_module(workflow) -> document_symbol(workflow)).
    async def document_symbol(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent(
            f"symbol {args['symbol']} of {args['module_doc']}", agent_type="documenter"
        )

    async def document_module(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow(
            "document_symbol", {"symbol": args["symbol"], "module_doc": args["module_doc"]}
        )

    workflows = (
        WorkflowRegistry()
        .register("document_symbol", document_symbol)
        .register("document_module", document_module)
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any | None]:
        results = await ctx.dag(
            [
                DagNode(
                    "pkg",
                    deps=[],
                    run=lambda d: ctx.agent("package mypkg", agent_type="documenter"),
                ),
                DagNode(
                    "mod_io",
                    deps=["pkg"],
                    run=lambda d: ctx.agent(f"module io | {d['pkg']}", agent_type="documenter"),
                ),
                # symbol node delegates to a sub-workflow nested two levels deep.
                DagNode(
                    "sym_read",
                    deps=["mod_io"],
                    run=lambda d: ctx.workflow(
                        "document_module", {"symbol": "read", "module_doc": d["mod_io"]}
                    ),
                ),
            ]
        )
        return results

    result = await run_workflow(orchestrate, roster=roster, workflows=workflows, thread_id="t1")

    # Topological data-flow: each level's prompt embedded its predecessor's doc.
    assert result["pkg"] == "DOC[package mypkg]"
    assert result["mod_io"] == "DOC[module io | DOC[package mypkg]]"
    # sym_read ran through document_module -> document_symbol (nesting depth 2) and
    # carried the module doc down the chain.
    assert result["sym_read"] == "DOC[symbol read of DOC[module io | DOC[package mypkg]]]"

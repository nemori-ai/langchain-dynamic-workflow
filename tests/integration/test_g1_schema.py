"""Integration: agent(schema=...) over the real engine @task / roster.runnable_for path."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import Roster, run_workflow, run_workflow_from_source
from langchain_dynamic_workflow._journal import InMemoryJournalStore


class Verdict(BaseModel):
    refuted: bool
    reason: str


def _structured_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    # A fake leaf whose output state carries a structured_response matching the
    # bound response_format's schema — stands in for a create_deep_agent built
    # with response_format=ToolStrategy(Verdict).
    async def _call(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": Verdict(refuted=False, reason="solid"),
        }

    return RunnableLambda(_call)


async def test_engine_agent_schema_returns_structured_object() -> None:
    roster = Roster().register("skeptic", builder=_structured_builder)

    async def orchestrate(ctx: Any) -> Any:
        verdict = await ctx.agent("verify X", agent_type="skeptic", schema=Verdict)
        return {"refuted": verdict.refuted, "reason": verdict.reason}

    result = await run_workflow(orchestrate, roster=roster)
    assert result == {"refuted": False, "reason": "solid"}


async def test_schema_in_parallel_fanout() -> None:
    roster = Roster().register("skeptic", builder=_structured_builder)

    async def orchestrate(ctx: Any) -> Any:
        verdicts = await ctx.parallel(
            [
                lambda i=i: ctx.agent(f"claim {i}", agent_type="skeptic", schema=Verdict)
                for i in range(3)
            ]
        )
        return [v.refuted for v in verdicts if v is not None]

    result = await run_workflow(orchestrate, roster=roster)
    assert result == [False, False, False]


async def test_schema_resume_restores_object_from_journal() -> None:
    # Same journal across two runs: the second run hits the cache and the
    # structured object is restored via model_validate_json.
    roster = Roster().register("skeptic", builder=_structured_builder)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any) -> Any:
        v = await ctx.agent("verify", agent_type="skeptic", schema=Verdict)
        return v.reason

    first = await run_workflow(orchestrate, roster=roster, journal=journal)
    second = await run_workflow(orchestrate, roster=roster, journal=journal)
    assert first == second == "solid"


async def test_l2_script_inline_dict_schema_runs_through_gate() -> None:
    # L2 path: a script that passes the AST gate declares an inline dict-literal
    # schema, so structured output works even though imports are forbidden.
    roster = Roster().register("skeptic", builder=_structured_builder)
    source = """
async def orchestrate(ctx, args):
    v = await ctx.agent(
        "verify",
        agent_type="skeptic",
        schema={
            "type": "object",
            "properties": {"refuted": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["refuted", "reason"],
        },
    )
    return v.reason
"""
    result = await run_workflow_from_source(source, roster=roster, args={})
    assert result == "solid"

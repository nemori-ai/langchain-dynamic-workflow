"""Integration: agent(schema=...) over the real engine @task / roster.runnable_for path."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import Roster, run_workflow


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

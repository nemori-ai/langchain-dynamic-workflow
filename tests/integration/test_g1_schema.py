"""Integration: agent(schema=...) over the real engine @task / roster.runnable_for path."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from langchain_dynamic_workflow import Roster, run_workflow, run_workflow_from_source
from langchain_dynamic_workflow._journal import InMemoryJournalStore


class Verdict(BaseModel):
    refuted: bool
    reason: str


class Aliased(BaseModel):
    """A schema whose field carries a serialization alias distinct from its name."""

    text: str = Field(alias="claimText")


def _aliased_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    async def _call(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": Aliased(claimText="hello"),
        }

    return RunnableLambda(_call)


def _structured_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    # A fake leaf standing in for a create_deep_agent built with
    # response_format=ToolStrategy(M): it returns a structured_response that is an
    # instance of the *bound* schema (Verdict for the class path, the converted
    # DynamicSchema for the dict path), faithfully mirroring the real contract.
    async def _call(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        model = response_format.schema if response_format is not None else Verdict
        structured = model.model_validate({"refuted": False, "reason": "solid"})
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": structured,
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


async def test_aliased_schema_survives_resume() -> None:
    # The structured result is dumped to JSON for the journal and re-validated on
    # resume. A schema with a field alias only round-trips if the dump emits the
    # alias (model_validate_json validates by alias by default) — otherwise the
    # second run's cache hit fails to revalidate.
    roster = Roster().register("namer", builder=_aliased_builder)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any) -> Any:
        obj = await ctx.agent("name it", agent_type="namer", schema=Aliased)
        return obj.text

    first = await run_workflow(orchestrate, roster=roster, journal=journal)
    second = await run_workflow(orchestrate, roster=roster, journal=journal)
    assert first == second == "hello"


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

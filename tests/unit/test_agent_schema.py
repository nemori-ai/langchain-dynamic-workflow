"""Unit tests for Ctx.agent(schema=...) — structured output via a fake leaf_runner."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._roster import Roster


class Claim(BaseModel):
    text: str
    confident: bool


def _unused_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    # Never invoked: the Ctx below is driven by a faked leaf_runner, not the
    # roster's builder. Present only so the roster entry is registered with a
    # builder (the schema-capable shape) rather than a pre-built runnable.
    raise AssertionError("builder must not be invoked: leaf_runner is faked")


def _ctx_with_structured(structured: Claim, counter: list[int]) -> Ctx:
    async def _leaf(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
    ) -> LeafOutcome:
        counter[0] += 1
        # Honor the bound schema like a real create_deep_agent(response_format=...):
        # the structured_response is an instance of response_format.schema (the
        # dict path converts to a DynamicSchema), reusing the source object's data.
        obj: BaseModel = structured
        if response_format is not None:
            obj = response_format.schema.model_validate(structured.model_dump())
        return LeafOutcome(state={"messages": [], "structured_response": obj}, usage=7)

    return Ctx(
        roster=Roster().register("x", builder=_unused_builder),  # unused: leaf_runner faked
        journal=InMemoryJournalStore(),
        leaf_runner=_leaf,
    )


async def test_agent_pydantic_schema_returns_validated_object() -> None:
    claim = Claim(text="t", confident=True)
    ctx = _ctx_with_structured(claim, [0])
    out = await ctx.agent("extract", agent_type="x", schema=Claim)
    assert isinstance(out, Claim)
    assert out.text == "t" and out.confident is True


async def test_agent_dict_schema_returns_validated_object() -> None:
    claim = Claim(text="d", confident=False)
    ctx = _ctx_with_structured(claim, [0])
    out = await ctx.agent(
        "extract",
        agent_type="x",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "confident": {"type": "boolean"}},
            "required": ["text", "confident"],
        },
    )
    # Attribute access on the converted model: the dict overload returns a
    # ``BaseModel`` whose dynamically-built fields are read via ``getattr``.
    assert getattr(out, "text") == "d"  # noqa: B009
    assert getattr(out, "confident") is False  # noqa: B009


async def test_agent_schema_journal_roundtrip_caches_object() -> None:
    claim = Claim(text="cached", confident=True)
    counter = [0]
    ctx = _ctx_with_structured(claim, counter)
    first = await ctx.agent("extract", agent_type="x", schema=Claim)
    second = await ctx.agent("extract", agent_type="x", schema=Claim)  # same key -> hit
    assert counter[0] == 1  # leaf ran once; second served from journal
    assert isinstance(first, Claim)
    assert isinstance(second, Claim) and second.text == "cached"


async def test_agent_schema_partitions_journal_key() -> None:
    claim = Claim(text="t", confident=True)
    counter = [0]
    ctx = _ctx_with_structured(claim, counter)
    await ctx.agent("extract", agent_type="x", schema=Claim)
    await ctx.agent("extract", agent_type="x")  # no schema -> different key -> miss
    assert counter[0] == 2

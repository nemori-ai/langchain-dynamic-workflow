"""``agent()`` single-leaf end to end, with a structured-output schema handoff.

A one-leaf workflow asks a geography question and receives a schema-validated
answer object: the leaf hands back a ``structured_response`` and
``ctx.agent(schema=Answer)`` folds it out as a typed ``Answer`` — no prose
parsing. Runs fully offline with a deterministic fake.

    uv run python -m examples.features.agent_and_schema
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from examples._shared.offline_models import structured_builder
from langchain_dynamic_workflow import Ctx, Roster, run_workflow


class Answer(BaseModel):
    """A structured answer to a geography question."""

    capital: str
    country: str


async def main() -> None:
    roster = Roster().register(
        "geographer",
        builder=structured_builder(lambda: Answer(capital="Paris", country="France")),
        description="Answers geography questions as a structured object",
    )

    async def orchestrate(ctx: Ctx) -> Answer:
        return await ctx.agent(
            "What is the capital of France?", agent_type="geographer", schema=Answer
        )

    answer = await run_workflow(orchestrate, roster=roster, thread_id="agent-and-schema")
    print(f"structured answer: {answer!r}")
    assert isinstance(answer, Answer) and answer.capital == "Paris"
    print("OK: a single agent() leaf returned a schema-validated object.")


if __name__ == "__main__":
    asyncio.run(main())

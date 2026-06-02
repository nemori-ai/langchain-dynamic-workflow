"""Phase 3 demo: a budget-guarded loop with progress narration.

The orchestration script accumulates research leaves one at a time, checking the
shared token budget before each iteration and stopping gracefully once the pool
nears exhaustion (``while budget.total and budget.remaining() > THRESHOLD``). It
narrates progress with ``phase`` / ``log``; a usage-metering fake model draws
down the shared budget so the loop terminates without a real API call.

Run it twice against the same journal to see resume in action: the second run
serves every leaf from the journal (zero model calls) yet rebuilds the same
``spent()`` total, and the already-delivered progress narration is suppressed.

Set ``LDW_DEMO_REAL_MODEL`` to drive a real deepagent through OpenRouter instead
(model ``anthropic/claude-opus-4.8``; credentials from a local ``.env``). The
live path needs ``uv sync --group example``.

    uv run python examples/03_loop_until_budget.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from _demo_models import load_demo_env, real_model
from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    ProgressEntry,
    Roster,
    run_workflow,
)

TOPICS = ["batteries", "solar", "wind", "hydrogen", "geothermal", "nuclear", "tidal"]
TOTAL_BUDGET = 50
THRESHOLD = 10
TOKENS_PER_LEAF = 10


class _UsageModel(BaseChatModel):
    """Offline fake model that meters a fixed token cost per call."""

    prefix: str = "note"
    tokens_per_call: int = TOKENS_PER_LEAF

    @property
    def _llm_type(self) -> str:
        return "scripted-budget-demo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].text if messages else ""
        usage = UsageMetadata(
            input_tokens=self.tokens_per_call,
            output_tokens=0,
            total_tokens=self.tokens_per_call,
        )
        message = AIMessage(
            content=f"{self.prefix}({last})",
            usage_metadata=usage,
            response_metadata={"model_name": "scripted-budget-demo"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _build_model() -> Any:
    return real_model() or _UsageModel()


async def orchestrate(ctx: Ctx) -> dict[str, Any]:
    """Accumulate research leaves until the budget nears exhaustion."""
    ctx.phase("budgeted research")
    findings: list[str] = []
    for topic in TOPICS:
        # Loop-until-budget: stop before the shared pool is exhausted. With a
        # total set, remaining() is finite and the loop terminates gracefully.
        if not (ctx.budget.total and ctx.budget.remaining() > THRESHOLD):
            ctx.log(f"stopping: only {int(ctx.budget.remaining())} tokens left")
            break
        ctx.log(f"researching {topic} (remaining={int(ctx.budget.remaining())})")
        findings.append(await ctx.agent(f"Research {topic}", agent_type="researcher"))
    ctx.log(f"done: {len(findings)} findings, spent {ctx.budget.spent()} tokens")
    return {"findings": findings, "spent": ctx.budget.spent()}


async def main() -> None:
    load_demo_env()
    roster = Roster()
    roster.register(
        "researcher",
        create_deep_agent(model=_build_model()),
        description="Researches a single topic",
    )
    journal = InMemoryJournalStore()

    def show(entry: ProgressEntry) -> None:
        print(f"  [{entry.kind.value}] {entry.message}")

    print("first run:")
    first = await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="demo-3a",
        budget=TOTAL_BUDGET,
        on_progress=show,
    )
    print(f"  -> {len(first['findings'])} findings, spent {first['spent']} of {TOTAL_BUDGET}")

    print("resume (same journal, fresh thread):")
    second = await run_workflow(
        orchestrate,
        roster=roster,
        journal=journal,
        thread_id="demo-3b",
        budget=TOTAL_BUDGET,
        on_progress=show,
    )
    print(f"  -> spent {second['spent']} (rebuilt from journal; progress suppressed above)")
    assert first["spent"] == second["spent"], "spent() must rebuild identically on resume"


if __name__ == "__main__":
    asyncio.run(main())

"""Phase 2 demo: parallel + pipeline fan-out orchestrated by ``run_workflow``.

Two fan-out shapes in one script:

1. ``parallel`` — fan out N research leaves with a blocking barrier; a failed
   leaf lands as ``None`` and is filtered out (the call itself never raises).
2. ``pipeline`` — stream the surviving topics through a two-stage, no-barrier
   pipeline (research -> summarize); each item travels independently.

Both share one bounded concurrency gate. Runs offline with a built-in fake model
(no API key needed). Set ``LDW_DEMO_REAL_MODEL=anthropic:claude-haiku-4-5`` (and
an API key) to drive real deepagents instead.

    uv run python examples/02_fanout.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_dynamic_workflow import Ctx, Roster, run_workflow

TOPICS = ["batteries", "solar", "wind", "hydrogen", "geothermal"]


class _ScriptedModel(BaseChatModel):
    """Offline fake model echoing a per-agent prefix plus the last prompt."""

    prefix: str = "note"

    @property
    def _llm_type(self) -> str:
        return "scripted-fanout-demo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].text if messages else ""
        reply = f"{self.prefix}({last})"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _build_model(prefix: str) -> Any:
    spec = os.environ.get("LDW_DEMO_REAL_MODEL")
    if spec:
        from langchain.chat_models import init_chat_model

        return init_chat_model(spec)
    return _ScriptedModel(prefix=prefix)


async def main() -> None:
    roster = Roster()
    roster.register(
        "researcher",
        create_deep_agent(model=_build_model("research")),
        description="Researches a single topic",
    )
    roster.register(
        "summarizer",
        create_deep_agent(model=_build_model("summary")),
        description="Condenses research into a brief",
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        # 1) Parallel fan-out: N independent research leaves, barrier-joined.
        findings = await ctx.parallel(
            [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in TOPICS]
        )
        surviving = [f for f in findings if f is not None]

        # 2) No-barrier pipeline: each topic flows research -> summarize on its own.
        async def research_stage(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"Deep-dive {item}", agent_type="researcher")

        async def summarize_stage(prev: str, item: str, index: int) -> str:
            return await ctx.agent(f"Summarize: {prev}", agent_type="summarizer")

        briefs = await ctx.pipeline(TOPICS, research_stage, summarize_stage)

        return {"parallel_findings": surviving, "pipeline_briefs": briefs}

    result = await run_workflow(orchestrate, roster=roster, thread_id="demo-2", max_concurrency=4)
    print(f"parallel findings ({len(result['parallel_findings'])}):")
    for finding in result["parallel_findings"]:
        print(f"  - {finding!r}")
    print(f"pipeline briefs ({len(result['pipeline_briefs'])}):")
    for brief in result["pipeline_briefs"]:
        print(f"  - {brief!r}")


if __name__ == "__main__":
    asyncio.run(main())

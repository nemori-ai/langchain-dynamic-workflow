"""Phase 1 demo: a single deepagent leaf orchestrated by ``run_workflow``.

Runs offline with a built-in fake model (no API key needed). Set
``LDW_DEMO_REAL_MODEL`` to drive a real deepagent through OpenRouter instead
(model ``anthropic/claude-opus-4.8``; credentials from a local ``.env``). The
live path needs ``uv sync --group example``.

    uv run python examples/01_single_agent.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from _demo_models import demo_cache_middleware, load_demo_env, real_leaf_model
from deepagents import create_deep_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_dynamic_workflow import Ctx, Roster, run_workflow


class _ScriptedModel(BaseChatModel):
    """Offline fake model returning a fixed reply (supports ``bind_tools``)."""

    reply: str = "Paris"

    @property
    def _llm_type(self) -> str:
        return "scripted-demo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def _build_model() -> Any:
    return real_leaf_model() or _ScriptedModel(reply="Paris")


async def main() -> None:
    load_demo_env()
    leaf = create_deep_agent(model=_build_model(), middleware=demo_cache_middleware())
    roster = Roster().register("geographer", leaf, description="Answers geography questions")

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("What is the capital of France?", agent_type="geographer")

    answer = await run_workflow(orchestrate, roster=roster, thread_id="demo-1")
    print(f"workflow result: {answer!r}")


if __name__ == "__main__":
    asyncio.run(main())

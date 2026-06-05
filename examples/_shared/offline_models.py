"""Reusable deterministic fakes for the offline feature demos.

These fakes let every feature demo run with no API key and produce a fixed,
reproducible result. ``ScriptedModel`` is a parameterizable chat fake; the
``*_leaf`` factories build leaf runnables that append an ``AIMessage`` (and,
for the structured variants, a validated ``structured_response``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel


class ScriptedModel(BaseChatModel):
    """A deterministic chat fake that needs no API key.

    Args:
        reply: When set, every generation returns this exact text.
        prefix: Used when ``reply`` is ``None``; the generation echoes the last
            prompt as ``f"{prefix}({last_text})"``.
        tokens_per_call: When set, attaches ``UsageMetadata`` reporting this many
            input tokens, so the model draws down a workflow budget per call.
    """

    reply: str | None = None
    prefix: str = "note"
    tokens_per_call: int | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.reply is not None:
            text = self.reply
        else:
            last = messages[-1].text if messages else ""
            text = f"{self.prefix}({last})"
        if self.tokens_per_call is not None:
            usage = UsageMetadata(
                input_tokens=self.tokens_per_call,
                output_tokens=0,
                total_tokens=self.tokens_per_call,
            )
            message = AIMessage(
                content=text,
                usage_metadata=usage,
                response_metadata={"model_name": self._llm_type},
            )
        else:
            message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Ignore tools and return self (this fake never emits tool calls)."""
        return self


def echo_leaf(prefix: str) -> Runnable[Any, Any]:
    """Build a leaf that echoes a trimmed prompt behind a role prefix."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        last = inp["messages"][-1].text if inp["messages"] else ""
        return {"messages": [*inp["messages"], AIMessage(content=f"{prefix}: {last.strip()[:80]}")]}

    return RunnableLambda(_leaf)


def structured_leaf(structured: BaseModel, *, reply: str = "ok") -> Runnable[Any, Any]:
    """Build a leaf that appends a reply and a fixed validated ``structured_response``.

    Stands in for a ``create_deep_agent(response_format=...)`` leaf so a workflow's
    ``ctx.agent(schema=...)`` call can fold the structured object back out.
    """

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def structured_builder(
    make: Callable[[], BaseModel], *, reply: str = "ok"
) -> Callable[..., Runnable[Any, Any]]:
    """Build a roster ``builder=`` that yields a structured leaf, ignoring ``response_format``."""

    def builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        return structured_leaf(make(), reply=reply)

    return builder


class ToolCallThenReplyModel(BaseChatModel):
    """A fake that attempts one tool call, then gives a verdict after the result lands.

    On the first turn it emits a single ``tool_name`` call writing to ``file_path``;
    once a ``ToolMessage`` is present (e.g. the call was denied at the tool boundary),
    it returns ``verdict``. Drives the read-only-judge demo: the write is refused, yet
    the model still produces its judgement.

    Args:
        tool_name: The tool the fake tries to call (e.g. ``write_file``).
        file_path: The path the fake attempts to write.
        verdict: The text returned once the tool result is observed.
    """

    tool_name: str = "write_file"
    file_path: str = "/fixed.py"
    verdict: str = "Verdict: the original code is unsound; I could not write the fix."

    @property
    def _llm_type(self) -> str:
        return "tool-call-then-reply"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.verdict))])
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": {"file_path": self.file_path, "content": "fix"},
                    "id": "w",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Ignore tools and return self."""
        return self

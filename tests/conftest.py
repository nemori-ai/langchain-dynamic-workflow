"""Shared test fixtures: deterministic fake leaves for green, key-free e2e tests.

Two flavours:
- ``make_deep_leaf``: a real ``deepagents.create_deep_agent`` driven by a custom
  tool-calling-capable fake chat model (counts model calls).
- ``make_fake_leaf``: a lightweight ``RunnableLambda`` leaf (counts calls, can
  fail the first N invocations) for journal/success-only mechanics tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import PrivateAttr


class CountingFakeModel(BaseChatModel):
    """A fake chat model that returns a fixed reply and counts generations.

    Supports ``bind_tools`` (returns itself, ignoring tools) so it can drive a
    full ``create_deep_agent`` without a real tool-calling provider.
    """

    reply: str = "ok"
    _calls: int = PrivateAttr(default=0)

    @property
    def calls(self) -> int:
        """Number of times the model has generated a response."""
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "counting-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls += 1
        message = AIMessage(content=self.reply)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        """Ignore tools and return self (the fake never emits tool calls)."""
        return self


@pytest.fixture
def make_deep_leaf() -> Callable[[str], tuple[Runnable[Any, Any], CountingFakeModel]]:
    """Return a factory building a real deepagent leaf + its counting model."""

    def factory(reply: str) -> tuple[Runnable[Any, Any], CountingFakeModel]:
        model = CountingFakeModel(reply=reply)
        leaf = create_deep_agent(model=model)  # pyright: ignore[reportUnknownVariableType]
        return leaf, model

    return factory


class _FakeLeafState:
    """Mutable call-count holder for a lightweight fake leaf."""

    def __init__(self) -> None:
        self.calls = 0


@pytest.fixture
def make_fake_leaf() -> Callable[..., tuple[Runnable[Any, Any], _FakeLeafState]]:
    """Return a factory building a lightweight ``RunnableLambda`` leaf.

    The leaf appends an ``AIMessage(reply)`` to the input messages. If
    ``fail_times`` > 0 it raises on its first ``fail_times`` invocations.
    """

    def factory(reply: str, *, fail_times: int = 0) -> tuple[Runnable[Any, Any], _FakeLeafState]:
        state = _FakeLeafState()

        async def _call(inp: dict[str, Any]) -> dict[str, Any]:
            state.calls += 1
            if state.calls <= fail_times:
                raise RuntimeError("fake leaf boom")
            messages = [*inp["messages"], AIMessage(content=reply)]
            return {"messages": messages}

        return RunnableLambda(_call), state

    return factory

"""Integration: a read-only judge leaf runs through the real engine roster path.

Registers ``read_only_builder`` as a roster entry and drives ``ctx.agent`` through
``run_workflow`` — exercising a REAL ``create_deep_agent`` leaf end to end (the G1
schema tests use fake leaves; this is the first real-deepagent-through-the-engine
path). A fake model attempts a write inside the judge; the deny-write permission
blocks it and the judge falls back to a verdict, which the workflow returns.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_dynamic_workflow import Roster, read_only_builder, run_workflow


class _WriteThenJudgeModel(BaseChatModel):
    """Fake model: tries to write a file, then (after the deny) returns a verdict."""

    @property
    def _llm_type(self) -> str:
        return "fake-write-then-judge"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        if any(getattr(m, "type", "") == "tool" for m in messages):
            verdict = AIMessage(content="verdict: the code is unsound; I could not edit it")
            return ChatResult(generations=[ChatGeneration(message=verdict)])
        call = AIMessage(
            content="",
            tool_calls=[
                {"name": "write_file", "args": {"file_path": "/fix.py", "content": "x"}, "id": "w"}
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools: Any, **kw: Any) -> BaseChatModel:
        return self


async def test_read_only_judge_runs_through_the_engine() -> None:
    roster = Roster().register("judge", builder=read_only_builder(_WriteThenJudgeModel()))

    async def orchestrate(ctx: Any) -> Any:
        return await ctx.agent("Judge this code; fix it if you can.", agent_type="judge")

    result = await run_workflow(orchestrate, roster=roster)

    # The judge ran end to end through the engine and returned its verdict; its
    # attempt to edit was blocked by the read-only permission (it "could not edit").
    assert "verdict" in result.lower()
    assert "could not edit" in result.lower()

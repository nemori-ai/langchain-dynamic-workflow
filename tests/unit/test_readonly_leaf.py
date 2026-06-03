"""Unit tests for the read-only judge leaf helper.

A read-only leaf can read/grep/glob/ls but a deny-write FilesystemPermission stops
it writing or editing — so a judge built from it physically cannot "fix" what it is
meant only to assess. The test drives a fake model that tries to write and asserts
no file lands.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_dynamic_workflow import read_only_leaf


class _WriteAttemptModel(BaseChatModel):
    """A fake model that tries to write a file, then acknowledges the refusal."""

    @property
    def _llm_type(self) -> str:
        return "fake-write-attempt"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        # Once the deny surfaces as a tool message, stop and acknowledge.
        if any(getattr(m, "type", "") == "tool" for m in messages):
            reply = AIMessage(content="I could not write the file.")
            return ChatResult(generations=[ChatGeneration(message=reply)])
        call = AIMessage(
            content="",
            tool_calls=[
                {"name": "write_file", "args": {"file_path": "/x.txt", "content": "hi"}, "id": "w"}
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools: Any, **kw: Any) -> BaseChatModel:
        return self


async def test_read_only_leaf_cannot_write() -> None:
    leaf = read_only_leaf(_WriteAttemptModel())
    out = await leaf.ainvoke({"messages": [HumanMessage(content="write /x.txt please")]})
    # The deny-write permission blocks the write: no file lands in the state.
    assert "/x.txt" not in out.get("files", {})
    # The refusal surfaces to the model as a tool error, which it acknowledges.
    assert any(
        isinstance(m, AIMessage) and "could not write" in m.text.lower() for m in out["messages"]
    )


async def test_read_only_builder_forwards_response_format() -> None:
    # The builder form (for schema= judges) constructs a read-only leaf per
    # response_format, so it is a valid roster builder.
    from langchain_dynamic_workflow import read_only_builder

    builder = read_only_builder(_WriteAttemptModel())
    leaf = builder(response_format=None)
    out = await leaf.ainvoke({"messages": [HumanMessage(content="write /x.txt please")]})
    assert "/x.txt" not in out.get("files", {})

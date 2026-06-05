"""Integration: the shared offline fakes behave as the feature demos rely on."""

from __future__ import annotations

from examples._shared.offline_models import (
    ScriptedModel,
    ToolCallThenReplyModel,
    echo_leaf,
    structured_builder,
    structured_leaf,
)
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel


class _Verdict(BaseModel):
    ok: bool


async def test_scripted_model_fixed_reply_and_prefix_and_usage() -> None:
    fixed = ScriptedModel(reply="Paris")
    out = fixed.invoke([HumanMessage(content="capital of France?")])
    assert out.content == "Paris"  # pyright: ignore[reportUnknownMemberType]

    prefixed = ScriptedModel(prefix="note")
    out2 = prefixed.invoke([HumanMessage(content="solar")])
    assert out2.content == "note(solar)"  # pyright: ignore[reportUnknownMemberType]

    metered = ScriptedModel(prefix="r", tokens_per_call=10)
    out3 = metered.invoke([HumanMessage(content="x")])
    assert out3.usage_metadata is not None
    assert out3.usage_metadata["total_tokens"] == 10
    # bind_tools returns self so it can drive create_deep_agent.
    assert metered.bind_tools([]) is metered


async def test_echo_and_structured_leaves() -> None:
    leaf = echo_leaf("researcher")
    state = await leaf.ainvoke({"messages": [HumanMessage(content="energy storage")]})
    assert state["messages"][-1].content.startswith("researcher: energy storage")

    sleaf = structured_leaf(_Verdict(ok=True), reply="judged")
    sstate = await sleaf.ainvoke({"messages": [HumanMessage(content="claim")]})
    assert sstate["structured_response"] == _Verdict(ok=True)
    assert sstate["messages"][-1].content == "judged"

    builder = structured_builder(lambda: _Verdict(ok=False))
    built = builder(response_format=None)
    bstate = await built.ainvoke({"messages": [HumanMessage(content="claim")]})
    assert bstate["structured_response"] == _Verdict(ok=False)


async def test_tool_call_then_reply_model() -> None:
    model = ToolCallThenReplyModel(
        tool_name="write_file", file_path="/fixed.py", verdict="could not write; verdict: unsound"
    )
    first = model.invoke([HumanMessage(content="fix it")])
    assert first.tool_calls and first.tool_calls[0]["name"] == "write_file"
    after_deny = model.invoke(
        [HumanMessage(content="fix it"), ToolMessage(content="denied", tool_call_id="w")]
    )
    assert "verdict" in after_deny.content.lower()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    assert model.bind_tools([]) is model

"""Unit tests for leaf result folding."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from langchain_dynamic_workflow import fold_result
from langchain_dynamic_workflow._result import fold_structured


class _V(BaseModel):
    refuted: bool


def test_fold_returns_last_nonempty_ai_text() -> None:
    result = {"messages": [HumanMessage(content="q"), AIMessage(content="the answer")]}
    assert fold_result(result) == "the answer"


def test_fold_skips_trailing_empty_ai_message() -> None:
    result = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="real answer"),
            AIMessage(content=""),
        ]
    }
    assert fold_result(result) == "real answer"


def test_fold_returns_empty_when_no_ai_text() -> None:
    result = {"messages": [HumanMessage(content="q")]}
    assert fold_result(result) == ""


def test_fold_raises_without_messages_key() -> None:
    with pytest.raises(ValueError, match="messages"):
        fold_result({"files": {}})


def test_fold_structured_returns_structured_response() -> None:
    inst = _V(refuted=True)
    state = {"messages": [], "structured_response": inst}
    assert fold_structured(state, _V) is inst


def test_fold_structured_missing_response_fails_loud() -> None:
    with pytest.raises(ValueError, match="structured_response"):
        fold_structured({"messages": []}, _V)

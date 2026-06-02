"""Leaf result folding — the context-quarantine boundary.

Mirrors deepagents' own subagent result extraction: the only thing that crosses
back into the orchestration script is the leaf's folded final text — a single
string. The leaf's intermediate tool calls and messages never enter the caller's
context.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel


def fold_result(result: dict[str, Any]) -> str:
    """Extract the final text from a leaf agent's raw output state.

    Scans ``result["messages"]`` in reverse and returns the first non-empty
    ``AIMessage`` text (skipping a trailing empty ``end_turn`` message, as
    Anthropic models can emit). Returns ``""`` if no non-empty AI text exists.

    Args:
        result: The raw output state of a leaf agent ``ainvoke`` call.

    Returns:
        The folded final text.

    Raises:
        ValueError: If ``result`` has no ``messages`` key.
    """
    if "messages" not in result:
        raise ValueError("leaf result is missing the 'messages' key")
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            text = message.text.rstrip()
            if text:
                return text
    return ""


def fold_structured(result: dict[str, Any], schema: type[BaseModel]) -> BaseModel:
    """Extract the validated structured response from a leaf's output state.

    Used when ``agent()`` was called with a ``schema``: the leaf was built with a
    ``response_format`` so its output state carries a ``structured_response``
    already validated against ``schema``. The intermediate messages never cross
    back — only this object does.

    Args:
        result: The raw output state of a leaf agent ``ainvoke`` call.
        schema: The pydantic model the leaf was bound to (for the error message).

    Returns:
        The validated ``structured_response`` instance.

    Raises:
        ValueError: If ``result`` has no ``structured_response`` (the leaf was not
            built with the expected ``response_format``), or the response is not an
            instance of ``schema`` (the builder bound a mismatched ``response_format``).
    """
    response = result.get("structured_response")
    if response is None:
        raise ValueError(
            f"leaf result has no 'structured_response' for schema {schema.__name__!r}; "
            "the leaf was not built with a matching response_format "
            "(register the agent_type with a builder that forwards response_format)"
        )
    if not isinstance(response, schema):
        raise ValueError(
            f"leaf structured_response is a {type(response).__name__!r}, expected "
            f"{schema.__name__!r}; the agent_type's builder bound a response_format that does "
            "not match the requested schema"
        )
    return response

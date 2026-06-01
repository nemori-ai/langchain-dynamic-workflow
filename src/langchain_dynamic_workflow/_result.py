"""Leaf result folding — the context-quarantine boundary.

Mirrors deepagents' own subagent result extraction: the only thing that crosses
back into the orchestration script is the leaf's folded final text — a single
string. The leaf's intermediate tool calls and messages never enter the caller's
context.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


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

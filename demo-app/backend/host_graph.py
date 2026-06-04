"""Host deepagent graph factory for the interactive demo (minimal spike version).

``langgraph dev`` loads :func:`make_host_graph` (registered under the ``host`` graph
id in ``langgraph.json``) and serves it on the local API. This minimal version is a
bare deepagent host with no workflow wiring yet; later spike tasks layer a ``ui``
state channel and an inline workflow-run path on top.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from _models import resolve_host_model

HOST_INSTRUCTIONS = (
    "You are a helpful assistant for the dynamic-workflow demo. "
    "Answer the user's questions directly and concisely."
)


def make_host_graph() -> Any:
    """Build the host deepagent graph served by ``langgraph dev``.

    Resolves the host model from the environment (a real provider model when a key
    is present, an offline scripted fallback otherwise) so the graph builds and
    registers with or without credentials.

    Returns:
        The compiled deepagent host graph (a runnable LangGraph graph).
    """
    return create_deep_agent(
        model=resolve_host_model(),
        system_prompt=HOST_INSTRUCTIONS,
    )

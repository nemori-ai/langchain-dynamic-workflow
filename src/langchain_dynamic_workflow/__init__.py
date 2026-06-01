"""Deterministic, scripted, resumable multi-agent orchestration for LangChain deepagents.

A community port of Claude Code's Dynamic Workflows: a deterministic orchestration script
owns the control flow (loops, branching, fan-out) while leaf ``agent()`` calls delegate to
deepagents, each running in an isolated, discarded context, so only the final result reaches
the caller.
"""

from importlib.metadata import version

__version__ = version("langchain-dynamic-workflow")

__all__ = ["__version__"]

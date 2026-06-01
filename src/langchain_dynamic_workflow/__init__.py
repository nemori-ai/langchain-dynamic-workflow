"""Deterministic, scripted, resumable multi-agent orchestration for LangChain deepagents.

A community port of Claude Code's Dynamic Workflows: a deterministic orchestration script
owns the control flow (loops, branching, fan-out) while leaf ``agent()`` calls delegate to
deepagents, each running in an isolated, discarded context, so only the final result reaches
the caller.
"""

from importlib.metadata import version

from ._context import Ctx
from ._engine import Orchestrator, run_workflow
from ._errors import WorkflowBudgetExceededError, WorkflowDeterminismError
from ._journal import InMemoryJournalStore, JournalStore, journal_key
from ._result import fold_result
from ._roster import Roster, RosterEntry

__version__ = version("langchain-dynamic-workflow")

__all__ = [
    "Ctx",
    "InMemoryJournalStore",
    "JournalStore",
    "Orchestrator",
    "Roster",
    "RosterEntry",
    "WorkflowBudgetExceededError",
    "WorkflowDeterminismError",
    "__version__",
    "fold_result",
    "journal_key",
    "run_workflow",
]

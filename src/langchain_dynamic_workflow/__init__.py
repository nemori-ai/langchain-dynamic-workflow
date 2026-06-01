"""Deterministic, scripted, resumable multi-agent orchestration for LangChain deepagents.

A community port of Claude Code's Dynamic Workflows: a deterministic orchestration script
owns the control flow (loops, branching, fan-out) while leaf ``agent()`` calls delegate to
deepagents, each running in an isolated, discarded context, so only the final result reaches
the caller.
"""

from importlib.metadata import version

from ._budget import Budget
from ._context import Ctx
from ._engine import Orchestrator, run_workflow
from ._errors import WorkflowBudgetExceededError, WorkflowDeterminismError
from ._journal import InMemoryJournalStore, JournalRecord, JournalStore, journal_key
from ._progress import ProgressEntry, ProgressKind, ProgressSink
from ._result import fold_result
from ._roster import Roster, RosterEntry

__version__ = version("langchain-dynamic-workflow")

__all__ = [
    "Budget",
    "Ctx",
    "InMemoryJournalStore",
    "JournalRecord",
    "JournalStore",
    "Orchestrator",
    "ProgressEntry",
    "ProgressKind",
    "ProgressSink",
    "Roster",
    "RosterEntry",
    "WorkflowBudgetExceededError",
    "WorkflowDeterminismError",
    "__version__",
    "fold_result",
    "journal_key",
    "run_workflow",
]

"""Deterministic, scripted, resumable multi-agent orchestration for LangChain deepagents.

A community port of Claude Code's Dynamic Workflows: a deterministic orchestration script
owns the control flow (loops, branching, fan-out) while leaf ``agent()`` calls delegate to
deepagents, each running in an isolated, discarded context, so only the final result reaches
the caller.
"""

from importlib.metadata import version

from ._background import BgRunManager, BgRunQuotaExceededError, BgStatus, ResultStore
from ._budget import Budget
from ._context import Ctx
from ._engine import Orchestrator, run_workflow
from ._errors import (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowNestingError,
)
from ._journal import InMemoryJournalStore, JournalRecord, JournalStore, journal_key
from ._observability import Span, SpanKind, SpanSink
from ._progress import ProgressEntry, ProgressKind, ProgressSink
from ._result import fold_result
from ._roster import Roster, RosterEntry
from ._sandbox import SandboxManager
from ._workflows import WorkflowRegistry
from .middleware import create_workflow_middleware
from .skills import skill_files, skills_path
from .tool import create_workflow_tool

__version__ = version("langchain-dynamic-workflow")

__all__ = [
    "BgRunManager",
    "BgRunQuotaExceededError",
    "BgStatus",
    "Budget",
    "Ctx",
    "InMemoryJournalStore",
    "JournalRecord",
    "JournalStore",
    "Orchestrator",
    "ProgressEntry",
    "ProgressKind",
    "ProgressSink",
    "ResultStore",
    "Roster",
    "RosterEntry",
    "SandboxManager",
    "Span",
    "SpanKind",
    "SpanSink",
    "WorkflowBudgetExceededError",
    "WorkflowDeterminismError",
    "WorkflowNestingError",
    "WorkflowRegistry",
    "__version__",
    "create_workflow_middleware",
    "create_workflow_tool",
    "fold_result",
    "journal_key",
    "run_workflow",
    "skill_files",
    "skills_path",
]

"""Deterministic, scripted, resumable multi-agent orchestration for LangChain deepagents.

A community port of Claude Code's Dynamic Workflows: a deterministic orchestration script
owns the control flow (loops, branching, fan-out) while leaf ``agent()`` calls delegate to
deepagents, each running in an isolated, discarded context, so only the final result reaches
the caller.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from ._background import BgRunManager, BgRunQuotaExceededError, BgStatus, ResultStore
from ._budget import Budget
from ._codegen import compile_workflow_source, extract_meta, run_workflow_from_source
from ._context import Ctx
from ._engine import Orchestrator, run_workflow
from ._errors import (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowNestingError,
    WorkflowScriptError,
)
from ._journal import InMemoryJournalStore, JournalRecord, JournalStore, journal_key, race_key
from ._leaf_events import LeafEvent, LeafEventSink
from ._leaves import read_only_builder, read_only_leaf
from ._local_subprocess import (
    ExecDecision,
    ExecPolicy,
    ExecRequest,
    LocalSubprocessSandbox,
    RLimitProfile,
)
from ._observability import Span, SpanBegin, SpanBeginSink, SpanKind, SpanSink
from ._progress import ProgressEntry, ProgressKind, ProgressSink
from ._race_types import RaceCandidate, RaceResult
from ._reduce import (
    Consensus,
    Reconciled,
    ReviewItem,
    corroborate,
    dedup,
    reconcile,
    survives,
)
from ._result import fold_result
from ._roster import Roster, RosterEntry
from ._run_store import InMemoryRunStore, RunSpec, WorkflowRunStore
from ._sandbox import SandboxFactory, SandboxManager, local_subprocess_factory
from ._workflows import WorkflowRegistry
from ._worktree import InMemoryWorktreeProvider, WorktreeProvider
from .middleware import create_workflow_middleware
from .skills import skill_files, skills_path
from .tool import create_workflow_tool

if TYPE_CHECKING:
    # Surfaced to type checkers without importing the optional-dependency module
    # at runtime: a bare ``import langchain_dynamic_workflow`` must stay
    # dependency-free, so the concrete symbols are resolved lazily in __getattr__.
    from ._persistence import IncompatibleSchemaError as IncompatibleSchemaError
    from ._persistence import SqliteWorkflowStore as SqliteWorkflowStore

__version__ = version("langchain-dynamic-workflow")


def __getattr__(name: str) -> Any:
    """Resolve the optional sqlite-backed store lazily on first attribute access.

    Keeping ``SqliteWorkflowStore`` out of the eager imports lets a base install
    (without the ``[sqlite]`` extra) ``import langchain_dynamic_workflow`` with no
    sqlite dependency present. The concrete class is imported only when the name
    is actually accessed; if the extra is missing, the import guard in
    ``_persistence`` surfaces a clear "install the [sqlite] extra" ``ImportError``.

    Args:
        name: The attribute requested on the package module.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If ``name`` is not a lazily exported symbol.
        ImportError: If ``SqliteWorkflowStore`` is requested without the
            ``[sqlite]`` extra installed.
    """
    if name == "SqliteWorkflowStore":
        from ._persistence import SqliteWorkflowStore

        return SqliteWorkflowStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BgRunManager",
    "BgRunQuotaExceededError",
    "BgStatus",
    "Budget",
    "Consensus",
    "Ctx",
    "ExecDecision",
    "ExecPolicy",
    "ExecRequest",
    "InMemoryJournalStore",
    "InMemoryRunStore",
    "InMemoryWorktreeProvider",
    "JournalRecord",
    "JournalStore",
    "LeafEvent",
    "LeafEventSink",
    "LocalSubprocessSandbox",
    "Orchestrator",
    "ProgressEntry",
    "ProgressKind",
    "ProgressSink",
    "RLimitProfile",
    "RaceCandidate",
    "RaceResult",
    "Reconciled",
    "ResultStore",
    "ReviewItem",
    "Roster",
    "RosterEntry",
    "RunSpec",
    "SandboxFactory",
    "SandboxManager",
    "Span",
    "SpanBegin",
    "SpanBeginSink",
    "SpanKind",
    "SpanSink",
    "WorkflowBudgetExceededError",
    "WorkflowDeterminismError",
    "WorkflowNestingError",
    "WorkflowRegistry",
    "WorkflowRunStore",
    "WorkflowScriptError",
    "WorktreeProvider",
    "__version__",
    "compile_workflow_source",
    "corroborate",
    "create_workflow_middleware",
    "create_workflow_tool",
    "dedup",
    "extract_meta",
    "fold_result",
    "journal_key",
    "local_subprocess_factory",
    "race_key",
    "read_only_builder",
    "read_only_leaf",
    "reconcile",
    "run_workflow",
    "run_workflow_from_source",
    "skill_files",
    "skills_path",
    "survives",
]

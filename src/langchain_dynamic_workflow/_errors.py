"""Engine-level fail-loud exceptions.

The substrate (LangGraph) treats determinism as a convention and provides no
budget concept at all. The engine raises these explicitly so a replay divergence
or an exhausted token pool surfaces as a loud failure rather than silent
corruption.
"""

from __future__ import annotations


class WorkflowDeterminismError(RuntimeError):
    """Raised when a replay run's ``agent()`` call sequence diverges from the record.

    The journal records the ordered sequence of leaf call-keys on the first run.
    On replay the script must reproduce that exact sequence; if its k-th call-key
    differs from the recorded one (or it issues more calls than were recorded),
    the orchestration is no longer deterministic and feeding it a positionally
    misaligned cache entry would silently corrupt the result. The engine raises
    this instead.
    """


class WorkflowBudgetExceededError(RuntimeError):
    """Raised when a new ``agent()`` call is attempted after the budget is exhausted.

    The shared token budget is checked before a leaf is dispatched: once
    ``budget.spent()`` has reached ``budget.total``, any further ``agent()`` call
    raises this. Leaves already in flight are allowed to finish and their results
    are kept.
    """


class WorkflowNestingError(RuntimeError):
    """Raised when a ``ctx.workflow`` call would exceed the configured nesting cap.

    A workflow script may inline other workflows up to ``max_workflow_depth`` levels
    (default 8). The cap is a runaway-recursion backstop: a legitimate composition
    nests a handful of levels, while an unbounded recursion (a missing base case)
    trips it and fails loud. The cycle guard (``WorkflowCycleError``) catches the
    common recursive pattern earlier and more precisely; this error fires when
    distinct workflow names are nested beyond the cap.

    This is a control-flow signal: a depth-cap breach is a structural/runaway-recursion
    error that must fail loud inside a fan-out frame (``parallel`` / ``pipeline`` /
    ``race`` / ``dag``), not be masked as a ``None`` hole. It belongs in
    ``WORKFLOW_CONTROL_FLOW_SIGNALS`` alongside ``WorkflowDagError`` and
    ``WorkflowCycleError``.
    """


class WorkflowSignoffRequired(RuntimeError):
    """Raised by ``run_workflow`` when the script parked at an in-run sign-off gate.

    A script pauses for a human with ``ctx.checkpoint(ask)``: when the gate has no
    journaled decision and no resume value is pending, the call raises this and it
    propagates out of ``run_workflow`` so the caller (the background run manager)
    parks the run in an ``AWAITING_SIGNOFF`` state rather than settling it done. It
    is a control-flow signal, not a failure: the run is intact and resumes when the
    host supplies a value via ``run_workflow(..., resume=value)`` against the same
    journal, which records the decision and replays completed work at zero cost.

    Attributes:
        ask: The payload the script passed to ``ctx.checkpoint`` — the question or
            context shown to the human deciding the sign-off.
        tag: The optional label the script attached to the gate (empty when none).
        gate_key: The gate's content-hash journal key, for correlation.
    """

    def __init__(self, ask: object, *, tag: str = "", gate_key: str = "") -> None:
        """Carry the sign-off ask, its label, and the gate's journal key.

        Args:
            ask: The payload passed to ``ctx.checkpoint`` (the human-facing ask).
            tag: The optional gate label.
            gate_key: The gate's content-hash journal key for correlation.
        """
        super().__init__(f"workflow parked at sign-off gate (tag={tag!r})")
        self.ask = ask
        self.tag = tag
        self.gate_key = gate_key


class WorkflowCheckpointError(RuntimeError):
    """Raised when ``ctx.checkpoint`` is called from inside a fan-out frame.

    ``ctx.checkpoint`` is a depth-0 (sequential orchestration) primitive. A gate is
    identified by its ordinal position among the run's ``checkpoint`` calls; inside
    a ``parallel`` / ``pipeline`` / ``race`` frame that ordinal would race across
    concurrent frames and become non-deterministic, breaking the journal replay that
    serves an already-approved gate, and a set of concurrently-reached gates has no
    well-defined order to present for sequential human sign-off. The engine refuses
    it rather than producing an unresumable run.
    """


class WorkflowScriptError(RuntimeError):
    """Raised when an LLM-authored orchestration script is rejected before execution.

    The meta layer compiles an untrusted source string into an orchestration
    callable behind a static AST gate. This is raised when the source cannot be
    turned into a safe, runnable script: a syntax error, an AST-gate security
    violation (an import, dunder attribute traversal, a banned builtin, or a
    ``str.format`` attribute-injection vector), or a structural problem (the
    script does not define an ``async def orchestrate(ctx, args)`` coroutine). The
    message enumerates every violation found in one pass so the author can fix them
    all at once and resubmit.

    The AST gate guards against an honest model's slip, not an adversary: an
    in-process restricted ``exec`` is not a security sandbox. Only compile source
    you authored yourself; for adversarial input use an out-of-process isolation
    backend.
    """


class WorkflowDagError(RuntimeError):
    """Raised when a ``ctx.dag`` call is structurally invalid before scheduling.

    The DAG is validated eagerly at the top of ``ctx.dag`` — before any node runs —
    so a duplicate node id, a dependency on an unknown id, a node depending on
    itself, or a dependency cycle fails loud rather than scheduling a graph with no
    topological order. It is a control-flow signal: raised from inside a
    ``parallel`` / ``pipeline`` / ``race`` frame (a nested ``dag``) it must surface,
    never be masked as a ``None`` hole, because it is an author bug, not a leaf
    failure.
    """


class WorkflowCycleError(RuntimeError):
    """Raised when ``ctx.workflow`` would re-enter a workflow already being inlined.

    A workflow may inline other workflows up to ``max_workflow_depth`` levels, but a
    name that is already on the inlining stack (a workflow calling itself directly,
    or a mutual cycle such as A->B->A) has no engine-bounded base case and would
    recurse to the depth cap on every run. The engine refuses the cycle the moment a
    repeated name is seen, with a clearer diagnostic than the eventual depth-cap
    breach. Like the other structural signals it fails loud inside a fan-out frame.
    """


WORKFLOW_CONTROL_FLOW_SIGNALS: tuple[type[Exception], ...] = (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowCheckpointError,
    WorkflowDagError,
    WorkflowCycleError,
    WorkflowNestingError,
)
"""Engine signals that must fail loud inside a fan-out, never be masked as ``None``.

``parallel`` / ``pipeline`` / ``race`` / ``dag`` isolate an ordinary leaf failure as
a quiet ``None`` hole, but these signals indicate the run itself is no longer sound (an
exhausted budget, a replay divergence, a checkpoint reached from a fan-out frame, a
malformed dag graph, a workflow naming cycle, or a depth-cap breach). They are re-raised
after the barrier/drain settles instead of being swallowed, so the breach surfaces rather
than corrupting the result with a hole.
"""

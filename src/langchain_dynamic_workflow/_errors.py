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
    """Raised when a workflow nests another workflow more than one level deep.

    A workflow script may inline another workflow with ``ctx.workflow(name, args)``
    exactly one level; the inner workflow runs in the same durable-execution scope
    and shares the parent's journal, budget, and concurrency gate. Calling
    ``ctx.workflow(...)`` again from inside an already-nested workflow would create
    a second nesting level, which the engine refuses by raising this instead of
    silently allowing unbounded recursion.
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


WORKFLOW_CONTROL_FLOW_SIGNALS: tuple[type[Exception], ...] = (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowCheckpointError,
)
"""Engine signals that must fail loud inside a fan-out, never be masked as ``None``.

``parallel`` / ``pipeline`` / ``race`` isolate an ordinary leaf failure as a quiet
``None`` hole, but these signals indicate the run itself is no longer sound (an
exhausted budget, a replay divergence, or a checkpoint reached from a fan-out
frame). They are re-raised after the barrier/drain settles instead of being
swallowed, so the breach surfaces rather than corrupting the result with a hole.
"""

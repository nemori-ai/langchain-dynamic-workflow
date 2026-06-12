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


class WorkflowConcurrencyError(RuntimeError):
    """Raised when two depth-0 orchestration calls run concurrently at the top level.

    The determinism backstop records the ordered sequence of call-keys on the
    sequential (fan-out depth-0) path and validates that order on resume. Three
    primitives observe into that same ordered sequence at depth 0 — ``agent()``,
    ``race()`` and ``checkpoint()`` — and issuing two of them concurrently at depth 0
    (e.g. a hand-written orchestration doing a raw ``await asyncio.gather(branch_a(ctx),
    branch_b(ctx))`` where each branch calls ``ctx.agent(...)`` / ``ctx.race(...)``)
    observes their keys in wall-clock order, which flips run to run. A
    logically-deterministic resume (same leaves, all journal hits) would then observe
    a different order and trip a spurious :class:`WorkflowDeterminismError` at replay,
    leaving a correct workflow permanently un-resumable.

    The engine refuses this the moment two depth-0 observes overlap, on the *first*
    run, rather than letting the divergence surface as a confusing replay-time false
    positive. Concurrent fan-out must go through ``ctx.parallel()`` / ``ctx.dag()`` /
    ``ctx.race()``: those frames mark the fan-out so their leaves are excluded from
    the positional guard (their completion order is wall-clock-dependent by design)
    and the journal still guards each leaf by content hash.

    Best-effort by design: the guard fires when two depth-0 observes actually
    *overlap*. The only undetected case is two ``agent()`` / ``race()`` calls that are
    BOTH journal cache hits — every depth-0 ``await`` (including a cache hit's
    ``journal.get``) is a scheduling point, so the sibling interleaves and is observed
    while the first is still in flight, EXCEPT when both resolve their hits without the
    event loop preferring the sibling in between, in which case they run effectively
    sequentially in argument order: a deterministic order that resumes stably with no
    order-flip and no spurious determinism error (results stay correct via
    content-hash). That residual case is exactly the benign one. Any call that runs a
    live leaf overlaps and is caught.

    A concurrent ``checkpoint()`` that parks is NOT a concern, but the reason is the
    durable executor, NOT ``gather``. A bare ``asyncio.gather`` with the default
    ``return_exceptions=False`` does NOT cancel its sibling awaitables when one raises
    (per the CPython docs: the others "won't be cancelled and will continue to run") —
    verified directly on this runtime, where the sibling ran to completion. The
    no-work-past-park guarantee instead comes from the engine path: ``orchestrate``
    runs inside a LangGraph ``@entrypoint`` driven by the pregel executor, and when
    that node raises — a ``checkpoint`` park raises :class:`WorkflowSignoffRequired` —
    the durable executor tears the node down and cancels the run's still-pending child
    tasks. The orphaned gathered ``agent()`` task is therefore cancelled at its next
    ``await`` (its ``journal.get`` / leaf dispatch) BEFORE its leaf runs, during
    ``run_workflow``'s own unwind — so even a persistent host loop that stays open
    after the park never lets the leaf run (empirically verified, including a probe
    that kept the loop alive 100ms past the park: the agent was cancelled, the leaf
    never ran). Concurrent fan-out should still use ``ctx.parallel()`` / ``ctx.dag()``
    / ``ctx.race()``: those frames manage sibling teardown deterministically rather
    than relying on the executor's unwind timing.

    This is a depth-0-only signal: the in-flight counter that detects it is gated on
    fan-out depth 0, so it can never be raised from inside a ``parallel`` /
    ``pipeline`` / ``race`` / ``dag`` frame. It is listed in
    ``WORKFLOW_CONTROL_FLOW_SIGNALS`` alongside the other structural signals purely
    for consistency (it would fail loud rather than be masked as a ``None`` hole),
    not because any fan-out path can actually emit it.
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
    WorkflowConcurrencyError,
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

"""The orchestration context (``ctx``) injected into a workflow script.

The context exposes the deterministic fan-out primitives. ``agent()`` runs a
single leaf; ``parallel()`` fans out a list of thunks with a blocking barrier;
``pipeline()`` streams items through stages without a barrier. The content-hash
journal is consulted on every leaf call, so a hit returns the cached result with
zero model calls — that is what makes runs resumable. A shared concurrency gate
bounds the number of in-flight leaves across every fan-out path.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from ._budget import Budget
from ._concurrency import ConcurrencyGate, resolve_max_concurrency
from ._determinism import CallSequenceGuard
from ._errors import (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowNestingError,
)
from ._journal import JournalRecord, JournalStore, journal_key
from ._observability import SpanKind, SpanRecorder
from ._pipeline import Stage, run_pipeline
from ._progress import ProgressKind, ProgressLog
from ._result import fold_result
from ._roster import Roster
from ._sandbox import leaf_id_from_key


@dataclass(frozen=True, slots=True)
class LeafOutcome:
    """The result of invoking a leaf: its raw output state plus token usage.

    Attributes:
        state: The leaf runnable's raw output state (contains ``messages``).
        usage: Total tokens the leaf consumed, metered via the forwarded usage
            callback; ``0`` when the model reported no usage.
    """

    state: dict[str, Any]
    usage: int


class LeafRunner(Protocol):
    """Invokes a resolved leaf as a durable task and returns its outcome.

    The runner receives the leaf's effective model plus the sandbox-admission
    inputs the engine needs to lease the right backend: the derived per-leaf
    identity and whether the leaf requires an isolated execution sandbox. Both
    sandbox arguments are keyword-only with defaults so a runner that ignores
    isolation (e.g. a unit-test fake) stays a valid implementation.
    """

    def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
    ) -> Awaitable[LeafOutcome]:
        """Run the leaf and return its ``LeafOutcome`` (raw state + token usage)."""
        ...


class WorkflowResolver(Protocol):
    """Resolves a workflow name to its orchestration callable.

    Structurally satisfied by
    :class:`~langchain_dynamic_workflow._workflows.WorkflowRegistry`; declared as a
    Protocol here so the context never imports the concrete registry (which itself
    depends on this module), keeping the dependency one-directional.
    """

    def resolve(self, name: str) -> Callable[[Ctx, dict[str, Any]], Awaitable[Any]]:
        """Return the workflow callable registered under ``name``.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        ...


T = TypeVar("T")

_FANOUT_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "langchain_dynamic_workflow_fanout_depth", default=0
)
"""Per-task fan-out nesting depth.

``parallel`` / ``pipeline`` increment this for the duration of their body; an
``agent()`` call sees a non-zero depth exactly when it is dispatched from inside a
fan-out frame. The variable is a :class:`~contextvars.ContextVar` so the depth set
before a fan-out propagates into the child tasks that ``asyncio`` spawns for the
thunks / stage workers (each copies the current context at creation), without any
of those tasks observing a sibling's mutation. The determinism backstop records a
call-key only at depth ``0`` — the sequential path, where positional cache
misalignment is the genuine risk.
"""

_WORKFLOW_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "langchain_dynamic_workflow_workflow_depth", default=0
)
"""How many ``ctx.workflow()`` frames are currently on the stack.

The top-level orchestration script runs at depth ``0``. A ``ctx.workflow(name)``
call enters depth ``1`` (one level of nesting); attempting another
``ctx.workflow()`` from inside that frame would push depth ``2``, which the engine
refuses. The inner workflow runs inline within the parent's entrypoint body and
shares the parent context; its leaf ``agent()`` calls still execute as durable
``@task`` invocations and journal normally, so resumability is unaffected. The
variable is a :class:`~contextvars.ContextVar` so the depth is isolated per
asyncio task and restored on frame exit.
"""


class Ctx:
    """Deterministic orchestration context handed to a workflow script.

    Args:
        roster: The leaf registry.
        journal: The content-hash journal store.
        leaf_runner: Callable that invokes a resolved leaf as a durable task.
        gate: Shared concurrency gate bounding in-flight leaves; a bounded
            default is created when omitted.
        sequence_guard: Determinism backstop recording / validating the ordered
            leaf call-key sequence; a fresh recording guard is created when
            omitted.
        budget: Shared token budget; an unbounded budget is created when omitted.
        progress: Replay-idempotent progress log backing ``phase``/``log``; a
            fresh log delivering to a no-op sink is created when omitted.
        workflows: Optional resolver for ``ctx.workflow(name, args)`` inline
            nesting; when omitted, any ``workflow()`` call raises ``LookupError``.
        spans: Span recorder backing observability-by-default; every
            ``agent``/``parallel``/``pipeline`` call emits a completed span. A
            silent no-op recorder is created when omitted, so observability costs
            nothing until a sink is wired.
    """

    def __init__(
        self,
        *,
        roster: Roster,
        journal: JournalStore,
        leaf_runner: LeafRunner,
        gate: ConcurrencyGate | None = None,
        sequence_guard: CallSequenceGuard | None = None,
        budget: Budget | None = None,
        progress: ProgressLog | None = None,
        workflows: WorkflowResolver | None = None,
        spans: SpanRecorder | None = None,
    ) -> None:
        self._roster = roster
        self._journal = journal
        self._leaf_runner = leaf_runner
        self._gate = (
            gate if gate is not None else ConcurrencyGate(limit=resolve_max_concurrency(None))
        )
        self._sequence_guard = (
            sequence_guard if sequence_guard is not None else CallSequenceGuard(recorded=None)
        )
        self._budget = budget if budget is not None else Budget(total=None)
        self._progress = (
            progress
            if progress is not None
            else ProgressLog(delivered_count=0, sink=lambda _entry: None)
        )
        self._workflows = workflows
        self._spans = spans if spans is not None else SpanRecorder()

    @property
    def observed_call_sequence(self) -> list[str]:
        """The ordered leaf call-keys observed this run (for journal persistence)."""
        return self._sequence_guard.sequence

    @property
    def progress_entry_count(self) -> int:
        """How many progress entries were recorded this run (for journal persistence)."""
        return len(self._progress.entries)

    @property
    def budget(self) -> Budget:
        """The shared token budget for this run (``.total`` / ``.spent()`` / ``.remaining()``)."""
        return self._budget

    def phase(self, title: str) -> None:
        """Open a named progress phase grouping subsequent work.

        Delivery is replay-idempotent: a phase already emitted on a prior run is
        not re-delivered when the script replays on resume.

        Args:
            title: The phase title.
        """
        self._progress.emit(ProgressKind.PHASE, title)

    def log(self, message: str) -> None:
        """Emit a free-form progress narration line.

        Delivery is replay-idempotent: a line already emitted on a prior run is
        not re-delivered when the script replays on resume.

        Args:
            message: The narration text.
        """
        self._progress.emit(ProgressKind.LOG, message)

    async def workflow(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """Inline another workflow by name, exactly one level deep.

        Resolves ``name`` against the workflow registry and runs its orchestration
        callable inline against *this* context, so the inner workflow shares the
        parent's journal, budget, concurrency gate, and progress log — its leaves
        are deduped and budgeted as if written inline, and each still executes as a
        durable ``@task`` leaf so resumability is unaffected. Nesting is allowed
        exactly one level; a ``workflow()`` call from inside an already-nested
        workflow fails loud rather than recursing without bound.

        Args:
            name: The workflow name to resolve in the registry.
            args: Optional arguments passed to the inner orchestration callable;
                an empty mapping is used when omitted.

        Returns:
            Whatever the inner workflow returns.

        Raises:
            LookupError: If no workflow registry is wired into this context.
            KeyError: If ``name`` is not registered.
            WorkflowNestingError: If called from inside an already-nested workflow
                (a second level of inlining).
        """
        if self._workflows is None:
            raise LookupError(
                f"cannot resolve workflow {name!r}: no workflow registry was wired "
                "into this run (pass workflows=... to run_workflow)"
            )
        if _WORKFLOW_DEPTH.get() >= 1:
            raise WorkflowNestingError(
                f"cannot nest workflow {name!r}: workflows may inline another workflow "
                "exactly one level deep, and this call is already inside a nested "
                "workflow (refusing a second nesting level)"
            )
        workflow_fn = self._workflows.resolve(name)  # KeyError on unknown name
        token = _WORKFLOW_DEPTH.set(_WORKFLOW_DEPTH.get() + 1)
        try:
            return await workflow_fn(self, args or {})
        finally:
            _WORKFLOW_DEPTH.reset(token)

    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        model: str | None = None,
        isolation: str = "shared",
    ) -> str:
        """Run a leaf subagent and return its folded final text.

        Resolves ``agent_type`` against the roster, consults the journal, and on
        a miss invokes the leaf and persists the result (success-only).

        The effective model is the ``model`` override when supplied, otherwise the
        roster entry's ``default_model``. That effective value is what is folded
        into the journal key and threaded into the leaf config — so two calls that
        resolve to the same effective model share one cache entry, and the leaf
        sees a single, consistent ``configurable['model']``.

        Model handling is best described as *config propagation*, not a runtime
        model swap: the effective model reaches ``config['configurable']['model']``
        and is honored by config-aware leaves (those that read that key to pick a
        provider). A leaf whose model is bound at construction (e.g. a plain
        ``create_deep_agent``) ignores the config value and runs its built-in
        model; the override still keys the journal distinctly, so it remains a
        deliberate cache-partitioning knob even where it does not swap the model.

        Args:
            prompt: The prompt for the leaf.
            agent_type: The roster name to resolve.
            model: Optional per-call model override. When ``None`` the roster
                entry's ``default_model`` is used as the effective model.
            isolation: Isolation mode (part of the journal key).

        Returns:
            The leaf's folded final text.

        Raises:
            KeyError: If ``agent_type`` is not registered.
            WorkflowBudgetExceededError: If the shared budget is exhausted.
        """
        entry = self._roster.resolve(agent_type)  # fail fast on unknown agent_type
        # Resolve the effective model once: an explicit override wins, otherwise
        # the roster entry's registered default. Both the journal key and the leaf
        # config are derived from this single value so they can never disagree.
        effective_model = model if model is not None else entry.default_model
        key = journal_key(
            prompt=prompt,
            agent_type=agent_type,
            model=effective_model,
            schema=None,
            isolation=isolation,
        )
        # Observability-by-default: the whole leaf lifecycle (determinism check,
        # journal lookup, dispatch) runs inside a span so a trace shows the agent
        # type, the cache outcome, the token usage, and any failure — with no
        # instrumentation in the orchestration script. The span is emitted on exit
        # whether this returns cleanly or raises (e.g. a budget breach).
        with self._spans.span(SpanKind.AGENT, agent_type) as span:
            span.set("agent_type", agent_type)
            # Determinism backstop: record this call-key (fresh run) or validate it
            # against the recorded sequence (replay). A divergence fails loud here,
            # before any cache entry is served. Only the *sequential* agent() path
            # (fan-out depth 0) is recorded: calls dispatched inside parallel() /
            # pipeline() observe in wall-clock completion order, which varies run to
            # run under real (variable-latency) leaves, so recording them would trip
            # the backstop spuriously on a deterministic resume. The journal still
            # guards fan-out leaves by content hash; only their *ordering* is excluded.
            if _FANOUT_DEPTH.get() == 0:
                self._sequence_guard.observe(key)
            cached = await self._journal.get(key)
            if cached is not None:
                # Resume re-counts the cached leaf's usage from the journal record,
                # so spent() rebuilds to the first run's cumulative total without a
                # model call. A cache hit never consumes a budget slot beyond its
                # own usage. The span reports the hit so a trace distinguishes a
                # replayed leaf from a fresh one.
                self._budget.record(key, cached.usage)
                span.set("cached", True)
                span.set("usage_tokens", cached.usage)
                return cached.result
            # Cap is checked only before dispatching a *new* leaf: an exhausted pool
            # refuses fresh work while in-flight leaves finish and keep their results.
            self._budget.ensure_within_cap()
            # Sandbox admission: derive the leaf's stable identity from its
            # content-hash key and tell the runner whether this leaf needs an
            # isolated execution sandbox. The runner (engine side) leases the right
            # backend per leaf_id and threads it into the leaf config; reasoning
            # leaves allocate nothing. The identity is the same key that dedups the
            # journal, so it is stable across retry/resume by construction.
            leaf_id = leaf_id_from_key(key)
            # The gate bounds the number of leaves actually in flight; a journal hit
            # above never consumes a slot, keeping resume cheap. The leaf runner
            # receives the *effective* model so the config it threads matches the key.
            outcome = await self._gate.run(
                lambda: self._leaf_runner(
                    agent_type,
                    prompt,
                    effective_model,
                    leaf_id=leaf_id,
                    needs_execution=entry.needs_execution,
                )
            )
            folded = fold_result(outcome.state)
            # success-only: unreachable if the leaf raised. Usage is journaled so
            # the spend is reconstructable on resume.
            await self._journal.put(key, JournalRecord(result=folded, usage=outcome.usage))
            self._budget.record(key, outcome.usage)
            span.set("cached", False)
            span.set("usage_tokens", outcome.usage)
            return folded

    async def parallel(self, thunks: Sequence[Callable[[], Awaitable[T]]]) -> list[T | None]:
        """Fan out a list of thunks concurrently with a blocking barrier.

        Each thunk is a zero-argument callable returning an awaitable (typically
        a closure over an ``agent()`` call). Results are returned in input order.
        A thunk whose leaf fails lands as ``None`` at its position and never aborts
        the barrier, mirroring Claude Code's ``parallel`` semantics — filter the
        ``None`` holes downstream. Engine control-flow signals are the deliberate
        exception: a ``WorkflowBudgetExceededError`` or ``WorkflowDeterminismError``
        raised inside a thunk is **not** masked as ``None`` but re-raised once the
        barrier settles, so a budget breach or replay divergence inside fan-out
        fails loud rather than corrupting the result with a quiet hole.

        Concurrency is bounded by the shared gate, which is acquired by the leaf
        ``agent()`` calls inside the thunks — not by this fan-out layer itself.
        Gating only at the leaf is what keeps the cap correct under nesting: an
        orchestration frame (e.g. a thunk that itself calls ``parallel``) does not
        hold a slot while it awaits its children, so a ``parallel`` inside a
        ``parallel`` cannot starve the pool into deadlock nor leak slots past the
        cap. The barrier means this returns only once every thunk has settled.

        Args:
            thunks: The zero-argument awaitable factories to fan out.

        Returns:
            A list aligned to ``thunks`` input order; each entry is the thunk's
            result, or ``None`` if its leaf failed.

        Raises:
            WorkflowBudgetExceededError: If a thunk's ``agent()`` call trips the
                budget cap (re-raised after the barrier settles).
            WorkflowDeterminismError: If a thunk's ``agent()`` call diverges from
                the recorded replay sequence (re-raised after the barrier settles).
        """

        async def _guarded(thunk: Callable[[], Awaitable[T]]) -> T | None:
            try:
                return await thunk()
            except (WorkflowBudgetExceededError, WorkflowDeterminismError):
                # Engine control-flow signals must fail loud, never be masked as a
                # leaf failure: re-raise so the barrier surfaces them.
                raise
            except Exception:
                # Failure isolation: one bad leaf must not abort the barrier.
                return None

        # The barrier runs inside a PARALLEL span so a trace shows the fan-out width
        # and how many thunks survived, plus a loud control-flow failure if one
        # escapes the barrier.
        with self._spans.span(SpanKind.PARALLEL, "parallel") as span:
            span.set("thunk_count", len(thunks))
            if not thunks:
                span.set("surviving_count", 0)
                return []
            # Mark the fan-out frame: agent() calls inside the thunks must not
            # record into the determinism backstop (their observe order is
            # non-deterministic). The depth is set before the tasks are spawned so
            # each child copies it.
            token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
            try:
                # return_exceptions keeps the barrier intact (every thunk settles)
                # even when one raises a control-flow signal, so in-flight leaves
                # finish and journal before we fail loud.
                settled = await asyncio.gather(
                    *[_guarded(thunk) for thunk in thunks], return_exceptions=True
                )
            finally:
                _FANOUT_DEPTH.reset(token)
            results: list[T | None] = []
            control_flow_error: BaseException | None = None
            for outcome in settled:
                if isinstance(outcome, BaseException):
                    # Only re-raised control-flow signals reach here (leaf failures
                    # are already None via _guarded). Remember the first; fail loud.
                    control_flow_error = control_flow_error or outcome
                    results.append(None)
                else:
                    results.append(outcome)
            if control_flow_error is not None:
                raise control_flow_error
            span.set("surviving_count", sum(1 for r in results if r is not None))
            return results

    async def pipeline(self, items: Sequence[Any], *stages: Stage) -> list[Any | None]:
        """Stream ``items`` through ``stages`` without a barrier between stages.

        Each item travels through every stage independently — item A can reach the
        last stage while item B is still in the first. Each stage is
        ``(prev_result, original_item, index) -> next_result`` and typically calls
        ``agent()`` internally, so per-leaf journal caching applies: a resumed run
        replays completed leaves from the journal (zero model calls) and only the
        unfinished ones run live. A stage that raises drops that item to ``None``,
        which then skips the remaining stages. Results are returned in input order.

        Concurrency across all stages is bounded by the shared gate.

        Args:
            items: The input items.
            *stages: One or more stage functions applied in order.

        Returns:
            A list aligned to ``items`` input order; each entry is the item's
            final result, or ``None`` if any stage raised for it.

        Raises:
            ValueError: If no stages are supplied.
        """
        # The streaming run is wrapped in a PIPELINE span recording the input width
        # and surviving count (a stage that raises drops its item to None), so a
        # trace shows the pipeline shape without the script instrumenting it.
        with self._spans.span(SpanKind.PIPELINE, "pipeline") as span:
            span.set("item_count", len(items))
            # Mark the fan-out frame so agent() calls inside the stages skip the
            # determinism backstop: items interleave stages by per-leaf completion
            # timing, so the observe order is wall-clock-dependent and would diverge
            # run to run under real leaves. The depth is set before the stage workers
            # are spawned (inside run_pipeline) so each worker task inherits it.
            token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
            try:
                results = await run_pipeline(items, stages, gate=self._gate)
            finally:
                _FANOUT_DEPTH.reset(token)
            span.set("surviving_count", sum(1 for r in results if r is not None))
            return results

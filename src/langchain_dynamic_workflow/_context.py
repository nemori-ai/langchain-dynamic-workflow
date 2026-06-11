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
import contextlib
import contextvars
import json
import time
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
    Sized,
)
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast, overload

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

from ._budget import Budget
from ._concurrency import ConcurrencyGate, resolve_max_concurrency
from ._dag import DagNode, run_dag
from ._determinism import CallSequenceGuard
from ._errors import (
    WORKFLOW_CONTROL_FLOW_SIGNALS,
    WorkflowCheckpointError,
    WorkflowConcurrencyError,
    WorkflowCycleError,
    WorkflowNestingError,
    WorkflowSignoffRequired,
)
from ._journal import JournalRecord, JournalStore, journal_key, race_key, signoff_key
from ._observability import SpanKind, SpanRecorder
from ._pipeline import Stage, run_pipeline
from ._progress import BatchMetrics, ProgressKind, ProgressLog
from ._race_types import RaceCandidate, RaceResult
from ._result import fold_result, fold_structured
from ._roster import Roster
from ._sandbox import leaf_id_from_key
from ._schema import to_pydantic_model

WORKTREE_CHANGESET_KEY = "__ldw_worktree_changeset__"
"""Reserved key under which the engine folds a worktree leaf's authoritative diff.

For a real-git worktree execution leaf the engine collects the leaf's real
``git diff`` while the lease is still held and stashes it in the leaf's output
state under this key (so it rides into the content-hash journal). ``agent()`` then
treats that on-disk truth as the authoritative changeset, overriding any file bytes
the model self-reported in its schema. The double-underscore name keeps it from
colliding with a real leaf-state key.
"""

_WORKTREE_FILES_FIELD = "files"
"""Schema field whose value the authoritative worktree changeset overrides.

A worktree fixer's structured schema reports its change under ``files``; when the
engine folds in the real ``git diff``, ``agent()`` overrides exactly this field so
the model-claimed bytes can never win over the on-disk truth. Other metadata
fields (``summary`` and the like) are left untouched.
"""


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

    def to_payload(self) -> dict[str, Any]:
        """Return a msgpack-native mapping carrying this outcome's fields.

        A durable ``@task`` return crosses the checkpointer, whose serializer
        round-trips built-in containers (and registered LangChain message types
        nested inside ``state``) without an opaque object-reconstruction hop. A
        custom class instance, by contrast, is only revived through a deprecated
        unregistered-type path that newer serializers block. Returning this plain
        mapping from the task boundary keeps the wrapper itself strictly
        serializable while the engine still works with the typed
        :class:`LeafOutcome` everywhere else.

        The strict-safe guarantee is scoped to the :class:`LeafOutcome` wrapper
        and the *registered* state it carries (the LangChain message and container
        types nested in ``state``). It does NOT extend to an unregistered user
        type: a structured-output leaf leaves its validated pydantic model under
        ``state['structured_response']``, and under strict msgpack that model dumps
        fine but loads back as a plain ``dict`` (the unregistered-type revival path
        is blocked). This is a documented serialization boundary, not a replay
        defect: the headline zero-cost replay reads the content-hash journal, which
        stores the folded result *string* (a schema-bound leaf stores its
        ``model_dump_json``), never the checkpoint state — so a degraded
        ``structured_response`` in a persisted checkpoint never reaches the script.

        Returns:
            A mapping with ``state`` and ``usage`` keys.
        """
        return {"state": self.state, "usage": self.usage}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LeafOutcome:
        """Rebuild a :class:`LeafOutcome` from a :meth:`to_payload` mapping.

        Args:
            payload: A mapping produced by :meth:`to_payload`, carrying ``state``
                and ``usage``.

        Returns:
            The reconstructed outcome.
        """
        return cls(state=payload["state"], usage=payload["usage"])


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
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
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
M = TypeVar("M", bound=BaseModel)

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

The top-level orchestration script runs at depth ``0``. Each ``ctx.workflow(name)``
call increments this by one; the engine refuses a call that would push it past
``max_workflow_depth`` (a runaway-recursion backstop). The inner workflow runs
inline within the parent's entrypoint body and shares the parent context; its leaf
``agent()`` calls still execute as durable ``@task`` invocations and journal
normally, so resumability is unaffected. The variable is a
:class:`~contextvars.ContextVar` so the depth is isolated per asyncio task and
restored on frame exit.
"""

DEFAULT_MAX_WORKFLOW_DEPTH = 8
"""Default cap on ``ctx.workflow()`` inline nesting depth.

The cap is a runaway-recursion backstop, not a semantic limit: a legitimate
composition nests a handful of levels, while an unbounded recursion (a missing base
case) trips this and fails loud. The cycle guard catches the common case (a name
re-entering itself) earlier and more precisely.
"""

_WORKFLOW_NAME_STACK: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "langchain_dynamic_workflow_workflow_name_stack", default=frozenset()
)
"""Names of the workflows currently inlined on the ``ctx.workflow`` call stack.

A ``ctx.workflow(name)`` whose ``name`` is already in this set is a cycle (a workflow
re-entering itself, directly or via a mutual A->B->A) and is refused. The variable is a
:class:`~contextvars.ContextVar` so the stack is isolated per asyncio task and
restored on frame exit, exactly like ``_WORKFLOW_DEPTH``.
"""


class _Unset:
    """Sentinel type distinguishing "no resume value" from "resume with ``None``"."""


UNSET = _Unset()
"""Default for a sign-off resume: no human value is being injected this run."""


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
        pending_signoff: Optional human value injected at the next un-decided
            ``ctx.checkpoint`` sign-off gate (an approve resume sets it). Omitted
            (the ``UNSET`` default) on a fresh run or a crash-resume replay.
        max_workflow_depth: Cap on ``ctx.workflow`` inline nesting depth (a
            runaway-recursion backstop); defaults to ``DEFAULT_MAX_WORKFLOW_DEPTH``.
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
        pending_signoff: Any = UNSET,
        max_workflow_depth: int = DEFAULT_MAX_WORKFLOW_DEPTH,
    ) -> None:
        self._roster = roster
        self._journal = journal
        self._leaf_runner = leaf_runner
        # The human value to inject at the next un-decided sign-off gate this run
        # (set by an approve resume). Consumed by the first ``checkpoint`` that has
        # no journaled decision; gates already approved on a prior run replay their
        # decision from the journal and never consume it. ``UNSET`` (not ``None``)
        # marks "no value injected" so approving with ``None`` is honored.
        self._pending_signoff = pending_signoff
        # Ordinal of the next ``checkpoint`` call on the sequential path; keys each
        # gate's journaled decision so gates stay distinct and replay in order.
        self._signoff_count = 0
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
        self._max_workflow_depth = max_workflow_depth
        # Count of depth-0 sequence-guard ``observe`` sites currently in flight —
        # ``agent()``, ``race()``, and ``checkpoint()`` all record into the SAME
        # ordered determinism sequence at fan-out depth 0. The backstop validates that
        # order on resume; two such depth-0 observes running concurrently (e.g. a raw
        # ``asyncio.gather`` of two ``ctx.agent`` / ``ctx.race`` branches) would
        # observe their keys in wall-clock order, which flips run to run and spuriously
        # trips the backstop on a deterministic resume. This counter detects that and
        # fails loud on the first run instead. It is a plain instance attribute (not a
        # ContextVar) precisely because the gathered branches must see each other's
        # increment: ``asyncio`` copies the context into each spawned task, so a
        # ContextVar mutation in one branch would be invisible to its sibling, whereas
        # the shared ``ctx`` object is visible to both. The check and increment happen
        # synchronously (no await between them) so two concurrent entries cannot both
        # pass: the first increments before its first await, and the second's
        # synchronous check then sees the counter at ``1``. Every site that increments
        # decrements in a ``finally`` on every exit path. ``_observe_depth0`` is the
        # single choke point all three sites share.
        self._depth0_inflight = 0

    @property
    def observed_call_sequence(self) -> list[str]:
        """The ordered leaf call-keys observed this run (for journal persistence)."""
        return self._sequence_guard.sequence

    @property
    def progress_entry_count(self) -> int:
        """How many progress entries were recorded this run (for journal persistence)."""
        return len(self._progress.entries)

    @property
    def has_unconsumed_signoff(self) -> bool:
        """Whether an injected sign-off resume value was never consumed by a gate.

        ``True`` when the run was given a ``resume`` decision but completed without
        reaching an un-decided ``ctx.checkpoint`` gate to consume it — the engine uses
        this to fail loud rather than let a human sign-off decision vanish silently.
        """
        return not isinstance(self._pending_signoff, _Unset)

    @property
    def budget(self) -> Budget:
        """The shared token budget for this run (``.total`` / ``.spent()`` / ``.remaining()``)."""
        return self._budget

    def _observe_depth0(self, key: str) -> bool:
        """Record ``key`` into the determinism sequence, guarding depth-0 concurrency.

        The single choke point shared by every site that observes into the ordered
        determinism sequence at fan-out depth 0 (``agent`` / ``race`` / ``checkpoint``).
        At depth 0 it fails loud if another depth-0 observe is already in flight (two
        concurrent depth-0 calls observe in wall-clock order, which flips run to run
        and would trip a spurious determinism failure on a faithful resume), otherwise
        records the key and increments the in-flight counter. At depth > 0 it is a
        no-op (fan-out leaf ordering is excluded from the sequence by design).

        The check, ``observe``, and increment run synchronously (no ``await`` between
        them) so two concurrent depth-0 entries cannot both pass: the first increments
        before its caller's first ``await``, and the second's synchronous check then
        sees the counter at ``1``.

        Args:
            key: The content-hash key to record into the determinism sequence.

        Returns:
            ``True`` if this call incremented the in-flight counter — the caller MUST
            then decrement ``self._depth0_inflight`` in a ``finally`` on every exit
            path. ``False`` at depth > 0 (nothing was observed or incremented).

        Raises:
            WorkflowConcurrencyError: At depth 0, if another depth-0 observe is already
                in flight (concurrent fan-out at the top level).
        """
        if _FANOUT_DEPTH.get() != 0:
            return False
        if self._depth0_inflight >= 1:
            raise WorkflowConcurrencyError(
                "two depth-0 orchestration calls (agent() / race() / checkpoint()) are "
                "running concurrently at the top level (fan-out depth 0); this is "
                "non-deterministic for resume — their observe order flips run to run and a "
                "faithful resume would trip a spurious determinism failure. Concurrent "
                "fan-out must use ctx.parallel() / ctx.dag() / ctx.race(), which mark their "
                "fan-out so the determinism guard correctly excludes the leaf ordering (each "
                "leaf is still guarded by its content hash)."
            )
        self._sequence_guard.observe(key)
        self._depth0_inflight += 1
        return True

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
        """Inline another workflow by name, up to ``max_workflow_depth`` levels deep.

        Resolves ``name`` against the workflow registry and runs its orchestration
        callable inline against *this* context, so the inner workflow shares the
        parent's journal, budget, concurrency gate, and progress log — its leaves
        are deduped and budgeted as if written inline, and each still executes as a
        durable ``@task`` leaf so resumability is unaffected. A cycle (a workflow
        re-entering itself, directly or via a mutual A->B->A) is refused the moment
        a repeated name is seen; nesting beyond ``max_workflow_depth`` levels fails
        loud as a runaway-recursion backstop.

        Args:
            name: The workflow name to resolve in the registry.
            args: Optional arguments passed to the inner orchestration callable;
                an empty mapping is used when omitted.

        Returns:
            Whatever the inner workflow returns.

        Raises:
            LookupError: If no workflow registry is wired into this context.
            KeyError: If ``name`` is not registered.
            WorkflowCycleError: If ``name`` is already on the inlining stack (a
                workflow re-entering itself, directly or via a mutual cycle).
            WorkflowNestingError: If nesting depth would exceed ``max_workflow_depth``
                (a runaway-recursion backstop).
        """
        if self._workflows is None:
            raise LookupError(
                f"cannot resolve workflow {name!r}: no workflow registry was wired "
                "into this run (pass workflows=... to run_workflow)"
            )
        stack = _WORKFLOW_NAME_STACK.get()
        if name in stack:
            raise WorkflowCycleError(
                f"cannot inline workflow {name!r}: it is already on the inlining stack "
                f"{sorted(stack)} — a workflow that re-enters itself (directly or via a "
                "cycle) has no engine-bounded base case; refusing the cycle"
            )
        if _WORKFLOW_DEPTH.get() >= self._max_workflow_depth:
            raise WorkflowNestingError(
                f"cannot inline workflow {name!r}: nesting depth would exceed "
                f"max_workflow_depth={self._max_workflow_depth} (runaway-recursion backstop)"
            )
        workflow_fn = self._workflows.resolve(name)  # KeyError on unknown name
        depth_token = _WORKFLOW_DEPTH.set(_WORKFLOW_DEPTH.get() + 1)
        stack_token = _WORKFLOW_NAME_STACK.set(stack | {name})
        try:
            return await workflow_fn(self, args or {})
        finally:
            _WORKFLOW_NAME_STACK.reset(stack_token)
            _WORKFLOW_DEPTH.reset(depth_token)

    async def checkpoint(self, ask: Any, *, tag: str = "") -> Any:
        """Pause the run for a human sign-off and return the value they supply.

        Surfaces ``ask`` (the question or context for the person deciding) and
        parks the run until the host approves it with a value, which becomes this
        call's return value. The decision is journaled like a leaf result, keyed by
        the gate's ordinal position, so an approve re-runs the script from the top
        with completed leaves and already-approved gates replayed from the journal
        at zero model cost, and only this un-decided gate consumes the new value.
        Sign-off semantics are a pattern, not extra API: the script interprets the
        returned value (e.g. an ``{"approved": bool}`` mapping it agreed with the
        host).

        This is a sequential-orchestration-path (depth-0) primitive: calling it
        from inside a ``parallel`` / ``pipeline`` / ``race`` fan-out frame raises
        :class:`WorkflowCheckpointError`. A gate is identified by its ordinal among
        the run's ``checkpoint`` calls; inside a fan-out that ordinal would race
        across concurrent frames and become non-deterministic, breaking replay, and
        a set of concurrently-reached gates has no well-defined order to present for
        sequential human sign-off.

        Args:
            ask: The human-facing question/context surfaced while the run is
                parked; carried on the raised :class:`WorkflowSignoffRequired`.
            tag: Optional label for the gate, folded into its journal key and
                echoed back on the parked status.

        Returns:
            The value supplied on approve (``run_workflow(..., resume=value)``),
            normalized through a JSON round-trip. Because the decision is
            journaled as JSON and read back the same way, the script sees the
            identical post-round-trip shape on the approving run and on every
            replay (e.g. a tuple becomes a list, int dict-keys become str keys),
            so the value is type-stable across the human pause and across
            replays. Supply only JSON-stable shapes to a script that depends on
            the decision's runtime type.

        Raises:
            WorkflowCheckpointError: If called from inside a fan-out frame.
            WorkflowSignoffRequired: If this gate has no journaled decision and no
                resume value is pending — the run parks here.
        """
        if _FANOUT_DEPTH.get() > 0:
            raise WorkflowCheckpointError(
                "ctx.checkpoint must run on the sequential orchestration path, not "
                "inside a parallel/pipeline/race fan-out frame: a gate is keyed by its "
                "ordinal position, which would race across concurrent frames and break "
                "deterministic replay, and concurrently-reached gates have no order to "
                "present for sequential human sign-off."
            )
        position = self._signoff_count
        self._signoff_count += 1
        key = signoff_key(position=position, tag=tag)
        # Determinism backstop: record this gate's key in the SAME ordered sequence as
        # leaf calls, so a replay whose gate order/identity drifts (an inserted/removed
        # leaf or gate before it, or a different tag at this position) fails loud here
        # rather than silently binding a journaled decision to a different gate — a leaf
        # diverges loud, and a gate must too. A gate's identity is (position, tag): the
        # ``ask`` is excluded from the key because it may carry non-deterministic content
        # (e.g. a model-produced assessment), so DISTINCT gates MUST use DISTINCT tags,
        # and editing the script's gate/leaf structure invalidates a parked journal (same
        # boundary as leaf keys). The guard catches drift across a full resume; across a
        # single park→approve the sequence is not yet persisted (it records, not validates).
        # ``_observe_depth0`` shares the depth-0 concurrency guard with agent()/race():
        # checkpoint already refuses depth > 0 above, so this always observes + increments
        # at depth 0; the finally below decrements on every exit (cache replay, approve,
        # JSON error, park) so a depth-0 checkpoint concurrent with another depth-0 observe
        # fails loud rather than racing the shared sequence.
        counted = self._observe_depth0(key)
        try:
            # A gate approved on a prior run replays its decision from the journal at
            # zero cost, keeping the run deterministic across the human pause.
            recorded = await self._journal.get(key)
            if recorded is not None:
                return json.loads(recorded.result)
            # The first un-decided gate consumes the pending resume value (an approve);
            # record it so a later resume replays this gate instead of re-asking.
            if not isinstance(self._pending_signoff, _Unset):
                decision = self._pending_signoff
                # Serialize BEFORE consuming the pending value: the decision is journaled
                # as the gate's record, so a non-JSON-serializable decision must fail with
                # a clear error and leave the gate un-decided (still re-approvable) rather
                # than losing the value and parking unjournaled.
                try:
                    serialized = json.dumps(decision)
                except TypeError as exc:
                    raise WorkflowCheckpointError(
                        "ctx.checkpoint decision (the resume value) must be JSON-serializable "
                        f"— it is journaled as the gate's recorded decision: {exc}"
                    ) from exc
                self._pending_signoff = UNSET
                await self._journal.put(key, JournalRecord(result=serialized, usage=0))
                # Return the JSON-normalized shape the replay branch above returns
                # (json.loads of the same serialized record), not the raw object: a
                # replay reads the decision back via json.loads, so handing the script
                # the un-round-tripped object here would make the SAME gate type-unstable
                # (raw tuple / int keys on the approve, list / str keys on every replay)
                # — replay drift the journal exists to prevent.
                return json.loads(serialized)
            # No decision and nothing to inject: park here for a human sign-off. The
            # finally still decrements so a resume's re-reached gate is not flagged.
            raise WorkflowSignoffRequired(ask, tag=tag, gate_key=key)
        finally:
            if counted:
                self._depth0_inflight -= 1

    @overload
    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        schema: type[M],
        model: str | None = ...,
        isolation: str = ...,
    ) -> M: ...

    @overload
    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        schema: dict[str, Any],
        model: str | None = ...,
        isolation: str = ...,
    ) -> BaseModel: ...

    @overload
    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        schema: None = ...,
        model: str | None = ...,
        isolation: str = ...,
    ) -> str: ...

    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        model: str | None = None,
        isolation: str = "shared",
    ) -> str | BaseModel:
        """Run a leaf subagent and return its folded result.

        Without ``schema`` the folded final text is returned. With ``schema`` (a
        pydantic ``BaseModel`` subclass or an inline JSON-schema ``dict``) the leaf
        is built with a matching ``response_format`` and the validated structured
        object is returned — the script reads it by attribute.

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
            schema: Optional structured-output schema (pydantic class or JSON-schema
                dict). Requires the roster entry to be registered with a builder.
            model: Optional per-call model override. When ``None`` the roster
                entry's ``default_model`` is used as the effective model.
            isolation: Isolation mode (part of the journal key).

        Returns:
            The folded final text, or the validated structured object when
            ``schema`` is given.

        Raises:
            KeyError: If ``agent_type`` is not registered.
            ValueError: If ``schema`` is given for a runnable-only roster entry, or
                a dict schema uses an unsupported construct.
            WorkflowBudgetExceededError: If the shared budget is exhausted.
        """
        entry = self._roster.resolve(agent_type)  # fail fast on unknown agent_type
        # isolation="worktree" needs an execution sandbox to seed into; a reasoning
        # leaf has none, so requesting a worktree on it is a contradiction — fail
        # loud rather than silently hand back an unseeded StateBackend.
        if isolation == "worktree" and not entry.needs_execution:
            raise ValueError(
                f"isolation='worktree' requires agent_type {agent_type!r} to be registered "
                "with needs_execution=True (a worktree is seeded into an execution sandbox; "
                "a reasoning leaf has none)"
            )
        # Normalize a supplied schema (pydantic class or JSON-schema dict) to a
        # concrete pydantic model. ``None`` keeps the schema-less text path.
        structured_model = to_pydantic_model(schema) if schema is not None else None
        # Resolve the effective model once: an explicit override wins, otherwise
        # the roster entry's registered default. Both the journal key and the leaf
        # config are derived from this single value so they can never disagree.
        effective_model = model if model is not None else entry.default_model
        # The schema is part of the journal key (via its JSON schema), so a
        # schema-bound call partitions distinctly from a schema-less one and from
        # a call bound to a different schema — resume restores the exact variant.
        key = journal_key(
            prompt=prompt,
            agent_type=agent_type,
            model=effective_model,
            schema=structured_model,
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
            # ``_observe_depth0`` also fails loud if a sibling depth-0 observe is
            # already in flight (raw concurrent fan-out), returning whether this call
            # incremented the shared counter so the finally below decrements exactly
            # the calls that did, on every exit path.
            counted = self._observe_depth0(key)
            try:
                cached = await self._journal.get(key)
                if cached is not None:
                    # Resume re-counts the cached leaf's usage from the journal record,
                    # so spent() rebuilds to the first run's cumulative total without a
                    # model call. A cache hit never consumes a budget slot beyond its
                    # own usage. The span reports the hit so a trace distinguishes a
                    # replayed leaf from a fresh one. A schema-bound result was stored
                    # as model_dump_json, so it is restored via model_validate_json.
                    self._budget.record(key, cached.usage)
                    span.set("cached", True)
                    span.set("usage_tokens", cached.usage)
                    if structured_model is not None:
                        return structured_model.model_validate_json(cached.result)
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
                # A schema binds the leaf to a ToolStrategy(model) so it emits a
                # validated structured_response; schema-less calls pass None. The same
                # response_format is threaded through to the roster's builder.
                response_format = (
                    ToolStrategy(structured_model, handle_errors=True)
                    if structured_model is not None
                    else None
                )
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
                        response_format=response_format,
                        isolation=isolation,
                        leaf_span_id=span.span_id,
                    )
                )
                # Authoritative changeset (R5): a real-git worktree execution leaf has its
                # real `git diff` folded into the leaf state by the engine. When present
                # it MUST be surfaced as the authoritative file source — the model's
                # self-reported file bytes can never win over what the leaf actually wrote
                # (mirroring M5's "gate on the real exit code, not the model's boolean").
                # A schema-less worktree leaf would collect a diff it can never surface,
                # so the boundary would be only half-closed; a worktree schema lacking the
                # `files` field, or typing it as anything but dict[str, str], would either
                # drop the diff or (under model_copy) journal a type-mismatched payload
                # that crashes on a later resume. All three fail loud here, at fold time.
                changeset = outcome.state.get(WORKTREE_CHANGESET_KEY)
                # With a schema, fold the validated structured object and journal its
                # canonical JSON; without one, fold the final text directly.
                folded_obj: str | BaseModel
                if structured_model is not None:
                    folded_obj = fold_structured(outcome.state, structured_model)
                    if changeset is not None:
                        if _WORKTREE_FILES_FIELD not in type(folded_obj).model_fields:
                            raise ValueError(
                                f"a git-worktree execution leaf (agent_type {agent_type!r}) must "
                                f"declare a schema with a {_WORKTREE_FILES_FIELD!r}: "
                                "dict[str, str] field to carry its authoritative changeset; the "
                                f"bound schema {structured_model.__name__!r} has no such field"
                            )
                        # Override `files` THROUGH validation (not model_copy, which
                        # bypasses it): a wrong-typed `files` field then fails loud here on
                        # the first run, never journaling a payload that would crash on a
                        # resume's model_validate_json.
                        folded_obj = type(folded_obj).model_validate(
                            {
                                **folded_obj.model_dump(by_alias=False),
                                _WORKTREE_FILES_FIELD: changeset,
                            }
                        )
                    # Dump by alias with round-trip semantics so a schema with field
                    # aliases survives resume: model_validate_json (the replay path)
                    # validates by alias by default, so the stored JSON must use aliases
                    # too. For alias-free models this is identical to a plain dump.
                    result_str = folded_obj.model_dump_json(by_alias=True, round_trip=True)
                else:
                    if changeset is not None:
                        raise ValueError(
                            f"a git-worktree execution leaf (agent_type {agent_type!r}) must "
                            f"declare a schema with a {_WORKTREE_FILES_FIELD!r}: dict[str, str] "
                            "field to carry its authoritative changeset (schema-less worktree "
                            "leaves are not supported)"
                        )
                    folded_obj = fold_result(outcome.state)
                    result_str = folded_obj
                # success-only: unreachable if the leaf raised. Usage is journaled so
                # the spend is reconstructable on resume.
                await self._journal.put(key, JournalRecord(result=result_str, usage=outcome.usage))
                self._budget.record(key, outcome.usage)
                span.set("cached", False)
                span.set("usage_tokens", outcome.usage)
                return folded_obj
            finally:
                # Release the depth-0 in-flight slot on both success and failure so the
                # counter reflects only calls actually running now. A depth-0 agent()
                # that itself errors (leaf raise, budget breach, worktree fold error)
                # still decrements here, so a subsequent sequential call is never falsely
                # flagged as concurrent. Only an entry that incremented decrements.
                if counted:
                    self._depth0_inflight -= 1

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
            except WORKFLOW_CONTROL_FLOW_SIGNALS:
                # Engine control-flow signals must fail loud, never be masked as a
                # leaf failure: re-raise so the barrier surfaces them. Includes a
                # checkpoint reached from inside a fan-out frame (a depth-0 primitive).
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
                if isinstance(outcome, asyncio.CancelledError):
                    # An INTERIOR CancelledError (a thunk awaited a child that was
                    # cancelled while this barrier's own task is NOT being torn down).
                    # return_exceptions=True can only place a CancelledError into the
                    # settled list for an interior raise: an EXTERNAL cancel of this
                    # task raises out of `await gather(...)` itself and never reaches
                    # here. So an interior CancelledError is masked as a leaf failure
                    # (None), like any other failed leaf — never re-raised.
                    results.append(None)
                elif isinstance(outcome, BaseException):
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

    async def dag(self, nodes: Sequence[DagNode]) -> dict[str, Any | None]:
        """Run a dependency graph in topological order; a node runs after its deps.

        Each :class:`~langchain_dynamic_workflow._dag.DagNode` declares an ``id``, its
        ``deps`` (ids it depends on), and a ``run(deps)`` callable that receives a
        ``{dep_id: result}`` mapping of its predecessors' results and typically calls
        ``agent()``. Ready nodes (all deps settled) run concurrently; there is no
        level barrier, so an independent branch races ahead of a slow one. A node
        whose ``run`` raises lands as ``None`` and every node that depends on it
        (transitively) is skipped to ``None``; a node that legitimately returns
        ``None`` does not skip its dependents. Engine control-flow signals
        (budget / determinism / a malformed graph) are re-raised loud after the
        in-flight nodes drain, never masked as a ``None`` hole.

        Concurrency is bounded at the leaf: the ``agent()`` calls inside a node's
        ``run`` acquire the shared gate, while this fan-out layer holds no slot — so a
        ``dag`` nested inside a node cannot starve the pool. Like ``parallel`` /
        ``pipeline`` / ``race``, the leaves inside the nodes are excluded from the
        determinism sequence guard (their completion order is wall-clock dependent);
        the content-hash journal still guards each by its inputs, so a resume replays
        completed nodes at zero model cost — the graph structure is script-defined and
        therefore deterministic, needing no dag-level journal entry of its own.

        Args:
            nodes: The graph's nodes.

        Returns:
            A mapping ``{node_id: result}``; a failed or skipped node maps to ``None``.

        Raises:
            WorkflowDagError: If the graph is structurally invalid (duplicate id,
                unknown / self dependency, or a cycle).
        """
        with self._spans.span(SpanKind.DAG, "dag") as span:
            span.set("node_count", len(nodes))
            if not nodes:
                span.set("surviving_count", 0)
                return {}
            # Mark the fan-out frame so agent() calls inside the nodes skip the
            # determinism backstop; set before run_dag spawns node tasks so each
            # child task inherits the depth.
            token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
            try:
                results = await run_dag(nodes)
            finally:
                _FANOUT_DEPTH.reset(token)
            span.set("surviving_count", sum(1 for v in results.values() if v is not None))
            return results

    async def loop_until[T](
        self,
        body: Callable[[int, list[T]], Awaitable[T]],
        *,
        done: Callable[[list[T]], bool],
        max_iters: int,
    ) -> list[T]:
        """Run ``body`` until ``done`` holds over the accumulated results, capped at ``max_iters``.

        A measured-stop loop with the two author disciplines baked in: every loop has
        a mandatory hard cap (``max_iters``), and the stop predicate is checked over
        the FULL accumulated list (so dedup / convergence is against *everything* seen,
        not just the last round). Each iteration calls ``body(iter_index, accumulated)``
        — where ``accumulated`` is a copy of the results so far — appends its result,
        then checks ``done(accumulated)``; the loop returns as soon as ``done`` holds.

        This is a sequential (depth-0) primitive: ``body``'s direct ``agent()`` calls
        record into the determinism guard, and the loop count derives from journaled
        leaf results, so a resume reproduces the same number of iterations and replays
        completed leaves at zero cost. If the cap is reached without ``done`` ever
        holding, a (replay-idempotent) ``log`` line is emitted and the accumulated
        results are returned — a graceful, non-silent stop rather than a raise.

        Args:
            body: ``(iter_index, accumulated_so_far) -> result`` for one iteration.
            done: Stop predicate over the full accumulated result list.
            max_iters: Mandatory hard cap on iterations (must be >= 1).

        Returns:
            The accumulated results, in iteration order.

        Raises:
            ValueError: If ``max_iters`` is less than 1.
        """
        if max_iters < 1:
            raise ValueError(f"loop_until requires max_iters >= 1, got {max_iters}")
        accumulated: list[T] = []
        for iteration in range(max_iters):
            accumulated.append(await body(iteration, list(accumulated)))
            if done(accumulated):
                return accumulated
        self.log(f"loop_until reached max_iters={max_iters} without satisfying done()")
        return accumulated

    async def _run_race_candidate(self, candidate: RaceCandidate) -> Any:
        """Dispatch one race candidate by reusing ``agent()`` verbatim.

        Forwarding through ``agent()`` (rather than re-implementing the leaf path)
        reuses the journal dedup, budget metering, sandbox admission, and span the
        leaf path already provides. The candidate runs at fan-out depth > 0 (the
        race frame is entered before this task is created), so it is excluded from
        the determinism sequence exactly like a ``parallel`` / ``pipeline`` leaf.
        """
        # Any-typed so the call is not matched against agent()'s overloads, which
        # are written per concrete schema type and do not accept the union the
        # candidate carries; the runtime forwarding is the same either way.
        agent_call: Any = self.agent
        return await agent_call(
            candidate.prompt,
            agent_type=candidate.agent_type,
            schema=candidate.schema,
            model=candidate.model,
            isolation=candidate.isolation,
        )

    async def race[T](
        self,
        candidates: Sequence[RaceCandidate],
        *,
        win: Callable[[T], bool],
        win_tag: str = "",
    ) -> RaceResult[T]:
        """Run candidates concurrently; the first whose result satisfies ``win`` wins.

        Best-of-N early exit: every candidate is dispatched via ``agent()`` at the
        same time, and the first to produce a result for which ``win`` returns
        ``True`` becomes the winner. The in-flight losers are then cancelled. When
        several candidates finish in the same scheduler wakeup the lowest input
        index wins, so the winner never depends on completion order.

        The decision is **journaled** under a content-hash race-key so resume is
        deterministic: a replayed race reproduces the recorded winner and dispatches
        **nothing** (the losers never re-run, so a resumed race is cheaper than the
        first one — by design). A race that produced no winner is **not** journaled,
        so a resume may retry it; use ``parallel`` when you want every result
        regardless of a predicate.

        All candidates must be homogeneous — either all schema-less (their results
        are ``str``) or all bound to the same schema (their results are that model)
        — so the winner's type is unambiguous.

        ``win_tag`` is folded into the race-key. Two races over the *same* candidates
        but with *different* win predicates **must** pass different ``win_tag``
        values; otherwise the second race replays the first's journaled decision and
        silently bypasses the changed predicate.

        Args:
            candidates: The agent-call specs to race; must be non-empty and
                homogeneous.
            win: Predicate over a candidate's result deciding whether it wins.
            win_tag: A label distinguishing this race's win criterion in the
                journal key (see the footgun note above). Defaults to ``""``.

        Returns:
            A :class:`RaceResult` carrying the winner and its index, or both
            ``None`` when no candidate satisfied ``win``.

        Raises:
            ValueError: If ``candidates`` is empty or not homogeneous.
            KeyError: If a candidate's ``agent_type`` is not registered.
            WorkflowBudgetExceededError: If a candidate's ``agent()`` trips the
                budget cap (re-raised after the in-flight losers are torn down).
            WorkflowDeterminismError: If the race-key diverges from the recorded
                replay sequence.
            Exception: Whatever ``win`` raises, re-raised after teardown (the
                predicate is script logic; a raise is a bug, not a leaf failure).
        """
        if not candidates:
            raise ValueError("race() requires at least one candidate; got an empty sequence")

        # Prelude (all synchronous, all before any dispatch): resolve each candidate
        # exactly as agent() will, derive its leaf key, and enforce homogeneity.
        leaf_keys: list[str] = []
        schema_signatures: set[str | None] = set()
        schema_model: type[BaseModel] | None = None
        for candidate in candidates:
            entry = self._roster.resolve(candidate.agent_type)  # fail fast on unknown agent_type
            if candidate.isolation == "worktree" and not entry.needs_execution:
                raise ValueError(
                    f"isolation='worktree' requires agent_type {candidate.agent_type!r} to be "
                    "registered with needs_execution=True (a worktree is seeded into an execution "
                    "sandbox; a reasoning leaf has none)"
                )
            candidate_model = (
                to_pydantic_model(candidate.schema) if candidate.schema is not None else None
            )
            effective_model = (
                candidate.model if candidate.model is not None else entry.default_model
            )
            leaf_keys.append(
                journal_key(
                    prompt=candidate.prompt,
                    agent_type=candidate.agent_type,
                    model=effective_model,
                    schema=candidate_model,
                    isolation=candidate.isolation,
                )
            )
            signature = (
                None
                if candidate_model is None
                else json.dumps(candidate_model.model_json_schema(), sort_keys=True)
            )
            schema_signatures.add(signature)
            schema_model = candidate_model
        if len(schema_signatures) != 1:
            raise ValueError(
                "race() candidates must be homogeneous: either all schema-less (text) or all "
                "bound to the same schema; got a mix, which would make RaceResult.winner's type "
                "ambiguous"
            )

        rkey = race_key(candidate_keys=leaf_keys, win_tag=win_tag)

        with self._spans.span(SpanKind.RACE, win_tag or "race") as span:
            span.set("candidate_count", len(candidates))
            # The race decision is one sequential step: its content-stable key is
            # recorded / validated once at depth 0 via the shared depth-0 choke point,
            # which also fails loud if a sibling depth-0 observe (another race / an
            # agent / a checkpoint) is concurrently in flight. The candidate agent()
            # calls run at depth > 0 and are excluded from the sequence (their
            # completion order varies run to run), mirroring leaves inside parallel() /
            # pipeline(). The finally decrements the in-flight slot on every exit
            # (cache replay, no-winner, control-flow re-raise, journaled win).
            counted = self._observe_depth0(rkey)
            try:
                # Replay: a journaled race decision reproduces the winner deterministically
                # and dispatches NOTHING — the losers never re-run, so a resumed race is
                # cheaper than the first (correct: the decision is already made). The
                # envelope is self-contained so replay needs no candidate leaf entry.
                cached = await self._journal.get(rkey)
                if cached is not None:
                    decision = json.loads(cached.result)
                    cached_index = int(decision["winner_index"])
                    cached_result_str = decision["result"]
                    # Reconstruct the winner's spend under its OWN leaf key — the key the
                    # fresh run counted it under — NOT the race-key. The journal hit on
                    # rkey guarantees identical candidates (rkey is derived from their leaf
                    # keys), so leaf_keys[cached_index] is the winner's key. Recording per
                    # leaf key keeps spend reconstructable AND idempotent: a later agent()
                    # with the winner's exact params hits the same key and is not
                    # double-counted (recording under rkey would, since rkey != leaf_key).
                    self._budget.record(leaf_keys[cached_index], cached.usage)
                    decoded: Any = (
                        schema_model.model_validate_json(cached_result_str)
                        if schema_model is not None
                        else cached_result_str
                    )
                    span.set("replayed", True)
                    span.set("won", True)
                    span.set("winner_index", cached_index)
                    return RaceResult[T](winner=cast(T, decoded), winner_index=cached_index)
                span.set("replayed", False)

                # Fresh run: dispatch all candidates concurrently; first to satisfy wins.
                # The depth is incremented just before the try and the tasks are created
                # INSIDE it, so the finally always tears the tasks down and resets the
                # depth even if task creation raises — mirroring parallel()/pipeline().
                winner_index: int | None = None
                winner_result: Any = None
                to_raise: BaseException | None = None
                tasks: list[asyncio.Task[Any]] = []
                token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
                try:
                    tasks = [
                        asyncio.ensure_future(self._run_race_candidate(candidate))
                        for candidate in candidates
                    ]
                    index_of = {task: index for index, task in enumerate(tasks)}
                    remaining = set(tasks)
                    while remaining and winner_index is None and to_raise is None:
                        done, remaining = await asyncio.wait(
                            remaining, return_when=asyncio.FIRST_COMPLETED
                        )
                        # Deterministic tie-break: decide same-wakeup completions in
                        # ascending candidate index, never set-iteration order.
                        for task in sorted(done, key=lambda finished: index_of[finished]):
                            error = task.exception()
                            if error is not None:
                                if isinstance(error, WORKFLOW_CONTROL_FLOW_SIGNALS):
                                    # Engine control-flow signal: fail loud, never mask.
                                    to_raise = error
                                    break
                                # Ordinary leaf failure: this candidate is out; others go on.
                                continue
                            candidate_result = task.result()
                            try:
                                satisfied = win(cast(T, candidate_result))
                            except Exception as predicate_error:
                                # Predicate raise is a script bug: fail loud after teardown.
                                to_raise = predicate_error
                                break
                            if satisfied:
                                winner_index = index_of[task]
                                winner_result = candidate_result
                                break
                finally:
                    # Teardown: cancel every still-running loser and await all tasks so
                    # none is orphaned and every gate slot is released. return_exceptions
                    # absorbs the CancelledErrors (and any loser exception) raised here.
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    _FANOUT_DEPTH.reset(token)

                if to_raise is not None:
                    raise to_raise

                if winner_index is None:
                    # No winner: do NOT journal a decision (a resume may retry the race).
                    span.set("won", False)
                    span.set("winner_index", None)
                    return RaceResult[T](winner=None, winner_index=None)

                # Journal a self-contained decision under the namespaced race-key: the
                # winner index plus the canonical result string agent() produced, carrying
                # the winner's usage so resume rebuilds spend. The winner's own leaf entry
                # already recorded its usage under its leaf key on this fresh run, so the
                # race-key is NOT re-recorded into the budget (that would double-count).
                winner_record = await self._journal.get(leaf_keys[winner_index])
                winner_usage = winner_record.usage if winner_record is not None else 0
                if schema_model is not None:
                    winner_result_str = cast(BaseModel, winner_result).model_dump_json(
                        by_alias=True, round_trip=True
                    )
                else:
                    winner_result_str = cast(str, winner_result)
                envelope = json.dumps({"winner_index": winner_index, "result": winner_result_str})
                await self._journal.put(rkey, JournalRecord(result=envelope, usage=winner_usage))
                span.set("won", True)
                span.set("winner_index", winner_index)
                return RaceResult[T](winner=cast(T, winner_result), winner_index=winner_index)
            finally:
                if counted:
                    self._depth0_inflight -= 1

    async def batch_map[X, T](
        self,
        items: Iterable[X] | AsyncIterable[X],
        fn: Callable[[X], Awaitable[T]],
        *,
        max_in_flight: int | None = None,
        total: int | None = None,
        label: str = "batch_map",
    ) -> list[T | None]:
        """Map ``fn`` over ``items`` with bounded streaming admission; collect in order.

        Streaming in, barrier out: items are admitted lazily through a bounded
        window (so N-thousand items never materialize N-thousand tasks), but
        results are collected into a list aligned to input order and returned
        only once every item has settled. ``fn`` is a single-argument async
        callable applied to each item, typically a single ``agent()`` call; a
        script that needs the index passes ``enumerate(items)`` and unpacks
        inside ``fn``. A failing ``fn`` lands ``None`` at its position and never
        aborts the barrier (filter the holes downstream, e.g. with
        ``dedup``/``survives``); engine control-flow signals (budget /
        determinism / checkpoint) are re-raised after a clean drain, never masked
        as ``None``.

        Concurrency is bounded at the leaf by the shared gate, while the admission
        ``window`` (``max_in_flight``, defaulting to the gate limit) bounds how
        many items are concurrently in ``fn`` and provides backpressure so memory
        stays decoupled from N. Like ``parallel`` / ``pipeline`` / ``dag``, the
        leaves inside ``fn`` run at a non-zero fan-out depth and are excluded from
        the determinism sequence guard (their completion order is wall-clock
        dependent); the content-hash journal still guards each by its inputs, so a
        resume replays completed items at zero model cost.

        Live ``BATCH`` progress (``completed`` / ``total`` / ``elapsed`` / ``eta``)
        is emitted to the progress sink as the batch advances, throttled to every
        ``K`` settled items or every second, plus one final settled entry. A
        ``Sized`` input takes ``len()`` for ``total`` automatically; a non-``Sized``
        source (a generator / async iterable) reports ``completed`` only unless a
        ``total`` hint is supplied, in which case the ETA becomes computable. The
        progress is transient and out-of-band: it is delivered to the sink but
        never recorded, journaled, or replayed, because its timestamps are
        non-deterministic.

        Args:
            items: The input items; a ``Sized`` ``Iterable``, a generator, or an
                ``AsyncIterable`` (e.g. a paged API). Consumed lazily through the
                admission window — never materialized.
            fn: The single-argument async callable applied to each item.
            max_in_flight: The admission window (items concurrently in ``fn``).
                ``None`` defaults to the shared gate limit.
            total: Optional item-count hint for a non-``Sized`` input, enabling
                the ETA. A ``Sized`` input takes ``len()`` automatically and
                ignores this hint.
            label: The span / progress label for this batch.

        Returns:
            A list aligned to ``items`` input order; each entry is ``fn``'s result,
            or ``None`` if ``fn`` raised for that item.

        Raises:
            ValueError: If ``max_in_flight`` is supplied and is less than 1.
        """
        # Fail fast on a meaningless window before opening a span or admitting work.
        if max_in_flight is not None and max_in_flight < 1:
            raise ValueError(f"batch_map requires max_in_flight >= 1, got {max_in_flight}")
        with self._spans.span(SpanKind.BATCH, label) as span:
            # A Sized input knows its length for free; a non-Sized source falls
            # back to the caller's total hint (which may be None -> no ETA).
            known_total = len(items) if isinstance(items, Sized) else total
            span.set("total", known_total)
            # The admission window bounds items concurrently in fn; default to the
            # gate limit (more than the limit only parks extra workers on the gate).
            window = max_in_flight if max_in_flight is not None else self._gate.limit

            # Live progress is engine-side, out-of-band, and never journaled: the
            # monotonic clock and the throttle counters live entirely in this
            # closure. asyncio is single-threaded, so a plain int increment in the
            # settle path needs no lock.
            start = time.monotonic()
            # Single-element lists so the throttle closure can mutate the count and
            # the last-emit timestamp in place. They are kept separate (not one
            # dict) so the count stays a strict int and the timestamp a strict
            # float — pyright would otherwise widen a mixed dict's values.
            completed_count: list[int] = [0]
            last_emit: list[float] = [start]
            # Emit on every K settled items OR every T seconds, whichever comes
            # first, plus always a final settled entry. K scales with total so a
            # large batch is not noisy; an unknown total uses a fixed step.
            throttle_step = max(1, known_total // 100) if known_total else 64
            throttle_seconds = 1.0

            def _emit_progress(completed: int, *, force: bool = False) -> None:
                now = time.monotonic()
                is_final = known_total is not None and completed >= known_total
                crossed_step = completed % throttle_step == 0
                crossed_time = (now - last_emit[0]) >= throttle_seconds
                if not (force or crossed_step or crossed_time or is_final):
                    return
                last_emit[0] = now
                elapsed = now - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                # A `total=` hint over a non-Sized source can under-count: once
                # `completed` exceeds it, the hint is proven wrong, so for THIS emit
                # treat the total as UNKNOWN (drop to the completed-only view) rather
                # than reporting a misleading "10/3" or clamping to "10/10" (which
                # would falsely imply completion). The Sized / accurate-hint happy
                # path (completed <= known_total) is unchanged. The eta guard requires
                # completed <= known_total so it can never be negative.
                effective_total = (
                    known_total if known_total is not None and completed <= known_total else None
                )
                eta = (
                    (effective_total - completed) / rate
                    if effective_total is not None and rate > 0
                    else None
                )
                if effective_total is not None:
                    message = f"{label}: {completed}/{effective_total}"
                else:
                    message = f"{label}: {completed}"
                # Progress is out-of-band, transient, best-effort observability:
                # delivered to the sink but never recorded, journaled, or replayed. A
                # host sink that raises is the host's own telemetry bug; isolating it at
                # this single emit chokepoint means EVERY call site — the per-item
                # `_stage` finally AND the post-pipeline final forced emit below — is
                # covered, so a sink fault can never corrupt a result (run_pipeline's
                # `except Exception` would otherwise demote a successful item to None)
                # nor turn a computationally-successful batch into a propagated failure.
                # Only emit_transient can raise (the throttle/metrics arithmetic above
                # cannot), so the suppress wraps it alone — the computation is
                # unaffected: a failed fn still lands None, and the engine's control-flow
                # signals (budget / determinism / checkpoint) are raised by
                # `await fn(item)` in `_stage`'s try body, never inside this emit.
                with contextlib.suppress(Exception):
                    self._progress.emit_transient(
                        message,
                        metrics=BatchMetrics(
                            completed=completed,
                            elapsed_seconds=elapsed,
                            rate=rate,
                            total=effective_total,
                            eta_seconds=eta,
                        ),
                    )

            async def _stage(_payload: Any, item: Any, _index: int) -> Any:
                # One stage = the whole map: run fn, then advance the shared
                # completed counter and conditionally emit progress. The counter
                # advances whether fn succeeds or raises (a raise drops the item to
                # None in run_pipeline, but it still settled), so progress tracks
                # settled work, not just successes.
                try:
                    return await fn(item)
                finally:
                    completed_count[0] += 1
                    # The emit is isolated centrally inside _emit_progress (a sink fault
                    # is suppressed at the chokepoint), so this is a plain call: fn's own
                    # exceptions / control-flow signals propagate from `await fn(item)`
                    # in the try body, never masked, while a progress-sink fault can
                    # neither corrupt this item's result nor abort the batch.
                    _emit_progress(completed_count[0])

            # Mark the fan-out frame BEFORE the engine spawns its workers so each
            # worker task inherits the depth and its agent() calls skip the
            # determinism sequence guard; reset in finally even if the run raises.
            token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
            try:
                results = await run_pipeline(
                    items,
                    [_stage],
                    gate=self._gate,
                    queue_maxsize=window,
                )
            finally:
                _FANOUT_DEPTH.reset(token)
            # Always emit one final settled entry (force past the throttle) so the
            # last state is exact even when the throttle skipped it — critical for an
            # unknown total, where is_final never fires. Skip only the empty input.
            if completed_count[0] > 0:
                _emit_progress(completed_count[0], force=True)
            span.set("admitted_count", completed_count[0])
            span.set("surviving_count", sum(1 for r in results if r is not None))
            return cast(list[T | None], results)

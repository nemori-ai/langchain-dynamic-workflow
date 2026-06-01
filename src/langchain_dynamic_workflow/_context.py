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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from ._budget import Budget
from ._concurrency import ConcurrencyGate, resolve_max_concurrency
from ._determinism import CallSequenceGuard
from ._journal import JournalRecord, JournalStore, journal_key
from ._pipeline import Stage, run_pipeline
from ._progress import ProgressKind, ProgressLog
from ._result import fold_result
from ._roster import Roster


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


LeafRunner = Callable[[str, str, "str | None"], Awaitable[LeafOutcome]]
"""Invokes a leaf: ``(agent_type, prompt, model) -> LeafOutcome`` (state + usage)."""

T = TypeVar("T")


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

        Args:
            prompt: The prompt for the leaf.
            agent_type: The roster name to resolve.
            model: Optional model override; folded into the journal key *and*
                threaded into the leaf invocation so it reaches execution.
            isolation: Isolation mode (part of the journal key).

        Returns:
            The leaf's folded final text.

        Raises:
            KeyError: If ``agent_type`` is not registered.
            WorkflowBudgetExceededError: If the shared budget is exhausted.
        """
        self._roster.resolve(agent_type)  # fail fast on unknown agent_type
        key = journal_key(
            prompt=prompt,
            agent_type=agent_type,
            model=model,
            schema=None,
            isolation=isolation,
        )
        # Determinism backstop: record this call-key (fresh run) or validate it
        # against the recorded sequence (replay). A divergence fails loud here,
        # before any cache entry is served.
        self._sequence_guard.observe(key)
        cached = await self._journal.get(key)
        if cached is not None:
            # Resume re-counts the cached leaf's usage from the journal record, so
            # spent() rebuilds to the first run's cumulative total without a model
            # call. A cache hit never consumes a budget slot beyond its own usage.
            self._budget.record(key, cached.usage)
            return cached.result
        # Cap is checked only before dispatching a *new* leaf: an exhausted pool
        # refuses fresh work while in-flight leaves finish and keep their results.
        self._budget.ensure_within_cap()
        # The gate bounds the number of leaves actually in flight; a journal hit
        # above never consumes a slot, keeping resume cheap.
        outcome = await self._gate.run(lambda: self._leaf_runner(agent_type, prompt, model))
        folded = fold_result(outcome.state)
        # success-only: unreachable if the leaf raised. Usage is journaled so the
        # spend is reconstructable on resume.
        await self._journal.put(key, JournalRecord(result=folded, usage=outcome.usage))
        self._budget.record(key, outcome.usage)
        return folded

    async def parallel(self, thunks: Sequence[Callable[[], Awaitable[T]]]) -> list[T | None]:
        """Fan out a list of thunks concurrently with a blocking barrier.

        Each thunk is a zero-argument callable returning an awaitable (typically
        a closure over an ``agent()`` call). Results are returned in input order.
        A thunk that raises lands as ``None`` at its position; the call as a whole
        never raises, mirroring Claude Code's ``parallel`` semantics — filter the
        ``None`` holes downstream.

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
            result, or ``None`` if it raised.
        """

        async def _guarded(thunk: Callable[[], Awaitable[T]]) -> T | None:
            try:
                return await thunk()
            except Exception:
                # Failure isolation: one bad thunk must not abort the barrier.
                return None

        if not thunks:
            return []
        return await asyncio.gather(*[_guarded(thunk) for thunk in thunks])

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
        return await run_pipeline(items, stages, gate=self._gate)

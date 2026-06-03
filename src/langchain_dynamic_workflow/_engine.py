"""The workflow engine — Layer 0 substrate binding.

``run_workflow`` builds a LangGraph ``@entrypoint`` whose body constructs a
:class:`Ctx` and runs the user's orchestration callable. Each leaf invocation is
a ``@task`` (durable execution), with the content-hash journal layered on top so
completed leaves return cached results on resume.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

from ._budget import Budget, total_tokens_from_handler
from ._concurrency import (
    HARD_CEILING,
    ConcurrencyGate,
    resolve_max_concurrency,
    with_max_concurrency,
)
from ._context import Ctx, LeafOutcome, WorkflowResolver
from ._determinism import CallSequenceGuard

# Re-exported on the engine's public core entry so the host-facing surface
# (tool / middleware) constructs per-run journal stores through `run_workflow`'s
# module rather than reaching directly into the engine-internal `_journal`.
from ._journal import InMemoryJournalStore, JournalStore
from ._observability import SpanRecorder, SpanSink
from ._progress import ProgressEntry, ProgressLog, ProgressSink
from ._roster import Roster
from ._sandbox import SandboxManager, SharedArtifactStore, build_leaf_backend


def _default_progress_sink(entry: ProgressEntry) -> None:
    """Print a progress entry to stdout (the default narration sink)."""
    print(f"[{entry.kind.value}] {entry.message}")


Orchestrator = Callable[[Ctx], Awaitable[Any]]
"""A workflow script: ``async def orchestrate(ctx) -> result``."""


async def run_workflow(
    orchestrate: Orchestrator,
    *,
    roster: Roster,
    journal: JournalStore | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    thread_id: str = "default",
    max_concurrency: int | None = None,
    budget: int | None = None,
    on_progress: ProgressSink | None = None,
    on_span: SpanSink | None = None,
    sandbox_manager: SandboxManager | None = None,
    workflows: WorkflowResolver | None = None,
) -> Any:
    """Run an orchestration script to completion and return its final result.

    Args:
        orchestrate: The async orchestration callable, called with a ``Ctx``.
        roster: The leaf registry resolved by ``ctx.agent(agent_type=...)``.
        journal: Content-hash journal store; defaults to an in-memory store.
            Pass the *same* instance across calls to get cached-result resume.
        checkpointer: LangGraph checkpointer; defaults to an in-memory saver.
        thread_id: Durable-execution thread id.
        budget: Optional shared token ceiling for all leaves in this run. When a
            leaf would push spend past the cap the next ``agent()`` raises
            :class:`~langchain_dynamic_workflow.WorkflowBudgetExceededError`;
            ``None`` (the default) leaves the budget unbounded so the
            loop-until-budget idiom (``while ctx.budget.remaining() > T``) is a
            no-op cap. The spend is rebuilt from journal usage on resume.
        on_progress: Optional sink receiving each newly-delivered ``phase``/``log``
            entry; defaults to printing to stdout. Delivery is replay-idempotent —
            entries already delivered on a prior run are not re-emitted on resume.
        on_span: Optional sink receiving a completed observability ``Span`` for
            every ``agent``/``parallel``/``pipeline`` call (observability-by-default).
            Each span carries the primitive kind, a name, and attributes (a leaf's
            ``agent_type`` / ``cached`` / ``usage_tokens``, a fan-out's counts), a
            wall-clock duration, and an error if the body raised. Unlike progress,
            spans are *not* replay-suppressed: a resumed run re-emits a span for each
            replayed leaf, flagged ``cached=True``, so a trace reflects the resume.
            When omitted, span recording is a silent no-op (zero cost).
        sandbox_manager: Optional per-leaf sandbox lifecycle manager. When
            supplied, a leaf whose roster entry is ``needs_execution`` is leased an
            isolated execution backend (keyed by its derived, resume-stable
            identity) that is threaded into the leaf config; a pure-reasoning leaf
            allocates no sandbox. After the script settles (clean return or raise)
            the engine stops every execution sandbox it leased this run, so the
            manager holds no live sandboxes once the run completes. When omitted,
            leaves run without sandbox admission: no backend is leased and none is
            threaded into the leaf config.
        max_concurrency: Optional explicit concurrency cap on in-flight leaves.
            LangGraph has no default (``None`` means unbounded at the substrate),
            so the engine always sets both layers explicitly. The shared
            :class:`ConcurrencyGate` is the *authoritative* bound (resolved to
            ``min(16, cores - 2)`` when omitted). The substrate ``max_concurrency``
            is pinned to the hard ceiling so it is never unbounded yet never
            throttles below the gate — setting both to the same value makes the two
            semaphores interleave and the effective cap fall one below the target.
        workflows: Optional named-workflow resolver enabling ``ctx.workflow(name,
            args)`` one-level inline nesting. The inner workflow shares this run's
            journal, budget, gate, and progress log. When omitted, any
            ``ctx.workflow()`` call raises ``LookupError``.

    Returns:
        Whatever the orchestration callable returns.
    """
    journal_store: JournalStore = journal if journal is not None else InMemoryJournalStore()
    saver: BaseCheckpointSaver[Any] = checkpointer if checkpointer is not None else InMemorySaver()
    limit = resolve_max_concurrency(max_concurrency)
    gate = ConcurrencyGate(limit=limit)
    # One shared artifact store per run: every execution leaf's backend routes its
    # /shared/ paths here (producer-namespaced), so a producer leaf can hand an
    # artifact off to a consumer leaf while each leaf's non-shared files stay
    # private to its own isolated sandbox.
    shared_store = SharedArtifactStore()
    # Identities of every execution sandbox actually leased this run. A lease keeps
    # its sandbox live for find-or-create reuse across retries, so nothing reclaims
    # it on its own; the engine owns the lifecycle finale and stops each one after
    # the script settles. Reasoning leaves never lease, so they never land here.
    leased_execution_leaf_ids: set[str] = set()

    @task
    async def leaf_task(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str,
        needs_execution: bool,
        response_format: Any = None,
        isolation: str = "shared",
    ) -> LeafOutcome:
        roster.resolve(agent_type)  # fail fast on unknown agent_type
        # Resolve the runnable bound to the requested response_format: a schema-less
        # call (response_format=None) gets the schema-less variant; a schema call
        # gets the builder-constructed, per-format-cached variant. needs_execution
        # arrives as a parameter (the caller derives it from the roster entry), so
        # the resolved runnable is all this task needs from the roster.
        runnable = roster.runnable_for(agent_type, response_format=response_format)
        # Forward a usage callback into the leaf's config — the same callback
        # forwarding deepagents performs for its own subagents — so token usage
        # aggregates across every (possibly nested) model call the leaf makes,
        # even though we bypass the LLM-driven task tool and invoke directly.
        usage_handler = UsageMetadataCallbackHandler()
        configurable: dict[str, Any] = {}
        if model is not None:
            # Propagate the effective model into the leaf config so a config-aware
            # leaf (one that reads configurable['model'] to pick a provider) runs
            # with it. A leaf whose model is bound at construction ignores this and
            # runs its built-in model; the value still partitions the journal key,
            # so the override is a consistent cache-partitioning knob in either case.
            configurable["model"] = model

        async def _invoke() -> LeafOutcome:
            leaf_config: RunnableConfig = {
                "callbacks": [usage_handler],
                "configurable": configurable,
            }
            state: dict[str, Any] = await runnable.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}, config=leaf_config
            )
            return LeafOutcome(state=state, usage=total_tokens_from_handler(usage_handler))

        if sandbox_manager is None:
            # No sandbox admission configured: invoke the leaf directly without
            # leasing or threading a backend into its config.
            return await _invoke()
        # Lease the leaf's backend for the duration of the invocation. Tiered
        # admission lives in the manager: an execution leaf gets an isolated
        # sandbox (find-or-created by its stable identity), a reasoning leaf a
        # StateBackend with no allocation. The backend is threaded into the leaf
        # config under a dedicated key so a backend-aware leaf can run against it.
        async with sandbox_manager.lease(
            leaf_id=leaf_id, needs_execution=needs_execution, isolation=isolation
        ) as backend:
            if needs_execution:
                # Record the leased execution identity so the engine can stop its
                # sandbox after the run; the lease itself keeps it live for reuse.
                leased_execution_leaf_ids.add(leaf_id)
            if needs_execution and isinstance(backend, SandboxBackendProtocol):
                # Give the execution leaf the /shared/ hand-off route on top of its
                # isolated sandbox: non-shared paths stay private, /shared/ paths go
                # to the run-shared store under this leaf's producer namespace.
                configurable["sandbox_backend"] = build_leaf_backend(
                    isolated=backend, shared_store=shared_store, producer=leaf_id
                )
            else:
                # Reasoning leaf: hand its StateBackend through unwrapped (it
                # allocates no sandbox and does no /shared/ hand-off).
                configurable["sandbox_backend"] = backend
            return await _invoke()

    async def leaf_runner(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
    ) -> LeafOutcome:
        # The single durable leaf path, shared by agent / parallel / pipeline.
        # The shared gate (applied inside Ctx) bounds how many of these run at
        # once across every fan-out path in this run.
        return await leaf_task(
            agent_type,
            prompt,
            model,
            leaf_id=leaf_id,
            needs_execution=needs_execution,
            response_format=response_format,
            isolation=isolation,
        )

    recorded_sequence = await journal_store.get_sequence()
    delivered_progress = await journal_store.get_progress_count()
    progress_sink: ProgressSink = on_progress if on_progress is not None else _default_progress_sink
    # One span recorder per run. Spans are emitted live (not replay-suppressed), so
    # a resumed run re-emits a span for every replayed leaf flagged cached=True.
    span_recorder = SpanRecorder(sink=on_span)

    @entrypoint(checkpointer=saver)
    async def _run(_input: Any) -> Any:
        sequence_guard = CallSequenceGuard(recorded=recorded_sequence)
        # A fresh budget per run: spend is rebuilt as leaves resolve (cache hits
        # re-count journaled usage, new leaves count their metered usage), so a
        # resumed run reaches the first run's cumulative total.
        run_budget = Budget(total=budget)
        # On resume, suppress progress entries already delivered on the prior run
        # so phase/log narration is not repeated; new entries still flow.
        progress = ProgressLog(delivered_count=delivered_progress, sink=progress_sink)
        ctx = Ctx(
            roster=roster,
            journal=journal_store,
            leaf_runner=leaf_runner,
            gate=gate,
            sequence_guard=sequence_guard,
            budget=run_budget,
            progress=progress,
            workflows=workflows,
            spans=span_recorder,
        )
        try:
            result = await orchestrate(ctx)
        finally:
            # Lifecycle finale: stop every execution sandbox leased this run,
            # whether the script returned cleanly or raised. A lease deliberately
            # keeps its sandbox live for find-or-create reuse across retries, so
            # nothing reclaims it on its own — the engine owns teardown. stop() is
            # idempotent, so stopping an already-reclaimed (TTL) sandbox is a no-op.
            # Reasoning leaves never lease, so this leaves the active count at zero
            # after a run, satisfying the "stop() 被调用清理" lifecycle contract.
            if sandbox_manager is not None:
                for leaf_id in leased_execution_leaf_ids:
                    await sandbox_manager.stop(leaf_id)
                leased_execution_leaf_ids.clear()
        # Reconcile call count on a clean return: observe() catches forward
        # divergence (mismatched / extra calls) as it happens, but an
        # early-terminating replay (fewer calls than recorded) is only detectable
        # here, after the script has finished issuing calls. Runs before
        # put_sequence so a divergent replay never overwrites the record with a
        # shorter sequence.
        sequence_guard.finalize()
        # Persist run-level state only after the script completes successfully. This
        # is deliberately asymmetric with the per-leaf journal: each completed leaf
        # is journaled the moment it finishes (inside agent()), but the call-key
        # sequence and the progress-delivered count are persisted only here, on a
        # clean return. A run that raises mid-flight (e.g.
        # WorkflowBudgetExceededError) therefore leaves completed leaves cached but
        # the progress count at its prior value (0 on a first run). A subsequent
        # re-run replays leaves from the journal at zero model cost, but because the
        # progress count was never advanced it re-delivers the phase/log narration
        # emitted before the failure. This is intended: a failed run is logically
        # incomplete, and re-narrating the work that is about to be retried is
        # preferable to silently skipping it. Callers that must suppress duplicate
        # narration across a failed-then-retried run should make their progress sink
        # idempotent (e.g. de-dupe on entry identity).
        await journal_store.put_sequence(ctx.observed_call_sequence)
        await journal_store.put_progress_count(ctx.progress_entry_count)
        return result

    base_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    # The gate is the precise cap; the substrate semaphore is pinned high (never
    # None/unbounded) so it stays explicit without fighting the gate.
    config = with_max_concurrency(base_config, HARD_CEILING)
    result: Any = await _run.ainvoke({}, config=config)  # pyright: ignore[reportUnknownMemberType]
    return result

"""The workflow engine — Layer 0 substrate binding.

``run_workflow`` builds a LangGraph ``@entrypoint`` whose body constructs a
:class:`Ctx` and runs the user's orchestration callable. Each leaf invocation is
a ``@task`` (durable execution), with the content-hash journal layered on top so
completed leaves return cached results on resume.
"""

from __future__ import annotations

import asyncio
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
from ._context import (
    DEFAULT_MAX_WORKFLOW_DEPTH,
    UNSET,
    WORKTREE_CHANGESET_KEY,
    Ctx,
    LeafOutcome,
    WorkflowResolver,
)
from ._determinism import CallSequenceGuard
from ._errors import WorkflowCheckpointError, WorkflowSignoffRequired

# Re-exported on the engine's public core entry so the host-facing surface
# (tool / middleware / persistence) constructs per-run journal stores and reads
# journal records through `run_workflow`'s module rather than reaching directly
# into the engine-internal `_journal`. ``JournalRecord`` is re-exported here so a
# Layer-2 persistent store can type its rows against the public wall.
from ._journal import (
    InMemoryJournalStore,
    JournalStore,
)
from ._journal import (
    JournalRecord as JournalRecord,  # re-exported for Layer-2 persistence on the public wall
)
from ._leaf_events import LeafEventHandler, LeafEventSink
from ._local_subprocess import LocalSubprocessSandbox
from ._observability import CommandSink, SpanBeginSink, SpanRecorder, SpanSink
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
    resume: Any = UNSET,
    max_concurrency: int | None = None,
    budget: int | None = None,
    on_progress: ProgressSink | None = None,
    on_span: SpanSink | None = None,
    on_span_begin: SpanBeginSink | None = None,
    on_leaf_event: LeafEventSink | None = None,
    leaf_event_include_payloads: bool = False,
    on_command: CommandSink | None = None,
    command_include_payloads: bool = False,
    sandbox_manager: SandboxManager | None = None,
    workflows: WorkflowResolver | None = None,
    max_workflow_depth: int = DEFAULT_MAX_WORKFLOW_DEPTH,
) -> Any:
    """Run an orchestration script to completion and return its final result.

    Args:
        orchestrate: The async orchestration callable, called with a ``Ctx``.
        roster: The leaf registry resolved by ``ctx.agent(agent_type=...)``.
        journal: Content-hash journal store; defaults to an in-memory store.
            Pass the *same* instance across calls to get cached-result resume.
        checkpointer: LangGraph checkpointer; defaults to an in-memory saver.
        thread_id: Durable-execution thread id.
        resume: Value injected at the next un-decided ``ctx.checkpoint`` sign-off so
            the run continues past the gate (the value becomes ``checkpoint()``'s
            return and is journaled as that gate's decision). Omit it (the ``UNSET``
            default) for a fresh run or a crash-resume replay. Approving against the
            *same* journal the run parked with replays completed leaves and
            already-approved gates at zero model cost; a fresh journal still
            approves correctly but re-runs prior work, so the host reuses the
            origin run's journal.
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
        on_span_begin: Optional sink receiving a ``SpanBegin`` the instant each
            ``agent``/``parallel``/``pipeline``/``race`` span opens, before its body
            runs — the running edge for a live status and an elapsed timer
            (``now - started_at``). It carries the span's ``span_id`` (shared with the
            matching end span), so a consumer correlates the running and completed
            edges. The ``span_id`` is resume-stable only for the sequential (depth-0)
            path — where the script replays in the same source order, so a fresh run
            and an honest resume mint the identical id; a fan-out leaf
            (``parallel``/``pipeline``/``race``) opens its span in wall-clock order, so
            its id correlates begin↔end within a run but is not guaranteed identical
            across a resume (mirroring the determinism guard, which records only the
            sequential path). Like the end span, begin is live-only, not
            replay-suppressed: a resumed run re-emits a begin for every replayed leaf
            and its matching end span is flagged ``cached=True`` with a near-zero
            duration, so a cached leaf renders as an instant replayed hit rather than
            a stuck "running" chip. When omitted, begin recording is a silent no-op.
        on_leaf_event: Optional sink receiving a ``LeafEvent`` for each runtime edge
            in a leaf's own callback subtree (its model calls, tool calls, nested
            sub-agent steps), correlated to the owning leaf via ``leaf_span_id`` and
            reconstructable into the in-leaf run tree via ``run_id`` /
            ``parent_run_id``. Events are delivered out-of-band and never injected
            into the host LLM's context, so quarantine is preserved. The tap fires
            only on real execution: a leaf served from the journal runs no interior,
            so a replayed leaf emits no leaf events. When omitted, no per-leaf tap is
            attached (zero cost).
        leaf_event_include_payloads: When ``True``, each ``LeafEvent``'s ``detail``
            carries bounded raw payloads (truncated tool input/output, model text);
            the default ``False`` keeps ``detail`` shape-only (node kind/name/timing),
            so leaf internals are not streamed unless the caller opts in.
        on_command: Optional sink receiving a ``CommandEvent`` for each real shell
            ``execute`` an execution leaf runs, fired from inside the leaf's
            ``LocalSubprocessSandbox`` at the subprocess boundary: a begin edge
            (``command``, ``started_at``, ``exit_code=None``) before the subprocess
            and an end edge (``exit_code``, bounded ``output``, ``duration_s``) after.
            Both edges share a resume-stable ``command_id`` and carry the owning
            leaf's ``leaf_span_id`` (the AGENT span id), so a consumer files the
            terminal card under the right span and flips it pass/fail in place. The
            sink fires only on real execution: a leaf served from the journal runs no
            subprocess, so a replayed (cached) leaf emits no command events. It
            reaches the sandbox only when a ``sandbox_manager`` whose factory
            produces ``LocalSubprocessSandbox`` backends is wired (the offline
            in-memory backend runs no subprocess and emits none). When omitted, no
            command sink is attached (zero cost).
        command_include_payloads: When ``True``, each end ``CommandEvent``'s
            ``output`` carries the full captured output (bounded only by the sandbox
            output cap); the default ``False`` bounds it to a small honest tail and
            flags ``truncated``, so command output is not surfaced in full unless the
            caller opts in.
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
            args)`` inline nesting. The inner workflow shares this run's journal,
            budget, gate, and progress log. When omitted, any ``ctx.workflow()``
            call raises ``LookupError``.
        max_workflow_depth: Cap on ``ctx.workflow`` inline nesting depth (a
            runaway-recursion backstop); defaults to ``DEFAULT_MAX_WORKFLOW_DEPTH``.

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
        leaf_span_id: str = "",
    ) -> dict[str, Any]:
        # A durable @task return crosses the checkpointer, whose serializer
        # round-trips built-in containers (and the registered LangChain message
        # types nested in the leaf state) but only revives a custom class through
        # a deprecated unregistered-type path that newer serializers block. So
        # this task returns the LeafOutcome's msgpack-native payload mapping; the
        # caller rebuilds the typed LeafOutcome, keeping the engine's public
        # behavior identical. The strict-safe shape covers the wrapper and the
        # registered (message/container) state only: a structured-output leaf's
        # state['structured_response'] is an unregistered user pydantic model that
        # degrades to a plain dict under strict msgpack. That is a documented
        # serialization boundary, not a replay defect — the zero-cost replay reads
        # the content-hash journal (which stores the folded result string), never
        # this checkpoint state, so the degraded model never reaches the script.
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

        async def _invoke() -> dict[str, Any]:
            callbacks: list[Any] = [usage_handler]
            # Attach the per-leaf event tap on the SAME callbacks list deepagents
            # forwards to subagents, so the whole leaf subtree is observed. The
            # handler closes over THIS leaf's span id for correlation; metadata is
            # not used (the subagent boundary drops it). It is attached only on real
            # execution -- _invoke runs only on a journal miss, so a journal hit
            # never emits interior events.
            if on_leaf_event is not None and leaf_span_id:
                callbacks.append(
                    LeafEventHandler(
                        leaf_span_id=leaf_span_id,
                        sink=on_leaf_event,
                        include_payloads=leaf_event_include_payloads,
                    )
                )
            leaf_config: RunnableConfig = {
                "callbacks": callbacks,
                "configurable": configurable,
            }
            state: dict[str, Any] = await runnable.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}, config=leaf_config
            )
            return LeafOutcome(
                state=state, usage=total_tokens_from_handler(usage_handler)
            ).to_payload()

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
                # Thread the command sink into the real sandbox so each subprocess
                # execute fires begin/end CommandEvents correlated to THIS leaf's
                # span. Only a LocalSubprocessSandbox runs a real subprocess; the
                # offline in-memory backend has no execute boundary to observe. The
                # sink reaches the sandbox here on the leased (real-execution) path,
                # so a journal hit -- which never leases or runs the leaf body --
                # never wires it, honoring the miss-only replay policy by construction.
                if (
                    on_command is not None
                    and leaf_span_id
                    and isinstance(backend, LocalSubprocessSandbox)
                ):
                    backend.set_command_sink(
                        sink=on_command,
                        leaf_span_id=leaf_span_id,
                        include_payloads=command_include_payloads,
                    )
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
            payload = await _invoke()
            # Authoritative changeset for a real-git worktree leaf (R5): while the
            # lease is STILL HELD (the worktree not yet torn down on close), read
            # the real `git diff` of the leaf's tree and fold it into the leaf's
            # journaled state under a reserved key. The script then treats this real
            # on-disk truth as authoritative over any file bytes the model
            # self-reported in its schema — mirroring M5's "gate on the real exit
            # code, not the model's boolean". Folding it into the @task payload's
            # state makes it ride into the content-hash journal, so a resume replays
            # the same authoritative changeset with no real git re-run.
            git_provider = (
                sandbox_manager.git_worktree_provider
                if needs_execution and isolation == "worktree"
                else None
            )
            if git_provider is not None:
                # collect() is a blocking git subprocess; thread-offload it so it
                # never wedges the event loop (same defect class R8 fixed for the
                # worktree add and H3 fixed for teardown).
                payload["state"][WORKTREE_CHANGESET_KEY] = await asyncio.to_thread(
                    git_provider.collect, leaf_id
                )
            return payload

    async def leaf_runner(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
    ) -> LeafOutcome:
        # The single durable leaf path, shared by agent / parallel / pipeline.
        # The shared gate (applied inside Ctx) bounds how many of these run at
        # once across every fan-out path in this run. The task returns the
        # checkpointer-safe payload mapping; rebuild the typed LeafOutcome here so
        # the context layer keeps working with .state / .usage unchanged. The
        # owning AGENT span's id is threaded through so the leaf seam can correlate
        # the leaf's callback subtree to its span.
        payload = await leaf_task(
            agent_type,
            prompt,
            model,
            leaf_id=leaf_id,
            needs_execution=needs_execution,
            response_format=response_format,
            isolation=isolation,
            leaf_span_id=leaf_span_id,
        )
        return LeafOutcome.from_payload(payload)

    recorded_sequence = await journal_store.get_sequence()
    delivered_progress = await journal_store.get_progress_count()
    progress_sink: ProgressSink = on_progress if on_progress is not None else _default_progress_sink
    # One span recorder per run. Spans are emitted live (not replay-suppressed), so
    # a resumed run re-emits a span for every replayed leaf flagged cached=True.
    span_recorder = SpanRecorder(sink=on_span, begin_sink=on_span_begin)

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
            pending_signoff=resume,
            max_workflow_depth=max_workflow_depth,
        )
        try:
            result = await orchestrate(ctx)
        except WorkflowSignoffRequired:
            # A sign-off pause is a DESIGNED stop, not a crash: persist the progress
            # count (and only that) so the approve replay does not re-deliver the
            # phase/log narration already shown before the gate. The call sequence is
            # deliberately NOT persisted here — the approve issues MORE calls past the
            # gate, and a partial recorded sequence would trip the determinism guard's
            # "extra calls" check. The completed leaves are already journaled (inside
            # agent()), so the gate-front work still replays free on approve.
            await journal_store.put_progress_count(ctx.progress_entry_count)
            raise
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
        # Fail loud if a sign-off decision was injected but no gate consumed it: the run
        # completed (or had no un-decided gate) yet a human decision was supplied, so
        # silently dropping it would lose a sign-off — exactly the action that must not
        # fail quietly. (resume is UNSET on a fresh run / crash-resume, so this is a
        # no-op there.) This is the engine-level backstop for an approve aimed at a run
        # that was not actually paused at a gate.
        if resume is not UNSET and ctx.has_unconsumed_signoff:
            raise WorkflowCheckpointError(
                "a sign-off decision was supplied (resume=) but the run completed without "
                "an un-decided ctx.checkpoint gate to consume it — the decision was not "
                "applied. The run was likely not actually paused for a sign-off."
            )
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
    # A parked ``ctx.checkpoint`` raises WorkflowSignoffRequired straight out of the
    # entrypoint body (it is journaled, not a LangGraph interrupt), so the caller
    # sees it here without any in-band sentinel to unwrap. The resume value, when
    # set, was threaded into the Ctx and is consumed at the first un-decided gate.
    result: Any = await _run.ainvoke({}, config=config)  # pyright: ignore[reportUnknownMemberType]
    return result

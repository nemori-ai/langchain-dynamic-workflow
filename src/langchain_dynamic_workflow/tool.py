"""The host-facing workflow tool — the agent's single runtime surface.

The library's only runtime consumer is an AI agent, and an agent's only outward
action is a tool call. So the one runtime public surface is a single multi-command
tool: ``run`` launches a workflow in the background and returns a placeholder
``run_id`` immediately (the host turn is never blocked); ``status`` polls a run
and returns its settled result (offloaded behind a handle when large);
``resume`` re-runs a workflow against the same journal so completed leaves replay
at zero model cost; ``cancel`` stops an in-flight run.

The tool delegates background lifecycle to a :class:`BgRunManager` (shared with
the middleware so completion notices can be drained into the host context) and
resolves named workflows through a
:class:`~langchain_dynamic_workflow._workflows.WorkflowRegistry`.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from ._background import BgRunManager, BgRunQuotaExceededError, BgRunStateError, BgStatus

# The meta-layer codegen (AST gate + restricted exec) and its error type sit in
# Layer 2 alongside this tool; the `run_script` command compiles an untrusted
# source string through them. They never reach the engine internals directly.
from ._codegen import compile_workflow_source, extract_meta

# The journal-store type is imported from the engine's public core entry (not the
# internal `_journal`) so the host-facing tool depends only on the `run_workflow`
# surface, preserving the one-directional Layer 2 -> Layer 0/1 boundary that
# import-linter mechanically guards.
from ._engine import JournalStore, run_workflow
from ._errors import WorkflowScriptError
from ._roster import Roster

# The run registry (spec persistence + per-run journals) is injected as a
# WorkflowRunStore; the in-memory default keeps the base install dependency-free
# while a sqlite-backed store extends durability across process restarts.
from ._run_store import InMemoryRunStore, RunSpec, WorkflowRunStore
from ._workflows import WorkflowFn, WorkflowRegistry

# The tool is agnostic to the host's concrete context/state types, so the runtime
# and Command updates are parameterized with Any (the host state schema, not the
# tool, declares the concrete shapes).
_ToolRuntime = ToolRuntime[Any, Any]
_Command = Command[Any]

WORKFLOW_RUNS_STATE_KEY = "workflow_runs"
"""State-channel key tracking launched background runs (survives compaction)."""


class WorkflowToolSchema(BaseModel):
    """Arguments accepted by the multi-command workflow tool.

    ``runtime`` is injected by the tool node and is deliberately absent here so it
    is never advertised to the model; the model supplies only these fields.
    """

    command: Literal["run", "run_script", "status", "resume", "cancel", "runs", "approve"] = Field(
        description="Which workflow operation to perform."
    )
    workflow: str | None = Field(
        default=None,
        description="The registered workflow name to launch (required for 'run').",
    )
    script: str | None = Field(
        default=None,
        description=(
            "Orchestration script source to launch (required for 'run_script'): a "
            "self-contained 'async def orchestrate(ctx, args)' coroutine you author."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "The target run id (required for 'status' / 'resume' / 'cancel' / 'approve')."
        ),
    )
    args: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional arguments for the launched workflow (for 'run' / 'run_script'); "
            "for 'approve' it carries the human sign-off decision passed back into the "
            "paused run (e.g. {'approved': true, 'note': '...'})."
        ),
    )


_WORKFLOW_TOOL_DESCRIPTION = """\
Run a dynamic orchestration workflow in the background and poll it.

Commands (pass `command`):
- `run`: launch a registered workflow by `workflow` name (optionally with `args`).
    Returns immediately with a `run_id` placeholder; the workflow keeps running in
    the background and you may continue your turn. A completion notification is
    delivered before your next reply.
- `run_script`: launch an orchestration `script` you author on the spot — a
    self-contained `async def orchestrate(ctx, args)` coroutine — when no registered
    workflow fits. The script is checked by a security gate before it runs; if it is
    rejected, the specific violations come back so you can fix them and resubmit.
    Use only scripts you author yourself: the gate stops an accidental slip, not a
    determined adversary (it is not a security sandbox).
- `status`: given a `run_id`, return the run's current status and — once done —
    its result. A large result is summarized and offloaded behind a handle. A run
    that paused for a human sign-off reports `awaiting_signoff` and shows the `ask`.
- `resume`: given a finished or interrupted `run_id`, re-run the same workflow
    against its journal so completed steps replay at zero cost. Returns a new run.
    (Use `approve`, not `resume`, to answer a sign-off — `resume` injects no value.)
- `cancel`: given a `run_id`, stop an in-flight run (including one awaiting sign-off).
- `runs`: list every run you launched on this thread with its workflow label and
    live status (and a short outcome preview once settled), so you can see all of
    them at once instead of polling each `run_id`. Takes no arguments.
- `approve`: answer a run that is `awaiting_signoff`. Given its `run_id` and `args`
    as the decision (e.g. {"approved": true, "note": "..."}), feed the value back
    into the paused run so it continues under the same `run_id`. Completed steps and
    already-approved gates replay at zero cost; the run may pause again at a later
    gate or finish.
"""


def _host_thread_id(runtime: _ToolRuntime) -> str:
    """Read the host thread id from the tool runtime config (default ``'default'``)."""
    config = runtime.config or {}
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    return thread_id if isinstance(thread_id, str) else "default"


def create_workflow_tool(
    roster: Roster,
    *,
    manager: BgRunManager,
    workflows: WorkflowRegistry,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_concurrency: int | None = None,
    budget: int | None = None,
    store: WorkflowRunStore | None = None,
) -> StructuredTool:
    """Build the multi-command background workflow tool for a host agent.

    Args:
        roster: The leaf registry passed through to every launched ``run_workflow``.
        manager: The background run manager; share the *same* instance with
            :func:`~langchain_dynamic_workflow.middleware.create_workflow_middleware`
            so completion notices land in the host context.
        workflows: The named-workflow registry resolved by the ``run`` / ``resume``
            commands.
        checkpointer: Optional LangGraph checkpointer for launched runs.
        max_concurrency: Optional explicit concurrency cap forwarded to runs.
        budget: Optional shared token ceiling forwarded to runs.
        store: Optional run registry persisting each launch's spec and per-run
            journal. Defaults to a fresh :class:`InMemoryRunStore` (dependency-free,
            same-session only); inject a sqlite-backed store to make ``resume``
            survive a process restart.

    Returns:
        A :class:`StructuredTool` exposing the ``run`` / ``status`` / ``resume`` /
        ``cancel`` commands.
    """
    # The run registry persists each launch's spec (so `resume` can rebuild the
    # original callable, label, and thread) and hands out a per-run journal (so
    # completed leaves replay from cache at zero model cost). The default is the
    # in-memory store; a sqlite-backed store extends both across a process restart.
    run_store = store if store is not None else InMemoryRunStore()

    def _label_for_script(source: str) -> str:
        """Best-effort run label from a script's optional top-level ``meta['name']``."""
        try:
            meta = extract_meta(source)
        except WorkflowScriptError:
            return "ad-hoc-script"
        if isinstance(meta, dict):
            name = meta.get("name")
            if isinstance(name, str) and name:
                return name
        return "ad-hoc-script"

    def _resolve_spec(spec: RunSpec) -> WorkflowFn:
        """Rebuild a run's orchestration callable from its persisted spec.

        A ``"name"`` spec re-resolves the registered workflow; a ``"script"`` spec
        recompiles the persisted source (it passed the gate on first launch, so the
        recompile succeeds), realizing "the callable is transient, the source is
        durable".
        """
        if spec.kind == "script":
            return compile_workflow_source(spec.name_or_source)
        return workflows.resolve(spec.name_or_source)

    async def _launch(
        *,
        workflow_fn: WorkflowFn,
        spec: RunSpec,
        host_thread_id: str,
    ) -> str:
        """Detach a workflow run onto the background manager; return its run_id.

        The run id is minted up front (before ``manager.start``) so the spec and the
        manager slot are both keyed by the *same* id, keeping the registry
        consistent for a later (possibly cross-process) ``resume``.

        Identity is split deliberately. The ``host_thread_id`` is the current
        caller's thread; it keys the background manager slot so the caller who
        issued the launch can poll its run. The canonical origin id — the spec's
        ``journal_run_id`` when set, else this run's own freshly minted id — keys
        both the per-run journal (so a resume replays completed leaves for free)
        and the per-run LangGraph checkpoint thread (so a resume rejoins its
        checkpoint), independent of the host thread. A fresh launch stamps its own
        id as the canonical origin on the saved spec; a resume inherits the
        origin from the spec it relaunches.

        The spec is persisted *before* the run is admitted to the manager, so a
        crash between admission and save can never strand a run with no spec. If
        admission is refused at the concurrency quota, the pre-saved spec is
        deleted so the refusal leaves no unresumable orphan.

        Args:
            workflow_fn: The orchestration callable to run.
            spec: The launch description persisted under the new run id.
            host_thread_id: The current caller's host thread; keys the background
                manager slot so the caller can poll the launched run.

        Returns:
            The newly minted run id.

        Raises:
            BgRunQuotaExceededError: If admission is refused at the concurrency
                quota; the pre-saved spec is deleted before the error propagates.
        """
        run_id = uuid.uuid4().hex
        # The canonical origin id keys both the journal and the checkpoint thread.
        # A resume inherits it from the spec; a fresh launch adopts its own run id.
        canonical = spec.journal_run_id or run_id
        spec_to_save = (
            spec if spec.journal_run_id else dataclasses.replace(spec, journal_run_id=run_id)
        )
        # The per-run journal must be obtained before the coroutine starts so the
        # launched run records into the same journal a resume will replay from. A
        # resume points this at the origin run's journal; a fresh launch uses its
        # own (still-empty) journal keyed by its own id.
        journal: JournalStore = run_store.journal_for(canonical)
        args = spec.args

        async def _orchestrate(ctx: Any) -> Any:
            return await workflow_fn(ctx, args)

        async def _coro() -> str:
            result = await run_workflow(
                _orchestrate,
                roster=roster,
                journal=journal,
                checkpointer=checkpointer,
                # The CHECKPOINT thread is the per-run canonical id, not the host
                # thread, so distinct runs on one host thread never collapse onto
                # one checkpoint thread and a resume rejoins the origin's thread.
                thread_id=canonical,
                max_concurrency=max_concurrency,
                budget=budget,
                # Wire the same registry so a launched workflow may inline another
                # via ctx.workflow(name) one level deep.
                workflows=workflows,
            )
            return result if isinstance(result, str) else str(result)

        # Persist the spec BEFORE admission so an admitted run always has a spec.
        await run_store.save_spec(run_id, spec_to_save)
        # The label travels with the manager slot so the `runs` listing can name the
        # run authoritatively, not via this tool's bookkeeping. The slot is keyed by
        # the current caller's host thread so the caller can poll the run.
        try:
            manager.start(_coro(), run_id=run_id, thread_id=host_thread_id, label=spec.label)
        except BgRunQuotaExceededError:
            # The run was refused before it could run: roll back the spec we saved
            # above so a refused launch leaves no unresumable orphan in the registry.
            await run_store.delete_spec(run_id)
            raise
        return run_id

    def _launch_record(run_id: str, *, label: str) -> dict[str, str]:
        """The ``workflow_runs`` state record for a freshly launched run."""
        return {"run_id": run_id, "workflow": label, "status": BgStatus.RUNNING.value}

    async def _run_command(
        runtime: _ToolRuntime, *, workflow: str | None, args: dict[str, Any] | None
    ) -> str | _Command:
        if not workflow:
            return "run: the 'workflow' name is required."
        if workflow not in workflows:
            return f"run: unknown workflow {workflow!r}; nothing was launched."
        try:
            run_id = await _launch(
                workflow_fn=workflows.resolve(workflow),
                spec=RunSpec(
                    kind="name",
                    name_or_source=workflow,
                    args=args or {},
                    label=workflow,
                    journal_run_id=None,
                ),
                host_thread_id=_host_thread_id(runtime),
            )
        except BgRunQuotaExceededError as exc:
            # The manager's concurrent-run quota is full: surface it as a clear
            # refusal string (not a Command) so the host knows nothing was launched
            # and can retry after a run finishes, rather than fanning out unbounded.
            return f"run: {exc}"
        message = (
            f"Launched workflow {workflow!r} in the background. run_id: {run_id}. "
            "It is running now; you can continue, and a completion notification will "
            "arrive before your next reply. Use command='status' with this run_id to "
            "fetch the result."
        )
        return Command(
            update={
                "messages": [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
                WORKFLOW_RUNS_STATE_KEY: [_launch_record(run_id, label=workflow)],
            }
        )

    async def _run_script_command(
        runtime: _ToolRuntime, *, script: str | None, args: dict[str, Any] | None
    ) -> str | _Command:
        if not script:
            return "run_script: the 'script' source is required."
        try:
            # The single security checkpoint for an ad-hoc script: gate + compile.
            # On rejection the violations are returned verbatim (a plain string, not
            # a Command) so the host can fix them and resubmit — nothing is launched.
            workflow_fn = compile_workflow_source(script)
        except WorkflowScriptError as exc:
            return f"run_script: the script was rejected and nothing was launched.\n{exc}"
        label = _label_for_script(script)
        try:
            run_id = await _launch(
                workflow_fn=workflow_fn,
                spec=RunSpec(
                    kind="script",
                    name_or_source=script,
                    args=args or {},
                    label=label,
                    journal_run_id=None,
                ),
                host_thread_id=_host_thread_id(runtime),
            )
        except BgRunQuotaExceededError as exc:
            return f"run_script: {exc}"
        message = (
            f"Launched your authored script (labeled {label!r}) in the background. "
            f"run_id: {run_id}. It is running now; you can continue, and a completion "
            "notification will arrive before your next reply. Use command='status' with "
            "this run_id to fetch the result."
        )
        return Command(
            update={
                "messages": [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
                WORKFLOW_RUNS_STATE_KEY: [_launch_record(run_id, label=label)],
            }
        )

    def _status_command(runtime: _ToolRuntime, *, run_id: str | None) -> str:
        if not run_id:
            return "status: a 'run_id' is required."
        thread_id = _host_thread_id(runtime)
        status = manager.poll(run_id, thread_id=thread_id)
        if status == BgStatus.UNKNOWN:
            return f"status: unknown run_id {run_id!r} (never launched, or already reclaimed)."
        if status in {BgStatus.PENDING, BgStatus.RUNNING}:
            return f"status: run {run_id!r} is {status.value}; no result yet."
        if status == BgStatus.AWAITING_SIGNOFF:
            ask = manager.get_signoff(run_id, thread_id=thread_id)
            return (
                f"status: run {run_id!r} is awaiting sign-off.\nask: {ask}\n"
                "Answer it with command='approve', this run_id, and args as the decision."
            )
        result = manager.get_result(run_id, thread_id=thread_id)
        if result.status == BgStatus.FAILED:
            return f"status: run {run_id!r} failed: {result.detail or result.summary}"
        if result.status == BgStatus.CANCELLED:
            return f"status: run {run_id!r} was cancelled."
        # DONE: inline the value, or summary + handle for an offloaded large result.
        if result.handle is not None:
            return (
                f"status: run {run_id!r} done. Result was large and offloaded.\n"
                f"summary: {result.summary}\nhandle: {result.handle}"
            )
        return f"status: run {run_id!r} done.\nresult: {result.value}"

    async def _resume_command(runtime: _ToolRuntime, *, run_id: str | None) -> str | _Command:
        if not run_id:
            return "resume: a 'run_id' is required."
        spec = await run_store.load_spec(run_id)
        if spec is None:
            return f"resume: unknown run_id {run_id!r}; nothing to resume."
        workflow_fn = _resolve_spec(spec)
        label = spec.label
        # The spec carries the canonical origin (journal_run_id), so the relaunch
        # rejoins the ORIGIN's journal (completed leaves replay from cache) and the
        # ORIGIN's checkpoint thread — even when resuming a resume-issued run_id,
        # because each saved spec inherits the same origin. The manager slot is
        # keyed by the CURRENT caller's host thread so the caller who issued the
        # resume can poll the new run.
        new_run_id = await _launch(
            workflow_fn=workflow_fn,
            spec=spec,
            host_thread_id=_host_thread_id(runtime),
        )
        message = (
            f"Resumed {label!r} from run {run_id!r} against its journal. "
            f"New run_id: {new_run_id}. Completed steps replay at zero model cost."
        )
        return Command(
            update={
                "messages": [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
                WORKFLOW_RUNS_STATE_KEY: [_launch_record(new_run_id, label=label)],
            }
        )

    async def _approve_command(
        runtime: _ToolRuntime, *, run_id: str | None, args: dict[str, Any] | None
    ) -> str | _Command:
        if not run_id:
            return "approve: a 'run_id' is required."
        thread_id = _host_thread_id(runtime)
        status = manager.poll(run_id, thread_id=thread_id)
        # ONLY a run parked AND live on THIS process can be approved. The parked state
        # (which gate, the ask) lives in the in-memory manager, not the run store, so a
        # run that is UNKNOWN here — swept after settling, or parked in another (dead)
        # process — cannot be verified as genuinely awaiting a sign-off. Relaunching it
        # from its persisted spec would risk advancing a NON-parked run past a gate no
        # human saw, or silently dropping the decision; so refuse rather than guess.
        if status == BgStatus.UNKNOWN:
            return (
                f"approve: run {run_id!r} is not awaiting sign-off on this process "
                "(unknown, already settled, or parked elsewhere); a sign-off can only be "
                "approved where its parked run is live. Check command='runs'."
            )
        if status != BgStatus.AWAITING_SIGNOFF:
            return f"approve: run {run_id!r} is {status.value}, not awaiting sign-off."
        spec = await run_store.load_spec(run_id)
        if spec is None:
            return f"approve: unknown run_id {run_id!r}; nothing to approve."
        workflow_fn = _resolve_spec(spec)
        # Continue the SAME run in place (its slot, its run_id) so a host tracking the
        # run by id follows it across the pause. Completed leaves and already-approved
        # gates replay at zero cost from the origin journal; the decision is injected at
        # the next un-decided gate (and the engine fails loud if none consumes it).
        canonical = spec.journal_run_id or run_id
        origin_journal: JournalStore = run_store.journal_for(canonical)
        spec_args = spec.args
        decision = args or {}

        async def _orchestrate(ctx: Any) -> Any:
            return await workflow_fn(ctx, spec_args)

        async def _coro() -> str:
            result = await run_workflow(
                _orchestrate,
                roster=roster,
                journal=origin_journal,
                checkpointer=checkpointer,
                thread_id=canonical,
                resume=decision,
                max_concurrency=max_concurrency,
                budget=budget,
                workflows=workflows,
            )
            return result if isinstance(result, str) else str(result)

        try:
            manager.approve(_coro(), run_id=run_id, thread_id=thread_id)
        except (KeyError, BgRunStateError) as exc:
            return f"approve: {exc}"
        message = (
            f"Approved the sign-off for run {run_id!r}; it is continuing in the "
            "background under the same run_id. A completion notification will arrive "
            "before your next reply, or it may pause again at a later gate (check with "
            "command='status')."
        )
        return Command(
            update={
                "messages": [ToolMessage(message, tool_call_id=runtime.tool_call_id)],
                WORKFLOW_RUNS_STATE_KEY: [_launch_record(run_id, label=spec.label)],
            }
        )

    def _runs_command(runtime: _ToolRuntime) -> str:
        """List every run on the host thread with its label and live status.

        The label is read from the run's own manager slot (recorded at launch), so
        it is authoritative regardless of which tool instance launched the run; a
        run launched outside this surface with no label renders as ``?``.
        """
        thread_id = _host_thread_id(runtime)
        snapshots = manager.list_runs(thread_id)
        if not snapshots:
            return "runs: no runs on this thread yet."
        lines: list[str] = []
        for snap in snapshots:
            line = f"- {snap.run_id} · {snap.label or '?'} · {snap.status.value}"
            if snap.summary:
                line += f" · {snap.summary}"
            lines.append(line)
        return f"runs: {len(snapshots)} run(s) on this thread:\n" + "\n".join(lines)

    async def _cancel_command(runtime: _ToolRuntime, *, run_id: str | None) -> str:
        if not run_id:
            return "cancel: a 'run_id' is required."
        thread_id = _host_thread_id(runtime)
        if manager.poll(run_id, thread_id=thread_id) == BgStatus.UNKNOWN:
            return f"cancel: unknown run_id {run_id!r}; nothing to cancel."
        await manager.cancel(run_id, thread_id=thread_id)
        return f"cancel: run {run_id!r} was cancelled."

    async def workflow_tool(
        command: str,
        workflow: str | None = None,
        script: str | None = None,
        run_id: str | None = None,
        args: dict[str, Any] | None = None,
        *,
        runtime: _ToolRuntime,
    ) -> str | _Command:
        """Dispatch one workflow command (run / run_script / status / resume / cancel).

        ``runtime`` is keyword-only and injected by the tool node; it is never
        part of the model-facing schema.
        """
        if command == "run":
            return await _run_command(runtime, workflow=workflow, args=args)
        if command == "run_script":
            return await _run_script_command(runtime, script=script, args=args)
        if command == "status":
            return _status_command(runtime, run_id=run_id)
        if command == "resume":
            return await _resume_command(runtime, run_id=run_id)
        if command == "cancel":
            return await _cancel_command(runtime, run_id=run_id)
        if command == "runs":
            return _runs_command(runtime)
        if command == "approve":
            return await _approve_command(runtime, run_id=run_id, args=args)
        return (
            f"unknown command {command!r}; expected one of: "
            "run, run_script, status, resume, cancel, runs, approve."
        )

    # Under ``from __future__ import annotations`` the ``runtime`` annotation is a
    # string at runtime, so the tool's injected-arg detection (which reads the raw
    # ``__annotations__`` via ``inspect.signature``) would not recognize it and the
    # injected ToolRuntime would be stripped before reaching the coroutine. Pin the
    # resolved type on the runtime parameter so it is detected as an injected arg.
    workflow_tool.__annotations__["runtime"] = ToolRuntime

    # An explicit args_schema (infer_schema=False) keeps the injected ToolRuntime
    # parameter out of the model-facing schema; schema inference cannot handle it.
    # from_function is typed loosely upstream; the constructed value is a
    # StructuredTool, so narrow it back for our strict surface.
    tool: StructuredTool = StructuredTool.from_function(  # pyright: ignore[reportUnknownMemberType]
        coroutine=workflow_tool,
        name="workflow",
        description=_WORKFLOW_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=WorkflowToolSchema,
    )
    return tool

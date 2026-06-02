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

from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from ._background import BgRunManager, BgRunQuotaExceededError, BgStatus

# The meta-layer codegen (AST gate + restricted exec) and its error type sit in
# Layer 2 alongside this tool; the `run_script` command compiles an untrusted
# source string through them. They never reach the engine internals directly.
from ._codegen import compile_workflow_source, extract_meta

# Journal-store types are imported from the engine's public core entry (not the
# internal `_journal`) so the host-facing tool depends only on the `run_workflow`
# surface, preserving the one-directional Layer 2 -> Layer 0/1 boundary that
# import-linter mechanically guards.
from ._engine import InMemoryJournalStore, JournalStore, run_workflow
from ._errors import WorkflowScriptError
from ._roster import Roster
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

    command: Literal["run", "run_script", "status", "resume", "cancel"] = Field(
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
        description="The target run id (required for 'status' / 'resume' / 'cancel').",
    )
    args: dict[str, Any] | None = Field(
        default=None,
        description="Optional arguments for the launched workflow (for 'run' / 'run_script').",
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
    its result. A large result is summarized and offloaded behind a handle.
- `resume`: given a finished or interrupted `run_id`, re-run the same workflow
    against its journal so completed steps replay at zero cost. Returns a new run.
- `cancel`: given a `run_id`, stop an in-flight run.
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

    Returns:
        A :class:`StructuredTool` exposing the ``run`` / ``status`` / ``resume`` /
        ``cancel`` commands.
    """
    # Per-run journals so `resume` can re-run a workflow against the journal its
    # first run populated — completed leaves replay from cache at zero model cost.
    journals: dict[str, JournalStore] = {}
    # How each run can be re-launched on `resume`, as a tagged spec
    # ``(kind, name_or_source, args)`` where ``kind`` is "name" (a registered
    # workflow re-resolved by name) or "script" (an ad-hoc source recompiled).
    run_specs: dict[str, tuple[str, str, dict[str, Any]]] = {}

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

    def _resolve_spec(spec: tuple[str, str, dict[str, Any]]) -> tuple[WorkflowFn, dict[str, Any]]:
        """Rebuild a run's orchestration callable + args from its resume spec.

        A "name" spec re-resolves the registered workflow; a "script" spec
        recompiles the persisted source (it passed the gate on first launch, so the
        recompile succeeds), realizing "the callable is transient, the source is
        durable".
        """
        kind, name_or_source, args = spec
        if kind == "script":
            return compile_workflow_source(name_or_source), args
        return workflows.resolve(name_or_source), args

    def _launch(
        *,
        workflow_fn: WorkflowFn,
        args: dict[str, Any],
        spec: tuple[str, str, dict[str, Any]],
        thread_id: str,
        journal: JournalStore,
    ) -> str:
        """Detach a workflow run onto the background manager; return its run_id."""

        async def _orchestrate(ctx: Any) -> Any:
            return await workflow_fn(ctx, args)

        async def _coro() -> str:
            result = await run_workflow(
                _orchestrate,
                roster=roster,
                journal=journal,
                checkpointer=checkpointer,
                thread_id=thread_id,
                max_concurrency=max_concurrency,
                budget=budget,
                # Wire the same registry so a launched workflow may inline another
                # via ctx.workflow(name) one level deep.
                workflows=workflows,
            )
            return result if isinstance(result, str) else str(result)

        slot = manager.start(_coro(), thread_id=thread_id)
        journals[slot.run_id] = journal
        run_specs[slot.run_id] = spec
        return slot.run_id

    def _launch_record(run_id: str, *, label: str) -> dict[str, str]:
        """The ``workflow_runs`` state record for a freshly launched run."""
        return {"run_id": run_id, "workflow": label, "status": BgStatus.RUNNING.value}

    def _run_command(
        runtime: _ToolRuntime, *, workflow: str | None, args: dict[str, Any] | None
    ) -> str | _Command:
        if not workflow:
            return "run: the 'workflow' name is required."
        if workflow not in workflows:
            return f"run: unknown workflow {workflow!r}; nothing was launched."
        thread_id = _host_thread_id(runtime)
        try:
            run_id = _launch(
                workflow_fn=workflows.resolve(workflow),
                args=args or {},
                spec=("name", workflow, args or {}),
                thread_id=thread_id,
                journal=InMemoryJournalStore(),
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

    def _run_script_command(
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
        thread_id = _host_thread_id(runtime)
        try:
            run_id = _launch(
                workflow_fn=workflow_fn,
                args=args or {},
                spec=("script", script, args or {}),
                thread_id=thread_id,
                journal=InMemoryJournalStore(),
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

    def _resume_command(runtime: _ToolRuntime, *, run_id: str | None) -> str | _Command:
        if not run_id:
            return "resume: a 'run_id' is required."
        if run_id not in run_specs:
            return f"resume: unknown run_id {run_id!r}; nothing to resume."
        thread_id = _host_thread_id(runtime)
        spec = run_specs[run_id]
        workflow_fn, args = _resolve_spec(spec)
        label = _label_for_script(spec[1]) if spec[0] == "script" else spec[1]
        # Re-run against the same journal so completed leaves replay from cache.
        new_run_id = _launch(
            workflow_fn=workflow_fn,
            args=args,
            spec=spec,
            thread_id=thread_id,
            journal=journals[run_id],
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
            return _run_command(runtime, workflow=workflow, args=args)
        if command == "run_script":
            return _run_script_command(runtime, script=script, args=args)
        if command == "status":
            return _status_command(runtime, run_id=run_id)
        if command == "resume":
            return _resume_command(runtime, run_id=run_id)
        if command == "cancel":
            return await _cancel_command(runtime, run_id=run_id)
        return (
            f"unknown command {command!r}; expected one of: "
            "run, run_script, status, resume, cancel."
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

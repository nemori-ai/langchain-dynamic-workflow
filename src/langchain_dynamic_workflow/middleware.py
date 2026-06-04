"""The workflow middleware — tool contribution + in-band completion notification.

A single ``AgentMiddleware`` packages the host-facing surface: it contributes
the multi-command workflow tool (``.tools``) and, before the host's next model
call, drains any completed background runs for the host thread and injects an
in-band ``<workflow_notification>`` message (``abefore_model``). This is the
fire-and-notify delivery load-bearer: without a hosting harness, the only way to
push a completion into the host's context is to inject it before the next model
turn.

Launched runs are tracked in a dedicated ``workflow_runs`` state channel (mirroring
deepagents' ``async_tasks``) so the record survives context compaction and can be
inspected programmatically.

Scope note: this middleware is a host-turn cross-cut. It bounds nothing inside a
workflow run — the engine-internal ``@task`` / journal / budget / sandbox live in
a strictly different scope inside ``run_workflow``.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime

from ._background import BgRunManager, BgStatus, Notice
from ._roster import Roster
from ._workflows import WorkflowRegistry
from .tool import create_workflow_tool

WORKFLOW_NOTIFICATION_TAG = "workflow_notification"
"""The XML-ish tag wrapping an injected background-run completion notice."""


def merge_workflow_runs(
    existing: list[dict[str, Any]] | None, incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reducer for the ``workflow_runs`` channel: upsert records by ``run_id``.

    A launch writes ``{run_id, workflow, status: running}``; a later settle update
    writes ``{run_id, status: <terminal>}``. Merging field-wise by ``run_id``
    (rather than appending) keeps one record per run whose status tracks its
    lifecycle, so the channel reflects the terminal status instead of staying at
    the launch-time ``running``. First-seen order is preserved; a record for an
    unseen ``run_id`` appends.

    Args:
        existing: The accumulated records (``None`` / empty on the first write).
        incoming: The records to merge in.

    Returns:
        The merged records, one per ``run_id``, in first-seen order.
    """
    merged: dict[str, dict[str, Any]] = {}
    for record in existing or []:
        run_id = record.get("run_id")
        if isinstance(run_id, str):
            merged[run_id] = dict(record)
    for record in incoming or []:
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            continue
        merged[run_id] = {**merged[run_id], **record} if run_id in merged else dict(record)
    return list(merged.values())


class WorkflowState(AgentState):
    """Host agent state extended with a background-run tracking channel.

    Attributes:
        workflow_runs: Launched background runs (each a record carrying its
            ``run_id`` / ``workflow`` / ``status``), upserted by ``run_id`` across
            turns so the record survives context compaction and its status is
            rewritten from ``running`` to the run's terminal status on settle.
    """

    workflow_runs: NotRequired[Annotated[list[dict[str, Any]], merge_workflow_runs]]


def _host_thread_id(config: RunnableConfig | None) -> str:
    """Read the host thread id from the runnable config (default ``'default'``)."""
    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id")
    return thread_id if isinstance(thread_id, str) else "default"


def _format_notice(notice: Notice) -> str:
    """Render one completion notice as a single human-readable line."""
    if notice.status == BgStatus.DONE:
        return f"- run {notice.run_id}: done. summary: {notice.summary}"
    if notice.status == BgStatus.FAILED:
        return f"- run {notice.run_id}: failed. {notice.detail or notice.summary}"
    if notice.status == BgStatus.CANCELLED:
        return f"- run {notice.run_id}: cancelled."
    return f"- run {notice.run_id}: {notice.status.value}."


def _render_notification(notices: list[Notice]) -> str:
    """Wrap completion notices in a ``<workflow_notification>`` block for the host."""
    lines = "\n".join(_format_notice(n) for n in notices)
    return (
        f"<{WORKFLOW_NOTIFICATION_TAG}>\n"
        "One or more background workflows you launched have finished:\n"
        f"{lines}\n"
        "Use the workflow tool with command='status' and a run_id to fetch a result.\n"
        f"</{WORKFLOW_NOTIFICATION_TAG}>"
    )


class WorkflowMiddleware(AgentMiddleware[WorkflowState, Any, Any]):
    """Contributes the workflow tool and injects background-run completion notices.

    The contributed tool exposes both meta-layer paths: launching a registered
    workflow by name (``run``) and launching an ad-hoc script the host authors on
    the spot (``run_script``, gated and compiled before it runs). The package's
    built-in orchestration skill (which teaches a host to author such scripts) is
    loaded separately onto ``create_deep_agent`` via ``skills_path()`` /
    ``skill_files()``; this middleware only contributes the workflow tool and the
    completion-notice injection.

    Args:
        roster: The leaf registry forwarded to launched runs.
        workflows: The named-workflow registry the tool resolves.
        manager: The shared background run manager owning run lifecycle and the
            completion-notice queue.
        checkpointer: Optional checkpointer forwarded to launched runs.
        max_concurrency: Optional concurrency cap forwarded to launched runs.
        budget: Optional shared token ceiling forwarded to launched runs.
    """

    state_schema = WorkflowState

    def __init__(
        self,
        *,
        roster: Roster,
        workflows: WorkflowRegistry,
        manager: BgRunManager,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        max_concurrency: int | None = None,
        budget: int | None = None,
    ) -> None:
        super().__init__()
        self._manager = manager
        self.tools = [
            create_workflow_tool(
                roster,
                manager=manager,
                workflows=workflows,
                checkpointer=checkpointer,
                max_concurrency=max_concurrency,
                budget=budget,
            )
        ]

    @property
    def manager(self) -> BgRunManager:
        """The shared background run manager backing this middleware."""
        return self._manager

    async def abefore_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        state: WorkflowState,
        runtime: Runtime[Any],
        config: RunnableConfig,
    ) -> dict[str, Any] | None:
        """Inject a ``<workflow_notification>`` for any completed runs on this thread.

        Drains the manager's completion-notice queue for the host thread (read
        from ``config``) and, if any notices are pending, returns a state update
        appending a single in-band notification message. Draining is one-shot, so
        each completion is injected exactly once. Returns ``None`` when nothing is
        pending, leaving the host turn untouched.

        The ``config`` parameter is annotated as a bare ``RunnableConfig`` (not
        ``RunnableConfig | None``) on purpose: under ``from __future__ import
        annotations`` the runtime annotation is a string, and the graph node runner
        only injects ``config`` when that string matches its accepted set — the
        ``| None`` spelling is not in that set, so it would silently leave config
        unbound and the host thread id would fall back to the wrong queue.
        """
        thread_id = _host_thread_id(config)
        notices = self._manager.drain_notifications(thread_id)
        if not notices:
            return None
        # Besides injecting the notification, rewrite each settled run's record from
        # the launch-time `running` to its terminal status. The channel reducer
        # (merge_workflow_runs) upserts these by run_id, so workflow_runs reflects
        # the live outcome instead of a stale `running`.
        settle_updates = [{"run_id": n.run_id, "status": n.status.value} for n in notices]
        return {
            "messages": [HumanMessage(content=_render_notification(notices))],
            "workflow_runs": settle_updates,
        }


def create_workflow_middleware(
    roster: Roster,
    *,
    workflows: WorkflowRegistry,
    manager: BgRunManager | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_concurrency: int | None = None,
    max_concurrent_runs: int | None = None,
    budget: int | None = None,
) -> WorkflowMiddleware:
    """Build the workflow middleware (tool + completion-notice injection).

    Args:
        roster: The leaf registry forwarded to launched runs.
        workflows: The named-workflow registry the tool resolves.
        manager: Optional shared background run manager; a fresh one is created
            when omitted. Pass the *same* instance to the tool factory if you
            build the tool separately, so completion notices reach the host.
        checkpointer: Optional checkpointer forwarded to launched runs.
        max_concurrency: Optional concurrency cap forwarded to launched runs.
        max_concurrent_runs: Optional cap on concurrent host-initiated background
            runs, applied only to the default manager this factory builds when
            ``manager`` is omitted. The quota lives on the :class:`BgRunManager`, so
            passing both an explicit ``manager`` and ``max_concurrent_runs`` is a
            conflict (the parameter could not take effect) and is rejected loud —
            set the quota on your manager directly instead. When the quota is full
            the ``run`` command refuses with a clear message rather than launching
            unbounded.
        budget: Optional shared token ceiling forwarded to launched runs.

    Returns:
        A :class:`WorkflowMiddleware` contributing the workflow tool and injecting
        ``<workflow_notification>`` before the host's next model call.

    Raises:
        ValueError: If both an explicit ``manager`` and ``max_concurrent_runs`` are
            supplied — the quota would be silently ignored, so it fails loud.
    """
    if manager is not None and max_concurrent_runs is not None:
        raise ValueError(
            "max_concurrent_runs applies only to the default manager this factory "
            "builds; it cannot be combined with an explicit `manager`. Set the quota "
            "on that manager instead: BgRunManager(max_concurrent_runs=...)."
        )
    return WorkflowMiddleware(
        roster=roster,
        workflows=workflows,
        manager=(
            manager
            if manager is not None
            else BgRunManager(max_concurrent_runs=max_concurrent_runs)
        ),
        checkpointer=checkpointer,
        max_concurrency=max_concurrency,
        budget=budget,
    )

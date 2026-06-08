"""Host deepagent graph factory for the interactive demo.

``langgraph dev`` loads :func:`make_host_graph` (registered under the ``host`` graph
id in ``langgraph.json``) and serves it on the local API. The host carries a ``ui``
state channel (reduced by ``ui_message_reducer``) so Generative-UI components render
in the chat, and two tools that drive inline workflow runs whose progress/span hooks
flow through a :class:`~ui_adapter.UiAdapter`:

* ``run_hello_demo`` — runs the trivial ``hello_workflow`` and also pushes a
  ``hello_ui`` component, proving the Generative-UI round-trip from a node context.
* ``run_live`` — the generic entry: resolves a named preset workflow
  (``deep_research`` / ``capstone``) from :func:`~workflows.make_workflows`, runs it
  inline against the real roster, and streams its phase / fan-out / span events live.

Both tools capture the host node context up front (``ui_bridge.make_host_ui_emit``)
so events emitted from deep inside the nested engine run still target the host
graph's stream and ``ui`` channel. The shared inline-run mechanics live in
:func:`run_workflow_live`, which is decoupled from the tool decorator so it can be
tested without a node context.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from typing import Annotated, Any

from _meta_fixtures import AUTHORED_SCRIPT, META_TOPICS, REJECTED_SCRIPT
from _models import cache_middleware, is_offline, resolve_host_model
from deepagents import DeepAgentState, create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langgraph.prebuilt import InjectedState
from ui_adapter import UiAdapter
from ui_bridge import UiEmit, make_host_ui_emit
from workflows import hello_workflow, make_reasoning_roster, make_roster, make_workflows

from langchain_dynamic_workflow import (
    BgRunManager,
    BgStatus,
    Ctx,
    ExecDecision,
    ExecPolicy,
    ExecRequest,
    InMemoryJournalStore,
    JournalStore,
    SandboxManager,
    WorkflowScriptError,
    compile_workflow_source,
    local_subprocess_factory,
    run_workflow,
    run_workflow_from_source,
)

HOST_INSTRUCTIONS = (
    "You are a thorough assistant for the dynamic-workflow demo. A hard, multi-step "
    "task is best tackled by decomposing it into independent sub-work, investigating "
    "those parts in parallel, cross-checking the findings before committing to them, "
    "and only then synthesizing a clear answer — and that heavier orchestration runs "
    "as a workflow rather than being crammed into a single pass. When the user wants "
    "to drive a preset scenario live, run it through the live workflow tool and let "
    "its progress stream into the panel. Otherwise answer directly and concisely."
)

# The preset the offline host (and a key-free user) triggers by default. A richer,
# fan-out-heavy scenario than the hello smoke path, so the demo shows real
# control-flow inversion out of the box.
DEFAULT_LIVE_WORKFLOW = "deep_research"

# Resilience bounds for the real execution sandbox the fix_loop's code_fixer leaf runs
# its build/test commands in. A per-command timeout caps a runaway test; the output cap
# bounds a chatty failure so the captured tail (and the on_command terminal card) stays
# small. Generous enough for `bun test` on a tiny module, tight enough to fail fast.
_EXEC_TIMEOUT_SECONDS = 60
_EXEC_OUTPUT_CAP_BYTES = 64 * 1024

# Defense-in-depth admission allowlist for the demo's LLM-driven execution. The fix loop
# only ever needs the test runner, so the safe set is the program *prefixes* a realistic
# fixer shells out for: the test/build toolchain plus benign read/inspect ops. The leading
# program token of a command must be on this set for the command to be admitted.
_SAFE_PROGRAMS: frozenset[str] = frozenset(
    {
        "bun",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "tsc",
        "deno",
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "echo",
        "pwd",
        "which",
        "wc",
        "mkdir",
        "touch",
        "true",
        "test",
    }
)

# Obviously dangerous shapes refused regardless of the leading program — a hard denylist
# that takes precedence over the safe-program allowlist (so e.g. `cat /etc/passwd` is
# rejected even though `cat` is otherwise safe). Each pattern targets a destructive or
# exfiltrating shape: a recursive root/home delete, a pipe-to-shell of remote content, a
# fork bomb, a read of a sensitive system path, and a privilege escalation prefix.
_DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\b.*\s-[a-z]*[rf]"),  # rm -rf / -fr style recursive/force delete
    re.compile(r"[|;&]\s*(?:sh|bash|zsh)\b"),  # pipe/chain into a shell (curl ... | sh)
    re.compile(r":\(\)\s*\{"),  # classic fork bomb :(){ :|:& };:
    re.compile(r"/etc/(?:passwd|shadow)\b"),  # reading sensitive system files
    re.compile(r"~/\.ssh\b|/\.ssh/"),  # ssh key material
    re.compile(r"\b(?:sudo|doas|su)\b"),  # privilege escalation
    re.compile(r"\b(?:curl|wget|nc|ncat|telnet)\b"),  # network egress / reverse shell
)


def fix_loop_admission(request: ExecRequest) -> ExecDecision:
    """Admit the intended fix-loop command shape; reject obviously dangerous ones.

    A defense-in-depth admission hook on top of the per-leaf real sandbox: even with the
    fix-loop driven by an LLM, the only command that should reach the host's shell is the
    test/build toolchain plus benign read ops. A command is REJECTED when it matches a hard
    denylist (a recursive delete, a pipe-to-shell, a fork bomb, a sensitive-path read,
    privilege escalation, or network egress) — this takes precedence so a safe program used
    dangerously (``cat /etc/passwd``) is still refused. Otherwise it is admitted only when
    its leading program is on the safe allowlist; anything else is rejected with a reason.

    This is the demo's belt-and-suspenders boundary, not a security sandbox:
    ``LocalSubprocessSandbox`` runs with the calling user's permissions, so the hook reduces
    the blast radius of an LLM-driven command but does not contain a determined adversary.

    Args:
        request: The admission request for one ``execute``, carrying the command string.

    Returns:
        An :class:`ExecDecision` admitting the command (``"allow"``) or refusing it
        (``"reject"``) with a short reason, evaluated before any subprocess spawns.
    """
    command = request.command.strip()
    if not command:
        return ExecDecision(outcome="reject", reason="empty command")
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return ExecDecision(
                outcome="reject", reason=f"command matches a denied dangerous shape: {command!r}"
            )
    try:
        tokens = shlex.split(command)
    except ValueError:
        # An unbalanced quote / malformed shell word — refuse rather than guess.
        return ExecDecision(outcome="reject", reason=f"unparseable command: {command!r}")
    if not tokens:
        return ExecDecision(outcome="reject", reason="no program token")
    program = tokens[0].rsplit("/", 1)[-1]  # strip any leading path to the bare program
    if program in _SAFE_PROGRAMS:
        return ExecDecision(outcome="allow")
    return ExecDecision(
        outcome="reject", reason=f"program {program!r} is not on the demo's safe allowlist"
    )


# The admission hook the real-execution sandbox enforces, exported so a test can drive it
# directly and the fix-loop wiring can pin it onto the ExecPolicy.
FIX_LOOP_ADMISSION = fix_loop_admission


def _make_exec_policy() -> ExecPolicy:
    """Build the bounded :class:`ExecPolicy` the real-execution sandboxes run under.

    Carries the per-command timeout, output cap, and the :func:`fix_loop_admission`
    admission hook (the defense-in-depth allowlist). Factored out so a test can assert the
    hook is wired and exercise it through a real sandbox boundary.

    Returns:
        The :class:`ExecPolicy` for the demo's real local-subprocess execution.
    """
    return ExecPolicy(
        default_timeout=_EXEC_TIMEOUT_SECONDS,
        output_cap_bytes=_EXEC_OUTPUT_CAP_BYTES,
        before_execute=FIX_LOOP_ADMISSION,
    )


def _make_sandbox_manager() -> SandboxManager:
    """Build a per-run sandbox manager, real-execution only when a leaf key is in force.

    When a provider key is in force (:func:`~_models.is_offline` is ``False``) the returned
    :class:`SandboxManager` produces real ``LocalSubprocessSandbox`` backends under
    :func:`_make_exec_policy` (a per-command timeout, an output cap, and the
    :func:`fix_loop_admission` allowlist). A ``needs_execution`` leaf (the ``fix_loop``
    preset's ``code_fixer``) is leased one of these so its ``execute`` tool runs real shell
    commands for a true exit code; a pure-reasoning leaf allocates none.

    Offline (no key), no real sandbox is built: the manager falls back to the engine's
    default offline backend, so no ``LocalSubprocessSandbox`` is ever leased and no
    subprocess can run — the fake leaves return a structured result without shelling out.
    This keeps the dangerous real-execution opt-in scoped to a keyed session rather than
    leased on every (including offline) run.

    DANGEROUS OPT-IN: when keyed, ``LocalSubprocessSandbox`` runs real shell commands on the
    host with the calling user's permissions and is not a security sandbox; the admission
    hook reduces but does not eliminate that risk.

    Returns:
        A fresh per-run :class:`SandboxManager` — real-execution when keyed, the default
        offline backend otherwise.
    """
    if is_offline():
        # No key in force: never build a real subprocess backend. The default manager
        # seeds an offline in-memory backend, so no execute ever spawns a subprocess.
        return SandboxManager()
    return SandboxManager(sandbox_factory=local_subprocess_factory(_make_exec_policy()))


class _ResumeLane:
    """A per-(host-thread, workflow) durable lane for a resumable preset run.

    A preset run is a single :func:`run_workflow` invocation, and the engine's
    content-hash journal only yields cache hits when the *same* journal instance is
    reused across calls. To make "pick it back up" a real behavior — a second turn on
    the same host thread replaying the first run's leaves as zero-cost journal hits —
    the host must persist the journal (and its checkpointer + durable ``thread_id``)
    and feed it back into the next ``run_workflow`` on that lane.

    The lane is keyed on ``(host_thread_id, workflow_name)`` rather than the host
    thread alone: each preset records its own ordered leaf-call sequence in the
    journal, and the engine's determinism backstop replays that sequence on resume —
    so two different presets sharing one journal would trip a false divergence. A
    dedicated lane per workflow keeps each preset independently resumable.

    Attributes:
        journal: The persisted content-hash journal whose recorded leaf results the
            next run on this lane replays as cache hits.
        checkpointer: The persisted LangGraph checkpointer for the lane's durable
            ``@entrypoint`` thread.
        thread_id: The engine ``thread_id`` for the lane's durable run (distinct from
            the host graph's own thread id).
    """

    __slots__ = ("checkpointer", "journal", "thread_id")

    def __init__(self, thread_id: str) -> None:
        self.journal: JournalStore = InMemoryJournalStore()
        self.checkpointer: BaseCheckpointSaver[Any] = InMemorySaver()
        self.thread_id = thread_id


# Durable lanes keyed on "<host_thread_id>::<workflow_name>". Held at module scope so
# they survive across host turns within a process: the second turn on the same host
# thread reuses the first turn's journal and replays its leaves as cache hits. A fresh
# process (a server restart) starts empty, which is the honest in-memory-store bound.
_RESUME_LANES: dict[str, _ResumeLane] = {}


def _host_thread_id() -> str:
    """Return the host graph's durable thread id from the current node config.

    ``langgraph dev`` runs each chat thread under a ``configurable.thread_id``; that
    id keys the resume lanes so a follow-up turn on the *same* chat thread reuses its
    prior run's journal. Must be called from within the host node context. Falls back
    to ``"default"`` when no thread id is configured (e.g. a bare unit-test invocation).

    Returns:
        The host graph's configured thread id, or ``"default"`` when absent.
    """
    config = get_config()
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else "default"


def _resume_lane(workflow: str) -> _ResumeLane:
    """Return the durable resume lane for ``workflow`` on the current host thread.

    Creates the lane on first use and reuses it on every later turn for the same
    ``(host_thread_id, workflow)`` pair, so a second run replays the first run's
    journaled leaves. Must be called from within the host node context (it reads the
    host thread id from the node config).

    Args:
        workflow: The preset workflow name whose lane to resolve.

    Returns:
        The persisted :class:`_ResumeLane` for this host thread and workflow.
    """
    key = f"{_host_thread_id()}::{workflow}"
    lane = _RESUME_LANES.get(key)
    if lane is None:
        lane = _ResumeLane(thread_id=key)
        _RESUME_LANES[key] = lane
    return lane


class HostState(DeepAgentState):
    """Host graph state: the deepagent state plus a Generative-UI message channel.

    Attributes:
        ui: Generative-UI messages pushed via ``push_ui_message`` during a turn.
            Reduced by ``ui_message_reducer`` so pushes append/merge into the
            channel the frontend renders.
    """

    ui: Annotated[list[AnyUIMessage], ui_message_reducer]


def _anchor_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Return the AI message that UI events should attach to.

    The frontend renders a UI message under the chat message whose id matches the
    UI event's ``metadata.message_id``. The right anchor is the most recent AI
    message (the one whose tool call invoked the pushing tool), so pushed
    components appear inline beneath the host's turn.

    Args:
        messages: The conversation messages from graph state.

    Returns:
        The most recent :class:`AIMessage`, or ``None`` if there is none.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


# Component name for the one-time run-status banner the host emits each turn. Carries
# the real offline state (no provider key) so the frontend can surface its offline
# banner from a true backend signal rather than a hardcoded flag.
_RUN_STATUS = "run_status"


def _emit_run_status(emit: UiEmit) -> None:
    """Emit a one-time ``run_status`` UI event carrying the real offline state.

    The offline flag is resolved by :func:`~_models.is_offline` (the same provider-key
    gate :func:`~_models.resolve_host_model` uses), so the frontend's offline banner
    reflects true backend state. The ``event_id`` is fixed so a re-emit on the same
    turn dedupes via the SDK's ui-message reducer.

    Args:
        emit: The host-bound, non-blocking UI emit (rebinds the host node context).
    """
    emit(_RUN_STATUS, {"offline": is_offline(), "event_id": "run-status-1"})


# Component name for the meta-layer viewer: the authored orchestration source plus the
# AST-gate verdict. The frontend's MetaScriptViewer renders these props exactly —
# ``source`` / ``gate`` ("passed" | "failed") / optional ``reason`` / ``event_id`` — so
# the keys and the gate values here must match it byte-for-byte.
_META_SCRIPT = "meta_script"
_GATE_PASSED = "passed"
_GATE_FAILED = "failed"


def _emit_meta_script(emit: UiEmit, *, source: str, gate: str, reason: str | None) -> None:
    """Emit a ``meta_script`` UI event carrying the authored source and gate verdict.

    Pushed through the host-bound ``emit`` directly (not via the :class:`UiAdapter`,
    whose vocabulary is the engine's progress/span events). The props match the
    frontend MetaScriptViewer contract: ``source`` (the generated script), ``gate``
    (``"passed"`` or ``"failed"``), an optional ``reason`` present only on a failed
    gate, and a stable ``event_id``. The id is derived from the gate verdict and a short
    hash of the source so an honest same-turn re-emit of the identical verdict dedupes
    while a different verdict (a rejected attempt then a corrected one) gets its own id.

    Args:
        emit: The host-bound, non-blocking UI emit (rebinds the host node context).
        source: The orchestration script source the meta layer authored.
        gate: The AST-gate verdict, ``"passed"`` or ``"failed"``.
        reason: The line-numbered rejection message when the gate failed, else ``None``.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    props: dict[str, Any] = {
        "source": source,
        "gate": gate,
        "event_id": f"meta-{gate}-{digest}",
    }
    if reason is not None:
        props["reason"] = reason
    emit(_META_SCRIPT, props)


async def run_meta_script_live(*, submit_rejected: bool, adapter: UiAdapter, emit: UiEmit) -> str:
    """Author a script, gate it, and (when admitted) run it — streaming via ``adapter``.

    The engine-facing core of the :func:`run_meta_script` host tool, kept free of the
    node-context machinery so it can be exercised directly in tests. It selects one of
    the two demo fixtures and crosses the engine's AST security gate before anything
    runs:

    * Gate-pass (``submit_rejected=False``): the clean authored script. The gate admits
      it (verified via :func:`compile_workflow_source`), a ``meta_script`` event reports
      ``gate="passed"``, then the script runs via :func:`run_workflow_from_source` with
      the adapter's sinks wired to ``on_progress`` / ``on_span`` so its phases and
      parallel fan-out stream live, and its result is returned.
    * Gate-fail (``submit_rejected=True``): the import-bearing script. The gate raises
      :class:`WorkflowScriptError`; a ``meta_script`` event reports ``gate="failed"``
      with the exception's line-numbered ``reason``, and NOTHING executes — the tool
      returns a short rejection notice.

    Security boundary: the gate stops an accidental slip, not a determined adversary —
    an in-process restricted ``exec`` is not a security sandbox. Defense in depth on top
    of the gate: the admitted script runs against a REASONING-ONLY roster
    (:func:`~workflows.make_reasoning_roster`), which omits the ``code_fixer`` execution
    leaf. The gate validates the authored source but does NOT constrain which
    ``agent_type`` it calls, so without this an authored script could call
    ``ctx.agent(..., agent_type="code_fixer")`` and reach the real shell. With no
    execution leaf registered, an authored script has nothing to reach — the trusted
    preset path (``run_workflow_live``) keeps the full roster.

    Args:
        submit_rejected: Select the gate-fail fixture when ``True``, else the gate-pass
            fixture. The offline host drives the pass path; a real model decides for
            itself.
        adapter: The :class:`~ui_adapter.UiAdapter` whose sinks receive the admitted
            run's progress and span events (gate-pass path only).
        emit: The host-bound UI emit used to push the ``meta_script`` verdict event
            directly (the adapter handles only the engine's progress/span vocabulary).

    Returns:
        The authored script's result on the gate-pass path, or a short rejection notice
        on the gate-fail path.
    """
    source = REJECTED_SCRIPT if submit_rejected else AUTHORED_SCRIPT
    try:
        # Validate first so the verdict reflects the gate, not a downstream runtime
        # error: compile_workflow_source runs the AST gate and returns the WorkflowFn.
        compile_workflow_source(source)
    except WorkflowScriptError as exc:
        _emit_meta_script(emit, source=source, gate=_GATE_FAILED, reason=str(exc))
        return f"The authored script was rejected by the AST gate and nothing ran.\n{exc}"

    _emit_meta_script(emit, source=source, gate=_GATE_PASSED, reason=None)
    # REASONING-ONLY roster (no code_fixer / no needs_execution leaf): the gate does not
    # constrain the authored script's agent_type, so an execution-capable roster would let
    # an authored ctx.agent(agent_type="code_fixer") reach the real shell. The sandbox
    # manager is still passed for parity but is moot — no leaf on this roster can lease it.
    result = await run_workflow_from_source(
        source,
        roster=make_reasoning_roster(),
        args={"topics": META_TOPICS},
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
        on_span_begin=adapter.on_span_begin,
        on_leaf_event=adapter.on_leaf_event,
        leaf_event_include_payloads=False,
        on_command=adapter.on_command,
        sandbox_manager=_make_sandbox_manager(),
    )
    return str(result)


@tool
async def run_hello_demo(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Run the demo workflow, streaming its progress into the UI as it goes.

    Pushes a trivial ``hello_ui`` component to confirm the Generative-UI round-trip,
    then runs ``hello_workflow`` inline, feeding the engine's progress/span hooks
    through a :class:`~ui_adapter.UiAdapter`. The adapter maps each engine event to a
    Gen-UI component (``phase_timeline`` for progress, ``fanout_graph`` /
    ``agent_span`` / ``journal_badge`` for spans), stamps a stable content-based
    ``event_id`` so re-emits dedupe, and swallows transport failures so a UI push can
    never break orchestration. The host-bound ``emit`` rebinds the captured node
    context so events stream live from inside the nested engine run.

    Returns:
        A short human-readable summary of what ran.
    """
    # Capture the host node context now so UI events emitted from inside the nested
    # engine run still target the host graph's stream and ``ui`` channel.
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)
    _emit_run_status(emit)

    # Round-trip 1: a trivial local component renders from the node context.
    emit("hello_ui", {"text": "hello from backend", "event_id": "hello-1"})

    # Round-trip 2: inline workflow run whose progress/span hooks flow through the
    # adapter (event mapping + stable-id dedupe + non-blocking emit).
    adapter = UiAdapter(emit=emit)
    result = await run_workflow(
        hello_workflow,
        roster=make_roster(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
        on_span_begin=adapter.on_span_begin,
        on_leaf_event=adapter.on_leaf_event,
        leaf_event_include_payloads=False,
    )
    return f"Demo workflow finished: {result}"


async def run_workflow_live(
    name: str, args: dict[str, Any], *, adapter: UiAdapter, lane: _ResumeLane | None = None
) -> str:
    """Resolve a named preset workflow and run it inline, streaming via ``adapter``.

    This is the engine-facing core of the :func:`run_live` host tool, kept free of
    the node-context machinery so it can be exercised directly in tests. It resolves
    ``name`` from :func:`~workflows.make_workflows`, wraps the two-argument
    ``WorkflowFn`` into the single-argument orchestrator ``run_workflow`` expects
    (binding ``args``), and runs it against the real roster with the adapter's sinks
    wired to ``on_progress`` / ``on_span`` / ``on_span_begin`` / ``on_leaf_event`` —
    the span-open edge surfaces each leaf's running chip, which the matching span-close
    edge flips in place to its completed state, and ``on_leaf_event`` folds each
    freshly-executing leaf's interior run tree onto its ``agent_span`` as a bounded,
    shape-only drill-in subtree (``leaf_event_include_payloads=False`` so no raw tool
    args or model text are surfaced). The same registry is also passed as ``workflows=``
    so a preset that nests ``ctx.workflow(...)`` resolves too.

    When a ``lane`` is supplied its persisted journal / checkpointer / ``thread_id``
    are threaded into ``run_workflow`` so a second run on the same lane replays the
    first run's leaves as journal hits (the resume / "pick it back up" story). When
    omitted, the run uses the engine's per-call in-memory defaults (no resume), which
    keeps the helper directly testable without lane wiring.

    Args:
        name: The registered preset workflow name (e.g. ``"deep_research"``).
        args: Arguments forwarded to the workflow (an empty mapping is fine; presets
            fall back to their own defaults).
        adapter: The :class:`~ui_adapter.UiAdapter` whose sinks receive the run's
            progress and span events.
        lane: Optional durable :class:`_ResumeLane` whose journal / checkpointer /
            ``thread_id`` make this run resumable on a later turn.

    Returns:
        The workflow's result, coerced to ``str`` for the tool's text reply.

    Raises:
        KeyError: If ``name`` is not a registered preset (the registry lists the
            available names).
    """
    workflows = make_workflows()
    workflow_fn = workflows.resolve(name)  # fail-loud on an unknown name

    async def _orchestrate(ctx: Ctx) -> Any:
        return await workflow_fn(ctx, args)

    # Thread the durable lane so a follow-up run replays cached leaves; absent a lane
    # the engine falls back to its per-call in-memory journal (no cross-turn resume).
    durable: dict[str, Any] = (
        {"journal": lane.journal, "checkpointer": lane.checkpointer, "thread_id": lane.thread_id}
        if lane is not None
        else {}
    )
    result = await run_workflow(
        _orchestrate,
        roster=make_roster(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
        on_span_begin=adapter.on_span_begin,
        on_leaf_event=adapter.on_leaf_event,
        leaf_event_include_payloads=False,
        on_command=adapter.on_command,
        sandbox_manager=_make_sandbox_manager(),
        workflows=workflows,
        **durable,
    )
    return str(result)


@tool
async def run_live(
    state: Annotated[dict[str, Any], InjectedState],
    workflow: str = DEFAULT_LIVE_WORKFLOW,
    workflow_args: dict[str, Any] | None = None,
) -> str:
    """Run a named preset workflow live, streaming its orchestration into the UI.

    Resolves ``workflow`` from the preset registry and runs it inline against the real
    roster. The resolvable presets are:

    * ``deep_research`` (the default) — fan out web searches across sub-questions,
      cross-check the findings, and synthesize a cited answer.
    * ``capstone`` — a heavier multi-phase research-and-reconcile scenario.
    * ``fix_loop`` — an executable-verification loop for a "make it pass / get the tests
      green" request: it edits code in a worktree leaf, then actually builds and runs the
      tests, branches on the REAL exit code, and retries until green. Each real command
      streams an ``execution_command`` event the UI renders as a TerminalCard that flips
      pass/fail in place — pick this when the user wants the code proven to build and the
      tests proven to pass, not just reviewed as looking correct.

    Each engine progress/span event flows through a :class:`~ui_adapter.UiAdapter`, which
    maps it to a Gen-UI component (``phase_timeline`` for phases/logs, ``fanout_graph``
    for parallel / pipeline fan-out, ``agent_span`` / ``journal_badge`` for leaves,
    ``execution_command`` / TerminalCard for ``fix_loop``'s real build/test commands)
    with a stable content ``event_id`` for dedupe, and pushes it live from inside the
    same node context — so the chat shows the control-flow inversion as it happens.

    The run is threaded through a durable :class:`_ResumeLane` keyed on this host
    thread and the workflow name, so a second turn on the same thread (e.g. "pick it
    back up") reuses the first run's journal: every replayed leaf comes back
    ``cached=True`` and surfaces a ``journal_badge``, making the zero-cost resume
    visible rather than re-running the work.

    Args:
        state: Injected graph state (used to anchor UI events to the host turn).
        workflow: The preset workflow to run — ``deep_research`` (default), ``capstone``,
            or ``fix_loop`` (the executable-verification build/test loop).
        workflow_args: Optional workflow arguments (e.g. ``{"question": "..."}``).
            When omitted the preset uses its own defaults. Named ``workflow_args``
            rather than ``args`` on purpose: LangChain's tool-schema generation mangles
            a parameter literally named ``args`` into ``v__args`` (typed as an array),
            and then passes that mangled keyword back on invocation, raising
            ``unexpected keyword argument 'v__args'``. A non-reserved name avoids it.

    Returns:
        A short human-readable summary including the workflow's result.
    """
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)
    _emit_run_status(emit)
    adapter = UiAdapter(emit=emit)
    lane = _resume_lane(workflow)
    result = await run_workflow_live(workflow, workflow_args or {}, adapter=adapter, lane=lane)
    return f"Workflow {workflow!r} finished: {result}"


@tool
async def run_meta_script(
    state: Annotated[dict[str, Any], InjectedState],
    submit_rejected: bool = False,
) -> str:
    """Author an orchestration script on the spot, gate it, and run it if admitted.

    Demonstrates the engine's meta layer: rather than launching a pre-registered
    preset, the host authors an ``async def orchestrate(ctx, args)`` and submits the
    *source* across the AST security gate before it ever runs. A ``meta_script`` Gen-UI
    event surfaces the script and the gate verdict so the chat shows exactly what was
    authored and whether it was admitted.

    On the gate-pass path the admitted script runs live — its phases and parallel
    fan-out stream into the panel through the same :class:`~ui_adapter.UiAdapter` the
    preset tools use — and its result is returned. On the gate-fail path the gate
    rejects the script, the ``meta_script`` event reports the line-numbered violation,
    and nothing executes.

    Security boundary: the gate stops an accidental slip, not a determined adversary —
    an in-process restricted ``exec`` is not a security sandbox.

    Args:
        state: Injected graph state (used to anchor UI events to the host turn).
        submit_rejected: When ``True`` submit the import-bearing fixture to show the
            gate reject it and run nothing; when ``False`` (the default) submit the
            clean authored fixture, which the gate admits and runs.

    Returns:
        The authored script's result on the gate-pass path, or a short rejection notice
        on the gate-fail path.
    """
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)
    _emit_run_status(emit)
    adapter = UiAdapter(emit=emit)
    return await run_meta_script_live(submit_rejected=submit_rejected, adapter=adapter, emit=emit)


# A single process-wide background-run manager, shared by every ``run_background`` call
# (and by any middleware that drives the same run/status/notify loop) — the engine keys
# its run slots on a composite ``(thread_id, run_id)``, so one manager safely serves all
# host threads. Held at module scope so a launched run survives across host turns within
# a process; a fresh process starts empty (the honest in-memory-store bound, matching the
# resume lanes). HONEST LIMITATION: ``BgRunManager.start`` runs the coroutine in a
# DETACHED asyncio task that does NOT carry the host node context, and
# ``push_ui_message`` requires that context — so a background run CANNOT push live
# progress/fan-out to the host ``ui`` channel. The background scenario deliberately
# surfaces lifecycle status plus the final result only; it never wires the run's
# progress/span sinks to a host emit (that would silently no-op from the detached task).
_BG_MANAGER = BgRunManager()

# The preset a background run launches by default — the same fan-out-heavy deep-research
# scenario the inline path uses, so the background story runs real orchestration.
DEFAULT_BACKGROUND_WORKFLOW = "deep_research"


def launch_background_run(
    manager: BgRunManager, *, thread_id: str, workflow: str = DEFAULT_BACKGROUND_WORKFLOW
) -> str:
    """Launch a preset workflow in the background and return its run id immediately.

    Resolves ``workflow`` from the preset registry and submits a coroutine that runs it
    to completion (coercing the result to ``str``, which :meth:`BgRunManager.start`
    requires) onto ``manager`` as a DETACHED asyncio task. The call returns at once with
    the run's id so the host turn is not blocked on the work — the caller then polls
    :meth:`BgRunManager.poll` for lifecycle status and :meth:`BgRunManager.get_result`
    for the settled result.

    The detached task intentionally receives NO progress/span sinks: it does not carry
    the host node context, so a ``push_ui_message`` from inside it would target the wrong
    (or no) stream. Live streaming is the inline tools' job; the background path surfaces
    lifecycle status and the final result.

    Args:
        manager: The shared :class:`~langchain_dynamic_workflow.BgRunManager` to launch on.
        thread_id: The host thread id; the manager keys run slots on ``(thread_id,
            run_id)`` so concurrent host threads stay isolated.
        workflow: The preset workflow name to run; defaults to ``deep_research``.

    Returns:
        The launched run's id, usable with ``manager.poll`` / ``manager.get_result``.
    """
    workflows = make_workflows()
    workflow_fn = workflows.resolve(workflow)  # fail-loud on an unknown name

    async def _run() -> str:
        async def _orchestrate(ctx: Ctx) -> Any:
            return await workflow_fn(ctx, {})

        result = await run_workflow(_orchestrate, roster=make_roster(), workflows=workflows)
        return str(result)

    slot = manager.start(_run(), thread_id=thread_id)
    return slot.run_id


async def run_background_live(
    manager: BgRunManager, *, thread_id: str, workflow: str = DEFAULT_BACKGROUND_WORKFLOW
) -> str:
    """Launch a background run, await its settlement, and summarize its lifecycle + result.

    The engine-facing core of the :func:`run_background` host tool, kept free of the
    node-context machinery so it can be exercised directly in tests. It launches the run
    via :func:`launch_background_run` (returning immediately with a run id), then — for
    the demo's bounded offline two-turn shape — awaits settlement and folds the lifecycle
    and final result into one summary string. A real-model host would instead return the
    run id right away and follow up across turns via status polling and completion
    notifications; that fuller loop is out of scope for the scripted offline host.

    Args:
        manager: The shared :class:`~langchain_dynamic_workflow.BgRunManager`.
        thread_id: The host thread id keying the run slot.
        workflow: The preset workflow name to run; defaults to ``deep_research``.

    Returns:
        A human-readable summary naming the run id, its settled status, and the result
        (or the failure detail when the run did not complete).
    """
    run_id = launch_background_run(manager, thread_id=thread_id, workflow=workflow)
    await manager.wait(run_id, thread_id=thread_id)
    status = manager.poll(run_id, thread_id=thread_id)
    if status is not BgStatus.DONE:
        result = manager.get_result(run_id, thread_id=thread_id)
        return f"Background run {run_id} ended with status {status.value} ({result.detail})."
    result = manager.get_result(run_id, thread_id=thread_id)
    payload = result.value if result.value is not None else result.summary
    return f"Background run {run_id} status=done. Result:\n{payload}"


@tool
async def run_background(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Launch a heavy workflow in the background and report its lifecycle and result.

    Hands a multi-step job off to run on its own: it launches the preset on the shared
    background-run manager (returning the host turn immediately rather than blocking on
    the work), then reports the run's lifecycle status and, once it settles, the final
    result.

    Honest limitation: a background run executes in a DETACHED task that does not carry
    the host node context, so it CANNOT push live progress into the chat panel —
    ``push_ui_message`` needs that context. The background scenario therefore surfaces
    lifecycle status and the final result, not the live phase/fan-out stream the inline
    tools show. (The scripted offline host drives the bounded two-turn shape — launch,
    settle, summarize — within this one tool; a real model follows up across turns via
    status and completion notifications.)

    Args:
        state: Injected graph state (used to anchor UI events to the host turn).

    Returns:
        A short summary naming the run id, its settled status, and the result.
    """
    anchor = _anchor_message(state.get("messages", []))
    emit = make_host_ui_emit(anchor=anchor)
    _emit_run_status(emit)
    return await run_background_live(_BG_MANAGER, thread_id=_host_thread_id())


def _build_host_graph(*, checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Build the host deepagent graph, optionally with an explicit checkpointer.

    Resolves the host model once at build time. The provider is locked to OpenRouter and
    the model is fixed in code, so this is a per-call lazy model: it decides online vs.
    offline PER TURN inside the node from the in-force OpenRouter key (a per-session
    ``configurable.openrouter_api_key`` or the backend ``.env`` ``OPENROUTER_API_KEY``),
    driving the real fixed host model when keyed and a scripted offline host otherwise —
    so the same built graph serves a key-free boot and a per-session-keyed session. It
    also extends the deepagent state with a ``ui`` channel and registers the host tools
    so the host can drive the Generative-UI round-trip (``run_hello_demo``), inline preset
    runs (``run_live``), the meta layer (``run_meta_script``), and a background run
    (``run_background``).

    Kept OFF the ``langgraph dev`` factory path on purpose: the platform classifies a
    graph factory by its parameter signature and injects its own value for a recognized
    name like ``checkpointer`` (a config dict that fails ``create_deep_agent``'s compile).
    The registered factory :func:`make_host_graph` therefore takes no parameters; this
    builder carries the injectable one for in-process tests only.

    Args:
        checkpointer: Optional LangGraph checkpointer. ``None`` (the deployment path) lets
            ``langgraph dev`` attach its own persistence; an ``InMemorySaver`` makes the
            host thread durable in-process so a follow-up turn on the same ``thread_id``
            resumes the accumulated state, as tests exercise.

    Returns:
        The compiled deepagent host graph (a runnable LangGraph graph).
    """
    extra: dict[str, Any] = {"checkpointer": checkpointer} if checkpointer is not None else {}
    return create_deep_agent(
        model=resolve_host_model(),
        system_prompt=HOST_INSTRUCTIONS,
        tools=[run_hello_demo, run_live, run_meta_script, run_background],
        state_schema=HostState,
        middleware=cache_middleware(),
        **extra,
    )


def make_host_graph() -> Any:
    """Build the host deepagent graph served by ``langgraph dev`` (no injectable params).

    Registered under the ``host`` graph id in ``langgraph.json``. Takes NO parameters on
    purpose: ``langgraph dev`` classifies a graph factory by its signature and injects
    values for recognized parameter names, so any parameter here (notably ``checkpointer``)
    would receive an injected config dict and break compilation. The deployment lets the
    platform attach its own persistence; in-process tests call :func:`_build_host_graph`
    directly to supply a checkpointer.

    Returns:
        The compiled deepagent host graph (a runnable LangGraph graph).
    """
    return _build_host_graph()

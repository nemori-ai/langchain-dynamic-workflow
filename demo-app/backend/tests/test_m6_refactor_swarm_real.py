"""Real-model E2E acceptance for the M6 real-git refactor-swarm path (gated).

Offline tests prove the ``refactor_swarm`` wiring with deterministic fake fixers (which
still edit a real ``git worktree`` on disk, so the isolation, the authoritative git-diff
collect, and the real scratch-repo merge conflict are genuinely exercised) — but no real
model authors the fixes. This proves the REAL thing, and it proves it through the PRODUCT
SURFACE. The acceptance drives the REAL host graph end to end from the "Refactor swarm"
scenario MESSAGE (the exact copy the preset button sends), so the model — not the test —
selects the tool and the preset. A clean pass that bypassed host tool-selection (calling
``refactor_swarm`` directly) would skip the very surface the demo ships, the anti-pattern
the project's real-E2E discipline warns against; this routes through it.

Driving ``make_host_graph().astream(...)`` with the scenario message exercises the same
chain ``langgraph dev`` runs for the button press: the host model reads the request, routes
to the ``run_live`` tool with ``workflow="refactor_swarm"``, which (via
:func:`~host_graph.run_refactor_swarm_live`) constructs a real
:class:`~langchain_dynamic_workflow.GitWorktreeProvider` over a seeded temp repo, fans out
real ``git_fixer`` leaves — each in its OWN ``git worktree`` on its own ``leaf/<id>``
branch — has a two-vote read-only judge review each patch, then folds the approved patches
into one tree with a real three-way ``git merge``. Two fixers rewrite the SAME ``/calc.py``
line two DIFFERENT ways in isolated worktrees, so the fold hits a REAL merge conflict that a
real ``conflict_resolver`` leaf flattens. After the run the HOST opens the pull request
(host finalization, R1) and emits a ``pull_request`` Gen-UI event on the host ``ui`` channel.

It asserts the headline properties a fallback or an offline-fake-content run could not
produce through the product surface:

* the host routed to ``refactor_swarm`` (a ``run_live`` tool call naming the preset) — the
  real model genuinely selected the fix-swarm path from a natural request;
* a ``pull_request`` Gen-UI event landed on the host ``ui`` channel carrying a real PR ref
  (number / branch / url) — the host finalized the PR after the run, the headline outcome;
* the ``run_live`` tool reply for ``refactor_swarm`` carries the host's finalization summary
  containing ``"resolved a merge conflict"`` — emitted ONLY when
  ``RefactorResult.conflict_resolved`` is True (see
  :func:`~host_graph.run_refactor_swarm_live`), i.e. two real fixers' competing edits to one
  line forced a genuine three-way ``git merge`` conflict that a real resolver leaf resolved.
  This is THE property a clean-merge-only or offline run could not prove through the surface;
  and
* the same reply reports that patches were integrated (the swarm produced an integrated
  change, not an empty fold) — a supporting, robust signal.

Gating + fail-loud honesty. The module is skipped unless ``LDW_DEMO_REAL_MODEL`` is set, so
CI stays offline. When the gate IS set, an OpenRouter key MUST be in force (the backend
``.env`` ``OPENROUTER_API_KEY``, loaded via :func:`_load_backend_env`) — otherwise the host
graph would build the offline scripted host (which still routes the message but runs the
FAKE roster) and the assertions could pass on a FAKE-content run, defeating the acceptance.
So with the gate on and no key, the setup FAILS loudly rather than skipping or passing on a
fallback: a pass here means a real model genuinely authored the fixes through the real host
surface.

``git`` must be on ``PATH`` for the real ``git worktree`` / ``git merge`` to run; its
absence FAILS loudly in setup (the swarm is built on real git, so a missing toolchain is a
real failure of the acceptance environment, not a skip).

LangSmith tracing is kept ON (whatever the ``.env`` activates) so this real run is captured
for usage/billing visibility — do not disable it here.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_m6_refactor_swarm_real.py -q -s
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import pytest

# The opt-in gate. The rest of the repo uses this same env var as the real-model switch.
_REAL_MODEL_GATE = "LDW_DEMO_REAL_MODEL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_REAL_MODEL_GATE),
    reason=f"{_REAL_MODEL_GATE} not set; real-model refactor-swarm E2E is opt-in",
)

# The exact "Refactor swarm" scenario message from scenarios.json — the copy the preset
# button sends. Driving the host graph from this proves the model routes the natural request
# to the refactor_swarm preset (no preset name in the text), not a hand-picked call.
_REFACTOR_SWARM_MESSAGE = (
    "There are a few separate bugs in this little module I'd like fixed all at once. "
    "Please have several helpers each take a fix in parallel, review the patches, merge "
    "them together into one change — sorting out any conflicts — and open a pull request "
    "with the result."
)


def _load_backend_env() -> None:
    """Best-effort load of the backend ``.env`` so the OpenRouter + tracing vars apply.

    Mirrors the example harness's ``load_demo_env``: populates ``os.environ`` from a local
    ``.env`` when ``python-dotenv`` is installed, and is a silent no-op when it is not, so
    the offline path keeps running with no extra dependency. Loading the ``.env`` also
    activates LangSmith tracing (its standard vars), which we deliberately keep ON.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(find_dotenv(usecwd=True))


@pytest.fixture
def _real_refactor_swarm_setup() -> None:
    """Load the backend ``.env`` and fail loud unless a REAL refactor-swarm run is possible.

    Loads the local ``.env`` (so the OpenRouter key and the LangSmith tracing vars are in
    force) and then enforces the two preconditions a real run needs, failing loudly rather
    than skipping or letting a fake-content run masquerade as acceptance:

    * an OpenRouter key MUST be in force — otherwise the host graph builds the offline
      scripted host whose ``refactor_swarm`` run uses the deterministic fake leaves (the
      fixers' CONTENT is canned, no real model authors them), so the assertions could pass
      on a FAKE-content run; and
    * ``git`` MUST be on ``PATH`` — the swarm is built on real ``git worktree`` /
      ``git merge``, so a missing toolchain cannot run the real swarm.

    Tracing is deliberately left ON (whatever the ``.env`` activates) so the real run is
    captured in LangSmith for usage/billing visibility.
    """
    _load_backend_env()
    # Keep LangSmith tracing ON (do NOT disable it): the real run must be billable.

    from _models import is_offline

    if is_offline():
        pytest.fail(
            f"{_REAL_MODEL_GATE} is set but no OpenRouter key is in force "
            "(set OPENROUTER_API_KEY in the backend .env). The M6 refactor-swarm HEADLINE "
            "path must run REAL models against a REAL git worktree provider THROUGH the host "
            "surface — a fallback to the offline scripted host runs fake-content leaves, so "
            "it cannot be accepted as proof that real models authored the fixes and resolved "
            "a real merge conflict."
        )

    if shutil.which("git") is None:
        pytest.fail(
            "git is not on PATH but the M6 refactor-swarm acceptance runs a real "
            "`git worktree` / `git merge`; install git to run this gate."
        )


def _pull_request_props(ui_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull every ``pull_request`` event's props off the host ``ui`` channel.

    The host graph finalizes the PR after the run and maps it through the ``UiAdapter``
    (:meth:`~ui_adapter.UiAdapter.emit_pull_request`) into a ``pull_request`` Gen-UI message
    pushed onto the ``ui`` channel — the PullRequestCard payload. This collects their props
    (``number`` / ``branch`` / ``url`` / ``integration_branch`` / ...) so the assertions read
    the SAME data the frontend renders.

    Args:
        ui_messages: The graph state's ``ui`` channel entries (``{"name", "props", ...}``).

    Returns:
        One props mapping per ``pull_request`` event, in channel order.
    """
    return [(msg.get("props") or {}) for msg in ui_messages if msg.get("name") == "pull_request"]


def _iter_pull_request_props(chunk: Any) -> list[dict[str, Any]]:
    """Recursively pull every ``pull_request`` event's props out of a stream chunk.

    ``push_ui_message`` writes each Gen-UI message to the custom stream AS IT IS EMITTED, so
    the PR card delivery is observable here too. The custom chunk is the ``UIMessage`` dict
    (``{"type": "ui", "name", "props", ...}``); this walks the chunk tolerantly so a minor
    shape variation does not silently drop the event.

    Args:
        chunk: One ``stream_mode="custom"`` chunk from the host graph's ``astream``.

    Returns:
        The props of every ``pull_request`` UIMessage nested anywhere in the chunk.
    """
    found: list[dict[str, Any]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            props = obj.get("props")
            if obj.get("name") == "pull_request" and isinstance(props, dict):
                found.append(props)
                return
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                _walk(value)

    _walk(chunk)
    return found


async def test_real_model_refactor_swarm_through_host_graph_resolves_conflict_and_opens_pr(
    _real_refactor_swarm_setup: None,
) -> None:
    """The "Refactor swarm" message drives the real host graph through a real conflict + PR.

    Streams ``make_host_graph().astream(...)`` with the exact scenario message — the same
    chain a button press triggers — so the REAL host model selects the ``run_live`` tool
    with ``workflow="refactor_swarm"`` and the preset runs inline against the real roster +
    a real :class:`~langchain_dynamic_workflow.GitWorktreeProvider`. Real ``git_fixer``
    leaves edit isolated worktrees (two rewriting the SAME ``/calc.py`` line two ways), a
    two-vote judge reviews each patch, the integrate fold hits a REAL ``git merge`` conflict
    that a real ``conflict_resolver`` leaf flattens, and the HOST opens the pull request
    after the run, emitting a ``pull_request`` event on the host ``ui`` channel.

    Asserts the host-driven headline properties none of which a fallback or an offline
    fake-content run could produce through the surface: the model routed to
    ``refactor_swarm``; a ``pull_request`` card carrying a real PR ref landed on the ui
    channel; the tool reply reports a REAL merge conflict was resolved (the conflict-path
    signal, emitted only when ``RefactorResult.conflict_resolved`` is True); and the reply
    reports integrated patches.
    """
    from host_graph import make_host_graph
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    graph = make_host_graph()
    config = {"configurable": {"thread_id": "t-m6-refactor-swarm-real-host"}}
    inp = {"messages": [HumanMessage(content=_REFACTOR_SWARM_MESSAGE)]}

    # STREAM the run (don't ainvoke) so a transient card delivery is observable too: we
    # collect every pull_request event DELIVERED onto the host ui stream across the run, plus
    # the final values snapshot for the message/tool assertions. (Single fresh turn, fresh
    # thread — no resume replay, so the stream carries no stale-state confounder.)
    streamed_pr_props: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    async for mode, chunk in graph.astream(inp, config=config, stream_mode=["custom", "values"]):
        if mode == "custom":
            streamed_pr_props.extend(_iter_pull_request_props(chunk))
        elif mode == "values":
            final_state = chunk

    messages = final_state["messages"]

    # Headline 0: the REAL host model selected the refactor-swarm preset. The message names
    # no preset, so a run_live tool call carrying workflow="refactor_swarm" proves the model
    # routed the natural fix-and-merge-into-a-PR intent through the product surface.
    run_live_calls = [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "run_live"
    ]
    assert run_live_calls, (
        "the host model never called run_live: it must route the 'refactor swarm' request "
        f"to the live preset tool. Messages: {[type(m).__name__ for m in messages]}"
    )
    assert any(call["args"].get("workflow") == "refactor_swarm" for call in run_live_calls), (
        "the host model must route to the refactor_swarm preset (workflow='refactor_swarm'); "
        f"run_live calls saw: {[call['args'] for call in run_live_calls]}"
    )

    # Headline 1 (PR finalized): a pull_request Gen-UI event landed on the host ui channel
    # carrying a real PR ref. The host opens the PR AFTER the run (host finalization, R1) and
    # emits this card; an offline fake-content run still reaches it, but a run that never
    # completed (or never produced a RefactorResult) would emit none. We read the STREAMED
    # deliveries first, falling back to the final ui channel.
    pr_events = streamed_pr_props or _pull_request_props(final_state.get("ui", []))
    assert pr_events, (
        "no pull_request event on the host ui stream: the host must finalize the PR after "
        "the refactor_swarm run and push a PullRequestCard onto the ui channel"
    )
    pr_props = pr_events[-1]
    assert pr_props.get("number") is not None, f"the PR card must carry a number; got {pr_props!r}"
    assert pr_props.get("branch"), f"the PR card must carry a source branch; got {pr_props!r}"
    assert pr_props.get("url"), f"the PR card must carry a url; got {pr_props!r}"

    # The run_live tool reply for refactor_swarm carries the host's finalization summary
    # (run_refactor_swarm_live's return, wrapped by run_live as "Workflow 'refactor_swarm'
    # finished: <summary>"). The summary text is the source of the conflict + integration
    # signals asserted below.
    tool_replies = [
        str(message.content) for message in messages if isinstance(message, ToolMessage)
    ]
    refactor_reply = next((reply for reply in tool_replies if "refactor_swarm" in reply), "")
    assert refactor_reply, (
        f"no run_live tool reply mentioning refactor_swarm; tool replies: {tool_replies!r}"
    )

    # Headline 2 (THE conflict path, taken AND resolved by the real models): the host
    # finalization summary contains "resolved a merge conflict" ONLY when
    # RefactorResult.conflict_resolved is True (host_graph.run_refactor_swarm_live: the
    # conflict branch picks "resolved a merge conflict", the clean branch picks "no
    # conflicts"). Two real fixers rewrote the SAME /calc.py line two DIFFERENT ways in
    # isolated worktrees -> a REAL three-way git merge conflict -> a real conflict_resolver
    # leaf flattened it. A clean-merge-only run (or an offline run) could not surface this.
    assert "resolved a merge conflict" in refactor_reply, (
        "the refactor_swarm reply must report a REAL merge conflict was resolved "
        "(RefactorResult.conflict_resolved=True): two fixers' competing edits to one line "
        "must force a genuine git merge conflict a resolver leaf resolves. "
        f"Reply: {refactor_reply!r}"
    )
    assert "no conflicts" not in refactor_reply, (
        "the refactor_swarm fold must NOT report a clean (no-conflict) merge: the fixture "
        f"forces a real conflict. Reply: {refactor_reply!r}"
    )

    # Headline 3 (supporting, robust): the swarm produced an integrated change. The summary
    # reads "Refactor swarm integrated <N> patches (...)"; assert it reports integration so a
    # pass means real patches were folded, not an empty result.
    assert "integrated" in refactor_reply.lower(), (
        f"the refactor_swarm reply must report integrated patches; got {refactor_reply!r}"
    )

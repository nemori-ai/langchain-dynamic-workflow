"""Real-model E2E acceptance for the M5 in-loop executable-verification path (gated).

Offline tests prove the ``fix_loop`` wiring with a deterministic fake fixer that runs
no subprocess; this proves the REAL thing — and it proves it through the PRODUCT SURFACE.
The acceptance drives the REAL host graph end-to-end from the "Make it pass" scenario
MESSAGE (the exact copy the preset button sends), so the model — not the test — selects
the tool and the preset. A clean pass that bypassed host tool-selection (calling
``fix_loop`` directly) would skip the very surface the demo ships, the anti-pattern the
project's real-E2E discipline warns against; this routes through it.

Driving ``make_host_graph().ainvoke(...)`` with the scenario message exercises the same
chain ``langgraph dev`` runs for the button press: the host model reads the request,
routes to the ``run_live`` tool with ``workflow="fix_loop"``, which runs the preset inline
against the real roster + a real :class:`SandboxManager`, so a real ``code_fixer`` leaf
shells out via its ``execute`` tool, runs ``bun test`` against a seeded failing TypeScript
module for a TRUE exit code, fixes the source, and re-runs until green. Every real command
the engine's ``on_command`` sink delivers is mapped by the :class:`~ui_adapter.UiAdapter`
into an ``execution_command`` Gen-UI event on the host ``ui`` channel — the TerminalCard
payloads — so the test reads them off ``out["ui"]``, the same place the frontend reads.

It asserts the headline properties a fallback or an offline fake could not produce:

* the host routed to ``fix_loop`` (a ``run_live`` tool call naming the preset, then a
  finished tool result) — the model genuinely selected the executable-verification path;
* the engine's ``on_command`` sink fired a real ``running`` (begin) AND a terminal
  (end) ``execution_command`` edge for at least one command (terminal-card lifecycle
  from inside the real sandbox, streamed onto the host ``ui`` channel);
* at least one command ended NON-zero (a red ``bun test``) and at least one command
  ended zero (a green ``bun test``) — the real red->green transition the scenario forces;
  and
* the loop returned the GREEN result (the ``run_live`` tool reply carries the
  green-attempt summary, which ``fix_loop`` returns ONLY when a ``FixResult.exit_code``
  was zero, never on the "still red" fallback) — the final result reports tests passed.

Gating + fail-loud honesty. The module is skipped unless ``LDW_DEMO_REAL_MODEL`` is set,
so CI stays offline. When the gate IS set, an OpenRouter key MUST be in force (the
backend ``.env`` ``OPENROUTER_API_KEY``, loaded via :func:`_load_backend_env`) — otherwise
the host graph would build the offline scripted host (which routes the message but runs the
fake roster: no subprocess, no ``on_command``) and the assertions would pass on a FAKE run,
defeating the acceptance. So with the gate on and no key, the setup FAILS loudly rather
than skipping or passing on a fallback: a pass here means a real model genuinely ran real
commands through the real host surface.

``bun`` must be on ``PATH`` for the real ``bun test`` to run; its absence FAILS loudly in
setup (the seeded module is a Bun test, so a missing toolchain is a real failure of the
acceptance environment, not a skip).

LangSmith tracing is kept ON (whatever the ``.env`` activates) so this real run is captured
for usage/billing visibility — do not disable it here.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_m5_fix_loop_real.py -q -s
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
    reason=f"{_REAL_MODEL_GATE} not set; real-model fix-loop E2E is opt-in",
)

# The exact "Make it pass" scenario message from scenarios.json — the copy the preset
# button sends. Driving the host graph from this proves the model routes the natural
# request to the fix_loop preset (no preset name in the text), not a hand-picked call.
_MAKE_IT_PASS_MESSAGE = (
    "I've got a small TypeScript module with a couple of failing unit tests. Please "
    "actually fix the code and keep checking it against the tests until they genuinely "
    "pass — don't just tell me it looks right, prove it builds and the tests go green."
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
def _real_fix_loop_setup() -> None:
    """Load the backend ``.env`` and fail loud unless a REAL fix-loop run is possible.

    Loads the local ``.env`` (so the OpenRouter key and the LangSmith tracing vars are
    in force) and then enforces the two preconditions a real run needs, failing loudly
    rather than skipping or letting a fake run masquerade as acceptance:

    * an OpenRouter key MUST be in force — otherwise the host graph builds the offline
      scripted host whose ``fix_loop`` run uses the deterministic fake ``code_fixer`` (no
      subprocess, no ``on_command``) and the assertions would pass on a FAKE run; and
    * ``bun`` MUST be on ``PATH`` — the seeded module is a Bun test, so a missing
      toolchain cannot run the real ``bun test``.

    Tracing is deliberately left ON (whatever the ``.env`` activates) so the real run is
    captured in LangSmith for usage/billing visibility.
    """
    _load_backend_env()
    # Keep LangSmith tracing ON (do NOT disable it): the real run must be billable.

    from _models import is_offline

    if is_offline():
        pytest.fail(
            f"{_REAL_MODEL_GATE} is set but no OpenRouter key is in force "
            "(set OPENROUTER_API_KEY in the backend .env). The M5 fix-loop HEADLINE path "
            "must run a REAL model against a REAL sandbox THROUGH the host surface — a "
            "fallback to the offline scripted host runs the fake roster (no subprocess, no "
            "on_command), so it cannot be accepted as proof of in-loop executable "
            "verification."
        )

    if shutil.which("bun") is None:
        pytest.fail(
            "bun is not on PATH but the M5 fix-loop acceptance runs a real `bun test` "
            "against a seeded Bun module; install bun (https://bun.sh) to run this gate."
        )


def _execution_command_props(ui_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull every ``execution_command`` event's props off the host ``ui`` channel.

    The host graph maps each engine ``on_command`` edge through the ``UiAdapter`` into an
    ``execution_command`` Gen-UI message pushed onto the ``ui`` channel — the TerminalCard
    payloads. This collects their props (``status`` / ``exit_code`` / ``command`` /
    ``leaf_span_id`` / ...) so the assertions read the SAME data the frontend renders.

    Args:
        ui_messages: The graph state's ``ui`` channel entries (``{"name", "props", ...}``).

    Returns:
        One props mapping per ``execution_command`` event, in channel order.
    """
    return [
        (msg.get("props") or {}) for msg in ui_messages if msg.get("name") == "execution_command"
    ]


async def test_real_model_fix_loop_through_host_graph_runs_bun_until_green(
    _real_fix_loop_setup: None,
) -> None:
    """The "Make it pass" message drives the real host graph red->green through fix_loop.

    Invokes ``make_host_graph().ainvoke(...)`` with the exact scenario message — the same
    chain a button press triggers — so the REAL host model selects the ``run_live`` tool
    with ``workflow="fix_loop"`` and the preset runs inline against the real roster + a
    real :class:`SandboxManager`. The real ``code_fixer`` leaf shells out, sees a red
    ``bun test``, fixes ``src/sum.ts``, and re-runs until green; every real command lands
    on the host ``ui`` channel as an ``execution_command`` (TerminalCard) event.

    Asserts the host-driven headline properties none of which the offline fake (no
    subprocess, no command events) — or a direct ``fix_loop`` call that skipped the host
    surface — could produce: the model routed to ``fix_loop``; a begin (running) AND a
    terminal command edge fired; at least one command exited non-zero (red) and at least
    one exited zero (green); and the tool reply reports the green-attempt result (never the
    honest "still red" fallback).
    """
    from host_graph import make_host_graph
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    graph = make_host_graph()
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content=_MAKE_IT_PASS_MESSAGE)]},
        config={"configurable": {"thread_id": "t-m5-fix-loop-real-host"}},
    )

    messages = out["messages"]

    # Headline 0a: the REAL host model selected the executable-verification preset. The
    # message names no preset, so a run_live tool call carrying workflow="fix_loop" proves
    # the model routed the "make it pass" intent through the product surface, not the test.
    run_live_calls = [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "run_live"
    ]
    assert run_live_calls, (
        "the host model never called run_live: it must route the 'make it pass' request "
        f"to the live preset tool. Messages: {[type(m).__name__ for m in messages]}"
    )
    assert any(call["args"].get("workflow") == "fix_loop" for call in run_live_calls), (
        "the host model must route to the fix_loop preset (workflow='fix_loop'); "
        f"run_live calls saw: {[call['args'] for call in run_live_calls]}"
    )

    # Headline 0b: the loop returned the GREEN result through the tool reply. fix_loop
    # returns "green on attempt ..." ONLY when a FixResult.exit_code was zero; the all-red
    # budget-exhausted path returns "still red ...". The run_live tool wraps the workflow
    # result in its "Workflow 'fix_loop' finished: <result>" reply, so a green substring
    # there is the proof the final result reports tests passed -- a real exit-code-gated
    # pass surfaced to the user, not leaf prose.
    tool_replies = [
        str(message.content) for message in messages if isinstance(message, ToolMessage)
    ]
    fix_loop_reply = next((reply for reply in tool_replies if "fix_loop" in reply), "")
    assert fix_loop_reply, (
        f"no run_live tool reply mentioning fix_loop; tool replies: {tool_replies!r}"
    )
    assert "still red" not in fix_loop_reply.lower(), (
        f"the loop exhausted its budget without going green (tests never passed): "
        f"{fix_loop_reply!r}"
    )
    assert "green on attempt" in fix_loop_reply.lower(), (
        f"the loop must return the green-attempt result (exit_code 0); got {fix_loop_reply!r}"
    )

    # The engine's on_command sink fired through the host UI surface: TerminalCard payloads
    # land on the ui channel as execution_command events. An offline fake fixer runs no
    # subprocess and would emit ZERO of these.
    command_events = _execution_command_props(out.get("ui", []))
    assert command_events, (
        "no execution_command events on the host ui channel: a real execution leaf must "
        "shell out via its execute tool (an offline fake fixer runs none, and a direct "
        "fix_loop call that bypassed the host surface would never reach this channel)"
    )

    # Headline 1: a real begin (running) AND a terminal edge fired -- the card lifecycle
    # the TerminalCard renders, sourced from inside the real LocalSubprocessSandbox.
    statuses = {props.get("status") for props in command_events}
    assert "running" in statuses, (
        f"on_command must deliver a begin (running) edge; statuses seen: {statuses}"
    )
    assert {"passed", "failed"} & statuses, (
        f"on_command must deliver a terminal (passed/failed) edge; statuses seen: {statuses}"
    )
    assert all(props.get("leaf_span_id") for props in command_events), (
        "every execution_command event must correlate to its owning leaf via leaf_span_id"
    )

    # Headline 2: a REAL red->green transition in the exit codes carried on the terminal
    # edges. The seeded sum() returns 0, so the first `bun test` exits non-zero (red); the
    # real model fixes src/sum.ts and a later `bun test` exits zero (green). Only a real
    # subprocess can produce both; the fold-into-result fallback carries no exit code.
    terminal_exit_codes = [
        props["exit_code"]
        for props in command_events
        if props.get("status") in ("passed", "failed") and props.get("exit_code") is not None
    ]
    assert terminal_exit_codes, "every terminal execution_command must carry a real exit code"
    assert any(code != 0 for code in terminal_exit_codes), (
        f"expected at least one RED (non-zero) command exit; exit codes seen: {terminal_exit_codes}"
    )
    assert any(code == 0 for code in terminal_exit_codes), (
        f"expected at least one GREEN (zero) command exit after the fix; exit codes seen: "
        f"{terminal_exit_codes}"
    )

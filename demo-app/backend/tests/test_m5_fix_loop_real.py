"""Real-model E2E acceptance for the M5 in-loop executable-verification path (gated).

Offline tests prove the ``fix_loop`` wiring with a deterministic fake fixer that runs
no subprocess; this proves the REAL thing. It runs the ``fix_loop`` preset through the
SAME engine-facing path the host tool uses (``run_workflow`` + the real roster + a real
:class:`SandboxManager` whose factory produces ``LocalSubprocessSandbox`` backends), so
a real model leaf actually shells out via its ``execute`` tool, runs ``bun test`` against
a seeded failing TypeScript module for a TRUE exit code, fixes the source, and re-runs
until green. It asserts the headline properties a fallback or an offline fake could not
produce:

* the engine's ``on_command`` sink fired a real ``"start"`` AND ``"end"`` edge for at
  least one command (terminal-card lifecycle from inside the real sandbox);
* at least one command ended NON-zero (a red ``bun test``) and at least one command
  ended zero (a green ``bun test``) — the real red->green transition the scenario is
  built to force; and
* the loop returned the GREEN result (``fix_loop`` returns the green-attempt summary
  ONLY when a ``FixResult.tests_passed`` was true, never on the "still red" fallback) —
  i.e. the final result reports tests passed.

Gating + fail-loud honesty. The module is skipped unless ``LDW_DEMO_REAL_MODEL`` is set,
so CI stays offline. When the gate IS set, an OpenRouter key MUST be in force (the
backend ``.env`` ``OPENROUTER_API_KEY``, loaded via :func:`load_demo_env`) — otherwise
the run would silently route to the offline fake roster (which runs no subprocess and
emits no ``on_command``) and the assertions would pass on a FAKE run, defeating the
acceptance. So with the gate on and no key, the setup FAILS loudly rather than skipping
or passing on a fallback: a pass here means a real model genuinely ran real commands.

``bun`` must be on ``PATH`` for the real ``bun test`` to run; its absence FAILS loudly
in setup (the seeded module is a Bun test, so a missing toolchain is a real failure of
the acceptance environment, not a skip).

LangSmith tracing is kept ON (whatever the ``.env`` activates) so this real run is
captured for usage/billing visibility — do not disable it here.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_m5_fix_loop_real.py -q -s
"""

from __future__ import annotations

import os
import shutil

import pytest

# The opt-in gate. The rest of the repo uses this same env var as the real-model switch.
_REAL_MODEL_GATE = "LDW_DEMO_REAL_MODEL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_REAL_MODEL_GATE),
    reason=f"{_REAL_MODEL_GATE} not set; real-model fix-loop E2E is opt-in",
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

    * an OpenRouter key MUST be in force — otherwise the roster serves the deterministic
      fake ``code_fixer`` (no subprocess, no ``on_command``) and the assertions would
      pass on a FAKE run; and
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
            "must run a REAL model against a REAL sandbox — a fallback to the offline "
            "fake roster runs no subprocess and emits no on_command, so it cannot be "
            "accepted as proof of in-loop executable verification."
        )

    if shutil.which("bun") is None:
        pytest.fail(
            "bun is not on PATH but the M5 fix-loop acceptance runs a real `bun test` "
            "against a seeded Bun module; install bun (https://bun.sh) to run this gate."
        )


async def test_real_model_fix_loop_runs_bun_until_red_then_green(
    _real_fix_loop_setup: None,
) -> None:
    """A real fix-loop shells out, sees a red ``bun test``, fixes it, and goes green.

    Runs the ``fix_loop`` preset through ``run_workflow`` with the demo's real roster
    (a real ``code_fixer`` execution leaf) and a real :class:`SandboxManager`, capturing
    every ``CommandEvent`` the engine's ``on_command`` sink delivers. Asserts the real
    red->green transition: a start AND end edge fired for a real command, at least one
    command exited non-zero (red ``bun test``) and at least one exited zero (green), and
    the loop returned the green-attempt result (never the honest "still red" fallback) —
    none of which the offline fake (no subprocess, no command events) could produce.
    """
    from workflows import fix_loop, make_roster, make_workflows

    from langchain_dynamic_workflow import (
        CommandEvent,
        Ctx,
        ExecPolicy,
        SandboxManager,
        local_subprocess_factory,
        run_workflow,
    )

    # A real per-leaf sandbox: a needs_execution leaf (code_fixer) is leased a
    # LocalSubprocessSandbox so its execute tool runs real shell commands for a true
    # exit code. Bounds keep a runaway test/chatty failure in check.
    manager = SandboxManager(
        sandbox_factory=local_subprocess_factory(
            ExecPolicy(default_timeout=120, output_cap_bytes=64 * 1024)
        )
    )

    commands: list[CommandEvent] = []

    async def orchestrate(ctx: Ctx) -> str:
        # Drive the preset directly (the same body run_workflow_live invokes), with a
        # small retry budget — enough for a real model to fix a one-line bug.
        return await fix_loop(ctx, {"max_attempts": 3})

    result = await run_workflow(
        orchestrate,
        roster=make_roster(),
        workflows=make_workflows(),
        sandbox_manager=manager,
        on_command=commands.append,
        thread_id="t-m5-fix-loop-real",
    )

    # Headline 0: the loop returned the GREEN result. fix_loop returns this string ONLY
    # when a FixResult.tests_passed was true; the all-red budget-exhausted path returns
    # "still red ...". So a green return is the proof the final result reports tests
    # passed -- a real exit-code-gated pass, not leaf prose.
    assert isinstance(result, str)
    assert "still red" not in result.lower(), (
        f"the loop exhausted its budget without going green (tests never passed): {result!r}"
    )
    assert "green on attempt" in result.lower(), (
        f"the loop must return the green-attempt result (tests_passed true); got {result!r}"
    )

    # Headline 1: the on_command sink fired a real start AND end edge for a real command
    # -- the terminal-card lifecycle, sourced from inside the real LocalSubprocessSandbox.
    # The offline fake fixer runs no subprocess and would emit ZERO command events.
    assert commands, (
        "no on_command events fired: a real execution leaf must shell out via its "
        "execute tool (an offline fake fixer runs no subprocess and emits none)"
    )
    phases = {event.phase for event in commands}
    assert "start" in phases and "end" in phases, (
        f"on_command must deliver both a start and an end edge; phases seen: {phases}"
    )
    # Begin/end edges of one command share a command_id and correlate to the owning leaf.
    starts = [e for e in commands if e.phase == "start"]
    ends = [e for e in commands if e.phase == "end"]
    assert starts and ends, f"expected paired start/end edges; got {len(starts)}/{len(ends)}"
    assert all(e.leaf_span_id for e in commands), (
        "every command event must correlate to its owning leaf via a leaf_span_id"
    )
    paired_ids = {e.command_id for e in starts} & {e.command_id for e in ends}
    assert paired_ids, (
        "at least one command's start and end edge must share a command_id (one card flip)"
    )

    # Headline 2: a REAL red->green transition in the exit codes. The seeded sum()
    # returns 0, so the first `bun test` exits non-zero (red); the real model fixes
    # src/sum.ts and a later `bun test` exits zero (green). Only a real subprocess can
    # produce both -- the fold-into-result fallback never carries an exit code at all.
    exit_codes = [e.exit_code for e in ends if e.exit_code is not None]
    assert exit_codes, "every end edge must carry the real subprocess exit code"
    assert any(code != 0 for code in exit_codes), (
        f"expected at least one RED (non-zero) command exit; exit codes seen: {exit_codes}"
    )
    assert any(code == 0 for code in exit_codes), (
        f"expected at least one GREEN (zero) command exit after the fix; "
        f"exit codes seen: {exit_codes}"
    )

    # Teardown invariant: the engine stopped every leased real sandbox at settle.
    assert manager.active_count == 0, "every leased real sandbox must be torn down at settle"

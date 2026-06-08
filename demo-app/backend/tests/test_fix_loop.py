"""Offline checks for the ``fix_loop`` preset and its real-execution run wiring.

These run with no model key, so the roster serves a deterministic fake ``code_fixer``
leaf instead of the real deepagent (and runs no subprocess). The fake encodes the
loop's headline shape in code: attempt 1 goes red with a failure tail, attempt 2 goes
green — so the offline gate exercises the edit -> build -> test -> branch-on-exit-code
loop WITHOUT the real model or bun. The headline real path (a real ``LocalSubprocessSandbox``
executing ``bun test`` for a true exit code) is covered by the gated real-model E2E.

The run-wiring tests assert the engine receives a ``sandbox_manager`` and an
``on_command`` sink (the two additions that make real execution and terminal-card
streaming reachable) via a spy, not source inspection — a behavioral guard.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from ui_adapter import UiAdapter
from workflows import FixResult, fix_loop, make_roster, make_workflows

from langchain_dynamic_workflow import ProgressKind, run_workflow


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the roster serves the deterministic fake fixer."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LDW_DEMO_REAL_MODEL"):
        os.environ.pop(key, None)


def test_make_workflows_registers_fix_loop() -> None:
    """The registry resolves the ``fix_loop`` preset by name."""
    registry = make_workflows()
    assert registry.resolve("fix_loop") is fix_loop


async def test_fix_loop_offline_goes_red_then_green_and_returns_green_summary() -> None:
    """The offline fake loop fails attempt 1, passes attempt 2, returns the green summary.

    The headline of M5: the script branches on a per-attempt ``tests_passed`` and stops
    on the first green attempt. The offline fake fixer is red on attempt 1 (with a
    failure tail) and green on attempt 2, so the loop must run exactly two attempts and
    return the green summary — proving the red->green branch is reachable offline.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))

    result = await run_workflow(
        lambda ctx: fix_loop(ctx, {}),
        roster=make_roster(),
        workflows=make_workflows(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
    )

    assert isinstance(result, str)
    # The green summary is returned (not the "still red" fallback).
    assert result.strip()
    assert "still red" not in result.lower(), result
    assert "green on attempt 2" in result.lower() or "attempt 2" in result.lower(), result

    # Exactly two attempt phases were narrated (attempt 1 red, attempt 2 green): the
    # loop stopped on the first green rather than running the full budget.
    phase_titles = [
        props["message"]
        for comp, props in events
        if comp == "phase_timeline" and props["kind"] == ProgressKind.PHASE.value
    ]
    assert phase_titles == ["attempt 1", "attempt 2"], phase_titles


async def test_fix_loop_respects_max_attempts_and_returns_honest_still_red() -> None:
    """An all-red fixer exhausts the budget and returns an honest "still red" result.

    With ``max_attempts`` capped, a fixer that never goes green must NOT loop forever:
    the loop runs exactly ``max_attempts`` attempts, then returns a truthful "still red"
    message carrying the last failure tail — never a false "passed". Driven by a
    roster whose ``code_fixer`` always reports red, so the all-red branch is exercised
    deterministically.
    """
    # Build a roster whose code_fixer never goes green, leaving the rest of the demo
    # roster intact. The fake honors the schema so agent(schema=FixResult) folds it out.
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig, RunnableLambda

    roster = make_roster()

    def _always_red_builder(*, response_format: Any = None) -> Any:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            return {
                "messages": [*inp["messages"], AIMessage(content="still failing")],
                "structured_response": FixResult(
                    tests_passed=False,
                    failure_tail="FAIL src/sum.test.ts > adds negatives",
                    summary="could not make the tests pass",
                ),
            }

        return RunnableLambda(_leaf)

    roster.register(
        "code_fixer",
        builder=_always_red_builder,
        description="always-red fixer (test fixture)",
        needs_execution=True,
    )

    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))

    result = await run_workflow(
        lambda ctx: fix_loop(ctx, {"max_attempts": 2}),
        roster=roster,
        workflows=make_workflows(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
    )

    assert "still red" in result.lower(), result
    assert "FAIL" in result, "the still-red result must carry the last failure tail"

    phase_titles = [
        props["message"]
        for comp, props in events
        if comp == "phase_timeline" and props["kind"] == ProgressKind.PHASE.value
    ]
    assert phase_titles == ["attempt 1", "attempt 2"], phase_titles


async def test_run_workflow_live_wires_sandbox_manager_and_on_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live run path passes a ``sandbox_manager`` and ``on_command`` to ``run_workflow``.

    These two additions are what make real execution (a per-leaf real sandbox) and
    terminal-card streaming reachable; currently the run path passes neither, so real
    execution never happens. This spies on ``run_workflow`` as the host calls it and
    asserts both are wired — a behavioral guard, not source inspection. The fix_loop
    preset itself runs offline (fake fixer, no subprocess) so no bun is needed here.
    """
    import host_graph

    from langchain_dynamic_workflow import SandboxManager

    captured: dict[str, Any] = {}
    real_run_workflow = host_graph.run_workflow

    async def _spy_run_workflow(*args: Any, **kwargs: Any) -> Any:
        captured["sandbox_manager"] = kwargs.get("sandbox_manager")
        captured["on_command"] = kwargs.get("on_command")
        return await real_run_workflow(*args, **kwargs)

    monkeypatch.setattr(host_graph, "run_workflow", _spy_run_workflow)

    adapter = UiAdapter(emit=lambda _comp, _props: None)
    await host_graph.run_workflow_live("fix_loop", {}, adapter=adapter)

    assert isinstance(captured["sandbox_manager"], SandboxManager), (
        "run_workflow_live must pass a SandboxManager so a needs_execution leaf gets a real sandbox"
    )
    # A bound method is a fresh object per access, so compare by equality (same __self__
    # and __func__) rather than identity: this still pins the wired sink to THIS adapter.
    assert captured["on_command"] == adapter.on_command, (
        "run_workflow_live must wire the adapter's on_command sink so terminal cards stream"
    )


async def test_run_meta_script_live_wires_sandbox_manager_and_on_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The meta-script run path also passes ``sandbox_manager`` and ``on_command``.

    A meta-authored fix-loop script must reach the same real-execution + terminal-card
    streaming wiring as a preset run. Spies on ``run_workflow_from_source`` (the meta
    path's runner) and asserts both sinks are wired on the gate-pass path.
    """
    import host_graph

    from langchain_dynamic_workflow import SandboxManager

    captured: dict[str, Any] = {}
    real_runner = host_graph.run_workflow_from_source

    async def _spy_runner(*args: Any, **kwargs: Any) -> Any:
        captured["sandbox_manager"] = kwargs.get("sandbox_manager")
        captured["on_command"] = kwargs.get("on_command")
        return await real_runner(*args, **kwargs)

    monkeypatch.setattr(host_graph, "run_workflow_from_source", _spy_runner)

    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))
    await host_graph.run_meta_script_live(
        submit_rejected=False, adapter=adapter, emit=lambda _c, _p: None
    )

    assert isinstance(captured["sandbox_manager"], SandboxManager)
    # Bound-method equality (same __self__/__func__), not identity — see the sibling test.
    assert captured["on_command"] == adapter.on_command

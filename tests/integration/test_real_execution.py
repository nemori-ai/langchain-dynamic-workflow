"""Integration: real subprocess execution end-to-end through ``run_workflow``.

An execution leaf leased a :class:`LocalSubprocessSandbox` actually runs shell
commands in its per-leaf temp root; two parallel execution leaves stay isolated
(distinct temp dirs); and the run tears down every temp dir, leaving no straggler
process. The leaf model is an offline scripted fake (no API key) that reaches its
leased backend through the engine's ``sandbox_backend`` config seam and calls
``aexecute`` — so the *subprocess* is the real thing under test while the path it
travels (gate -> lease -> guarded composite -> threaded execute -> teardown
``close()``) is the genuine engine headline path.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import (
    Ctx,
    ExecPolicy,
    Roster,
    SandboxManager,
    local_subprocess_factory,
    run_workflow,
)
from langchain_dynamic_workflow._local_subprocess import LocalSubprocessSandbox
from langchain_dynamic_workflow._sandbox import SharedArtifactStore, build_leaf_backend


async def test_leased_real_backend_executes_a_real_command_through_the_guard() -> None:
    # Lease a real backend the way the engine does, wrap it in the guarded
    # composite, and assert execute() runs a real command in the leaf temp root.
    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))
    backend = manager.acquire(leaf_id="leaf-exec", needs_execution=True)
    try:
        assert isinstance(backend, LocalSubprocessSandbox)
        guarded = build_leaf_backend(
            isolated=backend, shared_store=SharedArtifactStore(), producer="leaf-exec"
        )
        result = guarded.execute(f'{sys.executable} -c "print(6 * 7)"')
        assert "42" in result.output
        assert result.exit_code == 0
    finally:
        await manager.stop("leaf-exec")


async def test_two_parallel_real_leaves_are_isolated_and_torn_down() -> None:
    # Distinct leaf ids => distinct temp roots => files written by one are
    # invisible to the other; after stop() both temp dirs are gone.
    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))
    a = manager.acquire(leaf_id="leaf-a", needs_execution=True)
    b = manager.acquire(leaf_id="leaf-b", needs_execution=True)
    assert isinstance(a, LocalSubprocessSandbox) and isinstance(b, LocalSubprocessSandbox)
    a.write("/only-a.txt", "secret")
    # b cannot see a's file (separate roots).
    miss = b.read("/only-a.txt")
    assert miss.error is not None
    roots = [a.root_path, b.root_path]
    await manager.stop("leaf-a")
    await manager.stop("leaf-b")
    assert all(not os.path.exists(r) for r in roots)


def _exec_leaf(command: str) -> Runnable[Any, Any]:
    """A fake execution leaf that runs ``command`` via its leased backend.

    It reaches the per-leaf sandbox backend the engine threaded into config (the
    same ``sandbox_backend`` seam a backend-aware deepagent reads), calls
    ``aexecute`` so the real ``LocalSubprocessSandbox.execute`` runs on a worker
    thread, and reports the combined output and exit code it observed — letting a
    test assert the real command output rode all the way back through the engine.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        # Execution leaves are handed a full SandboxBackendProtocol (the guarded
        # composite over the leased LocalSubprocessSandbox), so ``aexecute`` is
        # part of the contract the engine threads through this seam.
        backend: SandboxBackendProtocol = configurable["sandbox_backend"]
        result = await backend.aexecute(command)
        reply = f"exit={result.exit_code};out={result.output.strip()}"
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call)


async def test_run_workflow_drives_a_real_command_through_the_engine() -> None:
    # The headline path end-to-end: run_workflow leases a real backend via the
    # factory, wraps it in the guarded composite, and the leaf's aexecute runs a
    # real command whose actual stdout + exit code ride back into the result.
    roster = Roster().register(
        "runner", _exec_leaf(f'{sys.executable} -c "print(6 * 7)"'), needs_execution=True
    )
    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("run it", agent_type="runner")

    result = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-exec"
    )
    assert "exit=0" in result
    assert "42" in result


async def test_run_workflow_tears_down_real_temp_dirs_after_the_run() -> None:
    # After a run with real execution leaves, every per-leaf temp root the engine
    # leased is removed by teardown (stop() -> close()) and no leaf temp dir
    # survives the run — no leaked directories, no straggler workspace.
    seen_roots: list[str] = []

    def _probe_leaf(command: str) -> Runnable[Any, Any]:
        async def _call(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            configurable = (config or {}).get("configurable", {})
            backend: SandboxBackendProtocol = configurable["sandbox_backend"]
            # Record the isolated backend's real temp root so the test can assert
            # it is gone once the run (and its teardown) completes.
            isolated = getattr(backend, "_isolated", backend)
            assert isinstance(isolated, LocalSubprocessSandbox)
            seen_roots.append(isolated.root_path)
            await backend.aexecute(command)
            return {"messages": [*inp["messages"], AIMessage(content="done")]}

        return RunnableLambda(_call)

    roster = (
        Roster()
        .register("p", _probe_leaf(f'{sys.executable} -c "print(1)"'), needs_execution=True)
        .register("q", _probe_leaf(f'{sys.executable} -c "print(2)"'), needs_execution=True)
    )
    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("p", agent_type="p"),
                lambda: ctx.agent("q", agent_type="q"),
            ]
        )

    await run_workflow(orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-teardown")

    assert len(seen_roots) == 2
    assert seen_roots[0] != seen_roots[1]  # two leaves => two distinct temp roots
    assert all(not os.path.exists(root) for root in seen_roots)  # all torn down

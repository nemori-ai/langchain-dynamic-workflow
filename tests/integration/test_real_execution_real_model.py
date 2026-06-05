"""Real-model E2E acceptance gate for real subprocess execution (development-time gate).

Gated behind ``LDW_DEMO_REAL_MODEL`` + OpenRouter creds (a local ``.env``). Runs a
REAL model leaf that genuinely calls the builtin ``execute`` tool against a
:class:`LocalSubprocessSandbox`, proving the full headline path: host -> agent ->
real shell -> real output -> folded result. Offline-skippable: with no gate set,
the whole module is skipped, so CI stays offline.

The scenario is a smoking gun designed so an honest real model MUST shell out: it
asks for the SHA-256 of a fixed string the model cannot guess (a 64-hex digest),
so the only way to answer correctly is to run a shell command through ``execute``.
The expected digest is computed here with :mod:`hashlib`, never hand-pinned, so the
assertion tracks the real value rather than a copied constant.

The leaf's leased backend is the real :class:`LocalSubprocessSandbox`, threaded
into the leaf config by the engine under ``sandbox_backend``; a deepagents backend
factory reads it off the runtime config so the builtin ``execute`` tool runs the
command in the leaf's per-leaf temp root. An ``on_leaf_event`` tap (riding M1's
per-leaf observability) confirms a ``tool``-kind ``execute`` edge actually fired,
end-to-end -- the command observability the real backend reuses rather than adding
its own sink.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LDW_DEMO_REAL_MODEL"),
    reason="real-model gate: set LDW_DEMO_REAL_MODEL + OpenRouter creds to run",
)

# A fixed payload whose SHA-256 the model cannot guess: the only honest way to
# report a correct 64-hex digest is to actually run a shell command via execute.
_SHA_PAYLOAD = "ldw-m5-real-exec-acceptance-7f3a"


async def test_real_model_leaf_runs_a_real_shell_command_and_folds_the_real_output() -> None:
    from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
    from deepagents.backends.protocol import BackendProtocol
    from examples._shared.real_models import load_demo_env, real_leaf_model
    from langchain.tools import ToolRuntime  # pyright: ignore[reportUnknownVariableType]

    from langchain_dynamic_workflow import (
        Ctx,
        ExecPolicy,
        LeafEvent,
        Roster,
        SandboxManager,
        local_subprocess_factory,
        run_workflow,
    )

    load_demo_env()
    # Disable LangSmith tracing for this deepagent-heavy run (memory: real-e2e).
    os.environ.pop("LANGSMITH_TRACING", None)

    model = real_leaf_model()
    assert model is not None, "real leaf model must be available under the gate"

    expected_digest = hashlib.sha256(_SHA_PAYLOAD.encode()).hexdigest()

    def _leased_backend(runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """Resolve the per-leaf backend the engine threaded into the leaf config.

        deepagents calls this factory with the live tool runtime; the engine leases
        a real ``LocalSubprocessSandbox`` and threads it under ``sandbox_backend``,
        so returning it wires the builtin ``execute`` tool to the real subprocess
        backend in the leaf's private temp root.
        """
        configurable = (runtime.config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        return backend

    # A real deepagent whose builtin execute tool runs against the leased real
    # backend; the model must call execute to compute the digest it is asked for.
    # deepagents' BackendFactory alias is under-parameterized (a bare ToolRuntime),
    # so the strict checker cannot reconcile the closure's annotation with it.
    leaf = create_deep_agent(model=model, backend=_leased_backend)  # pyright: ignore[reportArgumentType]
    roster = Roster().register("shell_runner", leaf, needs_execution=True)

    manager = SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy()))

    events: list[LeafEvent] = []

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent(
            "Compute the SHA-256 hex digest of the exact string "
            f"{_SHA_PAYLOAD!r} (no trailing newline) and report ONLY the 64-character "
            "hex digest. You CANNOT know this value -- you MUST run a shell command "
            "with the execute tool to compute it (for example, "
            f"`python3 -c \"import hashlib; print(hashlib.sha256(b'{_SHA_PAYLOAD}')"
            '.hexdigest())"`). Do not guess.',
            agent_type="shell_runner",
        )

    result = await run_workflow(
        orchestrate,
        roster=roster,
        sandbox_manager=manager,
        on_leaf_event=events.append,
        thread_id="t-real-exec",
    )

    # Headline 1: the real digest -- only reachable by actually running the shell.
    assert expected_digest in result.lower(), (
        f"the folded result must carry the real sha256 digest {expected_digest!r}; got {result!r}"
    )

    # Headline 2: an execute tool edge actually fired (岔口 0 reuse of M1's per-leaf
    # observability -- the real backend adds no sink; command visibility rides here).
    assert events, "a real leaf must fire interior callback events"
    execute_edges = [e for e in events if e.kind == "tool" and e.name == "execute"]
    assert execute_edges, (
        "the real model must have called the builtin execute tool; "
        f"tool names seen: {sorted({e.name for e in events if e.kind == 'tool'})}"
    )
    phases = {e.phase for e in execute_edges}
    assert "start" in phases and "end" in phases, (
        f"the execute tool edge must show a start and an end; phases seen: {phases}"
    )

    # Teardown: the engine stopped every leased real sandbox at settle.
    assert manager.active_count == 0, "every leased real sandbox must be torn down at settle"

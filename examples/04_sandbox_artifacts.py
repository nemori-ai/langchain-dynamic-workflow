"""Phase 4 demo: per-leaf sandbox isolation + ``/shared/`` artifact hand-off.

Two ``needs_execution`` producer leaves each run in their own isolated sandbox,
writing the SAME isolated path (``/work/out.txt``) with DIFFERENT content — and
each reads back only its own write, proving the sandboxes are mutually invisible.
Each producer also drops an artifact under ``/shared/``; a third consumer leaf
then picks both up through the run-shared store, demonstrating the explicit
hand-off across otherwise-isolated leaves.

The leaves read the per-leaf backend the engine threads into
``config['configurable']['sandbox_backend']`` and call its file operations
directly, so the demo runs fully offline with no API key and no real sandbox
infrastructure — the in-memory sandbox backend stands in for a container.

    uv run python examples/04_sandbox_artifacts.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import Ctx, Roster, SandboxManager, run_workflow

ISOLATED_PATH = "/work/out.txt"


def _producer_leaf(*, label: str, shared_path: str) -> Runnable[Any, Any]:
    """A producer: writes an isolated file and a shared artifact, reads its own back."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        # Private write to the leaf's isolated sandbox (same path across producers).
        await backend.awrite(ISOLATED_PATH, f"private:{label}")
        # Explicit hand-off: drop an artifact in the run-shared store.
        await backend.awrite(shared_path, f"artifact-from-{label}")
        # Read the isolated file back: each producer sees only its own content.
        read = await backend.aread(ISOLATED_PATH)
        seen = read.file_data["content"] if read.file_data is not None else "<missing>"
        reply = f"{label}: isolated read-back={seen!r}, id={backend.id}"  # type: ignore[attr-defined]
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call)


def _consumer_leaf(shared_paths: list[str]) -> Runnable[Any, Any]:
    """A consumer: reads every producer's artifact back through ``/shared/``."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        parts: list[str] = []
        for path in shared_paths:
            read = await backend.aread(path)
            parts.append(read.file_data["content"] if read.file_data is not None else "<miss>")
        return {"messages": [*inp["messages"], AIMessage(content=", ".join(parts))]}

    return RunnableLambda(_call)


async def main() -> None:
    roster = (
        Roster()
        .register(
            "producer_a",
            _producer_leaf(label="A", shared_path="/shared/a.txt"),
            description="Writes an isolated file and a shared artifact",
            needs_execution=True,
        )
        .register(
            "producer_b",
            _producer_leaf(label="B", shared_path="/shared/b.txt"),
            description="Writes an isolated file and a shared artifact",
            needs_execution=True,
        )
        .register(
            "consumer",
            _consumer_leaf(["/shared/a.txt", "/shared/b.txt"]),
            description="Collects both shared artifacts",
            needs_execution=True,
        )
    )
    manager = SandboxManager()

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        ctx.phase("isolated production")
        produced = await ctx.parallel(
            [
                lambda: ctx.agent("produce A", agent_type="producer_a"),
                lambda: ctx.agent("produce B", agent_type="producer_b"),
            ]
        )
        ctx.phase("shared hand-off")
        collected = await ctx.agent("collect artifacts", agent_type="consumer")
        return {"produced": produced, "collected": collected}

    result = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=manager, thread_id="demo-4"
    )

    print("isolated production (each producer reads back only its own write):")
    for line in result["produced"]:
        print(f"  - {line}")
    print(f"shared hand-off (consumer picked up both artifacts): {result['collected']!r}")
    # The engine's lifecycle finale stops every execution sandbox it leased once
    # the script settles, so the manager holds zero live sandboxes after the run.
    print(f"active sandboxes still live after the run: {manager.active_count}")


if __name__ == "__main__":
    asyncio.run(main())

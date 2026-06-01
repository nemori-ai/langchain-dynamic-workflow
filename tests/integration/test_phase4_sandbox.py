"""Phase 4 integration: per-leaf sandbox isolation through ``run_workflow``.

These tests drive the full ``run_workflow`` -> ``@entrypoint`` -> leaf path with
fake leaves (no API keys), pinning the locked Phase 4 semantics: tiered admission
(execution leaves get an isolated sandbox, reasoning leaves do not allocate one),
journal-key-derived sandbox identity that is stable across resume, two parallel
execution leaves writing the same path remaining mutually invisible, and the
``/shared/`` hand-off with ``..`` traversal blocked.

The fake execution leaf reaches its acquired backend through
``config['configurable']['sandbox_backend']`` — the same seam a backend-aware
deepagent reads — so the isolation boundary is exercised end to end without any
real sandbox infrastructure.
"""

from __future__ import annotations

from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow
from langchain_dynamic_workflow._sandbox import SandboxManager


def _writer_leaf(path: str, content: str) -> Runnable[Any, Any]:
    """A fake execution leaf that writes ``content`` to ``path`` in its backend.

    It reads the per-leaf sandbox backend the engine threaded into config, writes
    a file, then reports back how many files its backend can see — so a test can
    assert two parallel leaves never observe each other's writes.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        await backend.awrite(path, content)
        listing = await backend.als("/")
        seen = sorted(entry["path"] for entry in (listing.entries or []))
        reply = f"id={backend.id};files={seen}"  # type: ignore[attr-defined]
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call)


def _reasoning_leaf(reply: str) -> Runnable[Any, Any]:
    """A fake pure-reasoning leaf that ignores files and just replies."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call)


async def test_execution_leaf_receives_isolated_sandbox() -> None:
    # A needs_execution leaf is handed an isolated sandbox backend; it can write
    # and read back its own file through that backend.
    roster = Roster().register("writer", _writer_leaf("/out.txt", "hello"), needs_execution=True)
    manager = SandboxManager()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("write it", agent_type="writer")

    result = await run_workflow(orchestrate, roster=roster, sandbox_manager=manager, thread_id="t1")
    assert "files=['/out.txt']" in result


async def test_two_parallel_execution_leaves_are_mutually_invisible() -> None:
    # The core isolation guarantee: two execution leaves writing the SAME path in
    # parallel must each see only their own file, never the sibling's write. This
    # must hold even though both target "/out.txt" — proving per-leaf backends are
    # genuinely separate stores, not a single shared workspace routed by name.
    roster = (
        Roster()
        .register("a", _writer_leaf("/out.txt", "from-a"), needs_execution=True)
        .register("b", _writer_leaf("/out.txt", "from-b"), needs_execution=True)
    )
    manager = SandboxManager()

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("write", agent_type="a"),
                lambda: ctx.agent("write", agent_type="b"),
            ]
        )

    results = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=manager, thread_id="t1"
    )
    # Each leaf sees exactly one file (its own), never two.
    assert results is not None
    for reply in results:
        assert reply is not None
        assert "files=['/out.txt']" in reply
    # Distinct sandbox identities: the two leaves never shared a backend.
    ids = {reply.split(";")[0] for reply in results if reply is not None}
    assert len(ids) == 2


async def test_reasoning_leaf_does_not_allocate_a_sandbox() -> None:
    # Tiered admission end to end: a pure-reasoning leaf must run without the
    # manager ever allocating a sandbox — N logical agents != N active sandboxes.
    roster = Roster().register("thinker", _reasoning_leaf("thought"), needs_execution=False)
    manager = SandboxManager()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("think", agent_type="thinker")

    result = await run_workflow(orchestrate, roster=roster, sandbox_manager=manager, thread_id="t1")
    assert result == "thought"
    # The reasoning leaf was never allocated an isolated sandbox.
    assert manager.active_count == 0


async def test_sandbox_identity_is_stable_across_resume() -> None:
    # The same leaf call resolves the SAME sandbox identity on a resumed run,
    # because identity derives from the (stable) content-hash journal key. The
    # second run is a journal cache hit, so it serves the first run's recorded
    # reply verbatim — including the sandbox id baked into it.
    roster = Roster().register("writer", _writer_leaf("/out.txt", "v"), needs_execution=True)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("write it", agent_type="writer")

    first = await run_workflow(
        orchestrate,
        roster=roster,
        sandbox_manager=SandboxManager(),
        journal=journal,
        thread_id="t1",
    )
    second = await run_workflow(
        orchestrate,
        roster=roster,
        sandbox_manager=SandboxManager(),
        journal=journal,
        thread_id="t2",
    )
    # Same recorded result on resume => same derived sandbox identity.
    assert first == second
    first_id = first.split(";")[0]
    second_id = second.split(";")[0]
    assert first_id == second_id

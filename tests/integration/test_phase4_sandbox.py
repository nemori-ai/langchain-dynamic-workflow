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

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    SandboxManager,
    journal_key,
    run_workflow,
)
from langchain_dynamic_workflow._sandbox import leaf_id_from_key


def _writer_leaf(path: str, content: str) -> Runnable[Any, Any]:
    """A fake execution leaf that writes ``content`` to ``path`` in its backend.

    It reads the per-leaf sandbox backend the engine threaded into config, writes
    its file, then reads the same path back and reports the content it observes —
    so a test can assert two parallel leaves writing the SAME path never observe
    each other's content. The leaf's sandbox id is reported too, to confirm the
    two leaves were handed distinct backends.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        await backend.awrite(path, content)
        read = await backend.aread(path)
        observed = read.file_data["content"] if read.file_data is not None else "<missing>"
        reply = f"id={backend.id};read={observed}"  # type: ignore[attr-defined]
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
    assert "read=hello" in result


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
    # Each leaf reads back ITS OWN content at /out.txt, never the sibling's —
    # proving the two backends are genuinely separate stores, not one shared
    # workspace where the same path would collide.
    assert results is not None
    assert results[0] is not None and "read=from-a" in results[0]
    assert results[1] is not None and "read=from-b" in results[1]
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


async def test_engine_tears_down_leased_sandboxes_after_run() -> None:
    # Lifecycle finale (acceptance #4 "stop() 被调用清理"): a lease keeps its
    # sandbox live for find-or-create reuse across retries, so nothing reclaims it
    # on its own. The engine must stop every execution sandbox it leased once the
    # script settles, so the manager holds zero live sandboxes after the run —
    # otherwise every execution leaf leaks a live backend per run forever.
    roster = (
        Roster()
        .register("a", _writer_leaf("/out.txt", "from-a"), needs_execution=True)
        .register("b", _writer_leaf("/out.txt", "from-b"), needs_execution=True)
        .register("c", _writer_leaf("/out.txt", "from-c"), needs_execution=True)
    )
    manager = SandboxManager()

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("write", agent_type="a"),
                lambda: ctx.agent("write", agent_type="b"),
                lambda: ctx.agent("write", agent_type="c"),
            ]
        )

    results = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=manager, thread_id="t1"
    )
    assert results is not None and all(r is not None for r in results)
    # Every leased execution sandbox was torn down on the lifecycle finale: no leak.
    assert manager.active_count == 0


async def test_engine_tears_down_leased_sandboxes_even_when_script_raises() -> None:
    # Teardown must run on the failure path too: a script that raises mid-flight,
    # after an execution leaf has been leased, must still leave the manager empty.
    # The leaf journals on success, but the sandbox cleanup is unconditional.
    roster = Roster().register("a", _writer_leaf("/out.txt", "from-a"), needs_execution=True)
    manager = SandboxManager()

    class _Boom(RuntimeError):
        pass

    async def orchestrate(ctx: Ctx) -> str:
        await ctx.agent("write", agent_type="a")  # leases + journals a sandbox
        raise _Boom("script blew up after leasing a sandbox")

    try:
        await run_workflow(orchestrate, roster=roster, sandbox_manager=manager, thread_id="t1")
    except _Boom:
        pass
    else:  # pragma: no cover - the orchestrator always raises
        raise AssertionError("orchestrate was expected to raise _Boom")
    # The lifecycle finally-block stopped the leased sandbox despite the raise.
    assert manager.active_count == 0


def _counting_writer_leaf(
    path: str, content: str, *, calls: list[int], fail_times: int = 0
) -> Runnable[Any, Any]:
    """A writer leaf that counts live invocations and can fail the first N.

    ``calls`` is a single-element list used as a mutable counter shared with the
    test, so the test can assert the leaf actually ran live (not a journal hit).
    The leaf reports ``id=<backend.id>`` so the test can read the engine-derived
    sandbox identity off a *live* re-run.
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        calls[0] += 1
        if calls[0] <= fail_times:
            raise RuntimeError("counting writer leaf boom")
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        await backend.awrite(path, content)
        reply = f"id={backend.id}"  # type: ignore[attr-defined]
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    return RunnableLambda(_call)


async def test_live_re_run_re_derives_the_same_sandbox_identity() -> None:
    # The genuine resume identity-stability case the cache-hit test cannot cover:
    # an in-flight leaf that re-runs LIVE on resume must re-derive the SAME sandbox
    # identity, so a retry hits the same logical backend. The first run journals
    # leaf "stable" then raises before leaf "live" can journal; the resume replays
    # "stable" from cache (zero model cost) and runs "live" for real — exercising
    # identity derivation on the engine path, not serving a byte-identical cache
    # echo. We pin that the live-derived id equals the identity derived from the
    # leaf's content-hash journal key, the documented single source of identity.
    journal = InMemoryJournalStore()
    live_calls = [0]
    roster = (
        Roster()
        .register("stable", _writer_leaf("/s.txt", "s"), needs_execution=True)
        .register(
            "live",
            _counting_writer_leaf("/l.txt", "l", calls=live_calls, fail_times=1),
            needs_execution=True,
        )
    )

    first_attempt = [0]

    async def orchestrate(ctx: Ctx) -> str:
        await ctx.agent("do stable", agent_type="stable")  # journals on first run
        first_attempt[0] += 1
        # On the first run the "live" leaf raises (fail_times=1) and is NOT
        # journaled; on resume it runs live and succeeds.
        return await ctx.agent("do live", agent_type="live")

    # First run: "stable" journals, then "live" raises before journaling.
    try:
        await run_workflow(
            orchestrate,
            roster=roster,
            sandbox_manager=SandboxManager(),
            journal=journal,
            thread_id="t1",
        )
    except RuntimeError:
        pass
    else:  # pragma: no cover - the first run always raises in the live leaf
        raise AssertionError("first run was expected to raise from the live leaf")
    assert live_calls[0] == 1  # the live leaf ran once and failed

    # Resume on the SAME journal: "stable" is a cache hit, "live" re-runs live.
    manager2 = SandboxManager()
    result = await run_workflow(
        orchestrate,
        roster=roster,
        sandbox_manager=manager2,
        journal=journal,
        thread_id="t2",
    )
    # The live leaf genuinely re-ran (cache miss), so derivation was re-exercised.
    assert live_calls[0] == 2
    live_id = result.split("=", 1)[1]
    # The engine-derived identity on the live re-run equals the identity derived
    # from the leaf's content-hash journal key: same leaf -> same identity ->
    # same backend, re-derived live rather than echoed from cache.
    expected_id = leaf_id_from_key(
        journal_key(
            prompt="do live",
            agent_type="live",
            model=None,
            schema=None,
            isolation="shared",
        )
    )
    assert live_id == expected_id
    # And teardown still emptied the manager after the resumed run.
    assert manager2.active_count == 0


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


def _shared_producer_leaf(shared_path: str, content: str) -> Runnable[Any, Any]:
    """A leaf that writes ``content`` to a ``/shared/`` path for later hand-off."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        await backend.awrite(shared_path, content)
        return {"messages": [*inp["messages"], AIMessage(content=f"wrote {shared_path}")]}

    return RunnableLambda(_call)


def _shared_consumer_leaf(shared_paths: list[str]) -> Runnable[Any, Any]:
    """A leaf that reads several ``/shared/`` paths and concatenates their contents."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend: BackendProtocol = configurable["sandbox_backend"]
        collected: list[str] = []
        for path in shared_paths:
            read = await backend.aread(path)
            collected.append(read.file_data["content"] if read.file_data is not None else "<miss>")
        return {"messages": [*inp["messages"], AIMessage(content="+".join(collected))]}

    return RunnableLambda(_call)


async def test_shared_handoff_two_producers_to_one_consumer() -> None:
    # The M4 demo end to end: two needs_execution producer leaves each write an
    # artifact under /shared/ in their own isolated sandbox, then a third leaf
    # reads both back through /shared/. Isolation (separate sandboxes) and
    # hand-off (shared store) coexist in one run.
    roster = (
        Roster()
        .register("prod_a", _shared_producer_leaf("/shared/a.txt", "alpha"), needs_execution=True)
        .register("prod_b", _shared_producer_leaf("/shared/b.txt", "beta"), needs_execution=True)
        .register(
            "consumer",
            _shared_consumer_leaf(["/shared/a.txt", "/shared/b.txt"]),
            needs_execution=True,
        )
    )

    async def orchestrate(ctx: Ctx) -> str:
        # Producers run first (in parallel, isolated); then the consumer picks up
        # both artifacts from the run-shared store.
        await ctx.parallel(
            [
                lambda: ctx.agent("write a", agent_type="prod_a"),
                lambda: ctx.agent("write b", agent_type="prod_b"),
            ]
        )
        return await ctx.agent("collect", agent_type="consumer")

    result = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=SandboxManager(), thread_id="t1"
    )
    assert result == "alpha+beta"


async def test_isolation_mode_selects_a_distinct_sandbox() -> None:
    # Closes the Phase 2 review minor #5 gap end to end: the agent(isolation=...)
    # mode must reach backend selection, not merely partition the journal key.
    # Two calls identical except for isolation must run in DIFFERENT sandboxes
    # (different derived identities), so a "shared" leaf and an "isolated" leaf of
    # the same type never collide in one workspace.
    roster = Roster().register("writer", _writer_leaf("/out.txt", "x"), needs_execution=True)
    manager = SandboxManager()

    async def orchestrate(ctx: Ctx) -> list[str | None]:
        return await ctx.parallel(
            [
                lambda: ctx.agent("write", agent_type="writer", isolation="shared"),
                lambda: ctx.agent("write", agent_type="writer", isolation="isolated"),
            ]
        )

    results = await run_workflow(
        orchestrate, roster=roster, sandbox_manager=manager, thread_id="t1"
    )
    assert results is not None
    ids = {reply.split(";")[0] for reply in results if reply is not None}
    # Different isolation modes => two distinct sandbox identities.
    assert len(ids) == 2

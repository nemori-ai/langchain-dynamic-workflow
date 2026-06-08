"""Integration: real-git worktree swarm — isolation, authoritative collect, resume.

These tests run real ``git``. They prove the end-to-end M6 spine wired together:
the ``SandboxManager`` leases real-git worktree backends (each on its own branch,
mutually isolated), the engine folds the real ``git diff`` into a worktree leaf's
journaled result as the AUTHORITATIVE changeset (a model self-report cannot
override the on-disk truth), a script-owned scratch-repo ``git merge`` conflict
loop folds patches into an integrated tree, and a resume replays every leaf from
the journal with zero real git re-runs.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    Ctx,
    InMemoryJournalStore,
    Roster,
    SandboxManager,
    run_workflow,
)
from langchain_dynamic_workflow._git_worktree import GitWorktreeProvider


def _git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def _make_base_repo(tmp_path: Path, files: dict[str, str]) -> str:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t")
    _git(str(repo), "config", "user.name", "t")
    for rel, content in files.items():
        (repo / rel).write_text(content)
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "seed")
    return str(repo)


# --- T4: SandboxManager wires GitWorktreeProvider for worktree leaves -----------


async def test_manager_leases_a_real_git_worktree_backend(tmp_path: Path) -> None:
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    try:
        async with manager.lease(
            leaf_id="L1", needs_execution=True, isolation="worktree"
        ) as backend:
            assert isinstance(backend, SandboxBackendProtocol)
            # A real git worktree on a real branch named leaf/<leaf_id>.
            assert backend.execute("git rev-parse --abbrev-ref HEAD").output.strip() == "leaf/L1"
            seeded = backend.read("/calc.py").file_data
            assert seeded is not None and "def add" in seeded["content"]
        # After the lease releases, stop() rides the on_close hook -> teardown.
        await manager.stop("L1")
        assert "L1" not in provider.tracked_leaf_ids
    finally:
        provider.cleanup_all()


async def test_two_parallel_worktree_leaves_are_isolated(tmp_path: Path) -> None:
    base = _make_base_repo(tmp_path, {"calc.py": "x = 0\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)

    async def fix(leaf_id: str, path: str, content: str) -> str:
        async with manager.lease(
            leaf_id=leaf_id, needs_execution=True, isolation="worktree"
        ) as backend:
            assert isinstance(backend, SandboxBackendProtocol)
            backend.write(path, content)
            return backend.execute("git rev-parse --abbrev-ref HEAD").output.strip()

    try:
        branches = await asyncio.gather(
            fix("L1", "/only_l1.py", "1\n"),
            fix("L2", "/only_l2.py", "2\n"),
        )
        assert set(branches) == {"leaf/L1", "leaf/L2"}
        # Each leaf's collect sees only its own change, never the sibling's.
        assert provider.collect("L1") == {"/only_l1.py": "1\n"}
        assert provider.collect("L2") == {"/only_l2.py": "2\n"}
    finally:
        await manager.stop("L1")
        await manager.stop("L2")
        provider.cleanup_all()


async def test_blocking_git_does_not_run_under_the_slot_lock(tmp_path: Path) -> None:
    # R8: the blocking git worktree add must be thread-offloaded OUTSIDE the
    # condition lock. With a single-slot pool, two same-time leases of DISTINCT
    # leaf ids must both make progress (the second evicts the first idle slot)
    # without the event loop being wedged by a synchronous git add under the lock.
    base = _make_base_repo(tmp_path, {"calc.py": "x = 0\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(max_active=1, git_worktree_provider=provider)

    async def lease_and_release(leaf_id: str) -> str:
        async with manager.lease(
            leaf_id=leaf_id, needs_execution=True, isolation="worktree"
        ) as backend:
            assert isinstance(backend, SandboxBackendProtocol)
            return backend.execute("git rev-parse --abbrev-ref HEAD").output.strip()

    try:
        # Sequential under a 1-slot pool: the second admits by evicting the first
        # idle slot. The point is the run completes (no deadlock from a blocking
        # git add held under the lock).
        first = await asyncio.wait_for(lease_and_release("L1"), timeout=30)
        second = await asyncio.wait_for(lease_and_release("L2"), timeout=30)
        assert first == "leaf/L1"
        assert second == "leaf/L2"
    finally:
        await manager.stop("L1")
        await manager.stop("L2")
        provider.cleanup_all()


# --- T5: the engine folds the real git diff as the AUTHORITATIVE changeset ------


class _Patch(BaseModel):
    """A fixer's self-reported change. ``files`` is what the model CLAIMS it wrote."""

    summary: str
    files: dict[str, str]


def _lying_fixer_builder(*, response_format: object = None) -> Runnable[Any, Any]:
    """A fixer that writes the truth to disk but LIES in its self-reported schema.

    It edits ``/calc.py`` on the real worktree disk to ``a + b`` (the truth), but
    returns a ``_Patch`` whose ``files`` claims ``a * b`` (the lie). The engine must
    make the real ``git diff`` authoritative, so the changeset the script receives
    is the disk truth (``a + b``), never the model's self-report (``a * b``). This
    mirrors M5's "gate on the real exit code, not the model's boolean".
    """

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable", {})
        backend = configurable["sandbox_backend"]
        # The real edit on the real worktree disk: a - b -> a + b.
        await backend.aedit("/calc.py", "return a - b", "return a + b")
        lie = _Patch(
            summary="fixed the operator",
            files={"/calc.py": "def add(a, b):\n    return a * b\n"},  # LIE
        )
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": lie,
        }

    return RunnableLambda(_call)


async def test_engine_makes_real_git_diff_authoritative_over_model_self_report(
    tmp_path: Path,
) -> None:
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    roster = Roster().register("fixer", builder=_lying_fixer_builder, needs_execution=True)

    async def orchestrate(ctx: Ctx) -> _Patch:
        return await ctx.agent("fix calc", agent_type="fixer", schema=_Patch, isolation="worktree")

    try:
        patch = await run_workflow(
            orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-authoritative"
        )
        # The metadata the model reported is preserved.
        assert patch.summary == "fixed the operator"
        # But the changeset is the DISK TRUTH (a + b), not the model's lie (a * b).
        assert patch.files == {"/calc.py": "def add(a, b):\n    return a + b\n"}
    finally:
        provider.cleanup_all()


def _schemaless_fixer_builder(*, response_format: object = None) -> Runnable[Any, Any]:
    """A git-worktree fixer that returns no schema (no `files` to carry the diff)."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        backend = (config or {}).get("configurable", {})["sandbox_backend"]
        await backend.aedit("/calc.py", "return a - b", "return a + b")
        return {"messages": [*inp["messages"], AIMessage(content="done")]}

    return RunnableLambda(_call)


class _BadPatch(BaseModel):
    """A worktree schema whose `files` field is the WRONG shape (list, not dict)."""

    summary: str
    files: list[str]


def _bad_shape_fixer_builder(*, response_format: object = None) -> Runnable[Any, Any]:
    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        backend = (config or {}).get("configurable", {})["sandbox_backend"]
        await backend.aedit("/calc.py", "return a - b", "return a + b")
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": _BadPatch(summary="x", files=["/calc.py"]),
        }

    return RunnableLambda(_call)


async def test_schemaless_git_worktree_leaf_fails_loud(tmp_path: Path) -> None:
    # H2: a schema-less worktree leaf collects a real diff that can never be
    # surfaced (no `files` field to carry it). The R5 trust boundary would be only
    # half-closed, so the engine must fail loud, not silently drop the changeset.
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    roster = Roster().register("fixer", builder=_schemaless_fixer_builder, needs_execution=True)

    async def orchestrate(ctx: Ctx) -> str:
        return await ctx.agent("fix", agent_type="fixer", isolation="worktree")

    try:
        with pytest.raises(ValueError, match="files"):
            await run_workflow(
                orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-schemaless"
            )
    finally:
        provider.cleanup_all()


async def test_wrong_files_shape_fails_loud_at_fold_not_on_resume(tmp_path: Path) -> None:
    # H1: a worktree schema whose `files` is list[str] must fail loud at FOLD on the
    # first run (when the override is validated), not silently journal a type-
    # mismatched payload that crashes only on a later resume's model_validate_json.
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    roster = Roster().register("fixer", builder=_bad_shape_fixer_builder, needs_execution=True)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> _BadPatch:
        return await ctx.agent("fix", agent_type="fixer", schema=_BadPatch, isolation="worktree")

    try:
        # FOLD-time failure on the FIRST run: pydantic's ValidationError (a ValueError
        # subclass) surfaced loud at fold, not a silently journaled mismatch.
        with pytest.raises(ValueError):
            await run_workflow(
                orchestrate,
                roster=roster,
                sandbox_manager=manager,
                journal=journal,
                thread_id="t-badshape",
            )
        # Success-only journaling: the bad payload never made it into the journal, so a
        # later run re-raises at fold rather than serving a poisoned cache entry that
        # would crash on model_validate_json. (A persisted bad payload would instead
        # short-circuit and crash on the cached-restore path.)
        with pytest.raises(ValueError):
            await run_workflow(
                orchestrate,
                roster=roster,
                sandbox_manager=manager,
                journal=journal,
                thread_id="t-badshape-2",
            )
    finally:
        provider.cleanup_all()


async def test_authoritative_changeset_survives_resume(tmp_path: Path) -> None:
    # H1 happy path: the validated authoritative changeset is journaled and replayed
    # byte-for-byte on resume (the override goes through validation, not model_copy).
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    roster = Roster().register("fixer", builder=_lying_fixer_builder, needs_execution=True)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> _Patch:
        return await ctx.agent("fix calc", agent_type="fixer", schema=_Patch, isolation="worktree")

    try:
        first = await run_workflow(
            orchestrate, roster=roster, sandbox_manager=manager, journal=journal, thread_id="t-r1"
        )
        second = await run_workflow(
            orchestrate, roster=roster, sandbox_manager=manager, journal=journal, thread_id="t-r2"
        )
        # Resume restores the authoritative (disk-truth) changeset, not the lie.
        assert first.files == {"/calc.py": "def add(a, b):\n    return a + b\n"}
        assert second.files == first.files
        assert second.summary == first.summary
    finally:
        provider.cleanup_all()


# --- T7: scratch-repo real `git merge` conflict loop + resume -------------------


class _MergeResult(BaseModel):
    """The outcome of a single scratch-repo three-way ``git merge``.

    Attributes:
        clean: ``True`` when the merge applied with no conflict.
        files: The merged tree (every file's content) on a clean merge; on a
            conflict, the working tree carrying the real ``<<<<<<<`` markers.
        conflicts: ``path -> conflicted content`` for each file git could not
            auto-merge (empty on a clean merge).
    """

    clean: bool
    files: dict[str, str]
    conflicts: dict[str, str]


def scratch_merge(
    base: dict[str, str], ours: dict[str, str], theirs: dict[str, str]
) -> _MergeResult:
    """Run a real three-way ``git merge`` in a throwaway repo (pure, resume-safe).

    Builds a disposable git repo from the inputs alone — commit ``base``, branch
    "ours" committing ``ours``, branch "theirs" committing ``theirs`` — then runs a
    real ``git merge ours <- theirs``. A clean merge returns the merged tree; a real
    conflict returns the working tree carrying git's real ``<<<<<<<`` markers plus a
    per-file conflict map. The function is a pure rebuild from its inputs (no
    persisted state across calls), so a merge leaf that calls it is resume-safe:
    every replay reconstructs the identical scratch repo and the identical result.

    Args:
        base: The common ancestor tree (``path -> content``).
        ours: The integrated-so-far tree to merge into.
        theirs: The incoming patch tree to merge in.

    Returns:
        A :class:`_MergeResult` capturing whether the merge was clean and the
        resulting (merged or conflicted) tree.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ldw-scratch-merge-") as repo:
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "t")

        def _write_tree(tree: dict[str, str], message: str) -> None:
            # Replace the whole tree deterministically so a removed path is honored.
            for existing in Path(repo).iterdir():
                if existing.name == ".git":
                    continue
                existing.unlink()
            for rel, content in tree.items():
                (Path(repo) / rel.lstrip("/")).write_text(content)
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", message, "--allow-empty")

        # Commit the common ancestor, then branch "ours" and "theirs" off it
        # EXPLICITLY by the base commit SHA, so the merge is a genuine three-way
        # merge independent of the init default branch name (main vs master).
        _write_tree(base, "base")
        base_sha = _git_out(repo, "rev-parse", "HEAD").strip()
        _git(repo, "checkout", "-qb", "ours", base_sha)
        _write_tree(ours, "ours")
        _git(repo, "checkout", "-qb", "theirs", base_sha)
        _write_tree(theirs, "theirs")
        _git(repo, "checkout", "-q", "ours")
        merge = subprocess.run(
            ["git", "-C", repo, "merge", "--no-edit", "theirs"],
            capture_output=True,
            text=True,
        )
        files: dict[str, str] = {}
        for path in sorted(Path(repo).rglob("*")):
            if ".git" in path.parts or not path.is_file():
                continue
            rel = "/" + str(path.relative_to(repo))
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        if merge.returncode == 0:
            return _MergeResult(clean=True, files=files, conflicts={})
        # Real conflict: enumerate the unmerged paths git reports.
        unmerged = _git_out(repo, "diff", "--name-only", "--diff-filter=U")
        conflicts = {"/" + rel: files["/" + rel] for rel in unmerged.splitlines() if rel.strip()}
        return _MergeResult(clean=False, files=files, conflicts=conflicts)


def _git_out(cwd: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", cwd, *args], check=True, capture_output=True, text=True
    ).stdout


def _resolve_conflict_markers(conflicted: str) -> str:
    """Deterministically resolve git conflict markers, keeping BOTH sides' intent.

    A real ``git`` conflict hunk looks like::

        <<<<<<< ours
        <ours lines>
        =======
        <theirs lines>
        >>>>>>> theirs

    A deterministic resolver (standing in for an LLM resolver leaf) drops the marker
    lines and keeps both bodies concatenated, so the resolution is reproducible and
    contains both contributions — enough to re-merge cleanly.

    Args:
        conflicted: File content carrying one or more conflict hunks.

    Returns:
        The resolved content with every conflict hunk flattened.
    """
    out: list[str] = []
    in_theirs = False
    for line in conflicted.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("<<<<<<<"):
            in_theirs = False
            continue
        if stripped.startswith("======="):
            in_theirs = True
            continue
        if stripped.startswith(">>>>>>>"):
            in_theirs = False
            continue
        out.append(line)
        _ = in_theirs  # both sides kept; marker lines stripped
    return "".join(out)


def _real_fixer_builder(path: str, old: str, new: str) -> Callable[..., Runnable[Any, Any]]:
    """A worktree fixer that makes a real edit on disk (collect is authoritative)."""

    def _builder(*, response_format: object = None) -> Runnable[Any, Any]:
        async def _call(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            backend = (config or {}).get("configurable", {})["sandbox_backend"]
            await backend.aedit(path, old, new)
            # The model self-reports nothing trustworthy in files; the engine
            # overrides it with the real git diff.
            patch = _Patch(summary=f"edited {path}", files={})
            return {
                "messages": [*inp["messages"], AIMessage(content="done")],
                "structured_response": patch,
            }

        return RunnableLambda(_call)

    return _builder


def _merge_builder(counter: list[int]) -> Callable[..., Runnable[Any, Any]]:
    """A merge leaf that runs the real scratch-repo merge; counts real invocations.

    The leaf reads ``{base, ours, theirs}`` from its (journal-keyed) prompt JSON and
    runs :func:`scratch_merge`. ``counter[0]`` increments on every REAL run, so a
    resume that short-circuits the leaf from the journal leaves the counter
    unchanged — proving no real git re-run.
    """

    def _builder(*, response_format: object = None) -> Runnable[Any, Any]:
        async def _call(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            counter[0] += 1
            payload = json.loads(inp["messages"][-1].content)
            result = scratch_merge(payload["base"], payload["ours"], payload["theirs"])
            return {
                "messages": [*inp["messages"], AIMessage(content="merged")],
                "structured_response": result,
            }

        return RunnableLambda(_call)

    return _builder


class _Resolution(BaseModel):
    files: dict[str, str]


def _resolver_builder(*, response_format: object = None) -> Runnable[Any, Any]:
    """A resolver leaf that flattens conflict markers deterministically."""

    async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        conflicts = json.loads(inp["messages"][-1].content)
        resolved = {path: _resolve_conflict_markers(text) for path, text in conflicts.items()}
        return {
            "messages": [*inp["messages"], AIMessage(content="resolved")],
            "structured_response": _Resolution(files=resolved),
        }

    return RunnableLambda(_call)


def _real_fixer_builder_new(path: str, content: str) -> Callable[..., Runnable[Any, Any]]:
    """A worktree fixer that creates a brand-new file (collect picks it up)."""

    def _builder(*, response_format: object = None) -> Runnable[Any, Any]:
        async def _call(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            backend = (config or {}).get("configurable", {})["sandbox_backend"]
            await backend.awrite(path, content)
            return {
                "messages": [*inp["messages"], AIMessage(content="done")],
                "structured_response": _Patch(summary=f"added {path}", files={}),
            }

        return RunnableLambda(_call)

    return _builder


async def _integrate(
    ctx: Ctx, base: dict[str, str], patches: list[dict[str, str]]
) -> tuple[dict[str, str], bool]:
    """Script-owned conflict loop folding patches into an integrated tree.

    Mirrors the M6 integration pattern: cross-leaf state lives in the script
    variable ``integrated_tree`` (initialized from ``base``), and each patch is
    folded by a journaled merge leaf running a real scratch-repo ``git merge``. A
    real conflict routes through a resolver leaf and re-merges; a clean merge folds
    directly. The script owns the loop — the deterministic control-flow inversion at
    the heart of the engine.

    Args:
        ctx: The orchestration context.
        base: The starting tree (the integration branch's base).
        patches: Each approved patch tree to fold in order.

    Returns:
        ``(integrated_tree, any_conflict)`` — the final tree and whether any merge
        actually hit (and resolved) a real conflict.
    """
    integrated = dict(base)
    any_conflict = False
    for patch in patches:
        # "theirs" is a real branch: the base tree with this patch's files applied,
        # so a patch touching only some files does not appear to delete the rest.
        theirs = {**base, **patch}
        merged = await ctx.agent(
            json.dumps({"base": base, "ours": integrated, "theirs": theirs}, sort_keys=True),
            agent_type="merge",
            schema=_MergeResult,
        )
        if merged.clean:
            integrated = merged.files
            continue
        any_conflict = True
        # Real conflict: a resolver leaf flattens git's markers; the resolved content
        # IS the merge resolution. Fold it into the merged working tree (which already
        # carries the auto-merged non-conflicting files) — completing the merge,
        # exactly as `git add` + `git commit` would after a hand-resolved conflict.
        resolution = await ctx.agent(
            json.dumps(merged.conflicts, sort_keys=True),
            agent_type="resolver",
            schema=_Resolution,
        )
        integrated = dict(merged.files)
        integrated.update(resolution.files)
    return integrated, any_conflict


async def test_swarm_isolation_and_clean_merge_through_run_workflow(tmp_path: Path) -> None:
    # (1) isolation + (2) clean merge: two parallel worktree fixers edit DIFFERENT
    # files; each authoritative changeset (real git diff) is folded; a scratch-repo
    # merge cleanly integrates both into one tree.
    base_files = {"calc.py": "def add(a, b):\n    return a - b\n"}
    base = _make_base_repo(tmp_path, base_files)
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    merge_calls = [0]
    roster = (
        Roster()
        .register(
            "fix_op",
            builder=_real_fixer_builder("/calc.py", "return a - b", "return a + b"),
            needs_execution=True,
        )
        .register(
            "add_helper",
            builder=_real_fixer_builder_new("/helper.py", "HELP = 1\n"),
            needs_execution=True,
        )
        .register("merge", builder=_merge_builder(merge_calls))
        .register("resolver", builder=_resolver_builder)
    )
    base_tree = {"/calc.py": "def add(a, b):\n    return a - b\n"}

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        patches = await ctx.parallel(
            [
                lambda: ctx.agent(
                    "fix operator", agent_type="fix_op", schema=_Patch, isolation="worktree"
                ),
                lambda: ctx.agent(
                    "add helper", agent_type="add_helper", schema=_Patch, isolation="worktree"
                ),
            ]
        )
        trees = [p.files for p in patches if p is not None]
        integrated, conflict = await _integrate(ctx, base_tree, trees)
        return {"integrated": integrated, "conflict": conflict}

    try:
        result = await run_workflow(
            orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-swarm-clean"
        )
        # Each fixer's authoritative diff was isolated (one edited calc, one added
        # helper) and both folded into the integrated tree with no conflict.
        assert result["conflict"] is False
        assert result["integrated"]["/calc.py"] == "def add(a, b):\n    return a + b\n"
        assert result["integrated"]["/helper.py"] == "HELP = 1\n"
    finally:
        provider.cleanup_all()


async def test_conflict_loop_is_actually_taken_and_resolved(tmp_path: Path) -> None:
    # (3) conflict loop (headline): two patches edit the SAME overlapping region of
    # the SAME file -> a real scratch-repo merge conflict -> a resolver leaf resolves
    # the markers -> a re-merge folds cleanly. Assert the conflict path was ACTUALLY
    # taken (not clean-short-circuited) and the final tree carries the resolution.
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    merge_calls = [0]
    roster = (
        Roster()
        .register(
            "fix_plus",
            builder=_real_fixer_builder("/calc.py", "return a - b", "return a + b"),
            needs_execution=True,
        )
        .register(
            "fix_times",
            builder=_real_fixer_builder("/calc.py", "return a - b", "return a * b"),
            needs_execution=True,
        )
        .register("merge", builder=_merge_builder(merge_calls))
        .register("resolver", builder=_resolver_builder)
    )
    base_tree = {"/calc.py": "def add(a, b):\n    return a - b\n"}

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        # Two worktree fixers edit the SAME line in their OWN isolated worktrees.
        plus = await ctx.agent(
            "make it plus", agent_type="fix_plus", schema=_Patch, isolation="worktree"
        )
        times = await ctx.agent(
            "make it times", agent_type="fix_times", schema=_Patch, isolation="worktree"
        )
        integrated, conflict = await _integrate(ctx, base_tree, [plus.files, times.files])
        return {"integrated": integrated, "conflict": conflict}

    try:
        result = await run_workflow(
            orchestrate, roster=roster, sandbox_manager=manager, thread_id="t-conflict"
        )
        # The conflict path was actually taken (the second patch overlapped the first).
        assert result["conflict"] is True
        # The resolved file kept BOTH contributions (the resolver flattened markers),
        # and the final merge folded cleanly so no markers remain.
        final = result["integrated"]["/calc.py"]
        assert "<<<<<<<" not in final and ">>>>>>>" not in final
        assert "return a + b" in final and "return a * b" in final
    finally:
        provider.cleanup_all()


async def test_resume_hits_journal_with_zero_real_git_reruns(tmp_path: Path) -> None:
    # (4) resume: the SAME journal across two run_workflow calls. The first run does
    # the real worktree fixes + real merges; the second run short-circuits every leaf
    # from the journal, so NO real git runs again (the merge counter is unchanged)
    # and the integrated tree is restored byte-for-byte.
    base = _make_base_repo(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    provider = GitWorktreeProvider(base_repo=base)
    manager = SandboxManager(git_worktree_provider=provider)
    merge_calls = [0]
    roster = (
        Roster()
        .register(
            "fix_op",
            builder=_real_fixer_builder("/calc.py", "return a - b", "return a + b"),
            needs_execution=True,
        )
        .register("merge", builder=_merge_builder(merge_calls))
        .register("resolver", builder=_resolver_builder)
    )
    base_tree = {"/calc.py": "def add(a, b):\n    return a - b\n"}
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        patch = await ctx.agent("fix", agent_type="fix_op", schema=_Patch, isolation="worktree")
        integrated, conflict = await _integrate(ctx, base_tree, [patch.files])
        return {"integrated": integrated, "conflict": conflict}

    try:
        first = await run_workflow(
            orchestrate,
            roster=roster,
            sandbox_manager=manager,
            journal=journal,
            thread_id="t-resume-1",
        )
        merges_after_first = merge_calls[0]
        assert merges_after_first >= 1  # the first run really merged
        assert first["integrated"]["/calc.py"] == "def add(a, b):\n    return a + b\n"

        # Second run on the SAME journal: every leaf is a journal hit, so no real
        # git (worktree add OR merge) runs again.
        second = await run_workflow(
            orchestrate,
            roster=roster,
            sandbox_manager=manager,
            journal=journal,
            thread_id="t-resume-2",
        )
        assert merge_calls[0] == merges_after_first  # ZERO real merge re-runs
        assert second["integrated"] == first["integrated"]  # restored byte-for-byte
    finally:
        provider.cleanup_all()

"""A real-git fix swarm with a script-owned merge/conflict loop.

Mirrors `features/worktree` (the in-memory fix swarm), but the leaves run on REAL
``git worktree`` trees on disk and the integration is a real three-way
``git merge``:

    fan out one fixer per target file
      -> each runs in its OWN real git worktree, branched from the base repo
         (isolation="worktree"); the engine folds the real `git diff` into the
         leaf's result as the AUTHORITATIVE changeset (the model's claimed bytes
         are overridden by the on-disk truth)
      -> 2-vote review per patch -> keep the approved patches
      -> the SCRIPT folds the approved patches into an integration tree with a
         deterministic merge leaf running a scratch-repo `git merge`; on a real
         conflict it dispatches a resolver leaf and folds the resolution in.

The control flow lives in the script, not in any model: the loop, the merge order,
and the conflict branch are deterministic Python over journaled leaves. Two of the
fixers edit the SAME overlapping region of the same file, so a real merge conflict
is actually hit and resolved. After the run, the HOST opens a pull request and
sweeps the worktree provider's temp tree. Everything is deterministic and offline —
the demo needs real ``git`` on PATH but no API key.

    uv run python -m examples.features.git_worktree
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    Ctx,
    GitWorktreeProvider,
    LocalPullRequestProvider,
    Roster,
    SandboxManager,
    run_workflow,
)

REVIEWERS_PER_PATCH = 2
APPROVALS_TO_KEEP = 2

# A tiny repo with one genuine bug per file. Two of the fixers below target the
# SAME line of /calc.py in their own isolated worktrees, so the integration step
# must resolve a real git merge conflict.
BASE_FILES: dict[str, str] = {
    "calc.py": "def add(a, b):\n    return a - b  # bug: should be a + b\n",
    "strutil.py": "def shout(s):\n    return s.lower() + '!'  # bug: should be s.upper()\n",
}

# Each fixer targets one file with one deterministic edit (find -> replace). The
# two /calc.py fixers below overlap on the same line on purpose, to exercise the
# conflict path. Keyed by roster agent_type.
FIXES: dict[str, tuple[str, str, str]] = {
    "fix_calc_plus": ("/calc.py", "return a - b  # bug: should be a + b", "return a + b"),
    "fix_calc_doc": (
        "/calc.py",
        "return a - b  # bug: should be a + b",
        "return a - b  # reviewed: keep subtraction for now",
    ),
    "fix_strutil": (
        "/strutil.py",
        "return s.lower() + '!'  # bug: should be s.upper()",
        "return s.upper() + '!'",
    ),
}


# -- structured leaf contracts (schema-as-handoff) ------------------------------


class Patch(BaseModel):
    """A fixer's proposed change to its worktree.

    The engine OVERRIDES ``files`` with the real ``git diff`` of the leaf's
    worktree, so ``files`` always carries the authoritative on-disk changeset
    regardless of what the model self-reports here.

    Attributes:
        summary: A short human-readable description of the change.
        files: ``path -> new content`` for every file the leaf changed. Filled by
            the engine from the real git diff; a model self-report is overridden.
    """

    summary: str
    files: dict[str, str]


class Vote(BaseModel):
    """One reviewer's ruling on a patch."""

    approve: bool
    reason: str


class MergeResult(BaseModel):
    """The outcome of a single scratch-repo three-way ``git merge``.

    Attributes:
        clean: ``True`` when the merge applied with no conflict.
        files: The merged tree on a clean merge; on a conflict, the working tree
            carrying git's real ``<<<<<<<`` markers.
        conflicts: ``path -> conflicted content`` for each file git could not
            auto-merge (empty on a clean merge).
    """

    clean: bool
    files: dict[str, str]
    conflicts: dict[str, str]


class Resolution(BaseModel):
    """A resolver leaf's flattened, marker-free resolution of conflicted files."""

    files: dict[str, str]


# -- the workflow ---------------------------------------------------------------


def _review_prompt(voter: int, patch: Patch) -> str:
    changes = "\n".join(
        f"--- {path} ---\n{content}" for path, content in sorted(patch.files.items())
    )
    return (
        f"Reviewer #{voter + 1}: does this patch make a sensible, self-contained "
        f"change and touch only files it should? Approve only if so.\n"
        f"Summary: {patch.summary}\n{changes}"
    )


async def _integrate(
    ctx: Ctx, base: dict[str, str], patches: list[dict[str, str]]
) -> tuple[dict[str, str], bool]:
    """Script-owned conflict loop folding patches into one integrated tree.

    Cross-leaf state lives in the script variable ``integrated`` (initialized from
    ``base``); each patch is folded by a journaled merge leaf running a real
    scratch-repo ``git merge``. A clean merge folds its merged tree directly; a real
    conflict routes through a resolver leaf whose resolved content is folded straight
    into the merged working tree — completing the merge exactly as ``git add`` +
    ``git commit`` would after a hand-resolved conflict (no second merge pass). The
    SCRIPT owns the loop, the merge order, and the conflict branch — control-flow
    inversion over journaled leaves.

    Args:
        ctx: The orchestration context.
        base: The starting tree (the integration branch's base).
        patches: Each approved patch tree, folded in order.

    Returns:
        ``(integrated_tree, any_conflict)`` — the final tree and whether any merge
        actually hit (and resolved) a real conflict.
    """
    integrated = dict(base)
    any_conflict = False
    for patch in patches:
        # "theirs" is a real branch: the base tree with this patch applied, so a
        # patch touching only some files does not appear to delete the rest.
        theirs = {**base, **patch}
        merged = await ctx.agent(
            json.dumps({"base": base, "ours": integrated, "theirs": theirs}, sort_keys=True),
            agent_type="merge",
            schema=MergeResult,
        )
        if merged.clean:
            integrated = merged.files
            continue
        any_conflict = True
        resolution = await ctx.agent(
            json.dumps(merged.conflicts, sort_keys=True),
            agent_type="resolver",
            schema=Resolution,
        )
        # The resolved content IS the merge resolution; fold it into the merged
        # working tree (which already carries the auto-merged non-conflicting files).
        integrated = dict(merged.files)
        integrated.update(resolution.files)
    return integrated, any_conflict


async def fix_swarm(ctx: Ctx, fixer_types: list[str]) -> dict[str, Any]:
    """Fan out one real-git-worktree fixer per target, review, then integrate.

    Each fixer makes a real edit on its own worktree; the engine folds the real
    ``git diff`` into the patch's ``files`` as the authoritative changeset. A 2-vote
    review keeps the approved patches, then the script folds them into one
    integration tree through a real ``git merge`` conflict loop.

    Args:
        ctx: The workflow context driving ``parallel``/``agent``/``log``/``phase``.
        fixer_types: The roster agent_types to fan out, one isolated worktree each.

    Returns:
        A dict with the approved patches, the ``integrated`` tree, and whether the
        conflict path was ``conflict``-taken.
    """
    ctx.phase("fix")
    patches = await ctx.parallel(
        [
            lambda t=t: ctx.agent(
                f"Fix the bug in your worktree for task {t}.",
                agent_type=t,
                isolation="worktree",
                schema=Patch,
            )
            for t in sorted(fixer_types)
        ]
    )
    collected = [patch for patch in patches if patch is not None]
    ctx.log(f"collected {len(collected)}/{len(fixer_types)} patches")

    ctx.phase("review")
    approved: list[Patch] = []
    for patch in collected:
        votes = await ctx.parallel(
            [
                lambda p=patch, v=v: ctx.agent(
                    _review_prompt(v, p), agent_type="reviewer", schema=Vote
                )
                for v in range(REVIEWERS_PER_PATCH)
            ]
        )
        approvals = sum(1 for vote in votes if vote is not None and vote.approve)
        kept = approvals >= APPROVALS_TO_KEEP
        verdict = "kept" if kept else "dropped"
        ctx.log(f"patch {verdict} ({approvals}/{REVIEWERS_PER_PATCH}): {patch.summary}")
        if kept:
            approved.append(patch)

    ctx.phase("integrate")
    base_tree = {"/" + rel: content for rel, content in BASE_FILES.items()}
    integrated, conflict = await _integrate(ctx, base_tree, [p.files for p in approved])
    ctx.log(f"integrated {len(approved)} patches (conflict path taken: {conflict})")
    return {
        "approved": [{"summary": p.summary, "files": p.files} for p in approved],
        "integrated": integrated,
        "conflict": conflict,
    }


# -- offline leaves (deterministic fakes) ---------------------------------------


def _git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def _git_out(cwd: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", cwd, *args], check=True, capture_output=True, text=True
    ).stdout


def _make_base_repo(root: Path, files: dict[str, str]) -> str:
    """Create a real, committed base git repo seeded with ``files``."""
    repo = root / "base"
    repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "demo@example.com")
    _git(str(repo), "config", "user.name", "ldw-demo")
    for rel, content in files.items():
        (repo / rel).write_text(content)
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "seed buggy codebase")
    return str(repo)


def _fixer_builder(agent_type: str) -> Callable[..., Runnable[Any, Any]]:
    """Build a fixer leaf for one target: edit its real worktree, return a Patch.

    The leaf reads its seeded worktree (proving the worktree was branched from the
    base repo), makes the deterministic edit on disk, and returns a ``Patch``. The
    ``files`` it reports is intentionally empty — the engine overrides it with the
    real ``git diff``, so the changeset the script sees is always the on-disk truth.
    """
    path, old, new = FIXES[agent_type]

    def _builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            backend = (config or {}).get("configurable", {})["sandbox_backend"]
            seeded = (await backend.aread(path)).file_data  # proves the worktree was seeded
            assert seeded is not None and old in seeded["content"], "worktree must be seeded"
            await backend.aedit(path, old, new)
            # The engine overrides files with the real diff; we self-report nothing.
            patch = Patch(summary=f"{agent_type}: edited {path}", files={})
            return {
                "messages": [*inp["messages"], AIMessage(content="patched")],
                "structured_response": patch,
            }

        return RunnableLambda(_leaf)

    return _builder


def _reviewer_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    """Build a reviewer leaf: offline always approves (deterministic)."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        vote = Vote(approve=True, reason="patch is a sensible, self-contained change")
        return {
            "messages": [*inp["messages"], AIMessage(content="reviewed")],
            "structured_response": vote,
        }

    return RunnableLambda(_leaf)


def _scratch_merge(
    base: dict[str, str], ours: dict[str, str], theirs: dict[str, str]
) -> MergeResult:
    """Run a real three-way ``git merge`` in a throwaway repo (pure, resume-safe).

    Builds a disposable git repo from the inputs alone — commit ``base``, branch
    "ours" and "theirs" off the base commit SHA, then run ``git merge``. A clean
    merge returns the merged tree; a real conflict returns the working tree carrying
    git's real ``<<<<<<<`` markers plus a per-file conflict map. The function is a
    pure rebuild from its inputs (no persisted state), so a merge leaf that calls it
    replays identically on resume.

    Args:
        base: The common-ancestor tree (``path -> content``).
        ours: The integrated-so-far tree to merge into.
        theirs: The incoming patch tree to merge in.

    Returns:
        A :class:`MergeResult` for the merge.
    """
    with tempfile.TemporaryDirectory(prefix="ldw-scratch-merge-") as repo:
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "demo@example.com")
        _git(repo, "config", "user.name", "ldw-demo")

        def _write_tree(tree: dict[str, str], message: str) -> None:
            for existing in Path(repo).iterdir():
                if existing.name == ".git":
                    continue
                existing.unlink()
            for rel, content in tree.items():
                (Path(repo) / rel.lstrip("/")).write_text(content)
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", message, "--allow-empty")

        # Branch ours/theirs off the base commit SHA so the merge is a genuine
        # three-way merge, independent of the default branch name (main vs master).
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
        for file in sorted(Path(repo).rglob("*")):
            if ".git" in file.parts or not file.is_file():
                continue
            files["/" + str(file.relative_to(repo))] = file.read_text(
                encoding="utf-8", errors="replace"
            )
        if merge.returncode == 0:
            return MergeResult(clean=True, files=files, conflicts={})
        unmerged = _git_out(repo, "diff", "--name-only", "--diff-filter=U")
        conflicts = {"/" + rel: files["/" + rel] for rel in unmerged.splitlines() if rel.strip()}
        return MergeResult(clean=False, files=files, conflicts=conflicts)


def _merge_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    """Build a merge leaf: runs the real scratch-repo ``git merge`` (deterministic)."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        payload = json.loads(inp["messages"][-1].content)
        result = _scratch_merge(payload["base"], payload["ours"], payload["theirs"])
        return {
            "messages": [*inp["messages"], AIMessage(content="merged")],
            "structured_response": result,
        }

    return RunnableLambda(_leaf)


def _resolve_conflict_markers(conflicted: str) -> str:
    """Flatten git conflict hunks, keeping BOTH sides' bodies (deterministic).

    A real ``git`` conflict hunk looks like::

        <<<<<<< ours
        <ours lines>
        =======
        <theirs lines>
        >>>>>>> theirs

    A deterministic resolver (standing in for an LLM resolver leaf) drops the marker
    lines and keeps both bodies, so the resolution is reproducible and contains both
    contributions.

    Args:
        conflicted: File content carrying one or more conflict hunks.

    Returns:
        The resolved content with every conflict hunk flattened, no markers left.
    """
    out: list[str] = []
    for line in conflicted.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        out.append(line)
    return "".join(out)


def _resolver_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    """Build a resolver leaf: flattens conflict markers deterministically."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        conflicts = json.loads(inp["messages"][-1].content)
        resolved = {path: _resolve_conflict_markers(text) for path, text in conflicts.items()}
        return {
            "messages": [*inp["messages"], AIMessage(content="resolved")],
            "structured_response": Resolution(files=resolved),
        }

    return RunnableLambda(_leaf)


# -- driver ---------------------------------------------------------------------


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ldw-git-worktree-demo-") as tmp:
        base_repo = _make_base_repo(Path(tmp), BASE_FILES)
        provider = GitWorktreeProvider(base_repo=base_repo)
        manager = SandboxManager(git_worktree_provider=provider)

        fixer_types = sorted(FIXES)
        roster = Roster()
        for agent_type in fixer_types:
            roster = roster.register(
                agent_type,
                builder=_fixer_builder(agent_type),
                description=f"Fixes one file on its own git worktree ({agent_type})",
                needs_execution=True,
            )
        roster = (
            roster.register("reviewer", builder=_reviewer_builder, description="Reviews a patch")
            .register("merge", builder=_merge_builder, description="Runs a scratch-repo git merge")
            .register("resolver", builder=_resolver_builder, description="Resolves merge markers")
        )

        async def orchestrate(ctx: Ctx) -> dict[str, Any]:
            return await fix_swarm(ctx, fixer_types)

        # The host owns the provider lifecycle: cleanup_all in a finally sweeps the
        # provider's temp workspace_root and any worktree a teardown path missed.
        try:
            result = await run_workflow(
                orchestrate, roster=roster, sandbox_manager=manager, thread_id="git-worktree-demo"
            )

            approved: list[dict[str, Any]] = result["approved"]
            integrated: dict[str, str] = result["integrated"]
            conflict: bool = result["conflict"]

            print(f"approved patches ({len(approved)}):")
            for patch in approved:
                print(f"  - {patch['summary']} -> {sorted(patch['files'])}")
            print(f"conflict path taken during integration: {conflict}")
            print("integrated tree:")
            for path in sorted(integrated):
                print(f"  - {path}: {integrated[path]!r}")

            # Host finalization: open a PR for the integration branch (after the run
            # returns, never inside the journaled replay).
            pr = LocalPullRequestProvider().open(
                branch="leaf/fix-swarm",
                title="Fix swarm: integrate reviewed patches",
                body="Parallel real-git-worktree fixers, reviewed and merged.",
                integration_branch=provider.integration_branch,
            )
            print(f"opened PR #{pr.number} -> {pr.url} (targets {pr.integration_branch})")

            # -- assertions that PROVE the mechanism (double as the smoke check) --

            # (1) Isolation: each fixer's authoritative changeset is its OWN file only.
            for patch in approved:
                touched = set(patch["files"])
                assert len(touched) == 1, f"a fixer touched >1 file: {touched}"

            # (2) Authoritative collect: the changeset is the real on-disk edit, not a
            # model self-report. The /calc.py "+" fixer's real edit is `return a + b`;
            # the /strutil.py fixer's is `s.upper()`. The doc fixer overlaps the same
            # /calc.py line, so exactly one of the two /calc.py patches carries `a + b`.
            calc_patches = [p for p in approved if "/calc.py" in p["files"]]
            strutil_patches = [p for p in approved if "/strutil.py" in p["files"]]
            assert len(calc_patches) == 2, "both /calc.py fixers should be approved"
            assert len(strutil_patches) == 1, "the /strutil.py fixer should be approved"
            assert any("return a + b" in p["files"]["/calc.py"] for p in calc_patches), (
                "the authoritative changeset must carry the real on-disk + fix"
            )
            assert "s.upper()" in strutil_patches[0]["files"]["/strutil.py"], (
                "the authoritative changeset must carry the real on-disk upper() fix"
            )

            # (3) Conflict path: the two /calc.py patches overlap the same line, so a
            # real merge conflict was hit and resolved (no markers left, both kept).
            assert conflict is True, "the overlapping /calc.py patches must hit a real conflict"
            final_calc = integrated["/calc.py"]
            assert "<<<<<<<" not in final_calc and ">>>>>>>" not in final_calc, (
                "the resolved /calc.py must carry no conflict markers"
            )
            assert "return a + b" in final_calc, "the resolved /calc.py must keep the + fix"
            assert "reviewed: keep subtraction" in final_calc, (
                "the resolved /calc.py must keep the doc-comment contribution too"
            )

            # (4) Integrated tree carries every fix.
            assert "s.upper()" in integrated["/strutil.py"], (
                "integration must carry the upper() fix"
            )

            # (5) The PR was recorded.
            assert pr.created and pr.number >= 1 and pr.url.startswith("local://pr/"), (
                "host finalization must record a pull request"
            )

            print(
                "OK: isolated real-git fixers, authoritative diffs, a resolved real "
                "merge conflict, an integrated tree, and a recorded PR."
            )
        finally:
            provider.cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())

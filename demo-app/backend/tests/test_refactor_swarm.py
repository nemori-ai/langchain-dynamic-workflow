"""Offline checks for the ``refactor_swarm`` preset and its real-git run wiring (M6).

These run with no model key, so the roster serves deterministic fake leaves — but the
fan-out fixers still run in REAL ``git worktree`` backends leased from a real
:class:`GitWorktreeProvider` over a temp repo seeded with the buggy fixture. The fake
fixer makes the real edit on the worktree disk, so the engine's authoritative ``git diff``
collect (not a model self-report) drives the changeset, exactly as a real fixer would.

The headline properties pinned here:

* **fan-out isolation** — two fixers edit the SAME overlapping region of ``/calc.py`` in
  their OWN isolated worktrees, so neither sees the other's edit;
* **the conflict path is ACTUALLY taken and resolved** — folding those two overlapping
  patches hits a real three-way ``git merge`` conflict, a resolver leaf flattens it, and
  the final integrated tree carries no markers (and keeps BOTH fixes); and
* **the PR is host-finalized idempotently** — the workflow returns a pure PR intent, and
  the host (here, the test driving the same path) opens the PR once via
  :class:`LocalPullRequestProvider`, idempotently across a re-run.

The real-model path (a real ``git_fixer`` deepagent editing a worktree) is the gated
real-model E2E; here the fakes drive the edits but the worktree provider, the git diff
collect, and the scratch-repo merge are all REAL.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from host_graph import _ResumeLane, run_workflow_live
from ui_adapter import UiAdapter
from workflows import (
    REFACTOR_BASE_TREE,
    REFACTOR_PR_BRANCH,
    REFACTOR_TARGETS,
    RefactorResult,
    make_reasoning_roster,
    make_roster,
    make_workflows,
    refactor_swarm,
)

from langchain_dynamic_workflow import (
    GitWorktreeProvider,
    LocalPullRequestProvider,
    SandboxManager,
    run_workflow,
)
from langchain_dynamic_workflow._git_worktree import GitWorktreeProvider as _Provider


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the roster serves deterministic fake leaves."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LDW_DEMO_REAL_MODEL"):
        os.environ.pop(key, None)


def _git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def _seed_base_repo(tmp_path: Path) -> str:
    """Seed a real git repo with the refactor fixture (the buggy base tree)."""
    repo = tmp_path / "base"
    repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t")
    _git(str(repo), "config", "user.name", "t")
    for path, content in REFACTOR_BASE_TREE.items():
        (repo / path.lstrip("/")).write_text(content)
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "seed")
    return str(repo)


def _make_worktree_manager(tmp_path: Path) -> tuple[GitWorktreeProvider, SandboxManager]:
    """Build a real GitWorktreeProvider + a SandboxManager wired to it."""
    provider = GitWorktreeProvider(base_repo=_seed_base_repo(tmp_path))
    manager = SandboxManager(git_worktree_provider=provider)
    return provider, manager


def test_make_workflows_registers_refactor_swarm() -> None:
    """The registry resolves the ``refactor_swarm`` preset by name."""
    assert make_workflows().resolve("refactor_swarm") is refactor_swarm


def test_fixture_targets_are_designed_to_conflict() -> None:
    """At least one pair of fix targets edits the SAME file (the headline conflict path).

    The conflict story depends on two fixers touching the same file's overlapping region.
    This pins the fixture so a future edit that accidentally spreads every target across
    distinct files (a no-conflict swarm) fails this guard loudly.
    """
    by_path: dict[str, int] = {}
    for target in REFACTOR_TARGETS:
        by_path[target.path] = by_path.get(target.path, 0) + 1
    conflicting = [path for path, count in by_path.items() if count >= 2]
    assert conflicting, f"no two targets share a file; fixture cannot conflict: {by_path}"
    # And the base tree actually carries the bugs those targets fix.
    for target in REFACTOR_TARGETS:
        assert target.old in REFACTOR_BASE_TREE[target.path], target


async def test_refactor_swarm_isolation_conflict_resolved_through_run_workflow(
    tmp_path: Path,
) -> None:
    """The full swarm: isolated fan-out, a REAL conflict resolved, no markers remain.

    Drives ``refactor_swarm`` through ``run_workflow`` against a real GitWorktreeProvider.
    The two ``/calc.py`` fixers run in isolated worktrees and rewrite the SAME line two
    different ways, so the integrate fold hits a real ``git merge`` conflict; the resolver
    flattens it, and the final integrated tree keeps BOTH competing fixes (``a + b`` and
    ``abs(a) + abs(b)``) with no markers.
    """
    provider, manager = _make_worktree_manager(tmp_path)
    try:
        result = await run_workflow(
            lambda ctx: refactor_swarm(ctx, {}),
            roster=make_roster(),
            workflows=make_workflows(),
            sandbox_manager=manager,
            thread_id="t-refactor-swarm",
        )
        assert isinstance(result, RefactorResult)
        # (1) The conflict path was ACTUALLY taken and resolved (two patches overlapped).
        assert result.conflict_resolved is True
        # (2) The final tree carries NO conflict markers (the resolver flattened them) and
        # keeps BOTH competing /calc.py fixes — proving isolated fixers' work folded together.
        calc = result.integrated_tree["/calc.py"]
        assert "<<<<<<<" not in calc and ">>>>>>>" not in calc
        assert "return a + b" in calc, calc  # fixer A's correction
        assert "return abs(a) + abs(b)" in calc, calc  # fixer B's correction
        # (3) The clean (non-conflicting) /helper.py patch also folded in.
        assert result.integrated_tree["/helper.py"] == "VALUE = 42\n"
        # Every patch was approved offline (the judge never refutes), one per target.
        assert len(result.approved) == len(REFACTOR_TARGETS)
        assert result.rejected == []
        # The workflow returns a PR INTENT — it does NOT open the PR itself (R1).
        assert result.pr.branch == REFACTOR_PR_BRANCH
        assert result.pr.title
        assert result.pr.body
    finally:
        provider.cleanup_all()


async def test_two_calc_fixers_are_isolated_in_their_own_worktrees(tmp_path: Path) -> None:
    """Each ``/calc.py`` fixer's authoritative diff carries ONLY its own correction.

    Fan-out isolation made explicit: the engine folds the real ``git diff`` of each leaf's
    OWN worktree into its ``GitPatch.files``. The two fixers rewrite the SAME line two
    different ways in SEPARATE worktrees, so fixer A's changeset must carry ``a + b`` and
    NOT ``abs(...)``, and fixer B's must carry ``abs(a) + abs(b)`` and NOT plain ``a + b`` —
    if they shared a tree the diffs would cross-contaminate (and there'd be no conflict).
    """
    provider, manager = _make_worktree_manager(tmp_path)

    # Run the two competing calc fixers directly through the engine to inspect each diff.
    from workflows import GitPatch, _refactor_fix_prompt

    plus_target, abs_target = REFACTOR_TARGETS[0], REFACTOR_TARGETS[1]
    try:

        async def two_fixers(ctx: Any) -> tuple[GitPatch, GitPatch]:
            patches = await ctx.parallel(
                [
                    lambda: ctx.agent(
                        _refactor_fix_prompt(plus_target),
                        agent_type="git_fixer",
                        schema=GitPatch,
                        isolation="worktree",
                    ),
                    lambda: ctx.agent(
                        _refactor_fix_prompt(abs_target),
                        agent_type="git_fixer",
                        schema=GitPatch,
                        isolation="worktree",
                    ),
                ]
            )
            return patches[0], patches[1]  # type: ignore[return-value]

        plus_patch, abs_patch = await run_workflow(
            two_fixers,
            roster=make_roster(),
            sandbox_manager=manager,
            thread_id="t-isolation",
        )
        # Each leaf's authoritative diff carries ONLY its own correction (isolated worktrees).
        plus_calc = plus_patch.files["/calc.py"]
        abs_calc = abs_patch.files["/calc.py"]
        assert "return a + b" in plus_calc and "abs(" not in plus_calc, plus_calc
        assert "return abs(a) + abs(b)" in abs_calc, abs_calc
    finally:
        provider.cleanup_all()


async def test_host_finalizes_pr_idempotently_after_run(tmp_path: Path) -> None:
    """The host opens the PR once (after the run), idempotently across a re-run (R1).

    The workflow returns only the PR intent; the HOST opens the PR via
    LocalPullRequestProvider after ``run_workflow`` returns. Opening the same branch twice
    must return the SAME PR (``created=False`` on the second), never a duplicate — so host
    finalization is safe to run unconditionally on every turn / resume.
    """
    provider, manager = _make_worktree_manager(tmp_path)
    pr_provider = LocalPullRequestProvider()
    try:
        result = await run_workflow(
            lambda ctx: refactor_swarm(ctx, {}),
            roster=make_roster(),
            workflows=make_workflows(),
            sandbox_manager=manager,
            thread_id="t-pr-finalize",
        )
        assert isinstance(result, RefactorResult)
        first = pr_provider.open(
            branch=result.pr.branch,
            title=result.pr.title,
            body=result.pr.body,
            integration_branch=provider.integration_branch,
        )
        second = pr_provider.open(
            branch=result.pr.branch,
            title=result.pr.title,
            body=result.pr.body,
            integration_branch=provider.integration_branch,
        )
        assert first.created is True
        assert second.created is False  # idempotent re-open, no duplicate PR
        assert second.number == first.number
        assert first.integration_branch == provider.integration_branch
    finally:
        provider.cleanup_all()


async def test_run_workflow_live_runs_refactor_swarm_and_finalizes_pr_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_workflow_live`` runs the swarm against a worktree provider and emits a PR card.

    The host path: ``run_workflow_live("refactor_swarm", ...)`` constructs a real
    GitWorktreeProvider over a seeded temp repo, runs the swarm, opens the PR after the run
    (host finalization), emits a ``pull_request`` Gen-UI card, and tears the provider down.
    Here the temp base repo is injected so the test owns its lifecycle; the assertion is
    that a ``pull_request`` card carrying the PR ref is emitted AFTER the run completes.
    """
    base_repo = _seed_base_repo(tmp_path)
    # The host builds its own provider over a base repo; inject the seeded one so the test
    # controls the fixture's git state without reaching into the host's temp-dir creation.
    monkeypatch.setenv("LDW_DEMO_REFACTOR_BASE_REPO", base_repo)

    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))
    lane = _ResumeLane(thread_id="t::refactor_swarm")

    done = await run_workflow_live("refactor_swarm", {}, adapter=adapter, lane=lane)
    assert "refactor" in done.lower() or "pull request" in done.lower() or "pr" in done.lower()

    pr_cards = [p for c, p in events if c == "pull_request"]
    assert pr_cards, f"a pull_request card must be emitted after the run; events={events}"
    card = pr_cards[-1]
    assert card["branch"] == REFACTOR_PR_BRANCH
    assert card.get("number") is not None
    assert card.get("url")
    assert card.get("integration_branch")


def test_git_fixer_is_host_trusted_only_not_in_reasoning_roster() -> None:
    """The needs_execution ``git_fixer`` is on the host roster ONLY, never reasoning (R6).

    The AST gate does not constrain an authored script's ``agent_type``, so the
    git-worktree execution leaf must be unreachable from the untrusted reasoning roster —
    otherwise an authored ``ctx.agent(agent_type="git_fixer")`` could lease a real worktree.
    """
    full = make_roster()
    reasoning = make_reasoning_roster()
    # The trusted roster resolves git_fixer; the reasoning roster must NOT.
    full.resolve("git_fixer")
    full.resolve("conflict_resolver")
    full.resolve("merge")
    with pytest.raises(KeyError):
        reasoning.resolve("git_fixer")
    with pytest.raises(KeyError):
        reasoning.resolve("conflict_resolver")
    with pytest.raises(KeyError):
        reasoning.resolve("merge")


def test_provider_type_is_engine_export() -> None:
    """``GitWorktreeProvider`` is imported from the engine root (a stable public seam)."""
    assert _Provider is GitWorktreeProvider

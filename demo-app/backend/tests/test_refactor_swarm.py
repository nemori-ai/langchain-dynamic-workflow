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


# --- F-D3: conflict resolution is validated, retried, and fails loud (not blindly trusted) -


def _markered_then_clean_resolver_builder(calls: list[int]) -> Any:
    """A conflict_resolver fake that returns marker-laden files on attempt 1, clean on 2.

    Mirrors a real resolver that botches its first try: attempt 1 echoes the conflicted
    content verbatim (markers intact), attempt 2 flattens them. ``calls[0]`` counts real
    invocations so the test can assert the script RETRIED rather than accepting the first.
    """
    from workflows import Resolution, _flatten_conflict_markers, _parse_conflicts_from_prompt

    def _builder(*, response_format: Any = None) -> Any:
        async def _leaf(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
            from langchain_core.messages import AIMessage

            calls[0] += 1
            prompt = inp["messages"][-1].text if inp["messages"] else ""
            conflicts = _parse_conflicts_from_prompt(prompt)
            if calls[0] == 1:
                # Attempt 1: BLINDLY return the conflicted content with markers intact.
                resolved = dict(conflicts)
            else:
                # Attempt 2: actually flatten the markers (a clean resolution).
                resolved = {p: _flatten_conflict_markers(t) for p, t in conflicts.items()}
            return {
                "messages": [*inp["messages"], AIMessage(content="resolved")],
                "structured_response": Resolution(files=resolved),
            }

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_leaf)

    return _builder


def _never_clean_resolver_builder() -> Any:
    """A conflict_resolver fake that ALWAYS leaves markers (never produces a clean tree)."""
    from workflows import Resolution, _parse_conflicts_from_prompt

    def _builder(*, response_format: Any = None) -> Any:
        async def _leaf(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
            from langchain_core.messages import AIMessage

            prompt = inp["messages"][-1].text if inp["messages"] else ""
            conflicts = _parse_conflicts_from_prompt(prompt)
            return {
                "messages": [*inp["messages"], AIMessage(content="resolved")],
                "structured_response": Resolution(files=dict(conflicts)),  # markers intact
            }

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_leaf)

    return _builder


def _roster_with_resolver(resolver_builder: Any) -> Any:
    """Build the full host roster with ``conflict_resolver`` swapped for ``resolver_builder``."""
    roster = make_roster()
    roster.register(
        "conflict_resolver",
        builder=resolver_builder,
        description="test fixture resolver",
    )
    return roster


async def test_resolver_retries_when_first_attempt_leaves_markers(tmp_path: Path) -> None:
    """A resolver that returns markers on attempt 1 then clean on attempt 2 → retry succeeds.

    The fold must NOT blindly trust the resolver: it validates the resolved files carry no
    conflict markers and, on failure, retries the resolver leaf with feedback. A fake that
    botches its first try and fixes it on the second proves the retry loop runs and lands a
    clean tree (no markers), exactly M5's "gate on the real artifact, not the model's word".
    """
    calls = [0]
    provider, manager = _make_worktree_manager(tmp_path)
    roster = _roster_with_resolver(_markered_then_clean_resolver_builder(calls))
    try:
        result = await run_workflow(
            lambda ctx: refactor_swarm(ctx, {}),
            roster=roster,
            workflows=make_workflows(),
            sandbox_manager=manager,
            thread_id="t-resolver-retry",
        )
        assert isinstance(result, RefactorResult)
        assert result.conflict_resolved is True
        # The resolver was invoked at least twice (attempt 1 markered → retry → attempt 2 clean).
        assert calls[0] >= 2, f"the fold must RETRY a marker-laden resolution; calls={calls[0]}"
        # The final tree carries no markers (the validated clean resolution folded in).
        calc = result.integrated_tree["/calc.py"]
        assert "<<<<<<<" not in calc and "=======" not in calc and ">>>>>>>" not in calc, calc
    finally:
        provider.cleanup_all()


async def test_resolver_that_never_cleans_fails_loud(tmp_path: Path) -> None:
    """A resolver that always leaves markers exhausts the retry bound and FAILS LOUD.

    The fold must never fold a marker-laden tree (a broken merge) or open a PR over it. A
    resolver that never produces a clean tree must raise a clear error after the bound, not
    silently integrate conflict markers — the fail-loud half of "don't trust the model".
    """
    provider, manager = _make_worktree_manager(tmp_path)
    roster = _roster_with_resolver(_never_clean_resolver_builder())
    try:
        with pytest.raises(ValueError, match="conflict"):
            await run_workflow(
                lambda ctx: refactor_swarm(ctx, {}),
                roster=roster,
                workflows=make_workflows(),
                sandbox_manager=manager,
                thread_id="t-resolver-never-clean",
            )
    finally:
        provider.cleanup_all()


async def test_resolver_that_drops_a_conflicted_file_fails_loud(tmp_path: Path) -> None:
    """A resolver that omits a conflicted path is rejected (every conflict must be resolved)."""
    from workflows import Resolution

    def _dropping_builder(*, response_format: Any = None) -> Any:
        async def _leaf(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
            from langchain_core.messages import AIMessage

            # Return an EMPTY resolution: every conflicted path is missing.
            return {
                "messages": [*inp["messages"], AIMessage(content="resolved")],
                "structured_response": Resolution(files={}),
            }

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_leaf)

    provider, manager = _make_worktree_manager(tmp_path)
    roster = _roster_with_resolver(_dropping_builder)
    try:
        with pytest.raises(ValueError, match="conflict"):
            await run_workflow(
                lambda ctx: refactor_swarm(ctx, {}),
                roster=roster,
                workflows=make_workflows(),
                sandbox_manager=manager,
                thread_id="t-resolver-drops-file",
            )
    finally:
        provider.cleanup_all()


# --- F-D1: host finalization MATERIALIZES the integrated tree into the integration branch -


def _git_show(repo: str, ref: str, path: str) -> str:
    """Return the content of ``path`` at ``ref`` in ``repo`` via ``git show`` (or '' if absent)."""
    completed = subprocess.run(
        ["git", "-C", repo, "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
    )
    return completed.stdout if completed.returncode == 0 else ""


async def test_host_materializes_integrated_tree_into_integration_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host run writes the integrated tree to the integration branch in the base repo.

    F-D1: opening a PR ref is hollow unless a real branch carries the merged files. The host
    must materialize ``result.integrated_tree`` into ``provider.integration_branch`` in the
    base repo (real ``git`` commit) BEFORE opening the PR ref, so the PR references a branch
    that actually carries the merged content. Asserts ``git show <integration_branch>:calc.py``
    equals the merged calc (both competing fixes, no markers) and that helper.py is present.
    """
    base_repo = _seed_base_repo(tmp_path)
    monkeypatch.setenv("LDW_DEMO_REFACTOR_BASE_REPO", base_repo)

    events: list[tuple[str, dict[str, Any]]] = []
    adapter = UiAdapter(emit=lambda comp, props: events.append((comp, dict(props))))
    lane = _ResumeLane(thread_id="t::refactor_swarm_materialize")

    await run_workflow_live("refactor_swarm", {}, adapter=adapter, lane=lane)

    # The integration branch (default ldw/integration) must exist in the base repo and carry
    # the integrated tree — the merged calc.py keeps BOTH competing fixes with no markers.
    integration_branch = "ldw/integration"
    calc = _git_show(base_repo, integration_branch, "calc.py")
    assert calc, (
        f"the integration branch {integration_branch!r} must carry calc.py in the base repo; "
        "host finalization must MATERIALIZE the integrated tree, not just open a hollow PR ref"
    )
    assert "<<<<<<<" not in calc and ">>>>>>>" not in calc, calc
    assert "return a + b" in calc and "return abs(a) + abs(b)" in calc, calc
    # The clean helper.py fix is materialized too.
    helper = _git_show(base_repo, integration_branch, "helper.py")
    assert helper == "VALUE = 42\n", helper


# --- F-D2: the host-created base repo temp dir is cleaned up (no leak) --------------------


async def test_host_created_base_repo_is_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host run that creates its own base repo leaves no ``ldw-refactor-base-*`` dir behind.

    F-D2: ``_seed_refactor_base_repo`` mkdtemps ``ldw-refactor-base-*`` and ``cleanup_all``
    does NOT reclaim it; the host must remove it in ``finally`` (only when host-created, never
    a test-injected one). With no injected repo, the host creates its own — and must clean it
    up. Asserts no new ``ldw-refactor-base-*`` temp dir survives the run.
    """
    import tempfile as _tempfile

    # Ensure no injected repo, so the host creates (and must clean up) its own.
    monkeypatch.delenv("LDW_DEMO_REFACTOR_BASE_REPO", raising=False)
    tmp_root = Path(_tempfile.gettempdir())
    before = {p.name for p in tmp_root.glob("ldw-refactor-base-*")}

    adapter = UiAdapter(emit=lambda _c, _p: None)
    lane = _ResumeLane(thread_id="t::refactor_swarm_cleanup")
    await run_workflow_live("refactor_swarm", {}, adapter=adapter, lane=lane)

    after = {p.name for p in tmp_root.glob("ldw-refactor-base-*")}
    leaked = after - before
    assert not leaked, f"host run leaked base-repo temp dir(s): {leaked}"


async def test_injected_base_repo_is_NOT_removed_by_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test-injected base repo (via env) is the test's to own — the host must NOT remove it."""
    base_repo = _seed_base_repo(tmp_path)
    monkeypatch.setenv("LDW_DEMO_REFACTOR_BASE_REPO", base_repo)

    adapter = UiAdapter(emit=lambda _c, _p: None)
    lane = _ResumeLane(thread_id="t::refactor_swarm_injected")
    await run_workflow_live("refactor_swarm", {}, adapter=adapter, lane=lane)

    # The injected repo must still exist after the run (the host did not delete it).
    assert Path(base_repo).is_dir(), "the host must NOT remove a test-injected base repo"


# --- F-D4: cross-turn PR idempotency is realized through the host path ---------------------


async def test_host_path_pr_idempotent_across_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two host runs on the same branch → the second PR card is created=False (idempotent).

    F-D4: the production entry passes no ``pr_provider``, so a fresh provider per turn would
    always return ``created=True``/``number=1`` and the "PR updated" UI state could never
    render. A module-scope PR provider must persist across turns so the second finalization
    of the same branch returns the existing PR (``created=False``). Driven through the host
    path (``run_refactor_swarm_live`` with no injected provider) so it proves the production
    wiring, not just an injected provider.
    """
    import host_graph

    base_repo = _seed_base_repo(tmp_path)
    monkeypatch.setenv("LDW_DEMO_REFACTOR_BASE_REPO", base_repo)
    # Reset the module-scope PR provider so this test starts from a clean PR namespace.
    host_graph._REFACTOR_PR_PROVIDER = host_graph.LocalPullRequestProvider()

    cards: list[dict[str, Any]] = []
    adapter = UiAdapter(emit=lambda c, p: cards.append(dict(p)) if c == "pull_request" else None)
    lane = _ResumeLane(thread_id="t::refactor_swarm_idem")

    await host_graph.run_refactor_swarm_live({}, adapter=adapter, lane=lane)
    first_card = cards[-1]
    assert first_card["created"] is True, f"first finalization must create the PR: {first_card}"

    # A second host run on the SAME branch must re-open the SAME PR (created=False), proving
    # the PR provider persists across turns (module scope), not rebuilt per turn.
    adapter2 = UiAdapter(emit=lambda c, p: cards.append(dict(p)) if c == "pull_request" else None)
    lane2 = _ResumeLane(thread_id="t::refactor_swarm_idem")
    await host_graph.run_refactor_swarm_live({}, adapter=adapter2, lane=lane2)
    second_card = cards[-1]
    assert second_card["created"] is False, (
        f"second finalization of the same branch must re-open the SAME PR (created=False); "
        f"got {second_card}"
    )
    assert second_card["number"] == first_card["number"], "the re-opened PR must keep its number"


# --- F-D5: the worktree fixer backend runs under the same admission policy as code_fixer --


async def test_refactor_provider_constructed_with_exec_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host builds the GitWorktreeProvider with ``_make_exec_policy()`` (admission + cap).

    F-D5: a real ``git_fixer`` deepagent carries an ``execute`` tool, so its worktree backend
    must run under the SAME admission hook + output cap as the fix_loop's ``code_fixer`` (the
    denylist rejecting ``rm -rf`` / ``curl|sh`` / fork bombs / sudo / net egress + the output
    cap), not a bare default policy. Spies on the GitWorktreeProvider constructor and asserts
    a non-None ``policy`` carrying the admission hook is passed.
    """
    import host_graph

    captured: dict[str, Any] = {}
    real_provider_cls = host_graph.GitWorktreeProvider

    def _spy_provider(*args: Any, **kwargs: Any) -> Any:
        captured["policy"] = kwargs.get("policy")
        return real_provider_cls(*args, **kwargs)

    monkeypatch.setattr(host_graph, "GitWorktreeProvider", _spy_provider)
    monkeypatch.setenv("LDW_DEMO_REFACTOR_BASE_REPO", _seed_base_repo(tmp_path))

    adapter = UiAdapter(emit=lambda _c, _p: None)
    lane = _ResumeLane(thread_id="t::refactor_swarm_policy")
    await host_graph.run_refactor_swarm_live({}, adapter=adapter, lane=lane)

    policy = captured.get("policy")
    assert policy is not None, "the GitWorktreeProvider must be built with an ExecPolicy (F-D5)"
    assert policy.before_execute is not None, (
        "the worktree provider's policy must carry the admission hook (the same as code_fixer)"
    )


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

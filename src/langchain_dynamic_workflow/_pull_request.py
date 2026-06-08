"""Pull-request seam for idempotent host finalization of a workflow's result.

Materializing a pull request is a side effect that must NOT live inside the
deterministic replay: a content-hash journal short-circuits a completed leaf on
resume and returns its journaled result without re-running it, so a PR opened
inside the replayed orchestration would either be skipped (the leaf is cached) or
duplicated (a fresh journal re-runs it). Instead the workflow returns only its
pure, journaled ``integrated_tree``, and the host opens the PR once, after
``run_workflow`` returns, through this seam. The offline
:class:`LocalPullRequestProvider` is idempotent per branch so that host
finalization can run unconditionally and safely on every turn.

A real GitHub-backed provider is intentionally NOT shipped in the engine core; it
is a host concern (it spawns ``gh`` / pushes branches — a dangerous, networked
side effect). The shape such an implementation takes is sketched in
:class:`LocalPullRequestProvider`'s notes: push the branch, then ``gh pr create``
with a check-existing-then-create guard so a re-run does not open a duplicate PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PullRequestRef:
    """An immutable reference to a (real or local) pull request.

    Attributes:
        number: The PR number (monotonically assigned by the provider).
        branch: The source branch the PR is opened from.
        url: The PR's address (``local://pr/<number>`` for the offline provider).
        integration_branch: The branch the PR targets (the integration branch, not
            ``main``).
        created: ``True`` when this call newly created the PR; ``False`` when an
            existing PR for the same branch was returned (idempotent re-open).
    """

    number: int
    branch: str
    url: str
    integration_branch: str
    created: bool


@runtime_checkable
class PullRequestProvider(Protocol):
    """The host-finalization seam for opening a pull request.

    Structurally satisfied by :class:`LocalPullRequestProvider` (the offline
    default) and by any host-supplied real implementation (for example a
    ``gh``-backed one). Declared as a Protocol so a host can type against the seam
    and swap the concrete provider without the engine importing a networked one.
    """

    def open(
        self, *, branch: str, title: str, body: str, integration_branch: str
    ) -> PullRequestRef:
        """Open (or return an existing) pull request for ``branch``.

        Implementations must be idempotent per branch so host finalization can run
        unconditionally on every turn without minting duplicate PRs.

        Args:
            branch: The source branch to open the PR from.
            title: The PR title.
            body: The PR description body.
            integration_branch: The branch the PR targets (not ``main``).

        Returns:
            A :class:`PullRequestRef`; ``created`` distinguishes a fresh PR from an
            idempotent re-open of an existing one for the same branch.
        """
        ...


class LocalPullRequestProvider:
    """An in-process, offline pull-request provider (idempotent per branch).

    Records each PR as a local in-memory entry keyed by source branch and returns a
    ``local://pr/<number>`` ref. Re-opening the same branch returns the original
    ref with ``created=False`` rather than minting a duplicate, so host
    finalization (which runs after every ``run_workflow`` return, including a
    resume) is safe to call unconditionally. Used by tests, offline demos, and the
    demo-app's host finalization.

    A real GitHub-backed provider would implement the same :meth:`open` contract by
    pushing the branch and running ``gh pr create``, guarded by a
    ``gh pr list --head <branch>`` check so a re-run returns the existing PR
    instead of opening a second one — the same idempotency this offline provider
    enforces in memory.
    """

    def __init__(self) -> None:
        # Keyed by source branch so a re-open of the same branch is idempotent.
        self._by_branch: dict[str, PullRequestRef] = {}
        # Monotonic PR numbering, assigned only when a branch is first opened.
        self._next_number = 1

    def open(
        self, *, branch: str, title: str, body: str, integration_branch: str
    ) -> PullRequestRef:
        """Open or return the PR for ``branch`` (idempotent).

        Args:
            branch: The source branch to open the PR from.
            title: The PR title (unused by the offline record, kept for the seam).
            body: The PR description (unused by the offline record, kept for the seam).
            integration_branch: The branch the PR targets.

        Returns:
            The branch's :class:`PullRequestRef`; ``created=True`` on first open for
            the branch, ``created=False`` on a subsequent idempotent re-open.
        """
        existing = self._by_branch.get(branch)
        if existing is not None:
            # Idempotent: same branch -> the original PR, flagged as not newly created.
            return PullRequestRef(
                number=existing.number,
                branch=existing.branch,
                url=existing.url,
                integration_branch=existing.integration_branch,
                created=False,
            )
        number = self._next_number
        self._next_number += 1
        ref = PullRequestRef(
            number=number,
            branch=branch,
            url=f"local://pr/{number}",
            integration_branch=integration_branch,
            created=True,
        )
        self._by_branch[branch] = ref
        return ref

"""Unit tests for the offline ``LocalPullRequestProvider`` and the PR seam.

PR materialization is an idempotent host finalization step (moved out of the
deterministic replay), so the offline provider must be idempotent per branch: a
re-open of the same branch returns the original ref with ``created=False`` rather
than minting a duplicate PR. This is the anti-corruption floor for "PR finalization
is replay-safe".
"""

from __future__ import annotations

from langchain_dynamic_workflow._pull_request import (
    LocalPullRequestProvider,
    PullRequestProvider,
    PullRequestRef,
)


def test_open_records_and_is_idempotent() -> None:
    provider = LocalPullRequestProvider()
    r1 = provider.open(branch="leaf/x", title="t", body="b", integration_branch="ldw/integration")
    assert r1.created and r1.number == 1 and r1.branch == "leaf/x"
    assert r1.integration_branch == "ldw/integration"
    assert r1.url == "local://pr/1"
    # Same branch -> same PR, no duplicate (idempotent host finalization).
    r2 = provider.open(branch="leaf/x", title="t2", body="b2", integration_branch="ldw/integration")
    assert not r2.created and r2.number == 1 and r2.url == "local://pr/1"
    # A different branch mints the next number.
    r3 = provider.open(branch="leaf/y", title="t", body="b", integration_branch="ldw/integration")
    assert r3.created and r3.number == 2


def test_local_provider_satisfies_the_protocol() -> None:
    # The concrete offline provider structurally satisfies the seam, so a host can
    # type against the Protocol and swap in a real gh-backed implementation.
    provider: PullRequestProvider = LocalPullRequestProvider()
    ref = provider.open(branch="leaf/z", title="t", body="b", integration_branch="ldw/integration")
    assert isinstance(ref, PullRequestRef)


def test_ref_is_frozen() -> None:
    ref = PullRequestRef(
        number=1,
        branch="leaf/x",
        url="local://pr/1",
        integration_branch="ldw/integration",
        created=True,
    )
    try:
        ref.number = 2  # type: ignore[misc]
    except Exception as error:  # frozen dataclass raises on mutation
        assert "frozen" in str(error).lower() or "cannot assign" in str(error).lower()
    else:
        raise AssertionError("PullRequestRef must be immutable (frozen dataclass)")

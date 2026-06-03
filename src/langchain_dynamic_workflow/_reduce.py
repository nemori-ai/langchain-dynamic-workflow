"""Cross-leaf reduce — first-class helpers for folding many leaves' outputs into one.

The orchestration script fans out N leaves via ``ctx.parallel`` / ``ctx.pipeline``
and gets back a result list (a failed leaf is ``None``). These helpers turn the
recurring cross-leaf reductions — refute-by-default voting, de-duplication,
dual-blind reviewer reconciliation, cross-leaf corroboration — into named, tested,
fail-safe functions, so a script author never re-derives (and re-breaks) the
None-counting arithmetic. They are pure: no ``agent()`` call and no engine state,
so they are inherently replay-safe and never touch the journal or determinism guard.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, overload


def survives[T](votes: Sequence[T | None], *, against: Callable[[T], bool], kill_at: int) -> bool:
    """Return whether a thing survives a refute-by-default vote.

    Survives iff fewer than ``kill_at`` votes are 'against'. A ``None`` vote (a
    failed or absent leaf) ALWAYS counts as 'against' — the fail-safe so nothing is
    confirmed on missing verification. Covers adversarial-verify
    (``against=lambda v: v.refuted``) and judge-panel (``against=lambda v: not v.sound``).

    Args:
        votes: The leaves' verdicts in fan-out order; ``None`` marks a failed leaf.
        against: Predicate returning ``True`` when a (non-None) vote is against.
        kill_at: The number of 'against' votes that kills it (must be >= 1).

    Returns:
        ``True`` if the 'against' tally is below ``kill_at``.

    Raises:
        ValueError: If ``votes`` is empty (no verification ran), ``kill_at < 1``, or
            ``kill_at`` exceeds the vote count (an unreachable threshold that would
            confirm unconditionally — even an all-failed panel — defeating the
            fail-safe).
    """
    if not votes:
        raise ValueError("survives() requires at least one vote; got an empty sequence")
    if kill_at < 1:
        raise ValueError(f"kill_at must be >= 1, got {kill_at}")
    if kill_at > len(votes):
        raise ValueError(
            f"kill_at ({kill_at}) exceeds the vote count ({len(votes)}); the kill "
            "threshold is unreachable, so the thing would survive unconditionally "
            "(including an all-failed panel) — defeating the None fail-safe"
        )
    against_count = sum(1 for vote in votes if vote is None or against(vote))
    return against_count < kill_at


@overload
def dedup[H: Hashable](items: Iterable[H | None], *, key: None = ...) -> list[H]: ...
@overload
def dedup[T, K: Hashable](items: Iterable[T | None], *, key: Callable[[T], K]) -> list[T]: ...
def dedup(items: Iterable[Any], *, key: Callable[[Any], Hashable] | None = None) -> list[Any]:
    """Drop ``None`` and de-duplicate, preserving first-seen order.

    Args:
        items: The leaves' outputs; ``None`` (a failed leaf) is dropped.
        key: Maps an item to its identity (e.g. ``str.lower`` to merge case
            variants). Without it, the item itself is the key (items must be
            Hashable — enforced by the no-key overload, mirroring ``sorted(key=None)``).

    Returns:
        The kept items in first-seen order, one per distinct key.
    """
    seen: set[Hashable] = set()
    kept: list[Any] = []
    for item in items:
        if item is None:
            continue
        identity = item if key is None else key(item)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(item)
    return kept


@dataclass(frozen=True)
class ReviewItem[T, V]:
    """One item plus every reviewer's verdict on it (``None`` = that reviewer failed)."""

    item: T
    verdicts: Sequence[V | None]


@dataclass(frozen=True)
class Reconciled[T]:
    """The outcome of reconciling N independent reviewers over a set of items."""

    included: list[T]
    excluded: list[T]
    conflicts: list[T]


def reconcile[T, V](
    review_items: Sequence[ReviewItem[T, V]], *, include: Callable[[V], bool]
) -> Reconciled[T]:
    """Bucket items by independent-reviewer agreement (dual-blind screening, PRISMA-style).

    Per item: if any verdict is ``None`` or there are no verdicts, the item is a
    conflict (fail-safe: never auto-decide on missing review). Otherwise, if every
    reviewer would ``include`` it, it is included; if none would, it is excluded; a
    mix is a conflict to escalate.

    Args:
        review_items: Each item paired with its reviewers' verdicts.
        include: Predicate returning ``True`` when a verdict says 'include'.

    Returns:
        A :class:`Reconciled` partition into included / excluded / conflicts.
    """
    included: list[T] = []
    excluded: list[T] = []
    conflicts: list[T] = []
    for review in review_items:
        verdicts = review.verdicts
        if not verdicts or any(verdict is None for verdict in verdicts):
            conflicts.append(review.item)
            continue
        decisions = [include(verdict) for verdict in verdicts if verdict is not None]
        if all(decisions):
            included.append(review.item)
        elif not any(decisions):
            excluded.append(review.item)
        else:
            conflicts.append(review.item)
    return Reconciled(included=included, excluded=excluded, conflicts=conflicts)


@dataclass(frozen=True)
class Consensus[K: Hashable, T]:
    """A group of equivalent items that cleared the cross-leaf corroboration threshold."""

    key: K
    members: list[T]


def corroborate[T, K: Hashable](
    items: Iterable[T | None], *, key: Callable[[T], K], min_support: int = 2
) -> list[Consensus[K, T]]:
    """Group equivalent items by ``key``; keep groups with >= ``min_support`` members.

    Cross-leaf corroboration: an item is kept only if at least ``min_support``
    leaves independently produced an equivalent item (same key). ``None`` (failed
    leaves) are dropped before grouping. Groups are returned in first-seen key order.

    Args:
        items: The leaves' outputs; ``None`` is dropped.
        key: Maps an item to its equivalence key.
        min_support: Minimum corroborating members for a group to survive (>= 1).

    Returns:
        The surviving groups, in first-seen key order.

    Raises:
        ValueError: If ``min_support < 1``.
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support}")
    groups: dict[K, list[T]] = {}
    for item in items:
        if item is None:
            continue
        groups.setdefault(key(item), []).append(item)
    return [
        Consensus(key=group_key, members=members)
        for group_key, members in groups.items()
        if len(members) >= min_support
    ]

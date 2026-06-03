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

from collections.abc import Callable, Sequence


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
        ValueError: If ``votes`` is empty (no verification ran) or ``kill_at < 1``.
    """
    if not votes:
        raise ValueError("survives() requires at least one vote; got an empty sequence")
    if kill_at < 1:
        raise ValueError(f"kill_at must be >= 1, got {kill_at}")
    against_count = sum(1 for vote in votes if vote is None or against(vote))
    return against_count < kill_at

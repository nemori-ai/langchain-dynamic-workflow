"""Unit tests for the cross-leaf reduce helpers (pure functions over result lists)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from langchain_dynamic_workflow._reduce import (
    Consensus,
    Reconciled,
    ReviewItem,
    corroborate,
    dedup,
    reconcile,
    survives,
)


@dataclass
class _Vote:
    refuted: bool


def test_survives_when_refutes_below_kill_at() -> None:
    votes = [_Vote(refuted=False), _Vote(refuted=True), _Vote(refuted=False)]
    assert survives(votes, against=lambda v: v.refuted, kill_at=2) is True


def test_killed_when_refutes_reach_kill_at() -> None:
    votes = [_Vote(refuted=True), _Vote(refuted=True), _Vote(refuted=False)]
    assert survives(votes, against=lambda v: v.refuted, kill_at=2) is False


def test_none_vote_counts_as_against_failsafe() -> None:
    # Two failed leaves (None) + one explicit refute = 3 against >= kill_at -> killed.
    votes = [None, None, _Vote(refuted=False)]
    assert survives(votes, against=lambda v: v.refuted, kill_at=2) is False


def test_judge_panel_form_against_is_not_sound() -> None:
    @dataclass
    class _Ruling:
        sound: bool

    rulings = [_Ruling(sound=True), _Ruling(sound=True), _Ruling(sound=False)]
    # against = "not sound"; <2 unsound -> survives (2 of 3 sound).
    assert survives(rulings, against=lambda r: not r.sound, kill_at=2) is True


def test_empty_votes_raises() -> None:
    with pytest.raises(ValueError, match="at least one vote"):
        survives([], against=lambda v: v.refuted, kill_at=2)


def test_kill_at_below_one_raises() -> None:
    with pytest.raises(ValueError, match="kill_at must be >= 1"):
        survives([_Vote(refuted=False)], against=lambda v: v.refuted, kill_at=0)


def test_dedup_preserves_first_seen_order() -> None:
    assert dedup(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_dedup_drops_none() -> None:
    assert dedup(["a", None, "b", None]) == ["a", "b"]


def test_dedup_merges_by_key() -> None:
    # str.lower merges case variants; the first-seen original is kept.
    assert dedup(["Alpha", "alpha", "BETA"], key=str.lower) == ["Alpha", "BETA"]


def test_dedup_empty_and_all_none() -> None:
    assert dedup([]) == []
    assert dedup([None, None]) == []


@dataclass
class _Screen:
    keep: bool


def _items(*rows: tuple[str, list[_Screen | None]]) -> list[ReviewItem[str, _Screen]]:
    return [ReviewItem(item=name, verdicts=verdicts) for name, verdicts in rows]


def test_reconcile_three_buckets() -> None:
    review = _items(
        ("all-include", [_Screen(keep=True), _Screen(keep=True)]),
        ("all-exclude", [_Screen(keep=False), _Screen(keep=False)]),
        ("mixed", [_Screen(keep=True), _Screen(keep=False)]),
    )
    result = reconcile(review, include=lambda s: s.keep)
    assert result == Reconciled(
        included=["all-include"], excluded=["all-exclude"], conflicts=["mixed"]
    )


def test_reconcile_none_verdict_is_conflict_failsafe() -> None:
    review = _items(("had-a-failed-reviewer", [_Screen(keep=True), None]))
    result = reconcile(review, include=lambda s: s.keep)
    assert result.conflicts == ["had-a-failed-reviewer"]
    assert result.included == [] and result.excluded == []


def test_reconcile_empty_verdicts_is_conflict() -> None:
    review = _items(("no-reviews", []))
    assert reconcile(review, include=lambda s: s.keep).conflicts == ["no-reviews"]


def test_reconcile_empty_input() -> None:
    assert reconcile([], include=lambda s: s.keep) == Reconciled([], [], [])


def test_corroborate_keeps_groups_meeting_min_support() -> None:
    items = ["RAG", "rag", "long-ctx", "RAG"]  # key=str.lower: rag x3, long-ctx x1
    groups = corroborate(items, key=str.lower, min_support=2)
    assert groups == [Consensus(key="rag", members=["RAG", "rag", "RAG"])]


def test_corroborate_drops_none_then_groups() -> None:
    items = ["a", None, "A", None]
    assert corroborate(items, key=str.lower, min_support=2) == [
        Consensus(key="a", members=["a", "A"])
    ]


def test_corroborate_first_seen_key_order() -> None:
    items = ["b", "B", "a", "A"]
    keys = [g.key for g in corroborate(items, key=str.lower, min_support=2)]
    assert keys == ["b", "a"]


def test_corroborate_min_support_below_one_raises() -> None:
    with pytest.raises(ValueError, match="min_support must be >= 1"):
        corroborate(["a"], key=str.lower, min_support=0)

"""Unit tests for the shared token budget.

The budget rebuilds ``spent()`` from per-leaf usage so a resumed run reaches the
same cumulative total as the first run, enforces a hard cap that raises
:class:`WorkflowBudgetExceededError` once exhausted, and reports an unbounded
``remaining()`` when no total is configured (the loop-until-budget idiom).
"""

from __future__ import annotations

import math

import pytest

from langchain_dynamic_workflow import WorkflowBudgetExceededError
from langchain_dynamic_workflow._budget import Budget


def test_no_total_means_unbounded_remaining() -> None:
    budget = Budget(total=None)
    assert budget.total is None
    assert budget.remaining() == math.inf
    # An unbounded budget never trips the cap, no matter how much is spent.
    budget.record("k0", 10_000)
    assert budget.remaining() == math.inf
    budget.ensure_within_cap()  # must not raise


def test_spent_accumulates_distinct_leaf_usage() -> None:
    budget = Budget(total=100)
    budget.record("k0", 30)
    budget.record("k1", 20)
    assert budget.spent() == 50
    assert budget.remaining() == 50


def test_repeated_key_counts_once() -> None:
    # The same leaf (same call-key) served twice consumed model tokens only once;
    # the journal dedups by key, so spent() must not double-count it.
    budget = Budget(total=100)
    budget.record("k0", 40)
    budget.record("k0", 40)  # cache hit on resume: same key, recorded again
    assert budget.spent() == 40


def test_spent_rebuilds_identically_from_recorded_usage() -> None:
    # Resume reconstruction: replaying the same per-leaf usage in the same order
    # rebuilds spent() to exactly the first run's cumulative total.
    first = Budget(total=100)
    for key, usage in (("k0", 12), ("k1", 7), ("k2", 21)):
        first.record(key, usage)
    rebuilt = Budget(total=100)
    for key, usage in (("k0", 12), ("k1", 7), ("k2", 21)):
        rebuilt.record(key, usage)
    assert rebuilt.spent() == first.spent() == 40


def test_ensure_within_cap_raises_once_exhausted() -> None:
    budget = Budget(total=50)
    budget.record("k0", 50)
    assert budget.spent() == 50
    # spent() >= total: a new agent() dispatch must be refused.
    with pytest.raises(WorkflowBudgetExceededError, match="budget"):
        budget.ensure_within_cap()


def test_ensure_within_cap_allows_while_under_total() -> None:
    budget = Budget(total=50)
    budget.record("k0", 49)
    budget.ensure_within_cap()  # still one token under: a new dispatch is allowed


def test_remaining_floors_at_zero_on_overspend() -> None:
    # In-flight leaves can push usage past the cap before the next gate check;
    # remaining() must never report a negative figure.
    budget = Budget(total=50)
    budget.record("k0", 80)
    assert budget.remaining() == 0

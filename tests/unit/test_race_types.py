"""Unit tests for the race value types (pure frozen dataclasses)."""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow._race_types import RaceCandidate, RaceResult


def test_race_candidate_defaults() -> None:
    candidate = RaceCandidate(prompt="diagnose", agent_type="investigator")
    assert candidate.schema is None
    assert candidate.model is None
    assert candidate.isolation == "shared"


def test_race_candidate_is_frozen() -> None:
    candidate = RaceCandidate(prompt="p", agent_type="a")
    with pytest.raises((AttributeError, TypeError)):
        candidate.prompt = "mutated"  # type: ignore[misc]


def test_race_result_won_is_true_when_there_is_a_winner() -> None:
    result: RaceResult[str] = RaceResult(winner="root-cause", winner_index=0)
    assert result.won is True


def test_race_result_won_is_false_when_no_winner() -> None:
    result: RaceResult[str] = RaceResult(winner=None, winner_index=None)
    assert result.won is False

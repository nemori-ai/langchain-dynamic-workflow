"""Unit tests for the determinism backstop (journal-divergence oracle).

The backstop records the ordered sequence of ``agent()`` call-keys on the first
run and, on replay, fails loud the moment the script's k-th call-key diverges
from the recorded one — so a non-deterministic script can never be fed a
positionally-misaligned cache entry.
"""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow import WorkflowDeterminismError
from langchain_dynamic_workflow._determinism import CallSequenceGuard


async def test_recording_run_accepts_any_sequence() -> None:
    # First (recording) run: no prior sequence, every key is appended in order.
    guard = CallSequenceGuard(recorded=None)
    guard.observe("k0")
    guard.observe("k1")
    guard.observe("k2")
    assert guard.sequence == ["k0", "k1", "k2"]


async def test_replay_matching_sequence_passes() -> None:
    # Replay run with a recorded sequence: identical keys in identical order pass.
    guard = CallSequenceGuard(recorded=["k0", "k1", "k2"])
    guard.observe("k0")
    guard.observe("k1")
    guard.observe("k2")
    assert guard.sequence == ["k0", "k1", "k2"]


async def test_replay_divergent_key_fails_loud() -> None:
    # Replay where the 2nd call-key differs from the record: the backstop must
    # raise rather than silently serve a positionally-misaligned cache entry.
    guard = CallSequenceGuard(recorded=["k0", "k1", "k2"])
    guard.observe("k0")
    with pytest.raises(WorkflowDeterminismError) as excinfo:
        guard.observe("DIVERGED")
    message = str(excinfo.value)
    assert "1" in message  # the diverging position is reported
    assert "k1" in message  # the recorded (expected) key
    assert "DIVERGED" in message  # the actual (observed) key


async def test_replay_extra_call_beyond_record_fails_loud() -> None:
    # The script issued MORE calls on replay than were recorded: a divergence in
    # call count is just as much a determinism break as a key mismatch.
    guard = CallSequenceGuard(recorded=["k0"])
    guard.observe("k0")
    with pytest.raises(WorkflowDeterminismError) as excinfo:
        guard.observe("k1")
    assert "beyond" in str(excinfo.value).lower()

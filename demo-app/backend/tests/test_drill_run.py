"""Backend checks for the M3 background-run transport (drill-in + report-back).

A detached background run used to be event-dark: ``launch_background_run`` wired no
sinks, so nothing about the run's interior ever reached the host. These tests pin the
M3 transport mechanism: the launcher wires the manager's buffer sinks so a settled
run's slot holds replayable span events. They run offline (no model keys; the fake
roster leaves) against the real launcher and manager.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _no_model_keys() -> None:
    """Run with no provider key so the host stays on the offline path."""
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)


# The preset the offline tests launch: the same fan-out-heavy deep-research scenario the
# background path defaults to, served by the deterministic fake leaves.
_OFFLINE_PRESET = "deep_research"


async def test_launch_background_run_wires_event_sinks() -> None:
    """After launch + settle, the run's buffer holds span events — no longer event-dark.

    ``launch_background_run`` must pre-mint the run id, build the manager's buffer
    sinks for it, and thread them into the detached ``run_workflow`` — so the slot's
    bounded buffer captures the run's span edges for a later drill-in replay.
    """
    from host_graph import launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    events, dropped = manager.buffered_events(run_id, thread_id="t1")
    assert dropped == 0
    assert any(e.kind == "span" for e in events), [e.kind for e in events]

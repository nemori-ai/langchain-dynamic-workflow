"""Backend checks for the M3 background-run transport (drill-in + report-back).

A detached background run used to be event-dark: ``launch_background_run`` wired no
sinks, so nothing about the run's interior ever reached the host. These tests pin the
M3 transport mechanism: the launcher wires the manager's buffer sinks so a settled
run's slot holds replayable span events. They run offline (no model keys; the fake
roster leaves) against the real launcher and manager.
"""

from __future__ import annotations

import os
from typing import Any

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


async def test_drill_run_live_replays_buffered_events_as_cards() -> None:
    """A drill replays the run's buffer as the same card vocabulary an inline run shows.

    Launch + settle an offline background run, then drill: the captured emits must
    contain ``agent_span`` (the per-leaf interior, not an aggregate row), keyed by the
    engine-minted stable span_id, and the summary must name the run.
    """
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    captured: list[tuple[str, dict[str, Any]]] = []
    summary = await drill_run_live(
        manager,
        lambda name, props: captured.append((name, props)),
        thread_id="t1",
        target=run_id,
    )
    names = {name for name, _ in captured}
    assert "agent_span" in names, names
    assert run_id in summary, summary


async def test_drill_run_live_is_idempotent_across_ticks() -> None:
    """Two full replays of the same settled run yield the same set of event ids.

    The engine-minted stable span_id makes a replay upsert in place at the SDK reducer
    rather than stacking duplicate cards — so a poll loop re-replaying each tick (or a
    second drill) is idempotent.
    """
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    def _ids(emits: list[tuple[str, dict[str, Any]]]) -> set[str]:
        return {p["event_id"] for _, p in emits if "event_id" in p}

    first: list[tuple[str, dict[str, Any]]] = []
    await drill_run_live(manager, lambda n, p: first.append((n, p)), thread_id="t1", target=run_id)
    second: list[tuple[str, dict[str, Any]]] = []
    await drill_run_live(manager, lambda n, p: second.append((n, p)), thread_id="t1", target=run_id)
    assert _ids(first) == _ids(second) != set()


async def test_drill_run_live_resolves_label_and_reports_unknown_target() -> None:
    """A drill target resolves by label too, and an unknown target gets an honest reply."""
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager()
    run_id = launch_background_run(
        manager, thread_id="t1", workflow=_OFFLINE_PRESET, label="RAG study"
    )
    await manager.wait(run_id, thread_id="t1")

    summary = await drill_run_live(manager, lambda n, p: None, thread_id="t1", target="RAG study")
    assert run_id in summary, summary

    missing = await drill_run_live(manager, lambda n, p: None, thread_id="t1", target="ghost")
    assert "no background run" in missing.lower(), missing
    assert "RAG study" in missing, missing  # honest: lists what IS available


async def test_drill_run_live_surfaces_dropped_as_truncation_log() -> None:
    """Past the buffer cap, the drill surfaces the dropped count as a timeline log line."""
    from host_graph import drill_run_live, launch_background_run

    from langchain_dynamic_workflow import BgRunManager

    manager = BgRunManager(max_buffered_events=1)
    run_id = launch_background_run(manager, thread_id="t1", workflow=_OFFLINE_PRESET)
    await manager.wait(run_id, thread_id="t1")

    captured: list[tuple[str, dict[str, Any]]] = []
    await drill_run_live(
        manager, lambda n, p: captured.append((n, p)), thread_id="t1", target=run_id
    )
    timeline_msgs = [p.get("message", "") for n, p in captured if n == "phase_timeline"]
    assert any("dropped" in m for m in timeline_msgs), timeline_msgs

"""Unit tests for the in-memory run registry and its protocol surface.

These cover the dependency-free default (``InMemoryRunStore``) that backs the
workflow tool's run registry: spec round-trip, the unknown-run miss, the
same-instance journal guard that preserves same-session resume, and structural
conformance to the ``WorkflowRunStore`` protocol.
"""

from langchain_dynamic_workflow._engine import InMemoryJournalStore
from langchain_dynamic_workflow._run_store import (
    InMemoryRunStore,
    RunSpec,
    WorkflowRunStore,
)


def _spec() -> RunSpec:
    """A representative named-workflow spec for round-trip assertions."""
    return RunSpec(
        kind="name",
        name_or_source="incident_triage",
        args={"severity": "high", "alerts": [1, 2, 3]},
        label="Incident triage",
        journal_run_id="origin-run-id",
    )


async def test_save_then_load_round_trips_the_spec() -> None:
    """A saved spec loads back equal, with every field preserved."""
    store = InMemoryRunStore()
    spec = _spec()

    await store.save_spec("run-1", spec)
    loaded = await store.load_spec("run-1")

    assert loaded == spec
    assert loaded is not None
    assert loaded.kind == "name"
    assert loaded.name_or_source == "incident_triage"
    assert loaded.args == {"severity": "high", "alerts": [1, 2, 3]}
    assert loaded.label == "Incident triage"
    assert loaded.journal_run_id == "origin-run-id"


async def test_journal_run_id_defaults_to_none() -> None:
    """A fresh launch spec carries no journal lineage until one is stamped on."""
    spec = RunSpec(
        kind="name",
        name_or_source="wf",
        args={},
        label="wf",
    )

    assert spec.journal_run_id is None


async def test_delete_spec_drops_a_saved_spec() -> None:
    """``delete_spec`` removes a saved spec so a later load misses.

    This is the orphan-cleanup contract used when a launch is admitted-then-
    rejected: the spec persisted before ``manager.start`` must be deletable so a
    quota refusal leaves no unresumable orphan in the registry.
    """
    store = InMemoryRunStore()
    await store.save_spec("run-1", _spec())

    await store.delete_spec("run-1")

    assert await store.load_spec("run-1") is None


async def test_delete_spec_on_unknown_run_id_is_a_noop() -> None:
    """Deleting a run that was never saved is silent, not an error."""
    store = InMemoryRunStore()

    await store.delete_spec("never-saved")  # must not raise

    assert await store.load_spec("never-saved") is None


async def test_load_unknown_run_id_returns_none() -> None:
    """Loading a run that was never saved yields ``None`` rather than raising."""
    store = InMemoryRunStore()

    assert await store.load_spec("never-saved") is None


def test_journal_for_returns_same_instance_for_repeated_run_id() -> None:
    """Repeated ``journal_for`` on one run id returns the identical instance.

    This is the resume regression guard: a relaunch must reuse the journal that
    the original run populated, otherwise completed leaves would re-execute at
    new model cost instead of replaying for free.
    """
    store = InMemoryRunStore()

    first = store.journal_for("run-1")
    second = store.journal_for("run-1")

    assert first is second
    assert isinstance(first, InMemoryJournalStore)


def test_journal_for_isolates_distinct_run_ids() -> None:
    """Distinct run ids get distinct journal instances (no cross-run bleed)."""
    store = InMemoryRunStore()

    assert store.journal_for("run-1") is not store.journal_for("run-2")


def test_in_memory_store_satisfies_the_protocol() -> None:
    """The in-memory default is a structural ``WorkflowRunStore``."""
    assert isinstance(InMemoryRunStore(), WorkflowRunStore)


def test_run_spec_is_frozen() -> None:
    """``RunSpec`` is immutable so a persisted spec cannot drift after save."""
    spec = _spec()

    try:
        spec.journal_run_id = "mutated"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("RunSpec must be frozen")

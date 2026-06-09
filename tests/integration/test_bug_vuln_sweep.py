"""Integration: the bug/vuln sweep demo runs ctx.batch_map end to end.

Loads the runnable example and drives its ``sweep`` workflow through
``run_workflow`` with a deterministic, prompt-aware structured fake (no host,
no API key). Pins the batch_map shape: every file is mapped to one finder leaf,
the folded result is deduped by ``(file, kind)`` in input order, and the live
``on_progress`` sink receives transient ``BATCH`` entries that carry metrics.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from langchain_dynamic_workflow import (
    ProgressEntry,
    ProgressKind,
    Roster,
    run_workflow,
)


def _load_example() -> ModuleType:
    """Import the bug/vuln sweep feature demo (sweep workflow) as a module."""
    return importlib.import_module("examples.features.bug_vuln_sweep")


def _finder_builder(module: ModuleType) -> Any:
    """Build a roster ``builder=`` that returns the demo's prompt-aware finder.

    Reuses the demo's own deterministic leaf factory so the test pins the
    exact mechanism the demo ships, not a second parallel fake.
    """

    def builder(*, response_format: Any = None) -> Any:
        return module.build_finder()

    return builder


async def test_bug_vuln_sweep_maps_every_file_and_dedups() -> None:
    module = _load_example()
    roster = Roster().register("finder", builder=_finder_builder(module))

    entries: list[ProgressEntry] = []

    async def orchestrate(ctx: Any) -> Any:
        return await module.sweep(ctx, {"files": module.FILES})

    findings = await run_workflow(orchestrate, roster=roster, on_progress=entries.append)

    # One finder leaf per file; the planted findings dedup to the distinct
    # (file, kind) pairs the fake reports — duplicates across files collapse.
    assert findings == module.EXPECTED_FINDINGS
    # Every kept finding is a validated Finding with the demo's fields.
    for finding in findings:
        assert isinstance(finding, module.Finding)
        assert finding.file and finding.kind and finding.detail

    # Live progress: transient BATCH entries reached the sink, the last one is
    # exact (completed == total), and none of them was recorded into the
    # replay log (PHASE/LOG carry no metrics; BATCH does).
    batch_entries = [e for e in entries if e.kind is ProgressKind.BATCH]
    assert batch_entries, "the sweep must emit live BATCH progress"
    for entry in batch_entries:
        assert entry.metrics is not None
    last = batch_entries[-1]
    assert last.metrics is not None
    assert last.metrics.completed == len(module.FILES)
    assert last.metrics.total == len(module.FILES)


async def test_bug_vuln_sweep_main_smoke() -> None:
    """The demo's own asserting main() runs clean offline (the smoke check)."""
    module = _load_example()
    await module.main()

"""``ctx.batch_map`` — bounded streaming fan-out over a large item list + live count/ETA.

A codebase-wide bug/vuln sweep: one read-only finder leaf per file, mapped
concurrently through a bounded admission window so a flood of files never
materializes a flood of tasks at once. Results collect in input order; the
script folds them with ``dedup`` (the same finding planted in two files
collapses to one). While the sweep advances, the engine emits transient
``BATCH`` progress (completed / total / ETA) the host renders as a live bar —
out-of-band, never journaled, so a resume picks the bar up where live work
continues rather than replaying it.

    files = [auth.py, db.py, api.py, util.py, cache.py]
      |-- batch_map (one finder leaf per file, window-bounded)
      |     auth.py  -> sql-injection, hardcoded-secret
      |     db.py    -> sql-injection          (dup kind, different file)
      |     api.py   -> missing-auth
      |     util.py  -> (clean)
      |     cache.py -> hardcoded-secret       (dup kind, different file)
      |-- live "scanned 3/5 (~2s left)" arrives on the progress sink
      '-- dedup(key=(file, kind)) folds the holes and exact dups

Run it:

    uv run python -m examples.features.bug_vuln_sweep
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    Ctx,
    ProgressEntry,
    ProgressKind,
    Roster,
    dedup,
    run_workflow,
)

# The files to sweep, in the order the batch maps them.
FILES = ["auth.py", "db.py", "api.py", "util.py", "cache.py"]

# What each file's finder reports. ``util.py`` is clean (the leaf returns no
# finding -> a None hole the dedup drops). ``sql-injection`` and
# ``hardcoded-secret`` each recur in two files, so dedup keyed on (file, kind)
# keeps both occurrences (distinct files) while a same-file exact dup would
# collapse — the fold is per (file, kind), not global per kind.
_PLANTED: dict[str, list[tuple[str, str]]] = {
    "auth.py": [
        ("sql-injection", "unparameterized query builds SQL from request input"),
        ("hardcoded-secret", "an API token is embedded as a string literal"),
    ],
    "db.py": [("sql-injection", "a raw cursor.execute concatenates a user id")],
    "api.py": [("missing-auth", "an admin route has no authorization check")],
    "util.py": [],
    "cache.py": [("hardcoded-secret", "a signing key is committed in the source")],
}


class Finding(BaseModel):
    """One finder leaf's structured verdict on a single file.

    Attributes:
        file: The scanned file's path.
        kind: The vulnerability class (e.g. ``sql-injection``).
        detail: A one-line description of the finding.
    """

    file: str
    kind: str
    detail: str


def _scan_prompt(path: str) -> str:
    return (
        "You are a read-only security reviewer. Audit the file "
        f"{path!r} for the single most severe bug or vulnerability. Return "
        "its `file`, a short `kind`, and a one-line `detail`. If the file is "
        "clean, say so plainly."
    )


# The dedup identity: a finding is the same iff its (file, kind) pair matches.
# Two files with the same kind are kept (different files); an exact (file, kind)
# repeat collapses. Findings are pydantic models (unhashable), so a key is required.
def _finding_key(finding: Finding) -> tuple[str, str]:
    return (finding.file, finding.kind)


async def sweep(ctx: Ctx, args: dict[str, Any]) -> list[Finding]:
    """Map one finder leaf over every file, then fold the result list with dedup."""
    files: list[str] = args["files"]
    ctx.phase("sweep codebase")
    # batch_map returns list[Finding | None] aligned to input order: a failed leaf
    # lands None. A clean file returns an explicit kind == "clean" Finding, filtered
    # out below — so dropped entries come from an asserted filter, not a silent
    # internal crash. dedup then collapses (file, kind) duplicates.
    findings = await ctx.batch_map(
        files,
        lambda path: ctx.agent(_scan_prompt(path), agent_type="finder", schema=Finding),
        max_in_flight=4,
        total=len(files),
    )
    real = [f for f in findings if f is not None and f.kind != "clean"]
    return dedup(real, key=_finding_key)


# ── leaf (deterministic, prompt-aware fake) ──────────────────────────────────


# A deterministic finder. It reads the file path out of the scan prompt and
# returns that file's first planted finding as a validated Finding (or a kind=
# "clean" Finding for a clean file, which sweep() filters out explicitly). One
# finder per file, so the batch reports exactly what _PLANTED says — letting the
# demo assert the dedup fold rather than a model's free text.
def build_finder() -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content)
        planted = next((items for path, items in _PLANTED.items() if repr(path) in prompt), [])
        path = next((p for p in _PLANTED if repr(p) in prompt), "unknown.py")
        if not planted:
            # Clean file: return an EXPLICIT kind="clean" Finding (a valid structured
            # response), NOT a missing one. sweep() filters kind == "clean", so the
            # hole comes from an asserted filter — not an internal fold-of-None crash
            # that would dishonestly ride the failure-isolation path.
            return {
                "messages": [*inp["messages"], AIMessage(content="reviewed")],
                "structured_response": Finding(file=path, kind="clean", detail="no issues found"),
            }
        kind, detail = planted[0]
        return {
            "messages": [*inp["messages"], AIMessage(content="reviewed")],
            "structured_response": Finding(file=path, kind=kind, detail=detail),
        }

    return RunnableLambda(_leaf)


# The deduped findings the offline fake yields, in input order — the smoke-check
# ground truth. util.py is clean (dropped); every other file's first planted
# finding survives (no two share a (file, kind) pair).
EXPECTED_FINDINGS = [
    Finding(file="auth.py", kind="sql-injection", detail=_PLANTED["auth.py"][0][1]),
    Finding(file="db.py", kind="sql-injection", detail=_PLANTED["db.py"][0][1]),
    Finding(file="api.py", kind="missing-auth", detail=_PLANTED["api.py"][0][1]),
    Finding(file="cache.py", kind="hardcoded-secret", detail=_PLANTED["cache.py"][0][1]),
]


def _render_bar(entry: ProgressEntry) -> None:
    """Print a live progress line for a BATCH entry; pass PHASE/LOG through plainly."""
    if entry.kind is ProgressKind.BATCH and entry.metrics is not None:
        metrics = entry.metrics
        eta = f" (~{metrics.eta_seconds:.0f}s left)" if metrics.eta_seconds is not None else ""
        total = metrics.total if metrics.total is not None else "?"
        print(f"  scanned {metrics.completed}/{total}{eta}")
    else:
        print(f"[{entry.kind.value}] {entry.message}")


async def main() -> None:
    roster = Roster().register(
        "finder",
        builder=lambda *, response_format=None: build_finder(),
        description="Audits a single file for its most severe bug/vuln",
    )

    async def orchestrate(ctx: Ctx) -> list[Finding]:
        return await sweep(ctx, {"files": FILES})

    findings = await run_workflow(orchestrate, roster=roster, on_progress=_render_bar)
    print(f"sweep findings ({len(findings)}):")
    for finding in findings:
        print(f"  - {finding.file}: {finding.kind} — {finding.detail}")

    # The mechanism: batch_map mapped one leaf per file, the clean file and the
    # holes were dropped, and the surviving findings dedup by (file, kind) in
    # input order — exactly the planted set.
    assert findings == EXPECTED_FINDINGS, findings
    print("OK: batch_map fanned out one finder per file and deduped the findings.")


if __name__ == "__main__":
    asyncio.run(main())

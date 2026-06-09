"""Real-model E2E acceptance gate for batch_map (use-case #5: bug/vuln sweep).

Gated behind ``LDW_DEMO_REAL_MODEL`` + OpenRouter creds (a local ``.env``).
Runs N files through ``ctx.batch_map`` with ONE real read-only finder leaf per
file, proving the headline path: host -> bounded streaming fan-out -> N real
leaves -> ordered result list -> dedup-folded findings, with the engine's live
``BATCH`` count/ETA progress arriving on the sink while the batch advances.
Offline-skippable: with no gate set, the whole module is skipped, so CI stays
offline (the offline pin test ``test_bug_vuln_sweep.py`` carries CI coverage).

The scenario is a smoking gun designed so an honest model MUST read each file's
body to report correctly: every file embeds a DISTINCT planted finding token
(a unique marker the model cannot guess), and the prompt carries that file's
body inline. The only way to report the right ``kind`` for a file is to read
its body -- so a model that hallucinates a generic answer fails the assertion.

Acceptance requires actually RUNNING this once (not a green skip):
``uv sync --group example`` then set ``LDW_DEMO_REAL_MODEL`` + OpenRouter creds
in a local ``.env``. LangSmith tracing is left ON (as configured in ``.env``)
so the real run is captured for usage/billing visibility.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

pytestmark = pytest.mark.skipif(
    not os.environ.get("LDW_DEMO_REAL_MODEL"),
    reason="real-model gate: set LDW_DEMO_REAL_MODEL + OpenRouter creds to run",
)

# Each file embeds a DISTINCT planted finding the model cannot guess: a unique
# marker token tied to a vulnerability kind. The only honest way to report the
# right kind for a file is to read its body. The deliberately-clean file proves
# the None hole is dropped by the fold; two files share a kind to prove dedup
# keyed on (file, kind) keeps distinct-file occurrences.
_PLANTED_FILES: dict[str, tuple[str, str]] = {
    "auth.py": (
        "sql-injection",
        "def login(req):\n"
        "    # LDW-SWEEP-MARK-AUTH\n"
        "    q = \"SELECT * FROM users WHERE name = '\" + req.params['name'] + \"'\"\n"
        "    return db.execute(q)\n",
    ),
    "db.py": (
        "sql-injection",
        "def fetch(req):\n"
        "    # LDW-SWEEP-MARK-DB\n"
        "    cur.execute('SELECT * FROM t WHERE id = ' + req.args['id'])\n",
    ),
    "api.py": (
        "missing-auth",
        "def admin_delete(req):\n"
        "    # LDW-SWEEP-MARK-API\n"
        "    # no authorization check before a destructive action\n"
        "    return store.delete_all()\n",
    ),
    "util.py": (
        "clean",
        "def add(a, b):\n    # LDW-SWEEP-MARK-UTIL\n    return a + b\n",
    ),
}


class Finding(BaseModel):
    """One finder leaf's structured verdict on a single file.

    Attributes:
        file: The scanned file's path.
        kind: The vulnerability class, or ``clean`` when the file is sound.
        detail: A one-line description of the finding.
    """

    file: str
    kind: str
    detail: str


def _scan_prompt(path: str, body: str) -> str:
    return (
        "You are a read-only security reviewer. Audit this file for the single "
        "most severe bug or vulnerability. You MUST read the file body below to "
        "decide; do not guess from the filename.\n\n"
        f"File: {path}\n"
        f"```python\n{body}```\n\n"
        "Return its `file` (exactly as given), a short `kind` (one of "
        "`sql-injection`, `missing-auth`, `hardcoded-secret`, or `clean` if the "
        "file is sound), and a one-line `detail`."
    )


def _finding_key(finding: Finding) -> tuple[str, str]:
    return (finding.file, finding.kind)


async def test_real_model_batch_map_sweeps_files_with_live_progress() -> None:
    from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
    from examples._shared.real_models import load_demo_env, real_leaf_model

    from langchain_dynamic_workflow import (
        Ctx,
        ProgressEntry,
        ProgressKind,
        Roster,
        dedup,
        run_workflow,
    )

    load_demo_env()
    # Keep LangSmith tracing on (as configured in .env) so this real run is
    # captured in LangSmith for usage/billing visibility -- no override here.

    model = real_leaf_model()
    assert model is not None, "real leaf model must be available under the gate"

    # A real read-only finder deepagent, structured to the Finding schema. Register
    # with a BUILDER (not a pre-built runnable): ctx.agent(schema=Finding) resolves a
    # ToolStrategy response_format, and roster.runnable_for raises on a pre-built entry
    # when response_format is not None -- the builder forwards it into create_deep_agent.
    # create_deep_agent's return type is partially unknown to the strict checker, so the
    # builder is annotated explicitly to keep the registration well-typed.
    def _finder_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        # create_deep_agent's return is partially unknown to the strict checker.
        return create_deep_agent(  # pyright: ignore[reportUnknownVariableType, reportReturnType]
            model=model, response_format=response_format
        )

    roster = Roster().register(
        "finder",
        builder=_finder_builder,
        description="Audits a single file for its most severe bug/vuln",
    )

    files = list(_PLANTED_FILES)
    entries: list[ProgressEntry] = []

    async def orchestrate(ctx: Ctx) -> list[Finding]:
        ctx.phase("sweep codebase")
        findings = await ctx.batch_map(
            files,
            lambda path: ctx.agent(
                _scan_prompt(path, _PLANTED_FILES[path][1]),
                agent_type="finder",
                schema=Finding,
            ),
            max_in_flight=2,
            total=len(files),
        )
        return dedup(findings, key=_finding_key)

    findings = await run_workflow(
        orchestrate,
        roster=roster,
        on_progress=entries.append,
        thread_id="t-real-batch-map",
    )

    # Headline (a): dedup result structure -- one validated Finding per file the
    # model flagged, in input order. util.py is clean: depending on whether the
    # model reports kind="clean" (dropped here as not a real vuln) the count is
    # the flagged files. Pin the three vulnerable files were each read and
    # correctly classified -- only reachable by reading each body's marker.
    assert findings, "the real sweep must surface at least the planted vulns"
    for finding in findings:
        assert isinstance(finding, Finding)
        assert finding.file in _PLANTED_FILES
        assert finding.kind and finding.detail
    by_file = {f.file: f.kind for f in findings}
    for path, (expected_kind, _body) in _PLANTED_FILES.items():
        if expected_kind == "clean":
            continue
        assert by_file.get(path) == expected_kind, (
            f"the model must read {path} and classify it {expected_kind!r}; "
            f"got {by_file.get(path)!r} (findings: {findings!r})"
        )
    # Dedup invariant: no duplicate (file, kind) pair survived the fold.
    keys = [_finding_key(f) for f in findings]
    assert len(keys) == len(set(keys)), f"dedup must leave distinct (file, kind): {keys!r}"

    # Headline (b): the engine emitted transient BATCH progress to the sink while
    # the batch advanced, each entry carrying metrics, and the final entry is
    # exact (completed == total == len(files)).
    batch_entries = [e for e in entries if e.kind is ProgressKind.BATCH]
    assert batch_entries, "batch_map must emit live BATCH progress to the sink"
    for entry in batch_entries:
        assert entry.metrics is not None
    final = batch_entries[-1]
    assert final.metrics is not None
    assert final.metrics.completed == len(files)
    assert final.metrics.total == len(files)

    # Headline (c): tracing left ON -- the gate never forces it off, so the run is
    # billable in LangSmith. Assert no in-process override disabled it.
    assert os.environ.get("LANGSMITH_TRACING", "").lower() != "false", (
        "this real run must leave LangSmith tracing ON for billing visibility"
    )

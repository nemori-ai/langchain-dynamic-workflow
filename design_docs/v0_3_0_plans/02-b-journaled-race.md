# M2 · B — Journaled Race (early-exit / cancel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the orchestration engine an **early-exit / cancel** primitive — `ctx.race(candidates, *, win, win_tag="")` runs best-of-N agent calls concurrently, the first whose result satisfies the `win` predicate wins, the in-flight losers are cancelled — while preserving the engine's deterministic, resumable replay by **journaling the race decision** (content-hash keyed) so a resume reproduces the winner and dispatches nothing.

**Architecture:** Two pure value types (`RaceCandidate`, `RaceResult`) live in a new L1 module `_race_types.py` (mirroring `_reduce.py`); a content-hash `race_key` lives beside `journal_key` in `_journal.py`; the runtime method `Ctx.race` lives in `_context.py`. The method reuses `ctx.agent()` verbatim to dispatch each candidate (so journal dedup, budget metering, sandbox admission, and spans all come for free), runs an `asyncio.wait(FIRST_COMPLETED)` loop with an ascending-index deterministic tie-break, cancels losers in a `finally`, and persists a self-contained decision envelope `{winner_index, result}` under a namespaced race-key. The two value types are injected into the `run_script` namespace (host side) and exported from the package root (developer side).

**Tech Stack:** Python 3.12 (PEP 695 inline generics), asyncio, pydantic v2, pytest + pytest-asyncio (`asyncio_mode=auto`), ruff, pyright strict, import-linter. Spec: `docs/plans/2026-06-03-m2-b-journaled-race-design.md`.

**Branch:** Do all work on `feat/m2-journaled-race` (branch off `main` before Task 1).

**Scope note (B only):** This milestone delivers **B (race / early-exit / cancel)** only. **E (batch-map helper + count/ETA progress + streaming admission)** is split out into a later milestone, and true streaming output is an explicit **non-goal** for B (it conflicts with deterministic replay; the race buys the latency motivation B exists for). The roadmap is updated to reflect this split in Task 11.

---

## Setup (before Task 1)

- [ ] **Create the feature branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/m2-journaled-race
```

---

## File Structure

| File | Responsibility |
|---|---|
| `src/langchain_dynamic_workflow/_race_types.py` | **Create.** `RaceCandidate` + `RaceResult` frozen dataclasses. Pure L1 types, no engine coupling (mirrors `_reduce.py`). |
| `src/langchain_dynamic_workflow/_journal.py` | **Modify.** Add `race_key()` beside `journal_key()` (content-hash, `"race"`-namespaced, win_tag folded). |
| `src/langchain_dynamic_workflow/_observability.py` | **Modify.** Add `SpanKind.RACE`. |
| `src/langchain_dynamic_workflow/_context.py` | **Modify.** Add `Ctx.race` (+ private `_run_race_candidate`). The orchestration runtime primitive. |
| `src/langchain_dynamic_workflow/_codegen.py` | **Modify.** Inject `RaceCandidate` / `RaceResult` into the `run_script` namespace. |
| `src/langchain_dynamic_workflow/__init__.py` | **Modify.** Import + export `RaceCandidate`, `RaceResult`, `race_key`. |
| `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` | **Modify.** Add `ctx.race` to the DSL; add the race quality pattern + parallel-vs-race guidance + win_tag footgun note. |
| `examples/13_ai_sre_race_real_e2e.py` | **Create.** AI-SRE multi-hypothesis race demo (integration home for `race`). |
| `tests/unit/test_race_types.py` | **Create.** Pure unit tests for the two value types. |
| `tests/unit/test_journal.py` | **Modify.** Add `race_key` stability / namespace / win_tag tests. |
| `tests/unit/test_observability.py` | **Modify.** Add a `SpanKind.RACE` assertion. |
| `tests/unit/test_race.py` | **Create.** Fresh-path race mechanics + cancellation/gate-release unit tests. |
| `tests/integration/test_race.py` | **Create.** Replay reproduces winner + zero loser dispatch; nested-in-parallel resume; win_tag footgun. |
| `tests/unit/test_codegen.py` | **Modify.** Add a meta-layer injection test for the race types. |
| `tests/test_smoke.py` | **Modify.** Add a root-export assertion. |
| `tests/integration/test_ai_sre_race.py` | **Create.** Offline pin test for examples/13. |
| `design_docs/01-engine-mechanism.md`, `design_docs/02-architecture.md`, `design_docs/uml/02-class.md`, `design_docs/uml/03-sequence.md`, `README.md`, `README_zh.md`, `design_docs/v0_3_0_plans/00-roadmap.md` | **Modify.** Evergreen sync + roadmap (B/E split, streaming non-goal, M2 status). |

> **import-linter note (read before Task 5/6):** the "Layer 2 host-facing must not reach Layer 0/1 engine internals" contract forbids `tool` / `middleware` / `_background` / `_codegen` from importing `_journal`, `_budget`, `_determinism`, `_pipeline`, `_concurrency`, `_sandbox`, `_progress`, `_context`. `_race_types` is **not** in that list (it is a pure types module like `_reduce`), so `_codegen` importing `_race_types` is allowed — the injection depends on this. **Do NOT** add `_race_types` to the forbidden list, and **do NOT** make `_codegen` import `_journal` (it needs only the two types, never `race_key`).

---

## Task 1: `_race_types.py` — `RaceCandidate` + `RaceResult`

**Files:**
- Create: `src/langchain_dynamic_workflow/_race_types.py`
- Test: `tests/unit/test_race_types.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_race_types.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_race_types.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: FAIL — `ModuleNotFoundError: No module named 'langchain_dynamic_workflow._race_types'`.

- [ ] **Step 3: Create `_race_types.py`**

```python
"""Value types for ``ctx.race`` — the journaled best-of-N early-exit primitive.

A race enters several content-hashable agent-call specs (:class:`RaceCandidate`)
and returns the first whose result satisfies the win predicate
(:class:`RaceResult`), cancelling the in-flight losers. These types are pure data —
no engine state and no ``agent()`` call — so they live beside the runtime (like the
reduce helpers) rather than inside it, and they are injected into the ``run_script``
namespace so a host-authored script constructs and reads them by name without an
import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class RaceCandidate:
    """One content-hashable agent-call spec entered into a race.

    The fields mirror the keyed inputs of ``Ctx.agent`` so a candidate's journal
    key — and therefore the race's replay identity — is derived exactly as a direct
    ``agent()`` call would derive it.

    Attributes:
        prompt: The leaf prompt.
        agent_type: The roster name the candidate resolves to.
        schema: Optional structured-output schema (pydantic ``BaseModel`` subclass
            or inline JSON-schema ``dict``). All candidates in one race must agree:
            either all schema-less or all bound to the same schema.
        model: Optional per-candidate model override; ``None`` uses the roster
            entry's default model.
        isolation: Isolation mode (part of the journal key).
    """

    prompt: str
    agent_type: str
    schema: type[BaseModel] | dict[str, Any] | None = None
    model: str | None = None
    isolation: str = "shared"


@dataclass(frozen=True, slots=True)
class RaceResult[T]:
    """The outcome of a race: the first result that satisfied ``win``, or none.

    Attributes:
        winner: The winning candidate's result, or ``None`` if no candidate
            satisfied the win predicate.
        winner_index: The winning candidate's position in the input sequence, or
            ``None`` when there was no winner.
    """

    winner: T | None
    winner_index: int | None

    @property
    def won(self) -> bool:
        """Whether a candidate satisfied the win predicate."""
        return self.winner_index is not None
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_race_types.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: PASS (4 passed).

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/langchain_dynamic_workflow/_race_types.py tests/unit/test_race_types.py
uv run ruff format src/langchain_dynamic_workflow/_race_types.py tests/unit/test_race_types.py
uv run pyright src/langchain_dynamic_workflow/_race_types.py
```
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_race_types.py tests/unit/test_race_types.py
git commit -m "$(cat <<'EOF'
feat(race): RaceCandidate + RaceResult value types

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `race_key` — content-hash key for a race decision

**Files:**
- Modify: `src/langchain_dynamic_workflow/_journal.py`
- Test: `tests/unit/test_journal.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_journal.py`)

```python
from langchain_dynamic_workflow._journal import race_key  # add to the import block


def test_race_key_is_stable_for_identical_inputs() -> None:
    a = race_key(candidate_keys=["k0", "k1"], win_tag="t")
    b = race_key(candidate_keys=["k0", "k1"], win_tag="t")
    assert a == b


def test_race_key_changes_with_candidates_and_tag() -> None:
    base = race_key(candidate_keys=["k0", "k1"], win_tag="t")
    # Order matters (a different candidate ordering is a different race).
    assert race_key(candidate_keys=["k1", "k0"], win_tag="t") != base
    # A different candidate set is a different race.
    assert race_key(candidate_keys=["k0", "k2"], win_tag="t") != base
    # The win tag is folded in: same candidates, different criterion -> different key.
    assert race_key(candidate_keys=["k0", "k1"], win_tag="other") != base


def test_race_key_never_collides_with_a_leaf_key() -> None:
    # A race over a single candidate must not hash to that candidate's own leaf key:
    # the "race" namespace marker keeps the two key spaces disjoint, so a race
    # decision can never alias a leaf result (or vice versa).
    leaf = journal_key(prompt="hi", agent_type="x", model=None, schema=None, isolation="shared")
    assert race_key(candidate_keys=[leaf], win_tag="") != leaf
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_journal.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: FAIL — `ImportError: cannot import name 'race_key'`.

- [ ] **Step 3: Add `race_key` to `_journal.py`**

Add `from collections.abc import Sequence` to the imports (the module currently imports only `from typing import Any, Protocol, runtime_checkable`; `hashlib` and `json` are already imported). Then add `race_key` immediately after `journal_key`:

```python
def race_key(*, candidate_keys: Sequence[str], win_tag: str) -> str:
    """Compute the content-hash key for a ``ctx.race`` decision.

    The key is the SHA-256 of a canonical JSON encoding of the candidates' leaf
    keys (in order) plus the win tag, under a ``"race"`` namespace marker so it can
    never collide with a leaf :func:`journal_key`. ``win_tag`` is folded in
    deliberately: two races over the *same* candidates but with *different* win
    predicates must use different tags, or the second race replays the first's
    decision and silently bypasses the changed predicate.

    Args:
        candidate_keys: Each candidate's leaf journal key, in candidate order.
        win_tag: A caller-supplied label distinguishing this race's win criterion.

    Returns:
        A hex SHA-256 digest uniquely identifying this race decision.
    """
    payload: dict[str, Any] = {
        "kind": "race",
        "candidates": list(candidate_keys),
        "win_tag": win_tag,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_journal.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: PASS (existing journal tests + the 3 new ones).

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/langchain_dynamic_workflow/_journal.py tests/unit/test_journal.py
uv run pyright src/langchain_dynamic_workflow/_journal.py
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_journal.py tests/unit/test_journal.py
git commit -m "$(cat <<'EOF'
feat(journal): race_key() content-hash key, race-namespaced + win_tag-folded

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `SpanKind.RACE` — observability span kind

**Files:**
- Modify: `src/langchain_dynamic_workflow/_observability.py`
- Test: `tests/unit/test_observability.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_observability.py`)

```python
def test_recorder_emits_a_race_span() -> None:
    emitted: list[Span] = []
    recorder = SpanRecorder(sink=emitted.append)

    with recorder.span(SpanKind.RACE, "high-confidence-root-cause") as span:
        span.set("candidate_count", 4)
        span.set("replayed", False)
        span.set("won", True)
        span.set("winner_index", 1)

    assert len(emitted) == 1
    completed = emitted[0]
    assert completed.kind == SpanKind.RACE
    assert completed.attributes["candidate_count"] == 4
    assert completed.attributes["won"] is True
    assert completed.attributes["winner_index"] == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_observability.py::test_recorder_emits_a_race_span -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: FAIL — `AttributeError: RACE` (the enum member does not exist yet).

- [ ] **Step 3: Add `RACE` to `SpanKind`**

In `src/langchain_dynamic_workflow/_observability.py`, extend the `SpanKind` enum — add the member and document it in the class docstring `Attributes:` block (the docstring convention forbids documenting a field in two places, so update the existing `Attributes:` list, not a separate comment):

```python
class SpanKind(StrEnum):
    """The orchestration primitive a span describes.

    Attributes:
        AGENT: A single leaf ``agent()`` invocation (or a journal hit).
        PARALLEL: A ``parallel()`` blocking-barrier fan-out.
        PIPELINE: A ``pipeline()`` no-barrier streaming fan-out.
        RACE: A ``race()`` best-of-N early-exit fan-out (or a journaled-decision replay).
    """

    AGENT = "agent"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    RACE = "race"
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_observability.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: PASS.

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/langchain_dynamic_workflow/_observability.py tests/unit/test_observability.py
uv run pyright src/langchain_dynamic_workflow/_observability.py
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_observability.py tests/unit/test_observability.py
git commit -m "$(cat <<'EOF'
feat(observability): add SpanKind.RACE

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `Ctx.race` — fresh path (dispatch, first-to-satisfy wins, cancel losers)

**Files:**
- Modify: `src/langchain_dynamic_workflow/_context.py`
- Test: `tests/unit/test_race.py`

> **What this task builds:** the full `Ctx.race` method **minus the replay short-circuit** (added in Task 5). It resolves candidates, derives the race-key, observes the determinism backstop at depth 0, dispatches all candidates concurrently by reusing `agent()`, picks the first that satisfies `win` (ascending-index tie-break), cancels the losers, and journals a self-contained decision envelope. On a fresh run there is no journal hit, so the replay branch added in Task 5 is dead code here — the unit tests below exercise only fresh behavior.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_race.py`:

```python
"""Unit tests for ``Ctx.race`` — fresh-path mechanics + loser cancellation.

These build a ``Ctx`` directly with a fake ``leaf_runner`` (mirroring
``tests/unit/test_parallel.py``) so each candidate's result, usage, and failure is
controlled without a real model. Replay / nesting / footgun behaviour is covered by
``tests/integration/test_race.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow._budget import Budget
from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._race_types import RaceCandidate
from langchain_dynamic_workflow._roster import Roster

import pytest


def _noop_runnable() -> Runnable[Any, Any]:
    """A roster placeholder; race tests drive results through a custom leaf_runner."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        return {"messages": []}

    return RunnableLambda(_call)


def _text_runner(
    results: dict[str, str], *, raise_on: frozenset[str] = frozenset(), usage: int = 0
) -> Any:
    """A fake leaf_runner returning a text result keyed by prompt (or raising)."""

    async def _leaf(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
    ) -> LeafOutcome:
        if prompt in raise_on:
            raise RuntimeError("fake leaf boom")
        return LeafOutcome(state={"messages": [AIMessage(content=results[prompt])]}, usage=usage)

    return _leaf


def _ctx(leaf_runner: Any, *, budget: Budget | None = None, gate: ConcurrencyGate | None = None) -> Ctx:
    return Ctx(
        roster=Roster()
        .register("inv", _noop_runnable())
        .register("winner", _noop_runnable())
        .register("loser", _noop_runnable()),
        journal=InMemoryJournalStore(),
        leaf_runner=leaf_runner,
        gate=gate if gate is not None else ConcurrencyGate(limit=8),
        budget=budget,
    )


async def test_race_first_to_satisfy_win_wins() -> None:
    ctx = _ctx(_text_runner({"h0": "lose", "h1": "WIN", "h2": "lose"}))
    result = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1", "h2")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.won is True
    assert result.winner == "WIN"
    assert result.winner_index == 1


async def test_race_ascending_index_tiebreak() -> None:
    # Every candidate satisfies win; the lowest input index must win regardless of
    # completion / set-iteration order, so the winner is deterministic.
    ctx = _ctx(_text_runner({"h0": "WIN", "h1": "WIN", "h2": "WIN"}))
    result = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1", "h2")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.winner_index == 0


async def test_race_no_winner_returns_unwon_result() -> None:
    ctx = _ctx(_text_runner({"h0": "lose", "h1": "lose"}))
    result = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.won is False
    assert result.winner is None and result.winner_index is None


async def test_race_failed_candidate_is_skipped_others_continue() -> None:
    # The lower-index candidate's leaf raises; it cannot win, and the next candidate
    # that satisfies win takes the race.
    ctx = _ctx(_text_runner({"h1": "WIN"}, raise_on=frozenset({"h0"})))
    result = await ctx.race(
        [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.winner == "WIN"
    assert result.winner_index == 1


async def test_race_empty_candidates_raises() -> None:
    ctx = _ctx(_text_runner({}))
    with pytest.raises(ValueError, match="at least one candidate"):
        await ctx.race([], win=lambda text: True, win_tag="t")


async def test_race_mixed_schema_raises() -> None:
    # One schema-less candidate + one schema candidate would make the winner type
    # ambiguous; the homogeneity guard fails fast before any dispatch.
    ctx = _ctx(_text_runner({"h0": "x", "h1": "y"}))
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with pytest.raises(ValueError, match="homogeneous"):
        await ctx.race(
            [
                RaceCandidate(prompt="h0", agent_type="inv"),
                RaceCandidate(prompt="h1", agent_type="inv", schema=schema),
            ],
            win=lambda obj: True,
            win_tag="t",
        )


async def test_race_predicate_raise_fails_loud() -> None:
    # The win predicate is script logic; a raise is a bug, not a leaf failure — it
    # must propagate (after the in-flight losers are torn down), never be swallowed.
    ctx = _ctx(_text_runner({"h0": "WIN", "h1": "lose"}))

    def boom(_text: str) -> bool:
        raise ValueError("predicate boom")

    with pytest.raises(ValueError, match="predicate boom"):
        await ctx.race(
            [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
            win=boom,
            win_tag="t",
        )


async def test_race_budget_signal_fails_loud() -> None:
    # An exhausted budget makes each candidate's agent() raise the engine
    # control-flow signal; the race must fail loud rather than mask it as a loser.
    ctx = _ctx(_text_runner({"h0": "WIN", "h1": "lose"}), budget=Budget(total=0))
    with pytest.raises(WorkflowBudgetExceededError):
        await ctx.race(
            [RaceCandidate(prompt=p, agent_type="inv") for p in ("h0", "h1")],
            win=lambda text: text == "WIN",
            win_tag="t",
        )


async def test_race_cancels_losers_and_releases_gate_slots() -> None:
    # The winner completes immediately; the losers block forever until cancelled.
    # On a win the race must cancel them, await their teardown (no orphans), and
    # release every gate slot they held.
    gate = ConcurrencyGate(limit=4)
    cancelled = {"count": 0}

    async def _leaf(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
    ) -> LeafOutcome:
        if agent_type == "winner":
            return LeafOutcome(state={"messages": [AIMessage(content="WIN")]}, usage=0)
        try:
            await asyncio.Event().wait()  # block until cancelled
        except asyncio.CancelledError:
            cancelled["count"] += 1
            raise
        return LeafOutcome(state={"messages": []}, usage=0)  # pragma: no cover

    ctx = _ctx(_leaf, gate=gate)
    result = await ctx.race(
        [
            RaceCandidate(prompt="w", agent_type="winner"),
            RaceCandidate(prompt="l1", agent_type="loser"),
            RaceCandidate(prompt="l2", agent_type="loser"),
        ],
        win=lambda text: text == "WIN",
        win_tag="t",
    )
    assert result.winner == "WIN" and result.winner_index == 0
    assert cancelled["count"] == 2  # both losers received CancelledError during teardown

    # No slot leaked: all `limit` slots must be acquirable SIMULTANEOUSLY again. If a
    # cancelled loser had leaked its slot, this all-in-flight barrier would dead-lock
    # — wait_for bounds it so the test fails loud instead of hanging.
    entered = asyncio.Semaphore(0)
    release = asyncio.Event()

    async def _occupy() -> None:
        async with gate:
            entered.release()
            await release.wait()

    occupants = [asyncio.ensure_future(_occupy()) for _ in range(gate.limit)]
    try:
        for _ in range(gate.limit):
            await asyncio.wait_for(entered.acquire(), timeout=1.0)
    finally:
        release.set()
        await asyncio.gather(*occupants)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_race.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m2.log
```
Expected: FAIL — `AttributeError: 'Ctx' object has no attribute 'race'`.

- [ ] **Step 3: Add imports + `Ctx.race` + `_run_race_candidate` to `_context.py`**

First extend the imports at the top of `_context.py`:
- add `import json` (next to `import asyncio` / `import contextvars`);
- add `cast` to the `from typing import ...` line → `from typing import Any, Protocol, TypeVar, cast, overload`;
- add `race_key` to the journal import → `from ._journal import JournalRecord, JournalStore, journal_key, race_key`;
- add a new import line for the race types (sorted between `._progress` and `._reduce`/`._result` per the existing ordering — place it before `from ._reduce import ...` is N/A here since `_context` does not import `_reduce`; add it after `from ._progress import ProgressKind, ProgressLog`): `from ._race_types import RaceCandidate, RaceResult`.

Then add the two methods at the end of the `Ctx` class (after `pipeline`):

```python
    async def _run_race_candidate(self, candidate: RaceCandidate) -> Any:
        """Dispatch one race candidate by reusing ``agent()`` verbatim.

        Forwarding through ``agent()`` (rather than re-implementing the leaf path)
        reuses the journal dedup, budget metering, sandbox admission, and span the
        leaf path already provides. The candidate runs at fan-out depth > 0 (the
        race frame is entered before this task is created), so it is excluded from
        the determinism sequence exactly like a ``parallel`` / ``pipeline`` leaf.
        """
        # Any-typed so the call is not matched against agent()'s overloads, which
        # are written per concrete schema type and do not accept the union the
        # candidate carries; the runtime forwarding is the same either way.
        agent_call: Any = self.agent
        return await agent_call(
            candidate.prompt,
            agent_type=candidate.agent_type,
            schema=candidate.schema,
            model=candidate.model,
            isolation=candidate.isolation,
        )

    async def race[T](
        self,
        candidates: Sequence[RaceCandidate],
        *,
        win: Callable[[T], bool],
        win_tag: str = "",
    ) -> RaceResult[T]:
        """Run candidates concurrently; the first whose result satisfies ``win`` wins.

        Best-of-N early exit: every candidate is dispatched via ``agent()`` at the
        same time, and the first to produce a result for which ``win`` returns
        ``True`` becomes the winner. The in-flight losers are then cancelled. When
        several candidates finish in the same scheduler wakeup the lowest input
        index wins, so the winner never depends on completion order.

        The decision is **journaled** under a content-hash race-key so resume is
        deterministic: a replayed race reproduces the recorded winner and dispatches
        **nothing** (the losers never re-run, so a resumed race is cheaper than the
        first one — by design). A race that produced no winner is **not** journaled,
        so a resume may retry it; use ``parallel`` when you want every result
        regardless of a predicate.

        All candidates must be homogeneous — either all schema-less (their results
        are ``str``) or all bound to the same schema (their results are that model)
        — so the winner's type is unambiguous.

        ``win_tag`` is folded into the race-key. Two races over the *same* candidates
        but with *different* win predicates **must** pass different ``win_tag``
        values; otherwise the second race replays the first's journaled decision and
        silently bypasses the changed predicate.

        Args:
            candidates: The agent-call specs to race; must be non-empty and
                homogeneous.
            win: Predicate over a candidate's result deciding whether it wins.
            win_tag: A label distinguishing this race's win criterion in the
                journal key (see the footgun note above). Defaults to ``""``.

        Returns:
            A :class:`RaceResult` carrying the winner and its index, or both
            ``None`` when no candidate satisfied ``win``.

        Raises:
            ValueError: If ``candidates`` is empty or not homogeneous.
            KeyError: If a candidate's ``agent_type`` is not registered.
            WorkflowBudgetExceededError: If a candidate's ``agent()`` trips the
                budget cap (re-raised after the in-flight losers are torn down).
            WorkflowDeterminismError: If the race-key diverges from the recorded
                replay sequence.
            Exception: Whatever ``win`` raises, re-raised after teardown (the
                predicate is script logic; a raise is a bug, not a leaf failure).
        """
        if not candidates:
            raise ValueError("race() requires at least one candidate; got an empty sequence")

        # Prelude (all synchronous, all before any dispatch): resolve each candidate
        # exactly as agent() will, derive its leaf key, and enforce homogeneity.
        leaf_keys: list[str] = []
        schema_signatures: set[str | None] = set()
        schema_model: type[BaseModel] | None = None
        for candidate in candidates:
            entry = self._roster.resolve(candidate.agent_type)  # fail fast on unknown agent_type
            if candidate.isolation == "worktree" and not entry.needs_execution:
                raise ValueError(
                    f"isolation='worktree' requires agent_type {candidate.agent_type!r} to be "
                    "registered with needs_execution=True (a worktree is seeded into an execution "
                    "sandbox; a reasoning leaf has none)"
                )
            candidate_model = (
                to_pydantic_model(candidate.schema) if candidate.schema is not None else None
            )
            effective_model = candidate.model if candidate.model is not None else entry.default_model
            leaf_keys.append(
                journal_key(
                    prompt=candidate.prompt,
                    agent_type=candidate.agent_type,
                    model=effective_model,
                    schema=candidate_model,
                    isolation=candidate.isolation,
                )
            )
            signature = (
                None
                if candidate_model is None
                else json.dumps(candidate_model.model_json_schema(), sort_keys=True)
            )
            schema_signatures.add(signature)
            schema_model = candidate_model
        if len(schema_signatures) != 1:
            raise ValueError(
                "race() candidates must be homogeneous: either all schema-less (text) or all "
                "bound to the same schema; got a mix, which would make RaceResult.winner's type "
                "ambiguous"
            )

        rkey = race_key(candidate_keys=leaf_keys, win_tag=win_tag)

        with self._spans.span(SpanKind.RACE, win_tag or "race") as span:
            span.set("candidate_count", len(candidates))
            # The race decision is one sequential step: its content-stable key is
            # recorded / validated once at depth 0. The candidate agent() calls run
            # at depth > 0 and are excluded from the sequence (their completion order
            # varies run to run), mirroring leaves inside parallel() / pipeline().
            if _FANOUT_DEPTH.get() == 0:
                self._sequence_guard.observe(rkey)
            span.set("replayed", False)

            # Fresh run: dispatch all candidates concurrently; first to satisfy wins.
            token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
            tasks = [
                asyncio.ensure_future(self._run_race_candidate(candidate))
                for candidate in candidates
            ]
            index_of = {task: index for index, task in enumerate(tasks)}
            winner_index: int | None = None
            winner_result: Any = None
            to_raise: BaseException | None = None
            try:
                remaining = set(tasks)
                while remaining and winner_index is None and to_raise is None:
                    done, remaining = await asyncio.wait(
                        remaining, return_when=asyncio.FIRST_COMPLETED
                    )
                    # Deterministic tie-break: decide same-wakeup completions in
                    # ascending candidate index, never set-iteration order.
                    for task in sorted(done, key=lambda finished: index_of[finished]):
                        error = task.exception()
                        if error is not None:
                            if isinstance(
                                error, (WorkflowBudgetExceededError, WorkflowDeterminismError)
                            ):
                                # Engine control-flow signal: fail loud, never mask.
                                to_raise = error
                                break
                            # Ordinary leaf failure: this candidate is out; others go on.
                            continue
                        candidate_result = task.result()
                        try:
                            satisfied = win(cast(T, candidate_result))
                        except Exception as predicate_error:
                            # Predicate raise is a script bug: fail loud after teardown.
                            to_raise = predicate_error
                            break
                        if satisfied:
                            winner_index = index_of[task]
                            winner_result = candidate_result
                            break
            finally:
                # Teardown: cancel every still-running loser and await all tasks so
                # none is orphaned and every gate slot is released. return_exceptions
                # absorbs the CancelledErrors (and any loser exception) raised here.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                _FANOUT_DEPTH.reset(token)

            if to_raise is not None:
                raise to_raise

            if winner_index is None:
                # No winner: do NOT journal a decision (a resume may retry the race).
                span.set("won", False)
                span.set("winner_index", None)
                return RaceResult[T](winner=None, winner_index=None)

            # Journal a self-contained decision under the namespaced race-key: the
            # winner index plus the canonical result string agent() produced, carrying
            # the winner's usage so resume rebuilds spend. The winner's own leaf entry
            # already recorded its usage under its leaf key on this fresh run, so the
            # race-key is NOT re-recorded into the budget (that would double-count).
            winner_record = await self._journal.get(leaf_keys[winner_index])
            winner_usage = winner_record.usage if winner_record is not None else 0
            if schema_model is not None:
                winner_result_str = cast(BaseModel, winner_result).model_dump_json(
                    by_alias=True, round_trip=True
                )
            else:
                winner_result_str = cast(str, winner_result)
            envelope = json.dumps({"winner_index": winner_index, "result": winner_result_str})
            await self._journal.put(rkey, JournalRecord(result=envelope, usage=winner_usage))
            span.set("won", True)
            span.set("winner_index", winner_index)
            return RaceResult[T](winner=cast(T, winner_result), winner_index=winner_index)
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_race.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m2.log
```
Expected: PASS (9 passed).

- [ ] **Step 5: Lint + type-check + import contracts**

```bash
uv run ruff check src/langchain_dynamic_workflow/_context.py tests/unit/test_race.py
uv run ruff format src/langchain_dynamic_workflow/_context.py tests/unit/test_race.py
uv run pyright src/langchain_dynamic_workflow/_context.py
uv run lint-imports
```
Expected: clean. `_context` importing `_race_types` + `race_key` is within Layer 1 (no contract touches it).

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_context.py tests/unit/test_race.py
git commit -m "$(cat <<'EOF'
feat(race): Ctx.race fresh path — first-to-satisfy wins, cancel losers, journal decision

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `Ctx.race` — replay short-circuit + resume/nesting/footgun integration

**Files:**
- Modify: `src/langchain_dynamic_workflow/_context.py`
- Test: `tests/integration/test_race.py`

> **What this task adds:** the journal-hit short-circuit. After `observe(rkey)` and before the fresh dispatch, the method reads the journaled decision and, on a hit, decodes the envelope and returns the winner **without dispatching any candidate**. This is what makes resume reproduce the winner at zero model cost.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/test_race.py`:

```python
"""Integration: ``ctx.race`` through the full ``run_workflow`` resume loop.

Drives the engine with deterministic fake leaves (no API keys) to prove: a
journaled race decision reproduces the winner on resume and dispatches no
candidate; a race nested inside ``parallel`` resumes by content hash even though
its key is excluded from the determinism sequence; and the win_tag footgun — two
races over identical candidates with the same (default) tag alias one decision.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_dynamic_workflow import Ctx, InMemoryJournalStore, Roster, run_workflow
from langchain_dynamic_workflow._race_types import RaceCandidate


def _counting_runnable(calls: dict[str, int]) -> Runnable[Any, Any]:
    """A leaf that counts dispatches and replies WIN to the prompt ending in '0'."""

    async def _call(inp: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        prompt = inp["messages"][0].content
        content = "WIN" if str(prompt).endswith("0") else "lose"
        return {"messages": [*inp["messages"], AIMessage(content=content)]}

    return RunnableLambda(_call)


async def test_race_replay_reproduces_winner_and_dispatches_nothing() -> None:
    calls = {"n": 0}
    roster = Roster().register("inv", _counting_runnable(calls))
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Ctx) -> tuple[int | None, Any]:
        result = await ctx.race(
            [RaceCandidate(prompt=f"h{i}", agent_type="inv") for i in range(3)],
            win=lambda text: text == "WIN",
            win_tag="x",
        )
        return (result.winner_index, result.winner)

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched_on_first = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    # h0 is the only WIN, so the winner is index 0 regardless of completion order.
    assert first == second == (0, "WIN")
    assert dispatched_on_first >= 1  # at least the winner ran on the fresh run
    assert calls["n"] == dispatched_on_first  # replay dispatched NOTHING


async def test_nested_race_in_parallel_resumes_by_content_hash() -> None:
    # A race inside parallel runs at fan-out depth > 0, so its race-key is excluded
    # from the determinism sequence — yet the decision is still journaled by content
    # hash, so the whole workflow resumes and the nested race dispatches nothing.
    calls = {"n": 0}
    roster = Roster().register("inv", _counting_runnable(calls))
    journal = InMemoryJournalStore()
    items = ["a", "b"]

    async def orchestrate(ctx: Ctx) -> list[Any]:
        async def race_for(item: str) -> Any:
            result = await ctx.race(
                [RaceCandidate(prompt=f"{item}-h{j}", agent_type="inv") for j in range(2)],
                win=lambda text: text == "WIN",
                win_tag="nested",
            )
            return result.winner

        return await ctx.parallel([lambda item=item: race_for(item) for item in sorted(items)])

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched_on_first = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert first == second == ["WIN", "WIN"]
    assert calls["n"] == dispatched_on_first  # nested race replayed with zero dispatch


async def test_same_default_win_tag_aliases_the_decision_footgun() -> None:
    # The footgun: two races over identical candidates with the same (default """)
    # win_tag share one race-key, so the second replays the first's decision even
    # though its predicate differs. A distinct win_tag keeps them independent.
    calls = {"n": 0}
    roster = Roster().register("inv", _counting_runnable(calls))

    async def orchestrate(ctx: Ctx) -> tuple[Any, Any, Any]:
        cands = [RaceCandidate(prompt=f"h{i}", agent_type="inv") for i in range(2)]
        first = await ctx.race(cands, win=lambda text: text == "WIN")  # default win_tag ""
        aliased = await ctx.race(cands, win=lambda text: text == "lose")  # SAME key -> aliases
        independent = await ctx.race(cands, win=lambda text: text == "lose", win_tag="distinct")
        return (first.winner, aliased.winner, independent.won)

    out = await run_workflow(orchestrate, roster=roster)
    first_winner, aliased_winner, independent_won = out
    # The aliased race returns the first race's winner (WIN), NOT a "lose" result.
    assert first_winner == "WIN"
    assert aliased_winner == "WIN"
    # The distinctly-tagged race runs for real: no candidate replies "lose" at h0, so
    # h0 (the only one the WIN-style winner would pick) is "WIN" not "lose" -> the
    # winner is the candidate replying "lose", which is h1; it found a winner.
    assert independent_won is True
```

> Note on the last assertion: `_counting_runnable` replies `WIN` only for the prompt ending in `0` (h0) and `lose` for everything else (h1). So `win=lambda t: t == "lose"` is satisfied by h1; the distinctly-tagged race finds a real winner (`won is True`) rather than aliasing the WIN decision.

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/integration/test_race.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m2.log
```
Expected: FAIL — the replay/alias tests fail because the method always re-dispatches (no journal short-circuit yet): `test_race_replay_reproduces_winner_and_dispatches_nothing` sees `calls["n"]` grow on the second run, and `test_same_default_win_tag_aliases_the_decision_footgun` sees `aliased_winner` come back `lose` (re-run) instead of `WIN`.

- [ ] **Step 3: Insert the replay short-circuit into `Ctx.race`**

In `_context.py`, inside `race`, between the `if _FANOUT_DEPTH.get() == 0: self._sequence_guard.observe(rkey)` block and the `span.set("replayed", False)` line, insert the journal-hit branch (and delete the now-misplaced `span.set("replayed", False)` that followed `observe`, replacing it with the explicit `True`/`False` sets below):

```python
            # Replay: a journaled race decision reproduces the winner deterministically
            # and dispatches NOTHING — the losers never re-run, so a resumed race is
            # cheaper than the first (correct: the decision is already made). The
            # envelope is self-contained so replay needs no candidate leaf entry.
            cached = await self._journal.get(rkey)
            if cached is not None:
                self._budget.record(rkey, cached.usage)
                decision = json.loads(cached.result)
                cached_index = int(decision["winner_index"])
                cached_result_str = decision["result"]
                decoded: Any = (
                    schema_model.model_validate_json(cached_result_str)
                    if schema_model is not None
                    else cached_result_str
                )
                span.set("replayed", True)
                span.set("won", True)
                span.set("winner_index", cached_index)
                return RaceResult[T](winner=cast(T, decoded), winner_index=cached_index)
            span.set("replayed", False)
```

The resulting order inside the `with self._spans.span(...)` block is: `span.set("candidate_count", ...)` → `observe` (depth 0) → **journal-hit branch (new)** → `span.set("replayed", False)` → fresh dispatch.

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/integration/test_race.py tests/unit/test_race.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m2.log
```
Expected: PASS (3 integration + 9 unit). The fresh-path unit tests still pass (empty journal → no hit).

- [ ] **Step 5: Lint + type-check + import contracts**

```bash
uv run ruff check src/langchain_dynamic_workflow/_context.py tests/integration/test_race.py
uv run pyright src/langchain_dynamic_workflow/_context.py tests/integration/test_race.py
uv run lint-imports
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_context.py tests/integration/test_race.py
git commit -m "$(cat <<'EOF'
feat(race): journaled-decision replay short-circuit (zero loser dispatch on resume)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Meta-layer injection (host side)

**Files:**
- Modify: `src/langchain_dynamic_workflow/_codegen.py`
- Test: `tests/unit/test_codegen.py`

> **Layer note:** `_codegen` (Layer 2) importing `_race_types` is permitted — `_race_types` is a pure types module, NOT on the import-linter `forbidden_modules` list (which is `_journal`, `_budget`, `_determinism`, `_pipeline`, `_concurrency`, `_sandbox`, `_progress`, `_context`). Do NOT add it. `_codegen` needs only the two types, never `race_key`.

- [ ] **Step 1: Write the failing test** — add to `tests/unit/test_codegen.py`

```python
async def test_run_script_can_construct_injected_race_types() -> None:
    # A host-authored script reaches the race value types by name (no import — the
    # AST gate forbids imports). Proves the meta-layer namespace injection delivers
    # the race primitive's surface to on-the-fly scripts, not only to imported
    # developer workflows.
    from langchain_dynamic_workflow._codegen import compile_workflow_source

    source = (
        "async def orchestrate(ctx, args):\n"
        "    candidates = [RaceCandidate(prompt=h, agent_type='inv') for h in args['hyps']]\n"
        "    probe = RaceResult(winner='x', winner_index=0)\n"
        "    return (len(candidates), candidates[0].agent_type, probe.won)\n"
    )
    orchestrate = compile_workflow_source(source)
    # Run the compiled body (it touches only the injected race types + literals,
    # never ctx) and assert the computed result — proving the injected names resolve
    # AND execute, not merely that they are present.
    unused_ctx: Any = object()
    result = await orchestrate(unused_ctx, {"hyps": ["a", "b", "c"]})
    assert result == (3, "inv", True)
```

(`Any` is already imported in `test_codegen.py`.)

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_codegen.py::test_run_script_can_construct_injected_race_types -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: FAIL — `NameError: name 'RaceCandidate' is not defined` (raised when the compiled body runs).

- [ ] **Step 3: Inject the race types in `_codegen.py`**

Add the import (the internal-import block is sorted by module name; `_race_types` sorts after `_journal` is N/A — `_codegen` imports `_reduce`, so place this import directly before `from ._reduce import (...)`):

```python
from ._race_types import RaceCandidate, RaceResult
```

Add an injection mapping next to `_SCRIPT_REDUCE_API`:

```python
_SCRIPT_RACE_API: dict[str, Any] = {
    "RaceCandidate": RaceCandidate,
    "RaceResult": RaceResult,
}
"""Race value types injected as script globals so a host-authored script constructs
``RaceCandidate`` specs and reads ``RaceResult`` by name without an import (the AST
gate forbids imports). ``ctx.race`` is a method, so it needs no injection."""
```

Merge it into the namespace construction (the `namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **_SCRIPT_REDUCE_API}` line):

```python
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        **_SCRIPT_REDUCE_API,
        **_SCRIPT_RACE_API,
    }
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_codegen.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: PASS (existing codegen tests + the new one).

- [ ] **Step 5: Lint + type-check + import contracts**

```bash
uv run ruff check src/langchain_dynamic_workflow/_codegen.py tests/unit/test_codegen.py
uv run pyright src/langchain_dynamic_workflow/_codegen.py
uv run lint-imports
```
Expected: clean; import-linter all contracts kept (confirms `_codegen` -> `_race_types` is allowed).

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_codegen.py tests/unit/test_codegen.py
git commit -m "$(cat <<'EOF'
feat(meta): inject RaceCandidate/RaceResult into the run_script namespace

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Public exports (developer side)

**Files:**
- Modify: `src/langchain_dynamic_workflow/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test** — add to `tests/test_smoke.py`

```python
def test_race_surface_exported_from_package_root() -> None:
    import langchain_dynamic_workflow as ldw

    for name in ("RaceCandidate", "RaceResult", "race_key"):
        assert name in ldw.__all__, f"{name} missing from __all__"
        assert hasattr(ldw, name), f"{name} not importable from the package root"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_smoke.py::test_race_surface_exported_from_package_root -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
```
Expected: FAIL — names missing from `__all__`.

- [ ] **Step 3: Add the imports + `__all__` entries in `__init__.py`**

Add `race_key` to the existing journal import:

```python
from ._journal import InMemoryJournalStore, JournalRecord, JournalStore, journal_key, race_key
```

Add a new import line for the race types (the import block is sorted by module name; `_race_types` sorts after `_progress` and before `_reduce`):

```python
from ._race_types import RaceCandidate, RaceResult
```

Add three entries to `__all__` (keep it alphabetically sorted — RUF022 enforces it; `ruff check --fix` reorders): `"RaceCandidate"`, `"RaceResult"`, `"race_key"`.

- [ ] **Step 4: Run to verify pass + auto-sort + checks**

```bash
uv run ruff check --fix src/langchain_dynamic_workflow/__init__.py
uv run ruff format src/langchain_dynamic_workflow/__init__.py
uv run pytest tests/test_smoke.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m2.log
uv run pyright src/langchain_dynamic_workflow/__init__.py
uv run lint-imports
```
Expected: smoke tests PASS; ruff/pyright/import-linter clean.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_dynamic_workflow/__init__.py tests/test_smoke.py
git commit -m "$(cat <<'EOF'
feat(race): export RaceCandidate/RaceResult/race_key from the package root

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: SKILL.md — teach `ctx.race`

**Files:**
- Modify: `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md`
- Test: `tests/unit/test_skill_patterns.py` (must stay green)

> **Before editing:** `tests/unit/test_skill_patterns.py` extracts every ```` ```python ```` block, parses it, and runs it through the AST gate (bare snippets are wrapped in an `orchestrate` skeleton). Keep every edited/added block gate-clean: no imports, no dunder access, no `str.format`, only safe builtins + injected names (`RaceCandidate`, `RaceResult`, the reduce helpers) + `ctx`. The injected names are valid free names in a `run_script` script.

- [ ] **Step 1: Add `ctx.race` to the DSL list** — in the `## The DSL (ctx primitives)` section, after the `ctx.pipeline(...)` bullet, add:

```markdown
- `await ctx.race(candidates, *, win, win_tag="")` — run several `RaceCandidate`
  specs concurrently and return a `RaceResult` for the **first** whose result
  satisfies `win(result)`; the in-flight losers are cancelled. `RaceCandidate(prompt,
  agent_type, schema=None, model=None, isolation="shared")` mirrors an `agent()`
  call; all candidates must be homogeneous (all schema-less, or all the same
  `schema`). Read `result.won` / `result.winner` / `result.winner_index`. Use this
  over `parallel` when you only need the first good-enough answer and want to stop
  the rest (e.g. multi-hypothesis diagnosis). `win_tag` distinguishes the win
  criterion in the resume journal — see the footgun note in the race pattern below.
```

- [ ] **Step 2: Add the race quality pattern** — at the end of the `## Quality patterns` section (after the reduce-helper closing note), add:

````markdown
**Race to the first good-enough answer (`ctx.race`).** When several independent
attempts could each solve a task and you only need the first that clears a bar,
race them and cancel the rest the moment one wins — far cheaper than waiting for a
`parallel` barrier when the slow attempts are wasted work. The classic case is
multi-hypothesis diagnosis: investigate every hypothesis at once, confirm the root
cause on the first high-confidence result, drop the others.

```python
async def orchestrate(ctx, args):
    hypotheses = sorted(args["hypotheses"])
    result = await ctx.race(
        [
            RaceCandidate(
                prompt=f"Investigate whether the incident root cause is: {h}",
                agent_type="investigator",
                schema={
                    "type": "object",
                    "properties": {
                        "root_cause": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["root_cause", "confidence"],
                    "additionalProperties": False,
                },
            )
            for h in hypotheses
        ],
        win=lambda d: d.confidence >= 0.8,
        win_tag="high-confidence-root-cause",
    )
    if result.won:
        return result.winner.root_cause  # the other hypotheses were cancelled
    ctx.log("no hypothesis reached high confidence")
    return None
```

`race` journals its decision, so a resume reproduces the same winner and re-runs
nothing. **Footgun — always set a distinct `win_tag` when you reuse the same
candidates with a different `win`.** The journal key folds in `win_tag` but not the
predicate, so two races over identical candidates with the same tag share one cached
decision: the second silently replays the first's winner instead of applying your
new criterion. A race that finds **no** winner is not journaled (a resume may retry
it); if you want every result regardless of a bar, use `parallel`, not `race`.
````

- [ ] **Step 3: Run the SKILL.md pattern test**

```bash
uv run pytest tests/unit/test_skill_patterns.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m2.log
```
Expected: PASS. The new block uses only `ctx`, the injected `RaceCandidate`, an inline dict schema, an f-string, and `sorted` — all gate-clean. If it fails, fix the block (not the gate).

- [ ] **Step 4: Commit**

```bash
git add src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md
git commit -m "$(cat <<'EOF'
docs(skill): teach ctx.race — multi-hypothesis race pattern + win_tag footgun

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: examples/13 — AI-SRE multi-hypothesis race (real-model E2E + pin test)

**Files:**
- Create: `examples/13_ai_sre_race_real_e2e.py`
- Create: `tests/integration/test_ai_sre_race.py`

> **Pattern source:** mirror `examples/12_screening_reconcile_real_e2e.py` — `from _demo_models import load_demo_env, real_model`, a real path under `LDW_DEMO_REAL_MODEL`, deterministic structured fakes offline, `schema=` leaves. The pin test mirrors `tests/integration/test_screening_reconcile.py` (importlib load + structured fakes + `run_workflow`).

> **道/术 demo rule (AGENTS.md):** this example is host-wiring code (a developer-authored workflow run via `run_workflow`), not a host-prompt demo, so the persona rule does not apply here — but keep it reading like real product code, not a tutorial.

- [ ] **Step 1: Create the example** `examples/13_ai_sre_race_real_e2e.py`

```python
"""Demo: AI-SRE multi-hypothesis diagnosis with ctx.race (early-exit + cancel).

An incident comes in; the workflow investigates several root-cause hypotheses in
parallel and takes the FIRST one that clears a confidence bar, cancelling the rest:

    investigate (race: one investigator leaf per hypothesis, each returns a
                 structured Diagnosis(root_cause, confidence, evidence))
      -> win = confidence >= 0.8  (first high-confidence hypothesis wins)
      -> the remaining investigators are cancelled

Shows race as a first-class early-exit primitive: latency is bounded by the first
good-enough answer, not the slowest hypothesis, and the decision is journaled so a
resume reproduces the same winner and re-runs nothing.

Run it:

    uv sync --group example
    export LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8
    uv run --group example python examples/13_ai_sre_race_real_e2e.py

With LDW_DEMO_REAL_MODEL unset it runs fully offline on deterministic fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from _demo_models import load_demo_env, real_model
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import (
    Ctx,
    RaceCandidate,
    Roster,
    run_workflow,
)

INCIDENT = (
    "Checkout latency spiked 10x at 02:14 UTC; error rate is flat, CPU is normal, "
    "and the spike began minutes after a routine deploy."
)
HYPOTHESES = [
    "a database connection-pool exhaustion",
    "a slow downstream payment provider",
    "a regression in the deploy that added a synchronous call on the hot path",
    "a noisy-neighbor effect from a co-located batch job",
]


class Diagnosis(BaseModel):
    """One investigator's structured verdict on a single hypothesis."""

    root_cause: str
    confidence: float
    evidence: str


def _investigate_prompt(incident: str, hypothesis: str) -> str:
    return (
        f"You are an SRE investigating an incident. Incident: {incident}\n"
        f"Hypothesis to test: {hypothesis}\n"
        "Assess whether this hypothesis is the root cause. Return `root_cause` (a short "
        "statement), `confidence` in [0, 1], and one-sentence `evidence`."
    )


async def diagnose(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    """investigate (race over hypotheses) -> first high-confidence root cause wins."""
    incident: str = args["incident"]

    ctx.phase("investigate")
    result = await ctx.race(
        [
            RaceCandidate(
                prompt=_investigate_prompt(incident, hypothesis),
                agent_type="investigator",
                schema=Diagnosis,
            )
            for hypothesis in HYPOTHESES
        ],
        win=lambda diagnosis: diagnosis.confidence >= 0.8,
        win_tag="high-confidence-root-cause",
    )

    if result.won and result.winner is not None:
        ctx.log(
            f"confirmed root cause (hypothesis #{result.winner_index}, "
            f"confidence {result.winner.confidence:.2f}): {result.winner.root_cause}"
        )
        return {
            "root_cause": result.winner.root_cause,
            "confidence": result.winner.confidence,
            "evidence": result.winner.evidence,
            "winner_index": result.winner_index,
        }
    ctx.log("no hypothesis reached high confidence")
    return {"root_cause": None, "confidence": None, "evidence": None, "winner_index": None}


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_structured_leaf(structured: BaseModel, *, reply: str) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def _build_investigator(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline: every investigator returns a high-confidence diagnosis, so the race
    # has a clear winner (the lowest-index hypothesis, by the ascending tie-break).
    return _fake_structured_leaf(
        Diagnosis(
            root_cause="synchronous call added on the hot path by the recent deploy",
            confidence=0.9,
            evidence="latency onset aligns with the deploy and tracks the new call",
        ),
        reply="investigated",
    )


async def main() -> None:
    load_demo_env()
    roster = Roster().register(
        "investigator",
        builder=_build_investigator,
        description="Tests one root-cause hypothesis and returns a structured Diagnosis",
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        return await diagnose(ctx, {"incident": INCIDENT})

    print(f"incident: {INCIDENT}")
    print(f"mode: {'REAL (OpenRouter)' if real_model() is not None else 'offline (fake)'}")
    result = await run_workflow(orchestrate, roster=roster)
    print(f"root_cause: {result['root_cause']}")
    print(f"confidence: {result['confidence']}")
    print(f"winning hypothesis index: {result['winner_index']}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Lint + type-check the example**

```bash
uv run ruff check examples/13_ai_sre_race_real_e2e.py
uv run ruff format examples/13_ai_sre_race_real_e2e.py
uv run pyright examples/13_ai_sre_race_real_e2e.py
```
Expected: clean. (`Diagnosis` uses only `str`/`float` fields, so no `model_rebuild()` is needed despite `from __future__ import annotations`.)

- [ ] **Step 3: Write the offline pin test** `tests/integration/test_ai_sre_race.py`

```python
"""Integration: the AI-SRE workflow (examples/13) runs ctx.race end to end.

Loads the runnable example and drives its ``diagnose`` workflow through
``run_workflow`` with a deterministic structured fake (no host, no API key). Pins
the race shape: a high-confidence diagnosis wins, the winner is the lowest-index
hypothesis (ascending tie-break), and a resume reproduces the same winner.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import InMemoryJournalStore, Roster, run_workflow


def _load_example() -> ModuleType:
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "13_ai_sre_race_real_e2e.py"
    spec = importlib.util.spec_from_file_location("_ldw_ai_sre_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structured_builder(make: Any) -> Any:
    def builder(*, response_format: Any = None) -> Any:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            return {
                "messages": [*inp["messages"], AIMessage(content="ok")],
                "structured_response": make(),
            }

        return RunnableLambda(_leaf)

    return builder


async def test_ai_sre_race_picks_a_high_confidence_winner() -> None:
    module = _load_example()
    roster = Roster().register(
        "investigator",
        builder=_structured_builder(
            lambda: module.Diagnosis(
                root_cause="deploy regression", confidence=0.9, evidence="onset aligns"
            )
        ),
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.diagnose(ctx, {"incident": "I"})

    result = await run_workflow(orchestrate, roster=roster)
    # Every hypothesis clears the 0.8 bar -> the lowest-index hypothesis wins.
    assert result["winner_index"] == 0
    assert result["root_cause"] == "deploy regression"
    assert result["confidence"] == 0.9


async def test_ai_sre_race_resume_reproduces_the_winner() -> None:
    module = _load_example()
    calls = {"n": 0}

    def _counting_builder(*, response_format: Any = None) -> Any:
        async def _leaf(
            inp: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            calls["n"] += 1
            return {
                "messages": [*inp["messages"], AIMessage(content="ok")],
                "structured_response": module.Diagnosis(
                    root_cause="deploy regression", confidence=0.9, evidence="onset aligns"
                ),
            }

        return RunnableLambda(_leaf)

    roster = Roster().register("investigator", builder=_counting_builder)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any) -> Any:
        return await module.diagnose(ctx, {"incident": "I"})

    first = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t1")
    dispatched = calls["n"]
    second = await run_workflow(orchestrate, roster=roster, journal=journal, thread_id="t2")

    assert first["winner_index"] == second["winner_index"] == 0
    assert first["root_cause"] == second["root_cause"]
    assert calls["n"] == dispatched  # the resumed race dispatched no investigator
```

- [ ] **Step 4: Run the pin tests**

```bash
uv run pytest tests/integration/test_ai_sre_race.py -q > /tmp/ldw-m2.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m2.log
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add examples/13_ai_sre_race_real_e2e.py tests/integration/test_ai_sre_race.py
git commit -m "$(cat <<'EOF'
example(13): AI-SRE multi-hypothesis race demo (ctx.race) + pin tests

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Evergreen docs sync

**Files:**
- Modify: `design_docs/01-engine-mechanism.md`, `design_docs/02-architecture.md`, `design_docs/uml/02-class.md`, `design_docs/uml/03-sequence.md`, `README.md`, `README_zh.md`

- [ ] **Step 1: Read each doc's relevant section first**

```bash
uv run rg -n "parallel|pipeline|public (surface|API)|公共面|SpanKind|primitive|原语" design_docs/01-engine-mechanism.md design_docs/02-architecture.md design_docs/uml/02-class.md design_docs/uml/03-sequence.md README.md README_zh.md
```

- [ ] **Step 2: Add `race` to the L1 primitive list** in `design_docs/01-engine-mechanism.md` and `design_docs/02-architecture.md` — wherever `agent` / `parallel` / `pipeline` are enumerated as the orchestration primitives, add `race` with a one-line description: "`race` — best-of-N early exit: first result satisfying `win` wins, in-flight losers cancelled, the decision journaled (content-hash, `win_tag`-keyed) so resume reproduces the winner and dispatches nothing." Note the two patches it relies on (content-hash journal for the race-key; the determinism guard observing the race-key at depth 0 only).

- [ ] **Step 3: Add the public surface entries** in `design_docs/01-engine-mechanism.md` and `README.md` / `README_zh.md` — wherever the exported surface / reduce helpers are listed, add `RaceCandidate`, `RaceResult`, `race_key`, and `ctx.race`, with the one-line description from Step 2 and a note that `RaceCandidate` / `RaceResult` are injected into the `run_script` namespace.

- [ ] **Step 4: Update the UML class diagram** `design_docs/uml/02-class.md` — add a `_race_types` box (`RaceCandidate`, `RaceResult` + `.won`), add `race_key` to `_journal`, add `RACE` to `SpanKind`, add `Ctx.race` to the `Ctx` class, and draw `_codegen --> _race_types` (injection) and `_context --> _race_types` / `_context --> _journal.race_key`.

- [ ] **Step 5: Add a race sequence** to `design_docs/uml/03-sequence.md` — a fresh-run sequence (resolve candidates → race_key → observe at depth 0 → dispatch N agents → first-satisfies-win → cancel losers → journal envelope) and a replay sequence (race_key → journal hit → decode envelope → return, no dispatch).

- [ ] **Step 6: Commit**

```bash
git add design_docs/ README.md README_zh.md
git commit -m "$(cat <<'EOF'
docs(evergreen): sync ctx.race into design_docs + README + UML

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Roadmap update (B/E split, streaming non-goal) + full gate + real E2E + Codex review

**Files:**
- Modify: `design_docs/v0_3_0_plans/00-roadmap.md`
- Else: verification + review (no source changes beyond review fixes)

- [ ] **Step 1: Update the roadmap for the B/E split** in `design_docs/v0_3_0_plans/00-roadmap.md`:
  - In the **Gap backlog** table, change the **M2** row to **B only** — title `B 早退/取消（race）`, Plan cell `✅ 已落地 · `[`02-b-journaled-race.md`](02-b-journaled-race.md). Remove the bundled `+E` from the M2 title.
  - Add a new milestone row for **E** (batch-map helper + count/ETA progress + streaming admission) as a later milestone (place it after M2 or fold into the "轻量并入项 / 后续" section — choose the lighter edit that keeps the table coherent), marked `待写`.
  - In the **M2 · B** 里程碑详述 section: scope it to race only; add an explicit **非目标** line — "真流式输出（与确定性 replay 冲突；race 已替延迟动机买单）" and "混合 schema race"; note E is split to its own later milestone.
  - In the **状态** section, add an **M2（B race）：✅ 已落地** bullet summarizing the delivery (`ctx.race` + `_race_types` + `race_key` + `SpanKind.RACE` + injection + exports + SKILL.md race pattern + examples/13 AI-SRE), and note E is deferred.

```bash
git add design_docs/v0_3_0_plans/00-roadmap.md
git commit -m "$(cat <<'EOF'
docs(roadmap): split M2 into B (race, done) and E (deferred); streaming non-goal

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Run the full offline gate** (master orchestrator must re-run this independently — never trust a subagent's "green" self-report)

```bash
uv run pytest -q > /tmp/ldw-m2-full.log 2>&1; echo "EXIT=$?"; tail -15 /tmp/ldw-m2-full.log
grep -E "FAILED|ERROR" /tmp/ldw-m2-full.log || echo "no failures"
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports
```
Expected: all tests pass (existing count + new race unit/integration/codegen/smoke/example-pin tests), coverage gate (85%) met, ruff/pyright/import-linter clean.

- [ ] **Step 3: Real-model E2E acceptance** (per memory `per-gap-real-e2e-acceptance`; `.env` has `OPENROUTER_API_KEY` — print variable NAMES only, never values; model = `claude-opus-4.8` per the host-capability note)

```bash
export LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8
uv run --group example python examples/13_ai_sre_race_real_e2e.py > /tmp/ldw-m2-e2e13.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-m2-e2e13.log
unset LDW_DEMO_REAL_MODEL
```
Expected: EXIT=0; the run investigates the hypotheses, the trace shows the race cancelling the losers once a high-confidence Diagnosis lands, and it prints a `root_cause` + winning hypothesis index. (It is acceptable to interrupt the run once the trace confirms the race fired and a winner was selected — full completion is not required to validate the mechanism.)

- [ ] **Step 4: Codex cross-model review** — dispatch a `codex:codex-rescue` review of the full diff (`git diff main...HEAD`), focused on:
  - **cancellation teardown** — no orphaned tasks, every gate slot released, no "Task was destroyed but it is pending" warning; the `finally` cancel + `gather(return_exceptions=True)` is correct under a winner, a no-winner, a predicate raise, and a budget signal;
  - **determinism correctness** — race-key observed at depth 0 only; nested-in-fan-out race excluded from the sequence yet still resumable by content hash; no under-run / over-run on resume;
  - **journal envelope** — encode/decode round-trips for both text and schema winners; the budget is not double-counted on the fresh path and is rebuilt on replay;
  - **win_tag footgun** — the alias behaviour is intended and documented; no silent predicate bypass beyond the documented case;
  - **homogeneity guard** — mixed schema / mixed text+schema rejected before dispatch.

  Fix every HIGH/MEDIUM finding with a TDD cycle (failing test → fix → green) and commit. (Across v0.2.0 and M1, Codex caught a real HIGH every time — treat a "clean" report skeptically and probe the cancellation path specifically.)

- [ ] **Step 5: Final gate after fixes**

```bash
uv run pytest -q > /tmp/ldw-m2-final.log 2>&1; echo "EXIT=$?"; tail -15 /tmp/ldw-m2-final.log
uv run ruff check . && uv run pyright && uv run lint-imports
```
Expected: all green.

- [ ] **Step 6: Hand off for PR** — stop here and report to the user. Per repo workflow the user reviews/merges PRs; use the `github-pr` skill when they say "开 pr". Do NOT open the PR or merge without the user's go-ahead.

---

## Self-Review (filled by plan author)

**Spec coverage** (against `docs/plans/2026-06-03-m2-b-journaled-race-design.md`):
- §2 decisions → API form/naming/return type → T1 (`RaceCandidate`/`RaceResult`, `.won`, no `.settled`); no-winner-not-journaled → T4/T5; homogeneity → T4; journal-as-`JournalRecord`-envelope (no `JournalStore` change) → T4/T5; `win_tag` default + footgun → T4 (key) / T5 (alias test) / T8 (doc); cancel via `wait(FIRST_COMPLETED)` + ascending tie-break + `finally` gather → T4; determinism observe at depth 0 → T4/T5; module placement (`_race_types` L1, `race_key` in `_journal`) → T1/T2; streaming non-goal → T11 (roadmap).
- §3 API signature → T4 (exact `race[T]` + `_run_race_candidate`).
- §4 race-key derivation → T2 (incl. win_tag folding + namespace + leaf-key match via the same `journal_key`).
- §5 control flow (fresh / replay) → T4 (fresh) + T5 (replay short-circuit).
- §6 edge/error matrix → T4 unit tests (empty, failure isolation, no-winner, predicate raise, budget signal, homogeneity, cancellation/gate release) + T5 (nested depth>0, win_tag alias).
- §7 injection / observability / import-linter → T6 (inject) + T3 (`SpanKind.RACE`) + T4/T6 (`lint-imports`).
- §8 tests/acceptance → T4/T5 (unit+integration) + T9 (examples/13 + offline pin + resume pin + real E2E) + T11 (Codex).
- §9 evergreen sync → T10 + T11 (roadmap).
- §10 YAGNI boundaries → honored: no true streaming, no mixed-schema race, no `JournalStore` Protocol change, no race-over-sub-workflow, E deferred.

**Placeholder scan:** no TBD / "handle edge cases" / "similar to"; every code step carries complete code; every command has an expected result.

**Type consistency:** `RaceCandidate(prompt, agent_type, schema=None, model=None, isolation="shared")`, `RaceResult[T](winner, winner_index)` + `.won`, `race_key(*, candidate_keys, win_tag)`, `Ctx.race(candidates, *, win, win_tag="")`, `Ctx._run_race_candidate(candidate)`, `SpanKind.RACE`, injected names `RaceCandidate` / `RaceResult` — identical across `_race_types.py`, `_journal.py`, `_context.py`, `_codegen.py` injection, `__init__.py`, tests, SKILL.md, and examples/13. The two injected names equal the two new exported value-type names; `race_key` is exported but not injected (the script uses `ctx.race`, never the key directly).

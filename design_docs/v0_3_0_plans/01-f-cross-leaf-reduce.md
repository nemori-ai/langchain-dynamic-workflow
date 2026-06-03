# M1 · F — Cross-leaf Reduce Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the recurring cross-leaf reductions (refute-by-default voting, de-duplication, dual-blind reviewer reconciliation, cross-leaf corroboration) from error-prone hand-written script Python into four named, tested, fail-safe pure functions — reachable both by developer-authored registered workflows (via import) and by host-authored `run_script` scripts (via meta-layer namespace injection).

**Architecture:** A new `_reduce.py` module holds four pure functions (`survives`, `dedup`, `reconcile`, `corroborate`) plus three frozen dataclasses (`ReviewItem`, `Reconciled`, `Consensus`). They consume the result lists returned by `ctx.parallel` / `ctx.pipeline` (a failed leaf is `None`); they hold no engine state and make no `agent()` calls, so they are inherently replay-safe. The package root exports them for developer workflows; `_codegen.py` injects the same seven names into the restricted `exec` namespace so host-authored scripts call them without an `import` (which the AST gate forbids).

**Tech Stack:** Python 3.12, pydantic v2 (existing schema leaves), pytest + pytest-asyncio (`asyncio_mode=auto`), ruff, pyright strict, import-linter. Spec: `docs/plans/2026-06-03-m1-f-cross-leaf-reduce-design.md`.

**Branch:** Do all work on `feat/m1-cross-leaf-reduce` (branch off `main` before Task 1).

---

## Setup (before Task 1)

- [ ] **Create the feature branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/m1-cross-leaf-reduce
```

---

## File Structure

| File | Responsibility |
|---|---|
| `src/langchain_dynamic_workflow/_reduce.py` | **Create.** The four reduce functions + three dataclasses. Pure, no engine coupling. |
| `src/langchain_dynamic_workflow/__init__.py` | **Modify.** Import + export the seven public names (developer side). |
| `src/langchain_dynamic_workflow/_codegen.py` | **Modify.** Inject the seven names into the `run_script` namespace (host side). |
| `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` | **Modify.** Teach the helpers; rewrite three patterns; add two. |
| `examples/07_deep_research_real_e2e.py` | **Modify.** Verify phase → `survives`; extract → `dedup`. Integration home for `survives`/`dedup`. |
| `examples/12_screening_reconcile_real_e2e.py` | **Create.** Corroborated screening: `corroborate` + `reconcile`. Integration home for those two. |
| `tests/unit/test_reduce.py` | **Create.** Pure unit tests for all four functions. |
| `tests/unit/test_codegen.py` | **Modify.** Add a meta-layer reachability test. |
| `tests/integration/test_screening_reconcile.py` | **Create.** Offline pin test for examples/12. |
| `design_docs/01-engine-mechanism.md`, `02-architecture.md`, `uml/02-class.md`, `README.md`, `README_zh.md`, `design_docs/v0_3_0_plans/00-roadmap.md` | **Modify.** Evergreen sync + roadmap status. |

---

## Task 1: `survives` — refute-by-default vote

**Files:**
- Create: `src/langchain_dynamic_workflow/_reduce.py`
- Test: `tests/unit/test_reduce.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_reduce.py`:

```python
"""Unit tests for the cross-leaf reduce helpers (pure functions over result lists)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from langchain_dynamic_workflow._reduce import survives


@dataclass
class _Vote:
    refuted: bool


def test_survives_when_refutes_below_kill_at() -> None:
    votes = [_Vote(refuted=False), _Vote(refuted=True), _Vote(refuted=False)]
    assert survives(votes, against=lambda v: v.refuted, kill_at=2) is True


def test_killed_when_refutes_reach_kill_at() -> None:
    votes = [_Vote(refuted=True), _Vote(refuted=True), _Vote(refuted=False)]
    assert survives(votes, against=lambda v: v.refuted, kill_at=2) is False


def test_none_vote_counts_as_against_failsafe() -> None:
    # Two failed leaves (None) + one explicit refute = 3 against >= kill_at -> killed.
    votes = [None, None, _Vote(refuted=False)]
    assert survives(votes, against=lambda v: v.refuted, kill_at=2) is False


def test_judge_panel_form_against_is_not_sound() -> None:
    @dataclass
    class _Ruling:
        sound: bool

    rulings = [_Ruling(sound=True), _Ruling(sound=True), _Ruling(sound=False)]
    # against = "not sound"; <2 unsound -> survives (2 of 3 sound).
    assert survives(rulings, against=lambda r: not r.sound, kill_at=2) is True


def test_empty_votes_raises() -> None:
    with pytest.raises(ValueError, match="at least one vote"):
        survives([], against=lambda v: v.refuted, kill_at=2)


def test_kill_at_below_one_raises() -> None:
    with pytest.raises(ValueError, match="kill_at must be >= 1"):
        survives([_Vote(refuted=False)], against=lambda v: v.refuted, kill_at=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'survives'`.

- [ ] **Step 3: Create `_reduce.py` with the module docstring + `survives`**

```python
"""Cross-leaf reduce — first-class helpers for folding many leaves' outputs into one.

The orchestration script fans out N leaves via ``ctx.parallel`` / ``ctx.pipeline``
and gets back a result list (a failed leaf is ``None``). These helpers turn the
recurring cross-leaf reductions — refute-by-default voting, de-duplication,
dual-blind reviewer reconciliation, cross-leaf corroboration — into named, tested,
fail-safe functions, so a script author never re-derives (and re-breaks) the
None-counting arithmetic. They are pure: no ``agent()`` call and no engine state,
so they are inherently replay-safe and never touch the journal or determinism guard.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable)
H = TypeVar("H", bound=Hashable)


def survives(votes: Sequence[T | None], *, against: Callable[[T], bool], kill_at: int) -> bool:
    """Return whether a thing survives a refute-by-default vote.

    Survives iff fewer than ``kill_at`` votes are 'against'. A ``None`` vote (a
    failed or absent leaf) ALWAYS counts as 'against' — the fail-safe so nothing is
    confirmed on missing verification. Covers adversarial-verify
    (``against=lambda v: v.refuted``) and judge-panel (``against=lambda v: not v.sound``).

    Args:
        votes: The leaves' verdicts in fan-out order; ``None`` marks a failed leaf.
        against: Predicate returning ``True`` when a (non-None) vote is against.
        kill_at: The number of 'against' votes that kills it (must be >= 1).

    Returns:
        ``True`` if the 'against' tally is below ``kill_at``.

    Raises:
        ValueError: If ``votes`` is empty (no verification ran) or ``kill_at < 1``.
    """
    if not votes:
        raise ValueError("survives() requires at least one vote; got an empty sequence")
    if kill_at < 1:
        raise ValueError(f"kill_at must be >= 1, got {kill_at}")
    against_count = sum(1 for vote in votes if vote is None or against(vote))
    return against_count < kill_at
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: PASS (6 passed).

- [ ] **Step 5: Lint + type-check the new module**

```bash
uv run ruff check src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
uv run ruff format src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
uv run pyright src/langchain_dynamic_workflow/_reduce.py
```
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
git commit -m "$(cat <<'EOF'
feat(reduce): survives() refute-by-default vote with None fail-safe

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `dedup` — drop None + de-duplicate by key

**Files:**
- Modify: `src/langchain_dynamic_workflow/_reduce.py`
- Test: `tests/unit/test_reduce.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_reduce.py`)

```python
from langchain_dynamic_workflow._reduce import dedup  # add to the import block


def test_dedup_preserves_first_seen_order() -> None:
    assert dedup(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_dedup_drops_none() -> None:
    assert dedup(["a", None, "b", None]) == ["a", "b"]


def test_dedup_merges_by_key() -> None:
    # str.lower merges case variants; the first-seen original is kept.
    assert dedup(["Alpha", "alpha", "BETA"], key=str.lower) == ["Alpha", "BETA"]


def test_dedup_empty_and_all_none() -> None:
    assert dedup([]) == []
    assert dedup([None, None]) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: FAIL — `ImportError: cannot import name 'dedup'`.

- [ ] **Step 3: Add `dedup` to `_reduce.py`** (after `survives`)

```python
@overload
def dedup(items: Iterable[H | None], *, key: None = ...) -> list[H]: ...
@overload
def dedup(items: Iterable[T | None], *, key: Callable[[T], K]) -> list[T]: ...
def dedup(
    items: Iterable[Any], *, key: Callable[[Any], Hashable] | None = None
) -> list[Any]:
    """Drop ``None`` and de-duplicate, preserving first-seen order.

    Args:
        items: The leaves' outputs; ``None`` (a failed leaf) is dropped.
        key: Maps an item to its identity (e.g. ``str.lower`` to merge case
            variants). Without it, the item itself is the key (items must be
            Hashable — enforced by the no-key overload, mirroring ``sorted(key=None)``).

    Returns:
        The kept items in first-seen order, one per distinct key.
    """
    seen: set[Hashable] = set()
    kept: list[Any] = []
    for item in items:
        if item is None:
            continue
        identity = item if key is None else key(item)
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(item)
    return kept
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: PASS (10 passed).

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
uv run pyright src/langchain_dynamic_workflow/_reduce.py
```
Expected: clean (the two `@overload` stubs give callers `list[H]` / `list[T]`; the impl is `Any`-typed by design).

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
git commit -m "$(cat <<'EOF'
feat(reduce): dedup() drop-None + key-merge, first-seen order

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `reconcile` + `ReviewItem` + `Reconciled` — dual-blind reconciliation

**Files:**
- Modify: `src/langchain_dynamic_workflow/_reduce.py`
- Test: `tests/unit/test_reduce.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
from langchain_dynamic_workflow._reduce import (  # extend the import block
    Reconciled,
    ReviewItem,
    reconcile,
)


@dataclass
class _Screen:
    keep: bool


def _items(*rows: tuple[str, list[_Screen | None]]) -> list[ReviewItem[str, _Screen]]:
    return [ReviewItem(item=name, verdicts=verdicts) for name, verdicts in rows]


def test_reconcile_three_buckets() -> None:
    review = _items(
        ("all-include", [_Screen(keep=True), _Screen(keep=True)]),
        ("all-exclude", [_Screen(keep=False), _Screen(keep=False)]),
        ("mixed", [_Screen(keep=True), _Screen(keep=False)]),
    )
    result = reconcile(review, include=lambda s: s.keep)
    assert result == Reconciled(
        included=["all-include"], excluded=["all-exclude"], conflicts=["mixed"]
    )


def test_reconcile_none_verdict_is_conflict_failsafe() -> None:
    review = _items(("had-a-failed-reviewer", [_Screen(keep=True), None]))
    result = reconcile(review, include=lambda s: s.keep)
    assert result.conflicts == ["had-a-failed-reviewer"]
    assert result.included == [] and result.excluded == []


def test_reconcile_empty_verdicts_is_conflict() -> None:
    review = _items(("no-reviews", []))
    assert reconcile(review, include=lambda s: s.keep).conflicts == ["no-reviews"]


def test_reconcile_empty_input() -> None:
    assert reconcile([], include=lambda s: s.keep) == Reconciled([], [], [])
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: FAIL — `ImportError: cannot import name 'reconcile'`.

- [ ] **Step 3: Add `ReviewItem`, `Reconciled`, `reconcile` to `_reduce.py`** (after `dedup`)

```python
@dataclass(frozen=True)
class ReviewItem(Generic[T, V]):
    """One item plus every reviewer's verdict on it (``None`` = that reviewer failed)."""

    item: T
    verdicts: Sequence[V | None]


@dataclass(frozen=True)
class Reconciled(Generic[T]):
    """The outcome of reconciling N independent reviewers over a set of items."""

    included: list[T]
    excluded: list[T]
    conflicts: list[T]


def reconcile(
    review_items: Sequence[ReviewItem[T, V]], *, include: Callable[[V], bool]
) -> Reconciled[T]:
    """Bucket items by independent-reviewer agreement (dual-blind screening, PRISMA-style).

    Per item: if any verdict is ``None`` or there are no verdicts, the item is a
    conflict (fail-safe: never auto-decide on missing review). Otherwise, if every
    reviewer would ``include`` it, it is included; if none would, it is excluded; a
    mix is a conflict to escalate.

    Args:
        review_items: Each item paired with its reviewers' verdicts.
        include: Predicate returning ``True`` when a verdict says 'include'.

    Returns:
        A :class:`Reconciled` partition into included / excluded / conflicts.
    """
    included: list[T] = []
    excluded: list[T] = []
    conflicts: list[T] = []
    for review in review_items:
        verdicts = review.verdicts
        if not verdicts or any(verdict is None for verdict in verdicts):
            conflicts.append(review.item)
            continue
        decisions = [include(verdict) for verdict in verdicts if verdict is not None]
        if all(decisions):
            included.append(review.item)
        elif not any(decisions):
            excluded.append(review.item)
        else:
            conflicts.append(review.item)
    return Reconciled(included=included, excluded=excluded, conflicts=conflicts)
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: PASS (14 passed).

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
uv run pyright src/langchain_dynamic_workflow/_reduce.py
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
git commit -m "$(cat <<'EOF'
feat(reduce): reconcile() dual-blind reviewer partition (None=conflict)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `corroborate` + `Consensus` — cross-leaf corroboration

**Files:**
- Modify: `src/langchain_dynamic_workflow/_reduce.py`
- Test: `tests/unit/test_reduce.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
from langchain_dynamic_workflow._reduce import Consensus, corroborate  # extend import block


def test_corroborate_keeps_groups_meeting_min_support() -> None:
    items = ["RAG", "rag", "long-ctx", "RAG"]  # key=str.lower: rag x3, long-ctx x1
    groups = corroborate(items, key=str.lower, min_support=2)
    assert groups == [Consensus(key="rag", members=["RAG", "rag", "RAG"])]


def test_corroborate_drops_none_then_groups() -> None:
    items = ["a", None, "A", None]
    assert corroborate(items, key=str.lower, min_support=2) == [
        Consensus(key="a", members=["a", "A"])
    ]


def test_corroborate_first_seen_key_order() -> None:
    items = ["b", "B", "a", "A"]
    keys = [g.key for g in corroborate(items, key=str.lower, min_support=2)]
    assert keys == ["b", "a"]


def test_corroborate_min_support_below_one_raises() -> None:
    with pytest.raises(ValueError, match="min_support must be >= 1"):
        corroborate(["a"], key=str.lower, min_support=0)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: FAIL — `ImportError: cannot import name 'corroborate'`.

- [ ] **Step 3: Add `Consensus` + `corroborate` to `_reduce.py`** (after `reconcile`)

```python
@dataclass(frozen=True)
class Consensus(Generic[K, T]):
    """A group of equivalent items that cleared the cross-leaf corroboration threshold."""

    key: K
    members: list[T]


def corroborate(
    items: Iterable[T | None], *, key: Callable[[T], K], min_support: int = 2
) -> list[Consensus[K, T]]:
    """Group equivalent items by ``key``; keep groups with >= ``min_support`` members.

    Cross-leaf corroboration: an item is kept only if at least ``min_support``
    leaves independently produced an equivalent item (same key). ``None`` (failed
    leaves) are dropped before grouping. Groups are returned in first-seen key order.

    Args:
        items: The leaves' outputs; ``None`` is dropped.
        key: Maps an item to its equivalence key.
        min_support: Minimum corroborating members for a group to survive (>= 1).

    Returns:
        The surviving groups, in first-seen key order.

    Raises:
        ValueError: If ``min_support < 1``.
    """
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support}")
    groups: dict[K, list[T]] = {}
    for item in items:
        if item is None:
            continue
        groups.setdefault(key(item), []).append(item)
    return [
        Consensus(key=group_key, members=members)
        for group_key, members in groups.items()
        if len(members) >= min_support
    ]
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_reduce.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: PASS (18 passed).

- [ ] **Step 5: Lint + type-check**

```bash
uv run ruff check src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
uv run pyright src/langchain_dynamic_workflow/_reduce.py
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_reduce.py tests/unit/test_reduce.py
git commit -m "$(cat <<'EOF'
feat(reduce): corroborate() group-by-key cross-leaf corroboration

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Public exports (developer side)

**Files:**
- Modify: `src/langchain_dynamic_workflow/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test** — add to `tests/test_smoke.py` a root-import assertion

```python
def test_reduce_helpers_exported_from_package_root() -> None:
    import langchain_dynamic_workflow as ldw

    for name in (
        "survives",
        "dedup",
        "reconcile",
        "corroborate",
        "ReviewItem",
        "Reconciled",
        "Consensus",
    ):
        assert name in ldw.__all__, f"{name} missing from __all__"
        assert hasattr(ldw, name), f"{name} not importable from the package root"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_smoke.py::test_reduce_helpers_exported_from_package_root -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: FAIL — names missing from `__all__`.

- [ ] **Step 3: Add the import + `__all__` entries in `__init__.py`**

Add the import line (the import block is sorted by module name; `_reduce` sorts between `_progress` and `_result`, i.e. immediately before the `from ._result import fold_result` line):

```python
from ._reduce import (
    Consensus,
    Reconciled,
    ReviewItem,
    corroborate,
    dedup,
    reconcile,
    survives,
)
```

Add these seven entries to `__all__` (keep it alphabetically sorted — ruff's RUF rules enforce it; `ruff check --fix` will reorder if needed):
`"Consensus"`, `"Reconciled"`, `"ReviewItem"`, `"corroborate"`, `"dedup"`, `"reconcile"`, `"survives"`.

- [ ] **Step 4: Run to verify pass + auto-sort + checks**

```bash
uv run ruff check --fix src/langchain_dynamic_workflow/__init__.py
uv run ruff format src/langchain_dynamic_workflow/__init__.py
uv run pytest tests/test_smoke.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
uv run pyright src/langchain_dynamic_workflow/__init__.py
uv run lint-imports
```
Expected: smoke tests PASS; ruff/pyright clean; import-linter all contracts kept (`_reduce` is a pure L1 module; no layer is affected).

- [ ] **Step 5: Commit**

```bash
git add src/langchain_dynamic_workflow/__init__.py tests/test_smoke.py
git commit -m "$(cat <<'EOF'
feat(reduce): export survives/dedup/reconcile/corroborate + types from root

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Meta-layer injection (host side)

**Files:**
- Modify: `src/langchain_dynamic_workflow/_codegen.py`
- Test: `tests/unit/test_codegen.py`

> **Layer note:** `_codegen` (Layer 2) importing `_reduce` is permitted. The import-linter "Layer 2 must not reach Layer 0/1 internals" contract lists explicit `forbidden_modules` (`_journal`, `_budget`, `_determinism`, `_pipeline`, `_concurrency`, `_sandbox`, `_progress`, `_context`); `_reduce` is **not** among them (it is a pure helper, like `_result`). Do NOT add `_reduce` to that list — the injection depends on this import.

- [ ] **Step 1: Write the failing test** — add to `tests/unit/test_codegen.py`

```python
async def test_run_script_can_call_injected_reduce_helpers() -> None:
    # A host-authored script reaches the reduce helpers by name (no import — the AST
    # gate forbids imports). Proves the meta-layer namespace injection delivers F to
    # the host's on-the-fly scripts, not only to imported developer workflows.
    from langchain_dynamic_workflow._codegen import compile_workflow_source

    source = (
        "async def orchestrate(ctx, args):\n"
        "    votes = [{'refuted': False}, {'refuted': True}, {'refuted': False}]\n"
        "    kept = survives(votes, against=lambda v: v['refuted'], kill_at=2)\n"
        "    groups = corroborate(['a', 'A', 'b'], key=lambda s: s.lower(), min_support=2)\n"
        "    review = [ReviewItem(item='x', verdicts=[{'k': True}, {'k': True}])]\n"
        "    bucket = reconcile(review, include=lambda v: v['k'])\n"
        "    uniq = dedup(['a', 'a', 'b'])\n"
        "    return (kept, len(groups), bucket.included, uniq)\n"
    )
    orchestrate = compile_workflow_source(source)
    # The closure's __globals__ carries the injected names; call it to confirm.
    assert orchestrate.__globals__["survives"] is not None
    assert orchestrate.__globals__["ReviewItem"] is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/unit/test_codegen.py::test_run_script_can_call_injected_reduce_helpers -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: FAIL — `KeyError: 'survives'` (the name is not in the namespace yet).

- [ ] **Step 3: Inject the helpers in `_codegen.py`**

Add the import near the other internal imports (top of file, sorted block):

```python
from ._reduce import (
    Consensus,
    Reconciled,
    ReviewItem,
    corroborate,
    dedup,
    reconcile,
    survives,
)
```

Add the injection mapping next to `_SAFE_BUILTINS` (after its definition, ~line 76):

```python
_SCRIPT_REDUCE_API: dict[str, Any] = {
    "survives": survives,
    "dedup": dedup,
    "reconcile": reconcile,
    "corroborate": corroborate,
    "ReviewItem": ReviewItem,
    "Reconciled": Reconciled,
    "Consensus": Consensus,
}
"""Cross-leaf reduce helpers injected as script globals so a host-authored script
calls them by name without an import (the AST gate forbids imports)."""
```

Change the namespace construction (the `namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}` line, ~line 107) to merge the reduce API:

```python
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, **_SCRIPT_REDUCE_API}
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest tests/unit/test_codegen.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1.log
```
Expected: PASS (existing codegen tests + the new one).

- [ ] **Step 5: Lint + type-check + import contracts**

```bash
uv run ruff check src/langchain_dynamic_workflow/_codegen.py tests/unit/test_codegen.py
uv run pyright src/langchain_dynamic_workflow/_codegen.py
uv run lint-imports
```
Expected: clean; import-linter all contracts kept (confirms `_codegen` -> `_reduce` is allowed).

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_codegen.py tests/unit/test_codegen.py
git commit -m "$(cat <<'EOF'
feat(meta): inject reduce helpers into the run_script namespace

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: SKILL.md — teach the helpers

**Files:**
- Modify: `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md`
- Test: `tests/unit/test_skill_patterns.py` (verify it still passes after edits)

> **Before editing:** read `tests/unit/test_skill_patterns.py` to see how it extracts and validates the SKILL.md code blocks (it runs them through the AST gate / checks shape). Keep every edited code block gate-clean (no imports, no dunder access) so that test stays green. The injected names (`survives`, etc.) are now valid free names in a `run_script` script.

- [ ] **Step 1: Rewrite the "Adversarial verify" pattern** to use `survives`

Replace the hand-written tally
```python
        refutes = sum(1 for v in votes if v is None or v.refuted)
        if refutes < 2:  # survives a 3-skeptic majority
            confirmed.append(claim)
```
with
```python
        if survives(votes, against=lambda v: v.refuted, kill_at=2):
            confirmed.append(claim)
```
and add a sentence: "`survives` bakes in the fail-safe — a `None` (failed skeptic) counts as a refutation, so a claim is never confirmed on absent verification. It is available by name in a `run_script` script (no import needed)."

- [ ] **Step 2: Rewrite the "Judge panel" pattern** to use `survives`

Replace
```python
    passes = sum(1 for r in rulings if r is not None and r.sound)
    return "accepted" if passes >= 2 else "rejected"
```
with
```python
    return "accepted" if survives(rulings, against=lambda r: not r.sound, kill_at=2) else "rejected"
```

- [ ] **Step 3: Rewrite the "Fan out → reduce in Python" pattern** to use `dedup`

Replace
```python
    kept = sorted({f.strip() for f in findings if f})  # reduce in plain Python
```
with
```python
    kept = sorted(dedup(f.strip() for f in findings if f))  # dedup() drops None + de-dupes
```

- [ ] **Step 4: Add two new patterns** at the end of the quality-patterns section

````markdown
**Cross-leaf corroboration (`corroborate`).** When several leaves research the same
space, keep only what *more than one* of them independently produced. `corroborate`
groups equivalent items by a key and keeps groups with enough support — a far
stronger signal than any single leaf. Available by name in `run_script`.

```python
async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"State one fact about {t}", agent_type="researcher", schema={
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
            "additionalProperties": False,
        }) for t in topics]
    )
    groups = corroborate(findings, key=lambda f: f.fact.strip().lower(), min_support=2)
    return [g.members[0].fact for g in groups]  # one representative per corroborated group
```

**Dual-blind reconciliation (`reconcile`).** Two (or more) independent reviewers
screen each item; the script keeps only what they unanimously include, drops what
they unanimously exclude, and escalates disagreements (or a failed reviewer) as
conflicts. The fan-out stays explicit; `reconcile` just buckets the verdicts.

```python
async def orchestrate(ctx, args):
    records = sorted(args["records"])
    review = []
    for record in records:
        verdicts = await ctx.parallel(
            [lambda r=record, n=n: ctx.agent(
                f"Reviewer #{n + 1}: should this record be INCLUDED in the review? {r}",
                agent_type="screener",
                schema={
                    "type": "object",
                    "properties": {"keep": {"type": "boolean"}},
                    "required": ["keep"],
                    "additionalProperties": False,
                },
            ) for n in range(2)]
        )
        review.append(ReviewItem(item=record, verdicts=verdicts))
    result = reconcile(review, include=lambda v: v.keep)
    ctx.log(f"included {len(result.included)}, conflicts {len(result.conflicts)} to escalate")
    return result.included
```
````

- [ ] **Step 5: Update the closing note** about the reduce surface — after the "A judge in any of these patterns ideally cannot edit" paragraph, add:

```markdown
The reduce helpers — `survives`, `dedup`, `reconcile`, `corroborate` (and the
`ReviewItem` / `Reconciled` / `Consensus` types) — are available by name inside a
`run_script` script (injected into the namespace); you do not import them. They are
pure functions over the result list `ctx.parallel` / `ctx.pipeline` hands back, so
the fan-out stays explicit and the reduce stays correct.
```

- [ ] **Step 6: Run the SKILL.md pattern test + full suite spot-check**

```bash
uv run pytest tests/unit/test_skill_patterns.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m1.log
```
Expected: PASS. If the test compiles each block through `compile_workflow_source`, the injected names resolve and blocks stay gate-clean. If a block fails, fix the block (not the gate).

- [ ] **Step 7: Commit**

```bash
git add src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md
git commit -m "$(cat <<'EOF'
docs(skill): teach the reduce helpers; add corroborate + reconcile patterns

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: examples/07 integration — `survives` + `dedup`

**Files:**
- Modify: `examples/07_deep_research_real_e2e.py`
- Test: `tests/integration/test_phase7_deep_research.py` (must stay green — do not edit it)

- [ ] **Step 1: Swap the verify-phase tally to `survives`**

In `deep_research`, replace
```python
        refutes = sum(1 for verdict in verdicts if verdict is not None and verdict.refuted)
        survived = refutes < REFUTATIONS_TO_KILL
        mark = "kept" if survived else "killed"
        ctx.log(f"claim {mark} ({refutes}/{SKEPTICS_PER_CLAIM} refute): {claim.text.strip()[:50]}")
        if survived:
            confirmed.append(claim.text)
```
with
```python
        survived = survives(verdicts, against=lambda v: v.refuted, kill_at=REFUTATIONS_TO_KILL)
        mark = "kept" if survived else "killed"
        ctx.log(f"claim {mark}: {claim.text.strip()[:50]}")
        if survived:
            confirmed.append(claim.text)
```

- [ ] **Step 2: De-dupe claims across angles with `dedup`** before the verify loop

Replace
```python
    claims = [c for c in await ctx.pipeline(paired, _extract) if c is not None and c.checkable]
    ctx.log(f"extracted {len(claims)} checkable claims")
```
with
```python
    extracted = [c for c in await ctx.pipeline(paired, _extract) if c is not None and c.checkable]
    claims = dedup(extracted, key=lambda c: c.text.strip().lower())
    ctx.log(f"extracted {len(claims)} checkable claims ({len(extracted) - len(claims)} dups merged)")
```

- [ ] **Step 3: Add the import** at the top of `examples/07_deep_research_real_e2e.py`

Add `dedup` and `survives` to the `from langchain_dynamic_workflow import (...)` block (keep it sorted; ruff enforces).

- [ ] **Step 4: Run the pinned integration test — it MUST stay green**

```bash
uv run pytest tests/integration/test_phase7_deep_research.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m1.log
```
Expected: PASS (2 passed). The fake extractor yields distinct claim texts (`claim #1`..`#N`), so `dedup` merges nothing and the pinned `skeptic == angles * skeptics` count is preserved; `survives` is behaviour-identical to the old `refutes < REFUTATIONS_TO_KILL`. If counts shift, the example logic diverged — fix the example, not the test.

- [ ] **Step 5: Lint + type-check the example**

```bash
uv run ruff check examples/07_deep_research_real_e2e.py
uv run ruff format examples/07_deep_research_real_e2e.py
uv run pyright examples/07_deep_research_real_e2e.py
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add examples/07_deep_research_real_e2e.py
git commit -m "$(cat <<'EOF'
example(07): use survives() + dedup() in deep_research verify/extract

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: examples/12 — corroborated screening (`corroborate` + `reconcile`)

**Files:**
- Create: `examples/12_screening_reconcile_real_e2e.py`
- Create: `tests/integration/test_screening_reconcile.py`

> **Pattern source:** mirror `examples/07_deep_research_real_e2e.py` — `from _demo_models import load_demo_env, real_model`, a real path under `LDW_DEMO_REAL_MODEL`, deterministic fakes offline, structured (`schema`) leaves. The pin test mirrors `tests/integration/test_phase7_deep_research.py` (importlib load + counting fakes + `run_workflow`).

- [ ] **Step 1: Create the example** `examples/12_screening_reconcile_real_e2e.py`

```python
"""Demo: corroborated screening — corroborate() then reconcile().

A workflow that screens claims about a topic:

    gather (parallel: one source-leaf per source, each emits a candidate claim)
      -> corroborate (keep claims >= 2 sources independently produced — cross-leaf backing)
      -> screen (parallel: two independent screeners per corroborated claim, include/exclude)
      -> reconcile (unanimous-include kept, unanimous-exclude dropped, disagreement escalated)

Shows the cross-leaf reduce helpers as first-class, fail-safe functions: corroborate
groups equivalent claims and drops the unsupported; reconcile buckets the dual-blind
screening verdicts (a failed screener -> conflict, never a silent include).

Run it:

    uv sync --group example
    export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
    uv run python examples/12_screening_reconcile_real_e2e.py

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
    ReviewItem,
    Roster,
    corroborate,
    reconcile,
    run_workflow,
)

TOPIC = "the operational trade-offs of microservices vs a monolith"
SOURCES = ["a platform engineering report", "a postmortem blog", "a vendor whitepaper", "a survey"]
SCREENERS = 2


class Candidate(BaseModel):
    """One candidate claim a source-leaf surfaces about the topic."""

    claim: str


class Screen(BaseModel):
    """One screener's include/exclude decision on a corroborated claim."""

    keep: bool
    reason: str


def _gather_prompt(topic: str, source: str) -> str:
    return (
        f"From the perspective of {source}, state THE single most important claim about "
        f"{topic}. Answer with one short, self-contained sentence in `claim`."
    )


def _screen_prompt(topic: str, claim: str, n: int) -> str:
    return (
        f"You are screener #{n + 1}. Decide whether this claim about {topic} is specific and "
        f"defensible enough to INCLUDE in a briefing. Set `keep` and give one-sentence `reason`.\n"
        f"Claim: {claim}"
    )


async def screening(ctx: Ctx, args: dict[str, Any]) -> dict[str, list[str]]:
    """gather -> corroborate -> dual-blind screen -> reconcile."""
    topic: str = args["topic"]

    ctx.phase("gather")
    candidates = await ctx.parallel(
        [
            lambda s=s: ctx.agent(_gather_prompt(topic, s), agent_type="source", schema=Candidate)
            for s in SOURCES
        ]
    )

    # Cross-leaf corroboration: keep claims at least two sources independently produced.
    groups = corroborate(
        candidates, key=lambda c: c.claim.strip().lower(), min_support=2
    )
    corroborated = [group.members[0].claim for group in groups]
    ctx.log(f"corroborated {len(corroborated)} of {len(SOURCES)} source claims")

    ctx.phase("screen")
    review: list[ReviewItem[str, Screen]] = []
    for claim in corroborated:
        verdicts = await ctx.parallel(
            [
                lambda c=claim, n=n: ctx.agent(
                    _screen_prompt(topic, c, n), agent_type="screener", schema=Screen
                )
                for n in range(SCREENERS)
            ]
        )
        review.append(ReviewItem(item=claim, verdicts=verdicts))

    result = reconcile(review, include=lambda v: v.keep)
    ctx.log(
        f"included {len(result.included)}, excluded {len(result.excluded)}, "
        f"conflicts {len(result.conflicts)}"
    )
    return {
        "included": result.included,
        "excluded": result.excluded,
        "conflicts": result.conflicts,
    }


# ── leaves (real deepagents when env-gated, deterministic fakes offline) ──────


def _fake_structured_leaf(structured: BaseModel, *, reply: str) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content=reply)],
            "structured_response": structured,
        }

    return RunnableLambda(_leaf)


def _build_source(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline: every source emits the SAME claim, so it corroborates (>= 2 sources).
    return _fake_structured_leaf(
        Candidate(claim="Microservices trade local simplicity for operational complexity"),
        reply="surfaced a claim",
    )


def _build_screener(*, response_format: Any = None) -> Any:
    model = real_model()
    if model is not None:
        return create_deep_agent(model=model, response_format=response_format)
    # Offline screeners both include, so the corroborated claim is unanimously kept.
    return _fake_structured_leaf(Screen(keep=True, reason="specific and defensible"), reply="screened")


async def main() -> None:
    load_demo_env()
    roster = (
        Roster()
        .register("source", builder=_build_source, description="Surfaces one candidate claim")
        .register("screener", builder=_build_screener, description="Includes/excludes a claim")
    )

    async def orchestrate(ctx: Ctx) -> dict[str, list[str]]:
        return await screening(ctx, {"topic": TOPIC})

    print(f"topic: {TOPIC}")
    print(f"mode: {'REAL (OpenRouter)' if real_model() is not None else 'offline (fake)'}")
    result = await run_workflow(orchestrate, roster=roster)
    print(f"included: {result['included']}")
    print(f"excluded: {result['excluded']}")
    print(f"conflicts: {result['conflicts']}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Lint + type-check the example**

```bash
uv run ruff check examples/12_screening_reconcile_real_e2e.py
uv run ruff format examples/12_screening_reconcile_real_e2e.py
uv run pyright examples/12_screening_reconcile_real_e2e.py
```
Expected: clean.

- [ ] **Step 3: Write the offline pin test** `tests/integration/test_screening_reconcile.py`

```python
"""Integration: the screening workflow (examples/12) runs corroborate + reconcile end to end.

Loads the runnable example and drives its ``screening`` workflow through ``run_workflow``
with deterministic structured fakes (no host, no API key). Pins the reduce shape: every
source emits the same claim (so it corroborates), both screeners include (so it lands in
``included``), and a third path proves a failed screener routes the claim to ``conflicts``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import Roster, run_workflow


def _load_example() -> ModuleType:
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    path = examples_dir / "12_screening_reconcile_real_e2e.py"
    spec = importlib.util.spec_from_file_location("_ldw_screening_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _structured_builder(make: Any) -> Any:
    def builder(*, response_format: Any = None) -> Any:
        async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
            return {
                "messages": [*inp["messages"], AIMessage(content="ok")],
                "structured_response": make(),
            }

        return RunnableLambda(_leaf)

    return builder


async def test_screening_includes_a_corroborated_unanimous_claim() -> None:
    module = _load_example()
    roster = (
        Roster()
        .register(
            "source",
            builder=_structured_builder(
                lambda: module.Candidate(claim="Shared claim across sources")
            ),
        )
        .register(
            "screener",
            builder=_structured_builder(lambda: module.Screen(keep=True, reason="ok")),
        )
    )

    async def orchestrate(ctx: Any) -> Any:
        return await module.screening(ctx, {"topic": "T"})

    result = await run_workflow(orchestrate, roster=roster)
    # Every source emits the same claim -> corroborated; both screeners include -> kept.
    assert result["included"] == ["Shared claim across sources"]
    assert result["excluded"] == [] and result["conflicts"] == []
```

- [ ] **Step 4: Run the pin test**

```bash
uv run pytest tests/integration/test_screening_reconcile.py -q > /tmp/ldw-m1.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-m1.log
```
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add examples/12_screening_reconcile_real_e2e.py tests/integration/test_screening_reconcile.py
git commit -m "$(cat <<'EOF'
example(12): corroborated screening demo (corroborate + reconcile) + pin test

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Evergreen docs sync

**Files:**
- Modify: `design_docs/01-engine-mechanism.md`, `design_docs/02-architecture.md`, `design_docs/uml/02-class.md`, `README.md`, `README_zh.md`, `design_docs/v0_3_0_plans/00-roadmap.md`

- [ ] **Step 1: Read each doc's relevant section first**

```bash
uv run rg -n "fold_result|public (surface|API)|公共面|reduce" design_docs/01-engine-mechanism.md design_docs/02-architecture.md design_docs/uml/02-class.md README.md README_zh.md
```

- [ ] **Step 2: Add the reduce helpers to the public surface** in `design_docs/01-engine-mechanism.md` and `README.md` / `README_zh.md` — wherever `fold_result` / the exported surface is listed, add the four functions + three types with a one-line description: "cross-leaf reduce: `survives` (refute-by-default vote), `dedup`, `reconcile` (dual-blind), `corroborate` (cross-leaf corroboration); also injected into the `run_script` namespace."

- [ ] **Step 3: Add the `_reduce` module + dataclasses to** `design_docs/uml/02-class.md` (a `_reduce` box with `survives`/`dedup`/`reconcile`/`corroborate` + `ReviewItem`/`Reconciled`/`Consensus`; note `_codegen` depends on `_reduce` for injection).

- [ ] **Step 4: Mark M1 done** in `design_docs/v0_3_0_plans/00-roadmap.md` — change the M1 row's Plan cell from "首刀 · plan 待写" to "✅ 已落地" and update the 状态 section's M1 bullet.

- [ ] **Step 5: Lint the markdown-embedded code (if any) + commit**

```bash
git add design_docs/ README.md README_zh.md
git commit -m "$(cat <<'EOF'
docs(evergreen): sync reduce helpers into design_docs + README + roadmap

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Full gate + real-model E2E acceptance + Codex review

**Files:** none (verification + review)

- [ ] **Step 1: Run the full offline gate**

```bash
uv run pytest -q > /tmp/ldw-m1-full.log 2>&1; echo "EXIT=$?"; tail -12 /tmp/ldw-m1-full.log
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run lint-imports
```
Expected: all tests pass (existing count + the new `test_reduce.py` / codegen / screening tests), coverage gate met, ruff/pyright/import-linter clean.

- [ ] **Step 2: Real-model E2E acceptance** (per memory `per-gap-real-e2e-acceptance` — `.env` has `OPENROUTER_API_KEY`; print variable NAMES only, never values)

```bash
export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
uv run python examples/07_deep_research_real_e2e.py > /tmp/ldw-m1-e2e07.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1-e2e07.log
uv run python examples/12_screening_reconcile_real_e2e.py > /tmp/ldw-m1-e2e12.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-m1-e2e12.log
unset LDW_DEMO_REAL_MODEL
```
Expected: both EXIT=0; 07 produces a report driven by `survives`/`dedup`; 12 prints included/excluded/conflicts driven by `corroborate`/`reconcile`.

- [ ] **Step 3: Codex cross-model review** — dispatch a `codex:codex-rescue` review of the full diff (`git diff main...HEAD`), focused on: the None fail-safe correctness in all four helpers; the `dedup` overload typing; the meta-layer injection security (no new escape vector — pure functions + frozen dataclasses, AST gate still bans dunder access); the examples' offline determinism. Fix every HIGH/MEDIUM finding with a TDD cycle (failing test → fix → green) and commit.

- [ ] **Step 4: Final gate after fixes**

```bash
uv run pytest -q > /tmp/ldw-m1-final.log 2>&1; echo "EXIT=$?"; tail -12 /tmp/ldw-m1-final.log
uv run ruff check . && uv run pyright && uv run lint-imports
```
Expected: all green.

- [ ] **Step 5: Hand off for PR** — stop here and report to the user (per repo workflow, the user reviews/merges PRs; use the `github-pr` skill when they ask). Do NOT open the PR or merge without the user's go-ahead.

---

## Self-Review (filled by plan author)

**Spec coverage:** every spec section maps to a task — §3 module/exports → T1–T5; §3 meta-layer injection → T6; §4 signatures → T1–T4 (exact code); §5 边界/校验 → T1–T4 tests (empty/kill_at/min_support/None); §6 G1/G4 协同 → T8/T9 (schema leaves feed the helpers); §7 tests → T1–T4 (unit), T6 (reachability), T8 (07 pin), T9 (12 pin + real E2E); §8 docs → T7 (SKILL.md) + T10 (evergreen); Codex review → T11. No gaps.

**Placeholder scan:** no TBD/“handle edge cases”/“similar to”; every code step carries complete code; every command has an expected result.

**Type consistency:** `survives(votes, *, against, kill_at)`, `dedup(items, *, key)`, `reconcile(review_items, *, include)`, `corroborate(items, *, key, min_support)`, `ReviewItem(item, verdicts)`, `Reconciled(included, excluded, conflicts)`, `Consensus(key, members)` — identical across `_reduce.py`, `__init__.py`, `_codegen.py` injection, tests, SKILL.md, and both examples. The seven injected names equal the seven exported names.

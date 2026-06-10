# Design: E — Batch ergonomics (`batch_map` + streaming admission + count/ETA progress)

> Status: FINALIZED design spec — **DESIGN ONLY, not yet implemented** (brainstormed 2026-06-09).
> Committed as a **v0.4.0 roadmap target** (E, lifted from the v0.3.0 "後續里程碑" backlog — see
> [`00-roadmap.md`](00-roadmap.md)). SemVer rationale: `batch_map` is additive API (new `Ctx`
> method + new `ProgressKind` + new `SpanKind`) → minor bump, and v0.4.0 is already open (M1
> landed). Resolved decisions in §10. Writing-plans expands this into the Task 1..N TDD plan.

## 1. What this capability is

Three coupled pieces that make **large-scale fan-out** (thousands of leaves) ergonomic, delivered
as one milestone:

1. **`batch_map`** — a thin map primitive: apply one async `fn` (typically a single `agent()`) to
   every item of an iterable, concurrently, results collected in input order. The big-fan-out
   counterpart to `parallel` (which takes a pre-materialized thunk list).
2. **Streaming admission** — the input is consumed lazily through a bounded admission window, so a
   flood of N-thousand items never materializes N-thousand task objects at once. Memory stays
   bounded by the window, decoupled from N.
3. **Count/ETA progress** — `batch_map` auto-emits live `completed / total / elapsed / eta`
   progress as it advances, so a long batch is observable without the script instrumenting it.

`batch_map` is the surface a user touches; streaming admission is its engine; the progress is the
out-of-band signal it emits while advancing. The driving use case is the community's own open
question — use-case study #5 (codebase-wide bug/vuln sweep) and the explicit prompt *"thousands of
leaves vs CC's 16-concurrent / 1,000-total caps — what are our caps and batching ergonomics?"*

---

## 2. Where it lands in the 3-layer architecture

| Layer | Role in this capability |
|---|---|
| **Layer 0 — substrate** | Untouched. The bounded-queue streaming machinery already lives in the engine's own `run_pipeline` (LangGraph has no streaming no-barrier primitive). |
| **Layer 1 — orchestration runtime** | The home. `batch_map` is a new `Ctx` method; the streaming admission is a generalization of `run_pipeline` (`_pipeline.py`); the progress is a new `ProgressKind` delivered through the existing `ProgressSink` (`_progress.py`). |
| **Layer 2 — meta** | A consumer. `batch_map` is a `Ctx` method, so — like `race`/`dag`/`loop_until` — it needs **no `run_script` injection** (`_codegen.py:107,114`); only the AST gate must not ban the name. |

**Decision: the capability belongs to Layer 1**, built by generalizing the engine's existing
streaming scheduler rather than writing a new one.

---

## 3. The gap today (precise)

- **`parallel` eagerly materializes.** `parallel` builds `[_guarded(thunk) for thunk in thunks]`
  and hands the whole list to one `asyncio.gather` (`_context.py:779-781`). Real *leaf* concurrency
  is gate-bounded (the gate is acquired inside each thunk's `agent()`, not by the fan-out frame —
  `_context.py:728-734`), but at N-thousand items the **N-thousand task/coroutine objects are all
  created up front**. `parallel` also requires a `Sequence` (`len(thunks)` at `:766`).
- **`pipeline` streams between stages but not at the entrance.** `run_pipeline`'s feeder already
  feeds a bounded queue item-by-item (`_pipeline.py:151-161`) with backpressure — but the entrance
  is gated on `len`: `item_count = len(items)` (`:96`), `results = [None] * item_count` (`:97`),
  `_workers_per_stage` needs the count (`:62-64`). A generator must be listed first.
- **`ProgressSink` carries no metrics.** `ProgressKind` is `PHASE`/`LOG` only (`_progress.py:18-27`);
  `ProgressEntry` is `(kind, message)` (`:30-40`); `ProgressLog.emit` records every entry into an
  append-only list and suppresses replays by sequence position vs `delivered_count` (`:67-82`).
  There is no structured `completed/total/eta`, and the append-only model cannot host a value that
  refreshes thousands of times.

---

## 4. The public surface

```python
async def batch_map[X, T](
    self,
    items: Iterable[X] | AsyncIterable[X],
    fn: Callable[[X], Awaitable[T]],
    *,
    max_in_flight: int | None = None,
    total: int | None = None,
    label: str = "batch_map",
) -> list[T | None]:
    """Map ``fn`` over ``items`` with bounded streaming admission; collect in order.

    Streaming in, barrier out: items are admitted lazily through a bounded window
    (so N-thousand items never materialize N-thousand tasks), but results are
    collected into a list aligned to input order and returned only once every item
    has settled. A failing ``fn`` lands ``None`` at its position and never aborts
    the barrier (filter the holes downstream, e.g. with ``dedup``/``survives``);
    engine control-flow signals (budget / determinism / checkpoint) are re-raised
    after a clean drain, never masked as ``None``.
    """
```

- **`fn(item) -> awaitable[T]`** — single-arg, thin. A user who needs the index passes
  `enumerate(items)` as `items` and unpacks inside `fn`.
- **`max_in_flight`** — the admission window: the count of items concurrently in `fn`. `None`
  defaults to the shared gate limit. (More than the gate limit only parks extra workers on
  `gate.acquire`, so the gate limit is the natural default.)
- **`total`** — optional hint for a non-`Sized` input so ETA can be computed; a `Sized` input
  (`list`/`tuple`/...) takes `len()` automatically and needs no hint.
- **Reduce is not built in** — the returned `list[T | None]` is fed to the M1 reduce helpers
  (`survives` / `dedup` / `corroborate`) by the script. Keeps `batch_map` single-purpose.
- A new **`SpanKind.BATCH`** records `admitted_count` / `surviving_count` / `total`
  (mirrors the `PARALLEL` / `PIPELINE` / `RACE` / `DAG` spans in `_observability.py`).

---

## 5. Engine-side design (Layer 1 changes)

### 5.1 Streaming admission — generalize `run_pipeline`

The bounded-queue + worker-group + feeder machinery already exists; the only blockers are the
three `len` dependencies. Generalize them:

- **Feeder consumes an (async) iterator.** A `_drain(items)` adapter yields `(index, item)` from
  either an `Iterable` (`for ... enumerate`) or an `AsyncIterable` (`async for`), so the feeder is
  `async for index, item in _drain(items)`. The bounded queue (`maxsize = max_in_flight`) already
  provides backpressure: the feeder blocks on a full queue, so at any instant the number of live
  envelopes/tasks ≈ `worker_count + queue_maxsize` — **decoupled from N**. This is the streaming
  admission invariant.
- **Result collection drops the pre-allocation.** Replace `results = [None] * item_count` with a
  `dict[int, T | None]` keyed by index; at the end, flatten to a `list` of length `max(index)+1`,
  filling absent positions with `None`. Order-preserving without knowing the length up front.
- **Worker count without a length.** `Sized` input → `min(gate.limit, len, max_in_flight)`; unknown
  length → `max_in_flight` (default `gate.limit`).
- **`pipeline`'s public contract is unchanged.** A `Sequence` input still takes the `len`-fast
  path; the generalized `run_pipeline` serves both `pipeline` and `batch_map`.

### 5.2 `batch_map` = a single-stage generalized pipeline

`batch_map(items, fn)` is the generalized streaming run with one stage
`lambda _payload, item, _idx: fn(item)`. It inherits, for free: failure isolation (a raising stage
drops to `None`), control-flow drain-then-reraise (`_pipeline.py:122-138,179-182`), poison-pill
teardown, and the `_FANOUT_DEPTH` frame so inner `agent()` calls skip the determinism sequence
guard (their completion order is wall-clock-dependent — same reasoning as `parallel`/`pipeline`,
`_context.py:770-774,828-833`).

### 5.3 Count/ETA progress — extend `ProgressSink`

- **New `ProgressKind.BATCH`.** A new frozen `BatchMetrics(completed, elapsed_seconds, rate,
  total=None, eta_seconds=None)` rides on `ProgressEntry` via an optional `metrics: BatchMetrics | None = None`
  field (`PHASE`/`LOG` entries leave it `None` — backward-compatible). `message` stays a
  human-readable line (e.g. `"batch: 340/1000 (~12s left)"`).
- **ETA.** `rate = completed / elapsed`; `eta = (total - completed) / rate` when `total` is known;
  unknown `total` → emit `completed` only, `eta=None` (graceful degradation).
- **Throttle.** Emit on "every K items **or** every T seconds", not once per item (N-thousand
  emissions would be noise). Reasonable defaults (e.g. K=`max(1, total//100)`, T=`1.0s`); always
  emit a final 100%/settled entry so the last state is exact even if the throttle skipped it.
- **Transient delivery — the determinism boundary (load-bearing).** A `BATCH` entry is
  **out-of-band, transient, non-deterministic** and must NOT go through the append-only
  `ProgressLog` record path:
  - It is **delivered to the sink but not recorded** (`ProgressLog` does not append it to
    `_entries` and does not count it toward `delivered_count`). A fire-and-forget refresh; the
    consumer renders it by overwriting the previous one (a progress bar), exactly as M3.5 upserts
    `workflow_runs` and M1 taps leaf events out-of-band.
  - It **never enters the journal, the determinism guard, or the replay result**. The timestamps
    (`elapsed`/`eta`) are non-deterministic and must never reach a journal key.
  - Consequence: on resume the orchestration script **re-executes**, so `batch_map` re-runs and live
    BATCH progress is **re-emitted** — regenerated as a live view, not replayed-from-record
    (`emit_transient` is deliberately never replay-suppressed). This changes nothing in the journal:
    completed leaves still replay at zero model cost, and because BATCH entries never enter
    `_entries` / `delivered_count` / the journal / the determinism guard, the re-emitted progress
    carries no determinism weight. The load-bearing invariant is **not-recorded**, NOT
    **not-re-emitted** — progress is a live view, regenerated each run, never replayed.

### 5.4 Public surface (additive, keyword-only, defaulted)

`Ctx.batch_map` (new method) · `ProgressKind.BATCH` + `BatchMetrics` (new, exported from the package
root) · `SpanKind.BATCH` (new) · `ProgressEntry.metrics` (new optional field, default `None`). No
existing signature changes; `parallel`/`pipeline`/`ProgressSink` callers are unaffected.

---

## 6. Approaches (resolved during brainstorming, 2026-06-09)

- **Streaming admission lands in a new `batch_map`, NOT by mutating `parallel`** — though the
  roadmap wrote "修 parallel". Mutating `parallel` to take an iterator would cost its stable
  contract (`len`, index pre-allocation, span `thunk_count`) and violate preserve-public-signatures.
  Division of labor: `parallel` = small known set + barrier; `batch_map` = large streaming set +
  progress. (Rejected: change `parallel` in place; unify all three on one internal engine — the
  latter is cleanest but the widest blast radius, not justified now.)
- **`batch_map` is a thin map (`fn`), NOT a generalized multi-stage pipeline nor map+built-in
  reduce.** Multi-stage chaining is what `pipeline` is for (avoid two near-identical APIs); reduce
  stays a separate call on the result list (avoid coupling, YAGNI).
- **Progress reuses `ProgressSink` + a transient `BATCH` entry, NOT a new `on_batch_progress`
  hook.** Matches the roadmap's "扩 ProgressSink", adds no new API surface; the transient
  (not-recorded) semantics already isolate determinism. (Rejected: a separate metrics hook — a
  cleaner narrative/metrics split, but one more API.)
- **Input supports both `Iterable` and `AsyncIterable` from the start** — an async source (paged
  API, streaming line reader) is the point of streaming admission; the feeder is already async, so
  a unified async drain is a small increment.

---

## 7. Determinism & quarantine (cross-cutting invariants)

- **Determinism by index.** Inner `agent()` calls run under `_FANOUT_DEPTH` and skip the
  determinism sequence guard (non-deterministic completion order); results are collected by input
  index, so a resume reproduces the same ordered list. Completed leaves replay from the journal at
  zero model cost.
- **Failure isolation.** A raising `fn` → `None` at its position; the barrier is never aborted.
- **Control-flow signals.** Budget / determinism / checkpoint raised inside `fn` are drained then
  re-raised after teardown (the existing `run_pipeline` path), never masked as `None`.
- **AST gate / codegen.** `batch_map` is a `Ctx` method → not on the AST gate's banned-name list
  (like `dag`/`loop_until`); no `run_script` global injection (no new script-constructed value
  type — `BatchMetrics` is engine-produced, the script never builds it).

---

## 8. How a consumer uses it

```python
async def orchestrate(ctx, args):
    files = args["files"]                      # e.g. 3,500 paths
    findings = await ctx.batch_map(
        files,
        lambda path: ctx.agent(f"Audit {path} for bugs/vulns.", agent_type="finder", schema=Finding),
        max_in_flight=16,                      # bounded window; N is decoupled from memory
    )                                          # live "batch: 340/3500 (~9m left)" arrives on the sink
    return dedup(findings)                     # M1 reduce on the result list
```

The host wires `on_progress=...` (already a `run_workflow` parameter, `_engine.py:80`) and renders
`BATCH` entries as a live bar; everything else (status, drill-in) is M1's leaf observability.

---

## 9. Honest limitations

- **No true streaming output.** Results are collected and returned at the barrier, not yielded as
  they complete — true streaming output conflicts with deterministic replay (already a race
  non-goal). `batch_map` is streaming-IN, barrier-OUT.
- **ETA needs a total.** A pure generator/`AsyncIterable` with no `total` hint reports only
  `completed` — no ETA. Pass `total=` to recover it.
- **ETA is a naive linear estimate** (`remaining / mean rate`); it does not model variance or
  tail-heavy item cost. It is a hint, not a guarantee.
- **Throttled progress can skip intermediate counts**; the final settled entry is always exact.

---

## 10. Resolved decisions (2026-06-09)

1. Scope: all three pieces shipped as one coupled milestone.
2. Streaming admission via a new `batch_map`; `parallel` untouched; the internals generalize `run_pipeline`.
3. `batch_map` is a thin map; reduce stays a separate M1-helper call on the result list.
4. Input supports `Iterable` + `AsyncIterable`.
5. Progress reuses `ProgressSink` + a transient (not-recorded) `BATCH` entry.
6. Version: lands in **v0.4.0** as milestone E (additive API ⇒ minor bump; v0.4.0 already open).

---

## 11. Acceptance (per-gap delivery checklist)

1. **Full TDD** (Red→Green→Refactor), ruff + pyright strict green.
2. **Real-model E2E gate** — a scaled-down use-case #5 (codebase-wide bug/vuln sweep): N files,
   one read-only finder leaf per file via `batch_map`, live count/ETA progress, results folded with
   `dedup`. Real model (`LDW_DEMO_REAL_MODEL` + OpenRouter, per memory
   `real-e2e-must-exercise-headline-path` + `keep-langsmith-tracing-on-for-billing`); offline fake
   proves the mechanism, **including a streaming-admission invariant test** (N≫window ⇒ live tasks
   bounded by the window, never materializes N tasks). The gated test lands at
   `tests/integration/test_batch_map_real_model.py`, copied from the canonical
   `tests/integration/test_real_execution_real_model.py` (module-level `skipif` on
   `LDW_DEMO_REAL_MODEL`, `load_demo_env` + `real_leaf_model`, smoking-gun scenario, tracing ON).
3. **User-facing integration example** — ADD a new offline-only `examples/features/bug_vuln_sweep.py`
   (there is no existing sweep demo to extend); host wiring `ctx.batch_map` over a file list with one
   finder leaf each, an `on_progress` live bar, results folded with `dedup`. Mirrors the
   `features/dag.py` / `features/race.py` skeleton; update the `examples/AGENTS.md` §2 index + learning
   path + the `README.md` demo count (18 → 19) in the same task.
4. **Cross-model review** — one Codex round (per memory `independently-verify-gate-claims`).
5. **Evergreen docs sync** — `design_docs/{01,02}.md` + `uml/` + this roadmap (and fix the
   `00-roadmap.md:54` M7-drift while here: M7 is ✅, only E remains lifted from v0.3.0).

---

## 12. Key references

- `_context.py:715-797` — `parallel` (eager materialization at `:779-781`; gate-at-leaf at `:728-734`).
- `_context.py:799-839` — `pipeline` (`_FANOUT_DEPTH` frame at `:828-833`).
- `_pipeline.py` — `run_pipeline` streaming scheduler (feeder `:151-161`, `len` deps `:62-64,96-97`,
  control-flow drain `:122-138,179-182`).
- `_progress.py:18-82` — `ProgressKind` / `ProgressEntry` / `ProgressLog` (replay suppression `:79-82`).
- `_engine.py:80` — `on_progress` is already a `run_workflow` parameter.
- `_codegen.py:107,114` — `Ctx` methods need no `run_script` injection.

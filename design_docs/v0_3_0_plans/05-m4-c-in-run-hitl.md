# M4 · C — In-Run HITL Sign-off — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: layered TDD per `superpowers:test-driven-development` — Red → Green → Refactor, one behavior at a time. Steps use checkbox (`- [ ]`) syntax for tracking. Honor `.claude/rules/` (python_general, docstring, testing) on every task. Run the project gate (ruff + pyright-strict + pytest) yourself; never trust a self-reported green.

**Goal:** Let an orchestration script **pause mid-run for a human sign-off** between phases and continue once a person approves / supplies a value — superset over Claude Code (CC accepts no input mid-run). The script calls `ctx.checkpoint(ask)`; the run parks in a new `AWAITING_SIGNOFF` state; the host observes it (like `running` / `done`) and resumes it with the human's value, which becomes `checkpoint()`'s return value. Completed leaves before the gate replay from the journal at zero model cost.

**Architecture — the load-bearing insight (post-spike pivot):** Every workflow run executes in the **background** (`BgRunManager.start(_coro())`, `tool.py:267`); the host tool launches and returns immediately, never synchronously awaiting `run_workflow` inside its own graph turn. So the C8 fear (an interrupt propagating across the inner-`run_workflow` → host-graph boundary) is moot. The C8 spike then surfaced a deeper problem with using LangGraph's native `interrupt` (D1 below), so the as-built design **does not use `interrupt`/`Command` at all**. Instead a sign-off is **journaled like a leaf result**: `ctx.checkpoint(ask)` keys a gate by its ordinal position; an un-decided gate with no pending value raises `WorkflowSignoffRequired` straight out of the entrypoint body (the run parks at `AWAITING_SIGNOFF`); an `approve` re-launches the run with `resume=value` threaded into the `Ctx`, which records the decision at the first un-decided gate and replays completed leaves + already-approved gates from the journal at zero cost. This is **exactly M3's crash-resume model** (re-run `run_workflow` against the same journal, `checkpointer=None` works) plus a journaled human value. M4 = M3 resume infra + journaled gate + one new non-terminal status.

**Tech Stack:** Python 3.12+, async-first; the content-hash journal (no LangGraph `interrupt`); uv / ruff / pyright-strict / pytest.

---

## Design de-risk (C8 spike + the pivot it forced)

The spike (`/tmp/m4_spike.py`, entrypoint-with-task, real `interrupt`+resume on langgraph 1.2.2) verified the interrupt mechanics — and exposed why they are the **wrong** substrate here.

### D1 — LangGraph interrupt-resume misaligns with the content-hash journal (THE PIVOT)
- Spike facts: `interrupt()` in the entrypoint body makes `ainvoke` **return** `{'__interrupt__':[Interrupt(value,id)]}` (not raise); `Command(resume=v)` injects `v`; pre-interrupt `@task`s are replayed from the checkpointer cache on resume (`before` count stayed 1); an `interrupt` inside a `@task` re-runs that task body (count 1→2).
- **The killer:** the checkpointer's `@task` replay cache is **index-based** (Nth call to `leaf_task` → cache slot N). Our `agent()` **short-circuits on a journal hit before calling `leaf_task`** (`_context.py:512`). So on an interrupt-resume, leaf A's `agent()` returns from the journal and never calls `leaf_task`; leaf B's `leaf_task` becomes call #0 and the checkpointer hands it **A's cached payload** — B silently gets A's result (reproduced: `b == "finding-A"`). M3 dodged this by resuming with `checkpointer=None`; an interrupt-resume *requires* the same checkpointer, so the two caches collide. → **Reject native interrupt; journal the gate instead.**

### D2 — journaled gate = sign-off as a leaf-like record (the as-built)
- `ctx.checkpoint` keys a gate by `signoff_key(position, tag)` and consults the **journal**: a recorded decision replays (zero cost); the first un-decided gate consumes a pending resume value and journals it; otherwise it raises `WorkflowSignoffRequired`. Park and approve are separate `run_workflow` calls, each a fresh `.ainvoke` with its own per-call `InMemorySaver`, so the `@task` index cache never spans them and cannot misalign — the journal is the sole cross-call cache, identical to M3 crash-resume. **No checkpointer requirement** (the old H7 is gone); cross-process approve rides the persistent sqlite journal (T5).

### D3 — depth-0 only (fan-out guard)
- A gate is keyed by its ordinal among `checkpoint` calls; inside a `parallel`/`pipeline`/`race` frame that ordinal races across concurrent thunks (non-deterministic key → broken replay) and concurrently-reached gates have no order for sequential human sign-off. Guard `_FANOUT_DEPTH.get() > 0` → `WorkflowCheckpointError`. The guard is also added to the shared `WORKFLOW_CONTROL_FLOW_SIGNALS` so it fails loud (never masked as a `None` hole) if reached inside a fan-out.

### D4 — re-narration + /shared/ across a gate (same boundary as crash-resume)
- The park raises before `sequence_guard.finalize()` / `put_progress_count`, so an approve re-runs from the top: phase/log narration before the gate **re-delivers** (the documented re-narrate-on-retry behavior, `_engine.py` finalize note) — callers needing dedupe make their progress sink idempotent. The per-run `SharedArtifactStore` is rebuilt empty on approve; a pre-gate leaf replays its **result** but not its `/shared/` artifacts, so cross-gate state MUST go through **script variables** (the M5 `fix_loop` pattern), not `/shared/`. Real worktree persistence is M6. (A later refinement may persist the progress count at the park to suppress re-narration; deferred.)

---

## API design (as built)

1. **Primitive — `Ctx.checkpoint`.** One minimal primitive:
   ```python
   async def checkpoint(self, ask: Any, *, tag: str = "") -> Any: ...
   ```
   Surfaces `ask` and returns the injected resume value. Sign-off semantics (approve/reject) are a **documented pattern** — the script interprets the returned value (`if not decision.get("approved"): ...`) — not extra API. `tag` labels the gate (folded into its journal key). Guards `_FANOUT_DEPTH > 0` (D3). Journals the decision via `signoff_key(position, tag)`.
2. **Engine signal — `WorkflowSignoffRequired`** (`_errors.py`): carries `.ask`, `.tag`, `.gate_key`. Raised by `ctx.checkpoint` (not by a `run_workflow` interrupt detection); propagates out of `run_workflow` unchanged. Rationale: a clean, catchable control-flow signal the background wrapper turns into a park.
3. **Resume injection — `run_workflow(..., resume: Any = UNSET)`.** Threaded into `Ctx(pending_signoff=resume)`; consumed at the first un-decided gate. `UNSET` (a shared sentinel in `_context`, constant-named so it crosses the module wall cleanly) distinguishes "no resume" from "resume with `None`". `run_workflow` always invokes `_run.ainvoke({})` — no `Command`.
4. **Status — `BgStatus.AWAITING_SIGNOFF = "awaiting_signoff"`**, **non-terminal**. A parked run still occupies its run slot (counts toward `max_concurrent_runs`) until approved/cancelled — a parked sign-off is an open run. The slot stores the `ask` so `status` can show what to approve.
5. **Host command — new `approve`** (`approve <run_id> [value]`), distinct from `resume`. `resume` = replay-from-journal after crash/failure (no value injection); `approve` = inject a human value into a parked sign-off. **Local-only** (post-review): approves ONLY a run whose parked slot is live on this process (`AWAITING_SIGNOFF`); an `UNKNOWN` run (swept / cross-process) is refused — the parked state isn't persisted, so it can't be verified as genuinely at a gate (see Review hardening M1). `status` / `runs` surface `AWAITING_SIGNOFF` + the ask. All **术** — taught only via the tool description / `help` / SKILL.md, **never** a demo prompt (AGENTS.md 道/术线).

## Review hardening (Codex + in-house adversarial review, folded)

Both review passes (Codex `gpt-5.5` + in-house) corroborated; all real, no false positives. Fixed + tested:
- **H1/Codex#2 (HIGH, concurrency)** — `approve()` now flips the slot to `RUNNING` synchronously before scheduling, so a double-approve racing the loop is refused (no orphaned second continuation against one journal). Test: `test_double_approve_is_refused_no_orphaned_continuation`.
- **M1/Codex#3 (HIGH, auth/silent-loss)** — cross-process `approve` removed (the parked state isn't persisted, so it couldn't verify the run was at a gate). `approve` requires a local `AWAITING_SIGNOFF` slot; `UNKNOWN` is refused. Engine backstop: a `resume` decision that no gate consumes → fail-loud `WorkflowCheckpointError` (never a silent drop). Tests: `test_cross_process_approve_is_refused_*`, `test_resume_with_no_gate_to_consume_fails_loud`.
- **M5/Codex#1 (MED, gate-key drift)** — `checkpoint` observes the gate key in the determinism sequence (gate order/identity drift fails loud, like a leaf); `(position, tag)` keying documented, distinct gates need distinct tags.
- **M2 (MED, quota leak)** — abandoned parked runs expire via a `park_ttl_seconds` sweep (→ `CANCELLED`), a defended bound. Test: `test_park_ttl_expires_an_abandoned_signoff`.
- **M3 (MED, security)** — AST gate bans `ctx.checkpoint` in authored scripts (sign-off is a registered-workflow capability). Test: `test_rejects_ctx_checkpoint_in_authored_script`.
- **M4 (MED, re-narration)** — progress count persisted on park, so an approve does not re-deliver pre-gate narration. Test: `test_pre_gate_progress_not_renarrated_on_approve`.
- **Codex#4 (LOW, type honesty)** — `checkpoint` serializes the decision before consuming the pending value → a non-JSON decision fails clearly and stays re-approvable. Test: `test_non_json_decision_fails_loud_and_stays_re_approvable`.

---

## File structure / changes by layer

| Layer | File | Change |
|---|---|---|
| Engine | `_context.py` (done) | `Ctx.checkpoint()` (journal-based); `_FANOUT_DEPTH` guard; `UNSET` sentinel + `pending_signoff` |
| Engine | `_journal.py` (done) | `signoff_key(position, tag)` (namespaced gate key) |
| Engine | `_engine.py` (done) | `resume` kwarg → `Ctx(pending_signoff=...)`; no `interrupt`/`Command` |
| Engine | `_errors.py` (done) | `WorkflowSignoffRequired(ask, tag, gate_key)`, `WorkflowCheckpointError`, `WORKFLOW_CONTROL_FLOW_SIGNALS` |
| Engine | `__init__.py` (done) | export the two new errors |
| Runtime | `_background.py` | `BgStatus.AWAITING_SIGNOFF`; `_run_wrapped` catches `WorkflowSignoffRequired` → park; slot stores `ask`; `RunSnapshot`/`RunOutcome` carry it; `active_run_count` still counts a park |
| Host | `tool.py` | `_launch(resume_value=...)`; `approve` command; `status`/`runs` surface park + ask; description names `approve` |
| Persistence | `_run_store.py` / `_persistence.py` | (verify) a parked run's spec already persists for cross-process approve; add a regression test |
| Demo | `demo-app/backend/workflows.py` | a sign-off workflow preset (#6 security-audit: risk-tier → human sign-off → report) threading state via a script variable (H6) |
| Demo | `demo-app/backend/host_graph.py` / `ui_adapter.py` | wire the workflow; map the park + ask to a `SignoffGate` Gen-UI card; approve round-trips a host message → `approve` command |
| Demo | `demo-app/frontend/.../SignoffGate.tsx` + `registry.ts` | the card (approve / reject buttons); host-channel only (leaf-quarantine: see `leaf-quarantine-holds-at-ui-streaming-layer`) |
| Docs | `design_docs/{01,02}.md` + `uml/{02,03}.md` + this roadmap | document the HITL seam + sequence |

---

## TDD task breakdown

- [x] **T1 — `Ctx.checkpoint` primitive.** DONE. Journal-based gate (`signoff_key(position, tag)`); fan-out guard → `WorkflowCheckpointError` (added to `WORKFLOW_CONTROL_FLOW_SIGNALS`); `WorkflowSignoffRequired(ask, tag, gate_key)` in `_errors`; `UNSET` sentinel + `pending_signoff` on `Ctx`.
- [x] **T2 — `run_workflow` resume injection + journal replay (integration).** DONE. `run_workflow(..., resume=UNSET)` → `Ctx(pending_signoff=resume)`; `tests/integration/test_hitl_signoff.py` (4 tests: park→approve with pre-gate leaf journal-cached, multi-gate one-at-a-time, fan-out fails loud, resume value returns directly). Full gate green (478 passed, ruff+pyright clean, 93.6% cov).
- [x] **T3 — `BgRunManager` park + approve.** DONE. `BgStatus.AWAITING_SIGNOFF` (non-terminal, counts active, no TTL sweep); `_run_wrapped` catches `WorkflowSignoffRequired` → `_park` (stores ask, enqueues notice); `approve(coro, run_id, thread_id)` relaunches the parked slot in place (same run_id); `get_signoff`; `BgRunStateError`. `tests/integration/test_hitl_background.py` (manager park→approve, multi-gate one run_id).
- [x] **T4 — host `approve` command + status/runs surfacing.** DONE. `approve` command (dual path); `status`/`runs` surface `awaiting_signoff` + ask; `args` carries the decision; description names `approve`. Tested through the tool layer.
- [x] **T5 — cross-process approve.** DONE. No-local-slot → relaunch from persisted spec via `_launch(resume_value=...)` (`_NO_RESUME` sentinel). Test: two managers share one run store, the second approves the parked run.
- [x] **T6 — integration example.** DONE. `examples/features/signoff.py` (copy-pasteable host-wiring: park → approve proceeds / reject holds; pre-gate leaf replays free). `examples/AGENTS.md` index updated (16 feature demos).
- [x] **T7 — demo-app consumer slice.** DONE. `sign_off` preset; inline `run_workflow_live` park/resume via `_ResumeLane`; `SignoffGate.tsx` card + registry; `_models.py` offline routing; "Sign off mid-run" scenario (scenarios.json + ScenarioPanel + README, doc-sync green). `tests/test_sign_off.py` (13 tests incl. real-graph two-turn). UI leaf-quarantine held (`streamSubgraphs:false`; card on host ui channel via merge across turns).
- [x] **T8 — gated real-model E2E.** DONE & **ACTUALLY RUN** (`tests/test_m4_signoff_real.py`, `LDW_DEMO_REAL_MODEL=1`, OpenRouter + LangSmith tracing on, 1 passed in 46.95s). Real opus host routed the request to `sign_off`, parked, and answered the gate via the documented `signoff_decision` param (self-sufficiency proof) → approved → proceeded. Asserts via final thread-state per turn (no transient false-positive).
- [x] **T9 — evergreen docs sync + reviews.** DONE. Docs (`design_docs/01` §14 + `02` §11 + `uml/02-class` + `uml/03-sequence` G + this plan + roadmap). Codex cross-model + in-house adversarial reviews folded (see Review hardening above — 7 findings, all fixed + tested). Security audited (the threats — unauthorized approve, decision injection, park-flood, AST escape — each mitigated). Full gate re-run by me: engine 489 + demo 132 passed, ruff + pyright clean, real-model E2E re-passed (47.65s).

## Acceptance gate (per-gap delivery checklist)
1. Full TDD, ruff + pyright-strict green (run by me).
2. Real-model E2E (T8) actually run; offline tests pin the mechanism.
3. User-facing integration example (T6).
4. Codex cross-model review round.
5. Evergreen docs synced (T9).

## Non-goals (explicit)
- `checkpoint()` inside a leaf / fan-out frame (D3) — guarded, documented.
- LangGraph native `interrupt`/`Command` (D1) — rejected; the gate is journaled instead.
- True host-graph-native interrupt UI (the run is a background run; the gate surfaces as a host-channel card, not the agent-chat-ui native interrupt on the host graph).
- /shared/ artifact survival across a gate (D4) — script-variable state instead; real worktree persistence is M6.
- Cross-process / cross-session `approve` — the parked state (which gate, the ask) lives in the in-memory manager, not the run store, so approve requires a live local parked slot; a fresh process cannot safely approve a run parked in a dead one. Cross-session HITL (persisted park state) is a future milestone.
- Indefinite parked-run hold — bounded by `park_ttl_seconds` (expires to `CANCELLED`); within the TTL a parked run waits until approved/cancelled.

# M3 · D — Cross-Session / Cross-Process Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: dispatched via the `Workflow` tool (master-orchestrator mode) — sequential layered TDD agents, dependency order, each commits as it goes. Steps use checkbox (`- [ ]`) syntax for tracking. Honor `.claude/rules/` (python_general, docstring, testing) on every task.

**Goal:** Make resume/replay survive a process exit — a fresh process pointed at the same sqlite db file resumes a run by `run_id` and replays its completed leaves at **zero new model cost**. Superset over Claude Code (CC is same-session only).

**Architecture:** A persistent `JournalStore` + persistent host run-registry, both backed by **one unified sqlite db file** namespaced by `run_id`, plus a persistent LangGraph checkpointer (`AsyncSqliteSaver`) wired through `run_workflow`'s existing `checkpointer=` param. Persistence is gated behind an optional `[sqlite]` extra; the in-memory default stays dependency-free and behavior-identical.

**Tech Stack:** Python 3.12+, async-first; `aiosqlite` (autocommit + WAL) for the store; `langgraph-checkpoint-sqlite` (`AsyncSqliteSaver`) for the checkpointer; uv / ruff / pyright-strict / pytest.

---

## Load-bearing constraints (from the design de-risk — implementers MUST honor verbatim)

These were **empirically verified** during de-risk. Violating any one silently breaks the headline or hangs the host.

### C1 — AsyncSqliteSaver event-loop binding (HANG hazard)
- `AsyncSqliteSaver.__init__` calls `asyncio.get_running_loop()` and binds to it. Constructing it **outside** a running loop raises `RuntimeError('no running event loop')`. Reusing one instance across **different** loops (e.g. two `asyncio.run()` boundaries) **HANGS**.
- The host must construct the saver **inside** its single persistent event loop and reuse the *same* instance across all background runs / thread_ids (verified: 3 concurrent + 2 sequential runs on one instance, all correct).
- Construct via `AsyncSqliteSaver(conn)` over a **host-owned** `aiosqlite.Connection`. Do **NOT** use `AsyncSqliteSaver.from_conn_string(...)` for the long-lived host — it is an `@asynccontextmanager` that **closes the connection on `__aexit__`**, defeating cross-run / fresh-process resume.
- `setup()` is auto-called + idempotent (WAL + `CREATE TABLE IF NOT EXISTS`); calling it once on a fresh process is safe. Close the connection at host shutdown (else the program can hang on exit).

### C2 — aiosqlite write-through durability (ROLLBACK footgun)
- Default `isolation_level=''` **rolls back** uncommitted DML on close/crash. Verified: per-leaf `put()` then close-without-commit → reopened db shows **0 rows**.
- **MUST** open the store connection with `isolation_level=None` (autocommit) so every `put()` is durable on return with **zero** explicit `commit()` calls (verified: 50 rows survive close+reopen).
- Also set `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`.

### C3 — Two connections, one db file
- The store (autocommit) and `AsyncSqliteSaver` (explicit-commit + its own WAL regime) have **incompatible isolation regimes**; they MUST use **separate** `aiosqlite.Connection`s to the **same** db file. Cross-connection reads see committed writes under WAL (verified).

### C4 — Single connection serializes correctly; no asyncio.Lock
- Each `aiosqlite.Connection` serializes all ops through one worker thread. Concurrent multi-leaf `get`/`put` on one shared store connection is correct **without** any extra `asyncio.Lock` (verified, 50 concurrent leaves). Keep all queries indexed by PK `(run_id, key)` and tiny.

### C5 — Journal (not checkpointer) delivers zero-cost replay
- The native checkpointer is **index-based** and re-runs leaves on same-thread re-invoke; LangGraph re-executes the `@entrypoint` body in full on every `.ainvoke` (verified). The **content-hash JournalStore** is what makes completed leaves replay free. The headline "fresh process resumes at zero model cost" is delivered by the **journal alone** and is independently testable with `checkpointer=None` on the resume side. The persistent checkpointer is a robustness add-on (durable `@task` cache + interrupt/resume within a single in-flight run + cross-process resume-by-thread_id, all verified).

### C6 — Checkpointer serialization hazard (LeafOutcome / msgpack)
- `AsyncSqliteSaver` msgpack-serializes **every** `@task` return (`LeafOutcome` wrapping the leaf's raw state dict, incl. LangChain message objects). A de-risk probe surfaced `TypeError: Type is not msgpack serializable: LeafOutcome`. This MUST be proven against **realistic** deepagent leaf state **early** (Task 2 spike) — before the example/E2E depend on it. If it fails, adapt `LeafOutcome`/leaf-state serialization (msgpack-friendly shape or serde registration) as a sub-task of Task 2.

### C7 — journal_key is process-stable
- `journal_key` hashes `model_json_schema()` + `json.dumps(sort_keys=True)`, which is invariant across `PYTHONHASHSEED` for pydantic models and L2 dict-schemas (verified). No change needed; add a cross-subprocess key-stability regression test (Task 6).

### C8 — Layer boundary + packaging
- `_persistence.py` is a **Layer-2 host-wiring** module: add `langchain_dynamic_workflow._persistence` **and** `langchain_dynamic_workflow._run_store` to import-linter **Contract 1** `source_modules` (`[tool.importlinter]` in `pyproject.toml`). They MUST import `JournalStore`/`InMemoryJournalStore`/`JournalRecord` from `._engine` (the public wall), **never** `._journal`. `JournalRecord` is **not** currently re-exported by `_engine` — add it to the re-export at `_engine.py:35`.
- Optional-dep import guard: module-top `try/except ImportError` re-raising a clear "install `[sqlite]`" message. `_persistence` MUST NOT be eagerly imported by `__init__.py` or `_engine` (keeps base install dependency-free).
- pyright strict over `[src, tests]`: a module-top import of an uninstalled `langgraph.checkpoint.sqlite.aio` raises `reportMissingImports` and a `TYPE_CHECKING` guard does **not** suppress it. Resolution: install the extra in dev/CI so pyright checks `_persistence.py` for real (add the sqlite deps to the dev `[dependency-groups]` so `uv sync` brings them, **and** `uv sync --extra sqlite` in the CI quality-gate job). Mark the `except ImportError` branch `# pragma: no cover` to protect the 85% coverage gate.

### C9 — thread_id policy + race no-winner
- Persist `thread_id` in `run_specs` and replay it on resume (define the policy explicitly; test both same-thread and fresh-thread resume against the persistent saver).
- `race()` no-winner is **not** journaled → a cross-process resume re-dispatches candidates (new cost). The acceptance scenario MUST avoid a no-winner race (or assert the retry explicitly). Document as a known boundary.

---

## File structure

| File | Responsibility | Layer |
|---|---|---|
| `pyproject.toml` | `[sqlite]` extra; dev-group sqlite deps; import-linter Contract 1 += `_run_store`,`_persistence`; CI `--extra sqlite` | — |
| `.github/workflows/*.yml` | install `--extra sqlite` before pyright/pytest | — |
| `src/.../_engine.py` | add `JournalRecord` to the `._journal` re-export | 0/1 wall |
| `src/.../_run_store.py` | **NEW.** `WorkflowRunStore` Protocol + `RunSpec` dataclass + `InMemoryRunStore` (dep-free default) | 2 |
| `src/.../_persistence.py` | **NEW.** `SqliteWorkflowStore` (registry+journal+checkpointer facade) + `_RunScopedJournal` + import guard | 2 |
| `src/.../tool.py` | inject `store: WorkflowRunStore`; run_id-first launch; resume-from-store | 2 |
| `src/.../middleware.py` | forward `store` through `create_workflow_middleware` | 2 |
| `src/.../__init__.py` | export `WorkflowRunStore`/`RunSpec` (dep-free); lazy `__getattr__` for `SqliteWorkflowStore` | — |
| `tests/unit/test_run_store.py` | **NEW.** in-memory store + protocol round-trip | — |
| `tests/unit/test_persistence.py` | **NEW.** sqlite store CRUD/journal/write-through/concurrency (gated on extra) | — |
| `tests/integration/test_cross_process_resume.py` | **NEW.** true-subprocess cross-process resume (offline fakes) + checkpointer serialization | — |
| `examples/15_*.py` | **NEW or extend.** host wiring with `SqliteWorkflowStore` + cross-process resume (real-leaf via `LDW_DEMO_REAL_MODEL`) | — |
| `design_docs/{01,02}.md`, `uml/`, `00-roadmap.md` | evergreen sync; M3 ✅ | — |

---

## Tasks (dependency order — front-load external risk)

### Task 1 — Packaging foundation + engine re-export + import-linter contract
**Files:** `pyproject.toml`, `.github/workflows/*.yml`, `src/.../_engine.py`
- [ ] Add `[project.optional-dependencies] sqlite = ["langgraph-checkpoint-sqlite>=3.1,<4", "aiosqlite>=0.22"]` (PEP 621, uv-native). Note it transitively pulls `langgraph-checkpoint>=4.1,<5` and `sqlite-vec>=0.1.6` (a **compiled** extension — confirm it installs cleanly).
- [ ] Add the same two sqlite deps to the dev `[dependency-groups]` so local `uv sync` installs them (pyright/pytest see `_persistence.py` for real).
- [ ] Update the CI quality-gate workflow to `uv sync --extra sqlite` (or the dev group) before ruff/pyright/import-linter/pytest.
- [ ] `_engine.py:35` — add `JournalRecord` to `from ._journal import InMemoryJournalStore, JournalStore` → `..., JournalRecord, ...`. Update the re-export comment to mention `JournalRecord`.
- [ ] `pyproject.toml [tool.importlinter]` Contract 1 — append `langchain_dynamic_workflow._run_store` and `langchain_dynamic_workflow._persistence` to `source_modules`.
- [ ] Gate: `uv sync` clean; `uv run lint-imports` EXIT=0; `uv run pyright` 0 errors; `uv run pytest -q` still 358 passed. Commit: `chore(persistence): add [sqlite] extra + re-export JournalRecord + extend import-linter contract`.

### Task 2 — Checkpointer serialization spike (HIGHEST external risk; do early)
**Files:** `tests/integration/test_cross_process_resume.py` (start it here)
- [ ] **Red:** write a test that opens a real `aiosqlite.Connection`, wraps it `AsyncSqliteSaver(conn)` **inside the test's running loop**, and runs `run_workflow(..., checkpointer=saver)` over a **realistic fake leaf** whose returned state mirrors a deepagent (`{"messages": [HumanMessage(...), AIMessage(...)]}` plus typical keys). Assert the run completes and the checkpoint persists with **no** msgpack serialization error.
- [ ] Run it. If it raises `Type is not msgpack serializable: LeafOutcome` (C6), **fix the serialization** of `LeafOutcome` / the `@task` return so `AsyncSqliteSaver` can persist it (prefer a msgpack-friendly shape or register the type with the serde; smallest change that keeps the engine's public behavior). Re-run until green.
- [ ] Add a second assertion: a fresh `AsyncSqliteSaver` over a **new** connection to the **same** db file sees the prior thread's checkpoint (cross-process-by-thread_id, verified-feasible in de-risk).
- [ ] Gate: ruff + pyright + the new test green. Commit: `test(persistence): prove AsyncSqliteSaver round-trips a real run + adapt LeafOutcome serialization`.

### Task 3 — `_run_store.py`: WorkflowRunStore Protocol + RunSpec + InMemoryRunStore
**Files:** `src/.../_run_store.py` (new), `tests/unit/test_run_store.py` (new)
- [ ] Define `RunSpec` (frozen, slots dataclass): `kind: str` ("name"|"script"), `name_or_source: str`, `args: dict[str, Any]`, `label: str`, `thread_id: str`. (Replaces the bare `tuple[str,str,dict]` for clarity + carries label+thread_id for persistence, per C9.)
- [ ] Define `WorkflowRunStore` Protocol (`@runtime_checkable`): `save_spec(run_id: str, spec: RunSpec) -> None` (async), `load_spec(run_id: str) -> RunSpec | None` (async), `journal_for(run_id: str) -> JournalStore` (sync — returns a view/instance). Import `JournalStore`/`InMemoryJournalStore` from `._engine` (C8).
- [ ] `InMemoryRunStore`: wraps `_specs: dict[str, RunSpec]` + `_journals: dict[str, JournalStore]`; `journal_for` creates+caches an `InMemoryJournalStore` per `run_id` (so resume reuses the same instance — preserves current same-session behavior).
- [ ] **Tests** (meaningful, not filler): save/load round-trip; `load_spec` unknown → `None`; `journal_for` returns the **same** instance for a repeated `run_id` (regression guard for resume); `isinstance(InMemoryRunStore(), WorkflowRunStore)`.
- [ ] Gate: ruff + pyright + tests green. Commit: `feat(persistence): WorkflowRunStore protocol + RunSpec + in-memory default`.

### Task 4 — tool.py / middleware.py: inject the store (regression-safe)
**Files:** `src/.../tool.py`, `src/.../middleware.py`, `tests/unit/test_tool.py`, `tests/unit/test_middleware.py`
- [ ] `create_workflow_tool(..., store: WorkflowRunStore | None = None)` — default `InMemoryRunStore()`. Replace the closure-local `journals`/`run_specs` dicts with `store`.
- [ ] `_launch`: generate `run_id` up front (`uuid.uuid4().hex`); `journal = store.journal_for(run_id)`; build `_coro`; `manager.start(_coro(), run_id=run_id, thread_id=thread_id, label=label)`; `await store.save_spec(run_id, RunSpec(kind, name_or_source, args, label, thread_id))`. (Note: `save_spec` is async — `_launch`/its callers may need to `await`; thread the async through, or have the store offer a sync `save_spec` for the in-memory path. Decide: make `save_spec` async and `await` it in the command handlers, which are already async-capable.)
- [ ] `_resume_command`: `spec = await store.load_spec(run_id)`; `None` → "unknown run_id"; rebuild callable via `_resolve_spec` from `spec.kind`/`spec.name_or_source`/`spec.args`; `journal = store.journal_for(run_id)`; relaunch with `spec.thread_id` (C9). Cross-process: the spec now comes from sqlite, not the in-process dict.
- [ ] `create_workflow_middleware(..., store: WorkflowRunStore | None = None)` — forward to `create_workflow_tool`.
- [ ] **Tests:** existing tool/middleware tests still pass with the in-memory default (regression); a new test confirms `resume` reads the spec from an injected store; run_id is generated before `manager.start`; `save_spec` is invoked on launch with the right `RunSpec`.
- [ ] Gate: ruff + pyright + full suite green. Commit: `refactor(tool): route run registry through an injectable WorkflowRunStore`.

### Task 5 — `_persistence.py`: SqliteWorkflowStore (the headline)
**Files:** `src/.../_persistence.py` (new), `tests/unit/test_persistence.py` (new)
- [ ] Module-top optional-dep import guard (C8): `try: import aiosqlite; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver except ImportError as exc: raise ImportError("...install langchain-dynamic-workflow[sqlite]...") from exc`. Mark the except branch `# pragma: no cover`.
- [ ] `_RunScopedJournal(conn, run_id)` implementing the 6-method `JournalStore` Protocol, run_id-scoped SQL + UPSERTs (use the de-risk snippets verbatim). `get_progress_count` coalesces a missing row → **0** (C: matches InMemory semantics). Import `JournalRecord` from `._engine` (C8).
- [ ] `SqliteWorkflowStore` with **async factory** `@classmethod async def open(cls, db_path) -> SqliteWorkflowStore` (must run inside the host loop, C1): open store conn `await aiosqlite.connect(db_path, isolation_level=None)` (C2) → `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=5000`; bootstrap the 4 tables via `executescript(_SCHEMA_DDL)` (DDL from de-risk); open a **second** conn (C3) + `AsyncSqliteSaver(conn2)` (C1). Implements `WorkflowRunStore`: `save_spec`→UPSERT `run_specs` (args as `json.dumps`), `load_spec`→SELECT + rebuild `RunSpec`, `journal_for`→`_RunScopedJournal(self._store_conn, run_id)`. `@property checkpointer` → the saver. `async def aclose()` → close both conns.
- [ ] **Tests** (gated: skip if extra unavailable, but dev/CI installs it so they run): store spec CRUD persists across `aclose` + reopen; journal `put`/`get` round-trip; **write-through durability** (put then reopen WITHOUT explicit commit → row present, C2); `get_progress_count` missing → 0; 50 concurrent `put`/`get` on one store correct (C4); `journal_for` scopes by run_id (two runs don't collide); `isinstance(store, WorkflowRunStore)`.
- [ ] Gate: ruff + pyright (extra installed) + tests green. Commit: `feat(persistence): SqliteWorkflowStore — unified sqlite registry + journal + checkpointer`.

### Task 6 — Cross-process integration test + __init__ exports
**Files:** `tests/integration/test_cross_process_resume.py` (extend), `src/.../__init__.py`, `tests/unit/test_journal_key_stability.py` (new, optional)
- [ ] **True cross-process headline test (offline fakes):** subprocess A (real OS process via `subprocess`/`multiprocessing`) opens `SqliteWorkflowStore` on a temp db, launches a workflow with **offline fake leaves that count model invocations**, lets it complete, persists, exits. Subprocess B (fresh process) reopens the store from the same db file, `resume`s by `run_id`, asserts **zero** new leaf invocations (completed leaves replayed from journal — C5) and an identical result. Avoid a no-winner race (C9).
- [ ] **Failed-then-retried** variant: a run that raises mid-flight (e.g. budget exceeded) → re-run replays completed leaves free, determinism guard behaves as a fresh recording run (C: A3 risk).
- [ ] **Cross-subprocess journal_key stability** regression (C7): hash a representative pydantic model + an L2 dict-schema in two separate python processes; assert equal keys.
- [ ] `__init__.py`: export `WorkflowRunStore` + `RunSpec` (dep-free). Do **NOT** eagerly import `_persistence`; expose `SqliteWorkflowStore` via a lazy module-level `__getattr__` that imports `_persistence` on first access and surfaces the install-the-extra `ImportError` (C8). Confirm `import langchain_dynamic_workflow` works **without** the extra (dependency-free default).
- [ ] Gate: ruff + pyright + full suite green; `uv run lint-imports` EXIT=0. Commit: `test(persistence): true cross-process resume + journal-key stability; export run-store surface`.

### Task 7 — Integration example + evergreen docs + roadmap
**Files:** `examples/15_*.py` (new or extend a canonical example), `design_docs/01-*.md`, `design_docs/02-architecture.md`, `design_docs/uml/*`, `design_docs/v0_3_0_plans/00-roadmap.md`
- [ ] Copy-pasteable host-wiring example: `store = await SqliteWorkflowStore.open("workflows.db")`; build tool/middleware with `store=store, checkpointer=store.checkpointer`; launch a workflow; **simulate process restart** (close everything, reopen store from the db file) and `resume`. Real-leaf path via `LDW_DEMO_REAL_MODEL` (the headline E2E acceptance lands here per memory `real-e2e-demo-must-exercise-headline-path`). Honor the 道/术 rule (host prompt carries mental-model only).
- [ ] Evergreen sync: persistent store/checkpointer wiring in architecture doc; the two-connections-one-file + autocommit + event-loop-binding invariants; thread_id-on-resume policy. Update UML if a new module/relationship warrants. Mark M3 ✅ 已落地 in `00-roadmap.md` with a one-paragraph summary (incl. any deviation).
- [ ] Gate: ruff + pyright + full suite green. Commit: `example+docs(persistence): cross-process resume host wiring + evergreen sync (M3 ✅)`.

---

## Acceptance (per-gap deliverable checklist)
1. **Full TDD**, ruff + pyright-strict + import-linter all green; full suite passes.
2. **Real-model E2E** (`examples/15` on `LDW_DEMO_REAL_MODEL`): two real OS processes, one db file, resume replays completed deepagent leaves at zero new token cost (smoking gun = new metered tokens == 0 for replayed leaves; disable LangSmith tracing). Headline must exercise the **journal** path (C5), not a fallback.
3. **Integration example** — copy-pasteable host wiring (Task 7).
4. **Codex cross-model review** — fold into the review workflow (Phase D); independently re-run the full gate myself (memory `independently-verify-gate-claims`).
5. **Evergreen docs** — `design_docs/{01,02}` + `uml/` + roadmap status synced.

## Open questions / deviations to surface at review
- Did `LeafOutcome` need a serialization change for the persistent checkpointer (C6)? If yes, document the shape change and confirm it's behavior-preserving for the in-memory default.
- thread_id-on-resume policy (C9): same-thread vs fresh-thread — document the chosen policy and why.
- Global cross-run budget pool remains **out of scope** (roadmap open decision); per-run isolation preserved.

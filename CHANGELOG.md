# Changelog

All notable changes to `langchain-dynamic-workflow` are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Batch ergonomics — `ctx.batch_map` (v0.4.0 · E)** — a thin streaming map primitive: applies one
  async `fn` to every item of an `Iterable`/`AsyncIterable` through a bounded admission window, so a
  flood of N items never materializes N tasks (memory stays bounded by the window, decoupled from N);
  results are collected in input order (`list[T | None]`, a failing `fn` lands `None`). Auto-emits live
  count/ETA progress via a transient `ProgressKind.BATCH` entry + `BatchMetrics` (delivered to the
  `ProgressSink` but never recorded, so it stays out of the journal / determinism guard / replay).
  Adds `BatchMetrics`, `ProgressKind.BATCH`, `SpanKind.BATCH`, and `ProgressEntry.metrics`. `parallel`
  and `pipeline`'s public contracts are unchanged; the shared `run_pipeline` scheduler is generalized
  to consume an (async) iterable.
- **Background-run transport — drill-in + result report-back (v0.4.0 · M3)** — a detached background
  run is no longer UI-dark. `BgRunManager` buffers each run's raw runtime events on its slot
  (`BufferedEvent`, bounded by a per-run cap with a `dropped` counter; lock-guarded so an off-loop
  `on_command` edge cannot tear a read), exposed via `BgRunManager.event_sinks(run_id)` (five
  append-only sinks threaded into the detached `run_workflow`) and `buffered_events(run_id)`. A live
  turn replays a run's buffer through a fresh adapter — the same card vocabulary an inline run streams
  (`agent_span` / `fanout_graph` / `phase_timeline`), idempotent via stable span ids. Buffered events
  are transient telemetry: never journaled, never replayed, host-LLM-context isolation preserved.
  Adds `BufferedEvent`, `RunEventSinks`, and `BgRunManager.event_sinks` / `buffered_events` /
  `max_buffered_events`. (Demo consumer: a conversational `drill_run` tool replays one run's interior
  into the chat; the runs board's final report now carries per-run result substance, and a
  `fetch_run_result` tool fetches a run's full payload on demand.) `run_workflow`'s sink parameters
  are unchanged — the detached path simply started using them.

## [0.3.0] - 2026-06-09

A use-case-driven batch that approaches — and selectively exceeds — Claude Code's
Dynamic Workflows. Each milestone landed through full TDD, a real-model end-to-end
acceptance run, and a cross-model adversarial review. The public API stays
backward-compatible — every addition is keyword-only or a new symbol. Per-milestone
tracking lives in `design_docs/v0_3_0_plans/00-roadmap.md`.

### Added

- **Cross-leaf reduce helpers (M1 · F)** — `survives` (refute-by-default voting),
  `dedup`, `reconcile` (double-blind reviewer reconciliation), and `corroborate`
  (judge-panel aggregation), plus the `ReviewItem` / `Reconciled` / `Consensus` value
  types: pure functions that fold `parallel` / `pipeline` result lists. Exported from
  the package root and injected into the `run_script` namespace.
- **`ctx.race` best-of-N early exit + cancel (M2 · B)** — runs candidates
  concurrently; the first satisfying a `win` predicate wins, in-flight losers are
  cancelled; the decision is content-hash-journaled (`race_key`) so resume reproduces
  the winner and dispatches nothing new. Adds `RaceCandidate` / `RaceResult` and
  `SpanKind.RACE`.
- **Aggregate parallel-run observability (M3.5)** — a `runs` tool command +
  `BgRunManager.list_runs` / `RunSnapshot` enumerate a host thread's in-flight /
  finished runs; the `workflow_runs` state channel is rewritten to a terminal status
  on settle; the `max_concurrent_runs` quota stays on `BgRunManager` (the middleware
  fails loud on a conflicting double-source).
- **Cross-session / cross-process persistence (M3 · D, superset of Claude Code)** —
  a `WorkflowRunStore` protocol + `RunSpec` + `InMemoryRunStore` (default, zero-dep) +
  `SqliteWorkflowStore` (the `[sqlite]` extra: one sqlite db, run_id-namespaced
  journal + a persistent `AsyncSqliteSaver`). A fresh process pointed at the same db
  resumes a run by `run_id` and replays completed leaves at zero model cost.
- **Real local execution backend (M5 · A)** — `LocalSubprocessSandbox` (full
  `SandboxBackendProtocol`, stdlib-only): real subprocess exec, POSIX rlimits via a
  minimal `preexec_fn`, timeout → process-group kill (SIGTERM→grace→SIGKILL, exit
  124), bounded output drain, exit-code gating. Pluggable via
  `SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy(...)))`;
  `before_execute` admission (allow / reject / tighten); command observability via the
  `on_command` / `CommandEvent` out-of-band sink. DANGEROUS opt-in — not a security
  sandbox; the offline `InMemorySandbox` stays the zero-dependency default.
- **In-run human sign-off (M4 · C, superset of Claude Code)** — `ctx.checkpoint(ask,
  *, tag)` pauses a run for a human decision and resumes with it via a journal-driven
  gate (deliberately NOT LangGraph's native `interrupt`, whose index-based `@task`
  cache misaligns with the content-hash journal on resume): it raises
  `WorkflowSignoffRequired` when undecided, and `run_workflow(resume=...)` injects the
  decision. Adds `BgStatus.AWAITING_SIGNOFF`, `BgRunManager.approve`, an `approve`
  tool command, and `WorkflowCheckpointError`.
- **Real git worktree + branch/PR (M6 · I, superset of Claude Code)** — each
  file-mutating leaf runs in its own real `git worktree` / branch via
  `GitWorktreeProvider` (`SandboxManager(git_worktree_provider=...)`); its authoritative
  changeset is the real `git diff` collected inside the leaf task (a schema-less or
  wrong-typed `files` field fails loud). `LocalSubprocessSandbox(root=, on_close=)`
  roots a leaf sandbox in the real worktree. A `PullRequestProvider` protocol +
  `PullRequestRef` + offline `LocalPullRequestProvider` finalize a PR into an
  integration branch as host finalization, outside deterministic replay. DANGEROUS
  opt-in — real `git` subprocess.
- **`ctx.dag` topological fan-out + lifted nesting (M7 · H)** — `ctx.dag(nodes)` runs a
  dependency graph in topological order (the fourth fan-out frame): each
  `DagNode(id, deps, run)` runs once its predecessors settle and receives their results;
  ready nodes run concurrently with no level barrier, a failed node skips its dependents
  transitively (a legitimately-`None` node does not), and engine signals fail loud after
  an in-flight drain. `ctx.workflow()` nesting is lifted from a hard 1-level cap to a
  configurable `max_workflow_depth` (default 8) with name-stack cycle detection. Adds
  `ctx.loop_until(body, *, done, max_iters)` (a measured-stop loop with a mandatory cap
  and a full-accumulated stop predicate) and SKILL.md authoring patterns (M1.5). New
  symbols: `DagNode`, `SpanKind.DAG`, `WorkflowDagError`, `WorkflowCycleError`; all three
  structural errors (`WorkflowDagError` / `WorkflowCycleError` / `WorkflowNestingError`)
  join `WORKFLOW_CONTROL_FLOW_SIGNALS` so a depth/cycle/shape breach inside a fan-out
  fails loud rather than masking as a `None` hole.
- **Leaf-level live observability (Layer 1)** — `run_workflow` gains keyword-only,
  default-no-op out-of-band sinks: `on_span_begin` (the running edge with a
  resume-stable `span_id`), `on_leaf_event` (a leaf's callback subtree normalized to
  `LeafEvent`), and `on_command` (real-subprocess `CommandEvent`s), with
  `leaf_event_include_payloads` / `command_include_payloads`. All miss-only (a
  journal-cached leaf emits none) and out-of-band (the host LLM context is unchanged).

### Changed

- **`isolation="worktree"` gains a real git backend (M6)** — beyond the in-memory
  seeded copy (0.2.0 / G2), a configured `GitWorktreeProvider` roots the leaf in a real
  `git worktree` on its own branch; teardown is bound to the backend's `close()`, and
  blocking git is thread-offloaded out of the sandbox admission lock.
- **The background run registry can persist (M3)** — the host `journals` / `run_specs`
  registries move behind the `WorkflowRunStore` seam, so `resume` finds a run across
  processes when a persistent store is wired (the defaults stay in-memory /
  same-session).

## [0.2.0] - 2026-06-03

Capability release: four use-case-driven gaps (G1–G4) closed against Claude Code's
Dynamic Workflows, each landed through full TDD, a real-model end-to-end acceptance
run, and a cross-model adversarial review. The public API stays backward-compatible
— every addition is keyword-only or a new symbol.

### Added

- **`agent(schema=...)` structured output (G1)** — a leaf can now return a
  validated object instead of folded text. Pass a pydantic `BaseModel` subclass or
  an inline JSON-schema `dict`; the leaf is built with a `ToolStrategy` bound to the
  schema and the result is validated before it reaches the script. Overloaded so the
  static return type narrows to the model (or `str` without a schema). A
  JSON-schema-to-pydantic converter underpins the `dict` form, with collision,
  `additionalProperties`, `required ⊆ properties`, unknown-key, and resource-bound
  (depth / property / enum / cache) guards.
- **`read_only_leaf` / `read_only_builder` (G4)** — leaf constructors that force a
  deny-write `FilesystemPermission` over the whole tree, so a judge can read / grep /
  glob but never write. `read_only_builder` is a roster `builder` that threads
  `response_format`, so `agent(agent_type="judge", schema=Verdict)` yields a
  structured, read-only judge. Both reject `backend=` / `permissions=` overrides that
  would defeat the guarantee.
- **`WorktreeProvider` protocol + `InMemoryWorktreeProvider` (G2)** — the pluggable
  seam that seeds and collects a per-leaf workspace, wired into `SandboxManager` so
  `isolation="worktree"` provisions a real isolated copy.
- **Community quality-pattern library in the dynamic-workflow `SKILL.md` (G3)** —
  authoring guidance distilled from real community workflows: adversarial-verify
  (refute-by-default with a voter index), pipeline-by-default, fan-out → reduce-in-
  Python → synthesize, loop-until-dry (with a hard `MAX_ROUNDS` and a
  `budget.total is None` guard), judge-panel, model-routing, and no-silent-caps.
- **Examples** — `examples/09_quality_patterns.py` (adversarial-verify +
  loop-until-dry), `examples/10_worktree_fix_swarm.py` (parallel fixers in isolated
  worktrees returning `Patch` objects, reviewed with before/after), and
  `examples/11_readonly_judge_real_e2e.py` (a read-only judge observed writing zero
  files). `examples/07_deep_research_real_e2e.py` now wires its skeptic as a
  `read_only_leaf` to demonstrate the integrated host path.

### Changed

- **`isolation="worktree"` now has real semantics (G2)** — previously the value was
  accepted and folded into the journal key but provisioned no isolated workspace
  (it aliased shared state). It now seeds a per-leaf copy from a base snapshot via
  the `WorktreeProvider` seam, so parallel write-capable leaves no longer collide.
  Guarded by a fail-loud check: `isolation="worktree"` on a non-execution
  (`needs_execution=False`) leaf raises rather than silently degrading.

## [0.1.0] - 2026-06-02

First tagged release: the engine is feature-complete for 0.1.0 and the public API
is stable.

### Added

- **Orchestration runtime (Layer 1)** — the seven primitives on a `Ctx`:
  `agent()`, `parallel()` (blocking barrier), `pipeline()` (no-barrier streaming),
  `phase()` / `log()` (replay-idempotent progress), a shared token `budget`, and
  one-level inline `workflow()` nesting.
- **Library core** — `run_workflow(orchestrate, *, roster, ...)`, the developer /
  build-time entry that runs an orchestration callable to completion and returns
  only the final result.
- **Substrate binding (Layer 0)** — built on LangGraph durable execution
  (`@entrypoint` + `@task` + checkpointer) for resume / replay / cached-skip.
- **Content-hash journal** (success-only) with an `InMemoryJournalStore` and a
  `JournalStore` protocol, so completed leaves replay from cache at zero model
  cost across runs.
- **Fail-loud determinism guard** — a journal-divergence backstop that refuses to
  serve a positionally misaligned cache entry when a replay diverges.
- **Per-leaf sandbox isolation** — a `SandboxManager` with journal-key-derived
  identity, tiered admission (`needs_execution`), `/shared/` artifact hand-off
  with traversal guarding, TTL reclamation, and an engine-owned teardown finale.
- **Leaf roster** — `Roster` / `RosterEntry` resolving named deepagent leaves.
- **Host-facing surface (Layer 2)** — `create_workflow_tool` (the agent's single
  multi-command runtime surface: `run` / `run_script` / `status` / `resume` /
  `cancel`) and `create_workflow_middleware` (packages the tool + injects in-band
  completion notifications), backed by a self-built async background run mechanism
  (`BgRunManager`, `ResultStore`) with composite `(thread_id, run_id)` keying,
  large-result offload, and idle/hard TTL sweep.
- **Meta layer (Layer 2)** — an AST security gate + a single, confined restricted
  `exec` that compiles a host-authored source string into an orchestration
  callable: `compile_workflow_source`, `run_workflow_from_source`, `extract_meta`,
  and a `WorkflowScriptError` carrying the enumerated gate violations. The
  `run_script` tool command lets a host author an `async def orchestrate(ctx, args)`
  on the spot; a rejection returns the violations so the host can fix and resubmit.
  The engine stays source-unaware (it only runs callables); `exec` is confined to
  this one auditable seam. The gate is not a security sandbox (A1 boundary).
- **L2-as-skill teaching pack** — bundled orchestration `SKILL.md`, located via
  `skills_path()` or loaded disk-free via `skill_files()`.
- **Observability-by-default** — every `agent` / `parallel` / `pipeline` call
  emits a structured `Span` (kind, name, attributes, duration, error) to an
  opt-in `run_workflow(on_span=...)` sink; the default recorder is a silent no-op.
- **Bounded background fan-out** — `BgRunManager(max_concurrent_runs=...)`
  refuses a new run with `BgRunQuotaExceededError` (surfaced by the `run` command
  as a clear message) once the quota is full, rather than launching unbounded.
- **Architecture guards** — import-linter contracts enforcing the one-directional
  Layer 2 -> Layer 1 -> Layer 0 dependency, plus a coverage gate (line >= 85%).
- **Examples** — `examples/01`..`08`. `06_capstone.py` is the flagship: a
  multi-stage parallel-research -> pipeline-refine -> adversarial-verify ->
  synthesize workflow driven by a host agent in the background.
  `07_deep_research_real_e2e.py` has a live OpenRouter host launch a registered
  `deep_research` workflow; `08_meta_layer_run_script.py` has the host author an
  ad-hoc script and submit it via `run_script`. All examples run offline on fake
  models; a real-leaf variant is env-gated behind `LDW_DEMO_REAL_MODEL`.

[0.3.0]: https://github.com/nemori-ai/langchain-dynamic-workflow/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nemori-ai/langchain-dynamic-workflow/releases/tag/v0.2.0
[0.1.0]: https://github.com/nemori-ai/langchain-dynamic-workflow/releases/tag/v0.1.0

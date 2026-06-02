# Changelog

All notable changes to `langchain-dynamic-workflow` are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/nemori-ai/langchain-dynamic-workflow/releases/tag/v0.1.0

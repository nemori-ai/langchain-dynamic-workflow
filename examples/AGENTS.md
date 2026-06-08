# Guidance for `examples/`

This file is the authoritative guide for the example set: what lives here, how the
demos are organized, and how to add or change one without drifting. The root
`README.md` points here for the full demo index; keep this file in sync when you
add, rename, or remove a demo.

## 1. Purpose

`examples/` is a copy-paste-ready set of **host-wiring integration examples** and the
outward proof of the engine's capabilities. Each file shows a real host driving the
workflow runtime end to end — not a unit test and not a tutorial that narrates tool
mechanics. Read a demo to see exactly how a deployment would wire one capability into
a host, then lift the wiring into your own code.

## 2. Two-layer taxonomy

The set has exactly two layers, with different rules:

- **Flagships** (`flagship/`, 2 demos): the only real-model demos and the only ones
  that combine multiple mechanisms. They run a full deep-research pipeline (parallel
  search → extract → adversarial verify → synthesize) on live models with native web
  search and prompt caching. A flagship is expensive to maintain and is **not added
  casually** — the bar is a genuinely new end-to-end story, not a new single feature.
- **Feature demos** (`features/`, 16 demos): each demonstrates **exactly one**
  mechanism, runs **fully offline and deterministically**, and needs **no API key**.
  They use the shared scripted fakes in `_shared/offline_models.py`, so they are fast,
  reproducible, and safe to run in CI.

### Authoritative demo index

This table is the authoritative index of every demo. The two flagships:

| Demo | Mechanism |
|---|---|
| `flagship/deep_research_preset` | Real-model host drives the **registered** `deep_research` workflow: parallel search → extract → adversarial verify → synthesize, with native web search + prompt caching; schema-as-handoff and reduce are embedded in the registered workflow. |
| `flagship/deep_research_authored` | Real-model host **authors** the deep-research script live and submits it via `run_script` (AST-gate happy path), then runs it on the same web-search + caching leaf stack. |

The 16 feature demos, one mechanism each:

| Demo | Single mechanism |
|---|---|
| `features/agent_and_schema` | `agent()` single-leaf end to end + structured-output schema handoff. |
| `features/parallel` | `parallel()` barrier fan-out + filtering failed leaves (cross-reference pipeline). |
| `features/pipeline` | `pipeline()` no-barrier staged flow (cross-reference parallel). |
| `features/race` | `ctx.race` best-of-N early exit + in-flight cancel + journaled determinism (replay reproduces the winner and dispatches nothing new). |
| `features/nesting` | `workflow()` named nesting (one level of nesting). |
| `features/budget` | loop-until-budget / loop-until-dry via `ctx.budget.remaining()`. |
| `features/observability` | `phase()` / `log()` narration + the full tap set — `on_progress` / `on_span` for the trace, `on_span_begin` for the running edge, `on_leaf_event` for a leaf's run-tree subtree (the same surfaces the demo app consumes). |
| `features/reduce` | cross-leaf reduce family: gather → `dedup` → `corroborate` → `survives` (voting) → `reconcile` (double-blind reconciliation). |
| `features/journal_resume` | in-process resume: content-hash journal replay + fail-loud determinism guard. |
| `features/persistence` | cross-session sqlite store: proc1 launches → "restart" → proc2 resumes, zero-cost replay (an offline counter proves the replay is free). |
| `features/sandbox` | per-leaf sandbox isolation + `/shared/` artifact handoff. |
| `features/worktree` | worktree-isolated fix swarm (parallel fixers cannot see each other's edits). |
| `features/readonly_judge` | `read_only_leaf`: a judge is denied at the tool boundary even when asked to write. |
| `features/ast_gate` | meta-layer safety seam: an unsafe script → precise AST-gate rejection → rewrite → run. |
| `features/host_integration` | `create_workflow_tool` / `create_workflow_middleware` wired into a scripted host; the host launches N background runs and views the aggregate runs board. |
| `features/signoff` | in-run HITL sign-off: `ctx.checkpoint` parks the run for a human, the host catches `WorkflowSignoffRequired` and resumes with the decision (pre-gate leaf replays for free; approve proceeds, reject holds). |

### Suggested learning path

Read in this order to build up from a single leaf to the full flagships:

1. `agent_and_schema` — one leaf, structured handoff.
2. `parallel` → `pipeline` — fan-out with a barrier vs. a staged flow without one.
3. `budget` → `observability` → `journal_resume` — loop control, narration, replay.
4. `reduce` — synthesizing many leaf results into one.
5. `race` — best-of-N early exit and journaled cancel.
6. `nesting` — composing a named sub-workflow.
7. `host_integration` — wiring the runtime into a host and watching many runs.
8. `sandbox` → `worktree` → `readonly_judge` — isolation and tool-boundary guards.
9. `ast_gate` — the meta-layer reject-and-retry seam.
10. `flagship/deep_research_preset` → `flagship/deep_research_authored` — the full
    real-model combination, registered vs. authored live.

## 3. How to add a demo

A new **core feature** gets **one offline feature demo**:

- Put it in `features/<name>.py` with a descriptive, unnumbered name.
- Build its host and leaves from `examples._shared.offline_models` so it runs with
  **zero API key**, deterministically, via `uv run python -m examples.features.<name>`.
- End the demo with an **assertion that proves the mechanism** — that assertion doubles
  as the per-demo smoke check.

Respect the **道 / 术 line** from the root `AGENTS.md`. A host's `system_prompt`
carries only the **mental model** (the *why* and *when*): control-flow inversion,
decomposing a hard task into parallel sub-work and synthesizing, cross-checking before
committing, delegating heavy multi-step work to the background, composing a procedure
when no ready-made one exists. The prompt **must not teach mechanics**: no command
names (`run` / `run_script` / `status` / `resume` / `cancel`), no registered-workflow
names, no `args` shapes, no script-authoring steps, no AST-gate rules. **User messages
read like a real user's words** — a natural request, never tool instructions.

Feature demos **must not** carry an `LDW_DEMO_REAL_MODEL` real path. The real path
belongs to the flagships only.

## 4. Per-gap real-model acceptance

When a new gap is closed, run its real-model end-to-end acceptance as a
**development-time gate** — a temporary real run, or by extending a flagship to
exercise the new path — **not** as a resident real path inside a feature demo. The two
flagships carry the permanent real-model coverage; CI stays offline. If a real-model
host cannot complete a flagship from a 道-level prompt plus the skill plus the tool
description alone, that is a signal to improve the skill or the tool description, never
to drop 术 into the prompt.

## 5. Demo skeleton convention

Every feature demo follows the same shape:

1. A module docstring naming the **single mechanism** the demo proves.
2. An `async def main()` entry point.
3. Print the results so a reader sees what happened.
4. **Assert the mechanism** — this doubles as the smoke check.
5. A trailing comment with the `-m` run command.

Offline-scripted hosts encode their turn logic in **code**, not in a prompt, so they
are exempt from the prompt persona rule of §3 — but their **user messages still read
naturally**, like a real person's request.

## 6. Scaffolding, naming, and README sync

Shared scaffolding lives in `_shared/`:

- `_shared/offline_models.py` — reusable deterministic fakes for feature demos
  (`ScriptedModel`, `echo_leaf`, `structured_leaf`, `structured_builder`,
  `ToolCallThenReplyModel`). Prefer reusing these over inlining a new fake.
- `_shared/real_models.py` — `ChatOpenRouter` + provider lock + native web search +
  `load_demo_env` / `real_model` / `real_leaf_model` / `demo_cache_middleware`.
  **Flagship-only.**
- `_shared/prompt_caching.py` — `PromptCachingMiddleware`. **Flagship-only** (it is a
  no-op offline, so feature demos do not need it).

**Naming and placement.** Names are descriptive and unnumbered. Flagships go in
`flagship/`, feature demos in `features/`. Reading order is given by the learning path
in §2, so inserting a new demo never forces a renumbering.

**Packaging.** `examples/` and its subpackages carry `__init__.py` so demos run as
modules (`uv run python -m examples.features.<name>`); the repo root is on `sys.path`
and demos use absolute imports (`from examples._shared.offline_models import ...`).
`examples/` is not shipped in the wheel — the package exists only for `-m` imports.

**README sync.** When you add, rename, or remove a demo, update the table in §2 of this
file **and** the Examples pointer in the root `README.md`.

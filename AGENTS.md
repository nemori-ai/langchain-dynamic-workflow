# Development guidelines for langchain-dynamic-workflow

## Project Overview

`langchain-dynamic-workflow` is a **dynamic workflow orchestration engine** for the LangChain
[`deepagents`](https://github.com/langchain-ai/deepagents) ecosystem — a community port of Claude
Code's **Dynamic Workflows**. Instead of an LLM deciding control flow turn by turn, a deterministic
orchestration **script** owns the loops, branching, and fan-out, and only leaf `agent()` calls
delegate to deepagents — each running in an isolated, discarded context, so only the final result
reaches the caller.

The full architecture and design rationale are the **source of truth** in
[`docs/plans/2026-06-01-langchain-dynamic-workflow-engine-design.md`](docs/plans/2026-06-01-langchain-dynamic-workflow-engine-design.md).

### Core idea — control-flow inversion

| | Normal agent | Dynamic workflow |
|---|---|---|
| Who decides the next step | the LLM, turn by turn | the **script** (deterministic code) |
| Where intermediate results live | the model's context window | **script variables** |
| What reaches the caller's context | the whole trajectory | **only the final result** |

### Architecture — three layers

- **Layer 0 — substrate**: LangGraph durable execution (`@entrypoint` + `@task` + checkpointer). Provides resume / replay / cached-result-skip for free.
- **Layer 1 — orchestration runtime**: the primitives — `agent()`, `parallel()` (barrier), `pipeline()` (no barrier), `race()` (best-of-N early exit), `dag()` (topological fan-out), `loop_until()` (bounded sequential loop), `batch_map()` (streaming bounded-window large fan-out with count/ETA progress), `phase()`, `log()`, `budget`, `workflow()`, `checkpoint()` (in-run human sign-off) — plus two patches LangGraph lacks: a **content-hash journal** (LangGraph's native cache is index-based) and a **fail-loud determinism guard** (LangGraph treats determinism as convention, not invariant). A detached **background run**'s runtime events are buffered on its `BgRunManager` slot (`event_sinks` → `buffered_events`) so a live turn can replay its interior — the same observability sinks, made consumable after the launching turn ends.
- **Layer 2 — meta layer**: an LLM authors the Python orchestration script for a described task; an **AST gate** validates it (no imports / dunders / banned names) before execution.

Leaf `agent()` calls resolve a named deepagent from a **registry (roster)** and invoke it as a `@task`, reusing deepagents' context quarantine and sandbox backends.

## Repository Structure

```
langchain-dynamic-workflow/
├── pyproject.toml                  # uv-managed; ruff / pyright / pytest config
├── src/langchain_dynamic_workflow/ # source package (src layout)
│   ├── __init__.py
│   └── py.typed
├── tests/                          # pytest suite
├── docs/plans/                     # design docs (gitignored, not committed)
└── .claude/rules/                  # dev discipline rules
```

## Development Toolchain & Commands

| Tool | Purpose |
|------|---------|
| **uv** | Dependency management and virtual environment |
| **Ruff** | Linting and formatting |
| **Pyright** | Static type checking (strict mode) |
| **pytest** + pytest-asyncio | Test runner |

```bash
uv sync                      # install deps + create .venv
uv run pytest                # run tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pyright               # type check (strict)
```

## Core Development Principles

- **Python 3.12+, async-first.** Complete type hints; Pyright strict mode is enforced.
- **Industrial-grade resilience**: timeout protection on external/untrusted calls, fast-fail before
  expensive operations, bounded queues / resource-exhaustion guards, graceful degradation, no silent
  failures, intentional concurrency (sequential vs. parallel is a deliberate, defended choice).
- **Security**: never `eval`/`exec`/`pickle` on untrusted input outside the engine's sandboxed,
  AST-validated execution path; never bare `except:`; ensure resource cleanup for files, connections,
  and async tasks.
- **Stable interfaces**: preserve public signatures; new parameters are keyword-only with defaults.

Follow the conventions under `.claude/rules/`:

- `python_general.md` — Python style (Google style guide, py312, imports, enums).
- `docstring.md` — docstring conventions (English, Google style).
- `testing.md` — testing conventions (pytest + pytest-asyncio).

## Examples & Demos

The `examples/` demos must read like a real user driving the product, not a tutorial that
hand-holds the agent on tool mechanics. Draw the line at **道 vs 术** (mental model vs
technique):

- **Mental model (道) — allowed in the `system_prompt`.** A host prompt may carry
  workflow-related mental models and philosophy: the *why* and *when* — control-flow
  inversion, decomposing a hard task into parallel sub-work and synthesizing, cross-checking
  before committing, delegating heavy multi-step work to run in the background, composing a
  procedure when no ready-made one exists. This is the conceptual persona a real deployment
  would give its assistant.
- **Technique (术) — never in a prompt.** The same prompt must NOT teach *how* to drive the
  `workflow` tool: no command names (`run` / `run_script` / `status` / `resume` / `cancel`),
  no registered-workflow names, no `args` shapes, no script-authoring steps or AST-gate
  rules. That is the job of the tool's `description`, its `help` command, and the bundled
  `SKILL.md` (loaded via `skills=[...]`). A demo that coaches the mechanics hides whether
  those are self-sufficient — exactly what the demo should prove.
- **User messages are a real user's words** — a natural request, never tool instructions.
- **Offline scripted hosts** (deterministic `BaseChatModel` fakes) encode the turn logic in
  code, not in a prompt, so they are exempt from the persona rule — but their user messages
  should still read naturally.
- When a real-model demo (`LDW_DEMO_REAL_MODEL`) cannot complete from a 道-level prompt + the
  skill + the tool description alone, that is a signal to improve the skill or the tool
  description — never to drop down to 术 in the prompt.

## Design & Plan File Management

- **Temporary design/plan docs** go in `docs/plans/` — gitignored, not version-controlled.
- **Permanent design artifacts** (ADRs, design docs) are committed when introduced.
- Commit or push only when explicitly asked.

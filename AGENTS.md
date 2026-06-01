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
- **Layer 1 — orchestration runtime**: the primitives — `agent()`, `parallel()` (barrier), `pipeline()` (no barrier), `phase()`, `log()`, `budget`, `workflow()` — plus two patches LangGraph lacks: a **content-hash journal** (LangGraph's native cache is index-based) and a **fail-loud determinism guard** (LangGraph treats determinism as convention, not invariant).
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

## Design & Plan File Management

- **Temporary design/plan docs** go in `docs/plans/` — gitignored, not version-controlled.
- **Permanent design artifacts** (ADRs, design docs) are committed when introduced.
- Commit or push only when explicitly asked.

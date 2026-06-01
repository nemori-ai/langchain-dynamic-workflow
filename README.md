# langchain-dynamic-workflow

**English** | [Chinese](README_zh.md)

> Deterministic, scripted, resumable multi-agent orchestration for LangChain [`deepagents`](https://github.com/langchain-ai/deepagents) — a community port of Claude Code's **Dynamic Workflows**.

## What

A normal agent decides its control flow **turn by turn** — every loop, branch, and fan-out lives in the model's context window, burning tokens and accumulating intermediate state. `langchain-dynamic-workflow` **inverts** that: a deterministic orchestration **script** owns the control flow, and only leaf `agent()` calls delegate to deepagents — each running in an isolated, discarded context, so **only the final result reaches the caller's context**.

| | Normal agent | Dynamic workflow |
|---|---|---|
| Who decides the next step | the LLM, turn by turn | the **script** (deterministic code) |
| Where intermediate results live | the model's context window | **script variables** |
| What reaches the caller's context | the whole trajectory | **only the final result** |

## Why

The engine sits on top of LangGraph's durable execution (`@entrypoint` + `@task`), which already provides resume / replay / cached-result-skip. On top of it, this library adds the pieces that turn a deepagents stack into a **scriptable, fan-out-capable, resumable** orchestration runtime — and, optionally, a meta layer where an LLM writes the orchestration script for a task you describe.

## Architecture (three layers)

- **Layer 0 — substrate**: LangGraph durable execution (`@entrypoint` + `@task` + checkpointer).
- **Layer 1 — orchestration runtime**: the primitives — `agent()`, `parallel()` (barrier), `pipeline()` (no barrier), `phase()`, `log()`, `budget`, `workflow()` — plus a content-hash journal and a fail-loud determinism guard.
- **Layer 2 — meta layer**: an LLM authors the Python orchestration script; an AST gate validates it before execution.

Leaf `agent()` calls resolve a named deepagent from a **registry (roster)** and invoke it as a `@task`, reusing deepagents' context quarantine and sandbox backends.

The full design rationale lives in [`docs/plans/`](docs/plans/) (design baseline; gitignored, not version-controlled).

## Status

**Early stage — architecture locked, public API being built out.** Not yet published to PyPI.

## Development

```bash
uv sync                 # install dependencies + create .venv
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
```

Python 3.12+. Dependency management via [uv](https://docs.astral.sh/uv/).

## License

[MIT](LICENSE)

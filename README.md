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

## Quickstart

```bash
uv sync   # install dependencies + create .venv
```

Write an orchestration script (the `ctx` exposes the primitives), register your leaf agents in a roster, and run it. Leaves resolve to any runnable whose state has a `messages` key — typically a `deepagents.create_deep_agent(...)`:

```python
import asyncio

from deepagents import create_deep_agent
from langchain_dynamic_workflow import Ctx, Roster, run_workflow


async def main() -> None:
    # 1. Register leaf agents by name (build-time wiring; the agent never does this).
    roster = Roster().register(
        "researcher",
        create_deep_agent(model="anthropic:claude-haiku-4-5"),
        description="Researches one topic",
    )

    # 2. The orchestration script owns the control flow; only leaf agent() calls
    #    delegate to deepagents. parallel() is a blocking barrier; a failed leaf
    #    lands as None (filter the holes) and never aborts the barrier.
    async def orchestrate(ctx: Ctx) -> str:
        ctx.phase("research")
        findings = await ctx.parallel(
            [
                lambda t=topic: ctx.agent(f"Research {t}", agent_type="researcher")
                for topic in ("batteries", "solar", "wind")
            ]
        )
        surviving = [f for f in findings if f is not None]
        return f"synthesized {len(surviving)} findings: " + " | ".join(surviving)

    # 3. Run it. Only the final result reaches you — not the whole trajectory.
    result = await run_workflow(orchestrate, roster=roster)
    print(result)


asyncio.run(main())
```

Pass the **same** `journal=` across calls to get cached-result resume (completed leaves replay at zero model cost), `budget=` for a shared token ceiling, and `on_span=` for an observability trace. To let a **host agent** drive workflows in the background, attach `create_workflow_middleware(roster, workflows=...)` to a host `create_deep_agent` — the agent calls a single `workflow` tool to launch a registered workflow by name (`run`) or **author and submit an ad-hoc script on the spot** (`run_script` — the meta layer), poll it (`status`), `resume`, or `cancel`, and is notified when a run finishes.

`run_script` is the meta layer: the agent writes an `async def orchestrate(ctx, args)` and submits the source, which passes an **AST security gate** and runs under a restricted-builtins namespace. The gate stops an accidental slip — it is **not a security sandbox**; a determined adversarial script can still escape, so submit only scripts the agent authors itself (for adversarial input, run the engine behind an out-of-process isolation backend). Build-time code can do the same programmatically with `run_workflow_from_source(source, roster=...)`.

Every example under [`examples/`](examples/) runs **offline with no API key** (fake models). To drive real leaves through OpenRouter and capture LangSmith traces, install the demo extras with `uv sync --group example`, put `OPENROUTER_API_KEY` and the `LANGSMITH_*` settings in a local `.env`, then set `LDW_DEMO_REAL_MODEL` (model defaults to `anthropic/claude-opus-4.8`; set it to any OpenRouter slug to override). The flagship is [`examples/06_capstone.py`](examples/06_capstone.py): a host agent driving a background `parallel`-research → `pipeline`-refine → adversarial-verify → synthesize workflow. For a fully real run, [`examples/07_deep_research_real_e2e.py`](examples/07_deep_research_real_e2e.py) has a **live OpenRouter host agent** decide to launch a registered `deep_research` workflow (search → extract → adversarial-verify → synthesize) end to end. [`examples/08_meta_layer_run_script.py`](examples/08_meta_layer_run_script.py) shows the **meta layer**: the host *authors* an ad-hoc script and submits it via `run_script` (offline, it also demonstrates the gate rejecting an unsafe first attempt and the host fixing it).

```bash
uv run python examples/06_capstone.py

# fully real end-to-end (live OpenRouter host + leaves):
LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8 uv run python examples/07_deep_research_real_e2e.py
```

## Public API

The stable, public surface is exported from the package root and follows semantic versioning from `0.1.0`:

- **Library core**: `run_workflow` — the developer / build-time entry.
- **Meta layer**: `compile_workflow_source` / `run_workflow_from_source` / `extract_meta` — compile and run an LLM-authored source string through the AST gate.
- **Registries**: `Roster` / `RosterEntry`, `WorkflowRegistry`.
- **Host-facing**: `create_workflow_tool`, `create_workflow_middleware`, `skills_path` / `skill_files`.
- **Primitives**: exposed on the `Ctx` handed to your script — `agent` / `parallel` / `pipeline` / `phase` / `log` / `budget` / `workflow`.
- **Types & errors**: `Budget`, `JournalStore` / `InMemoryJournalStore` / `JournalRecord`, `SandboxManager`, `Span` / `SpanKind` / `SpanSink`, the `BgRunManager` family, and the `Workflow*Error` exceptions (including `WorkflowScriptError`).

Public signatures are stable; new parameters are added keyword-only with defaults. Names prefixed with `_` (modules and members) are internal and may change without notice.

## Status

**v0.1.0 — architecture locked, public API stable.** Not yet published to PyPI. See [`CHANGELOG.md`](CHANGELOG.md).

## Development

```bash
uv sync                 # install dependencies + create .venv
uv run pytest           # run tests (with coverage gate, line >= 85%)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
uv run lint-imports     # check the Layer 0/1/2 architecture boundaries
```

Python 3.12+. Dependency management via [uv](https://docs.astral.sh/uv/).

## License

[MIT](LICENSE)

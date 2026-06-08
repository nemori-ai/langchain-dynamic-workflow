# langchain-dynamic-workflow

[![CI](https://github.com/nemori-ai/langchain-dynamic-workflow/actions/workflows/ci.yml/badge.svg)](https://github.com/nemori-ai/langchain-dynamic-workflow/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Types: pyright strict](https://img.shields.io/badge/types-pyright%20strict-blue.svg)
![Status: alpha 0.2.0](https://img.shields.io/badge/status-alpha%200.2.0-orange.svg)

**English** | [中文](README_zh.md)

> Deterministic, scripted, resumable multi-agent orchestration for LangChain [`deepagents`](https://github.com/langchain-ai/deepagents) — a community port of Claude Code's **Dynamic Workflows**.

A normal agent decides its control flow **turn by turn**: every loop, branch, and fan-out lives in the model's context window, burning tokens and accumulating intermediate state. `langchain-dynamic-workflow` **inverts** that — a deterministic orchestration **script** owns the control flow, and only leaf `agent()` calls delegate to deepagents, each running in an isolated, discarded context, so **only the final result reaches the caller's context**.

|  | Normal agent | Dynamic workflow |
|---|---|---|
| Who decides the next step | the LLM, turn by turn | the **script** (deterministic code) |
| Where intermediate results live | the model's context window | **script variables** |
| What reaches the caller's context | the whole trajectory | **only the final result** |

## Table of contents

- [Why](#why)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Host agents and the meta layer](#host-agents-and-the-meta-layer)
- [Resume, budget, and observability](#resume-budget-and-observability)
- [Examples](#examples)
- [Public API](#public-api)
- [Development](#development)
- [Status](#status)
- [License](#license)

## Why

Turn-by-turn control flow has three costs that compound as a task grows: the context window fills with intermediate reasoning, the trajectory is non-deterministic, and an interrupted run cannot resume without replaying the model. Inverting control flow addresses all three at once — the script holds the loops and branches, intermediate results stay in plain variables, and a content-hash journal makes a resumed run replay completed work at zero model cost.

Reach for it when a task is **fan-out heavy** (research N angles, grade M candidates), **long and multi-step** (the trajectory would otherwise overflow the context), or needs **deterministic resume and a shared token budget** across many sub-agents.

## Highlights

- **Deterministic control flow** — loops, branching, and fan-out live in code, not in the model's head.
- **Context quarantine** — each leaf runs in a fresh, discarded deepagents context; only its folded result returns.
- **Parallel, pipeline, and race fan-out** — `parallel()` (blocking barrier), `pipeline()` (no-barrier streaming), and `race()` (best-of-N early exit: the first result satisfying `win` wins, in-flight losers are cancelled, the decision is journaled so resume reproduces the winner) over a shared concurrency gate.
- **Resumable by content hash** — a success-only journal replays completed leaves on resume at zero model cost.
- **Fail-loud determinism guard** — a replay whose `agent()` call sequence diverges raises rather than serving a positionally misaligned cache entry.
- **Shared token budget** — one ceiling across every leaf, with a `loop-until-budget` idiom.
- **Observability by default** — every `agent` / `parallel` / `pipeline` / `race` call emits a span to an opt-in sink (zero cost when unset).
- **Per-leaf sandbox isolation** — execution leaves lease isolated backends; a `/shared/` route enables explicit producer→consumer hand-off.
- **Meta layer** — a host agent authors an orchestration script at runtime; an AST gate validates it before a single restricted `exec`.
- **Strict engineering** — Python 3.12, async-first, pyright `strict`, and Layer 0/1/2 boundaries enforced by import-linter.

## Architecture

Three layers, with a one-directional dependency (Layer 2 → Layer 1 → Layer 0) that import-linter enforces mechanically:

- **Layer 0 — substrate**: LangGraph durable execution (`@entrypoint` + `@task` + checkpointer). Provides resume, replay, and cached-result skip.
- **Layer 1 — orchestration runtime**: the primitives — `agent()`, `parallel()` (barrier), `pipeline()` (no barrier), `race()` (best-of-N early exit), `phase()`, `log()`, `budget`, `workflow()` — plus the two patches LangGraph lacks: a **content-hash journal** (LangGraph's native cache is index-based) and a **fail-loud determinism guard** (LangGraph treats determinism as convention).
- **Layer 2 — meta layer**: an LLM authors a Python orchestration script; an **AST gate** validates it (no imports, dunders, or banned names) before a restricted-builtins `exec`.

Leaf `agent()` calls resolve a named deepagent from a **registry (roster)** and invoke it as a `@task`, reusing deepagents' context quarantine and sandbox backends.

## Installation

```bash
uv sync   # install dependencies + create .venv
```

Python 3.12+. Dependency management via [uv](https://docs.astral.sh/uv/). The package is not yet published to PyPI; install it from a clone or as a git dependency.

## Quickstart

Write an orchestration script (the `ctx` exposes the primitives), register your leaf agents in a roster, and run it. A leaf is any runnable whose state has a `messages` key — typically a `deepagents.create_deep_agent(...)`:

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

## Host agents and the meta layer

To let a **host agent** drive workflows in the background, attach `create_workflow_middleware(roster, workflows=...)` to a host `create_deep_agent`. The agent then controls runs through a single `workflow` tool:

| Command | Effect |
|---|---|
| `run` | Launch a **registered** workflow by name; returns a `run_id` immediately. |
| `run_script` | Launch an **ad-hoc script the agent authors on the spot** (the meta layer). |
| `status` | Poll a run and fetch its result (large results are offloaded behind a handle). |
| `resume` | Re-run against the same journal so completed leaves replay at zero cost. |
| `cancel` | Stop an in-flight run. |

A run executes in the background and a completion notice is injected before the host's next turn, so launching never blocks the conversation.

**The meta layer (`run_script`).** The host writes an `async def orchestrate(ctx, args)` and submits the source. It passes an **AST security gate** (no imports, dunder access, banned builtins, or `str.format` injection) and then runs under a restricted-builtins namespace via a single, confined `exec`. A rejected script returns its exact violations so the host can fix them and resubmit. Build-time code can do the same programmatically with `run_workflow_from_source(source, roster=...)`.

> **Security boundary.** The gate plus the restricted namespace stop an honest model's slip — they are **not a security sandbox**, and a determined adversarial script can still escape. Submit only scripts the host authors itself; for adversarial input, run the engine behind an out-of-process isolation backend.

> ## ⚠️ Real shell execution is a DANGEROUS opt-in — NOT a security sandbox
>
> By default, execution leaves use the offline in-memory backend and **run no real shell**. You can opt a `SandboxManager` into a real local-subprocess backend (`SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy(...)))`), which runs each leaf's shell command in a **private per-leaf temporary directory**. **This is not a security sandbox — read this before enabling it.**
>
> **What a command CAN still do — it runs as you, on your host:**
> - **Read and write any host path via absolute paths.** The per-leaf temp directory bounds only the *default working directory*, not the reachable filesystem. `cat ~/.ssh/id_rsa` or `rm -rf /some/path` is fully within reach.
> - **Use the network with your user's permissions** — exfiltrate, download, connect anywhere your account can.
> - **Consume host resources beyond the best-effort POSIX `rlimit`s** — the limits are a runaway guard, not a hard cgroup-grade containment.
>
> **What it DOES guarantee** (resilience, not confinement):
> - A private temp working directory per leaf — no accidental execution in the engine's own directory.
> - The engine file APIs stay rooted, with `..` traversal rejected.
> - A bounded effective timeout with best-effort process-group kill (SIGTERM → grace → SIGKILL on POSIX).
> - Bounded combined stdout+stderr output (flagged when truncated).
> - A bounded count of concurrent executions.
>
> **What it deliberately does NOT provide** (deferred sharp edges):
> - Hard filesystem confinement (container / chroot / nsjail / firejail).
> - Network-egress blocking.
> - cgroup-grade CPU / memory / process containment.
> - Strong orphan cleanup for processes that deliberately daemonize or escape the process group.
>
> **The single biggest risk is false security.** A per-leaf temp working directory *looks* isolated but is not. **Never run untrusted or adversarial commands through it.** For those, run the engine behind an out-of-process isolation backend (a container).
>
> **Windows is weaker still:** no `rlimit`s and no honest resource/security posture — only a timeout, an output cap, a temp working directory, best-effort termination, and the concurrency bound.

## Resume, budget, and observability

`run_workflow` takes a few keyword-only knobs that compose:

- **`journal=`** — pass the *same* journal store across calls for cached-result resume: completed leaves replay at zero model cost, and the determinism guard verifies the call sequence has not diverged.
- **`budget=`** — a shared token ceiling for all leaves; once spent, the next `agent()` raises `WorkflowBudgetExceededError`. Drives the `while ctx.budget.remaining() > T` idiom.
- **`on_span=`** — a sink receiving a span for every `agent` / `parallel` / `pipeline` call; resumed runs re-emit spans flagged `cached=True`.
- **`sandbox_manager=`** — leases an isolated execution backend per leaf that needs one; pure-reasoning leaves allocate nothing.

## Examples

The **15 feature demos** under [`examples/features/`](examples/features/) each isolate one mechanism and run **offline with no API key** (deterministic fake models). The two **flagships** under [`examples/flagship/`](examples/flagship/) carry the real end-to-end path: offline by default, they light up live OpenRouter leaves with native web search + prompt caching when you opt in. To drive the real path, install the demo extras with `uv sync --group example`, put `OPENROUTER_API_KEY` and the `LANGSMITH_*` settings in a local `.env`, then set `LDW_DEMO_REAL_MODEL` (defaults to `anthropic/claude-opus-4.8`; set it to any OpenRouter slug to override). The full taxonomy and the authoritative index of every demo live in **[`examples/AGENTS.md`](examples/AGENTS.md)**.

| Demo | Shows |
|---|---|
| **[`flagship/deep_research_preset`](examples/flagship/deep_research_preset.py)** | Flagship (real model): a host drives the registered `deep_research` workflow — parallel search → extract → adversarial verify → synthesize, with native web search + prompt caching. |
| **[`flagship/deep_research_authored`](examples/flagship/deep_research_authored.py)** | Flagship (real model): a host *authors* the deep-research script live and runs it via `run_script` (AST gate happy path), same web-search + caching leaf stack. |

The 15 offline feature demos (one mechanism each, no API key) and the full taxonomy live in **[`examples/AGENTS.md`](examples/AGENTS.md)**.

```bash
# any offline feature demo:
uv run python -m examples.features.parallel

# a flagship, offline by default:
uv run python -m examples.flagship.deep_research_preset

# fully real end-to-end (live OpenRouter host + leaves):
LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8 uv run python -m examples.flagship.deep_research_preset
```

## Public API

The stable, public surface is exported from the package root and follows semantic versioning from `0.1.0`:

- **Library core** — `run_workflow`: the developer / build-time entry.
- **Meta layer** — `compile_workflow_source` / `run_workflow_from_source` / `extract_meta`: compile and run an LLM-authored source string through the AST gate.
- **Registries** — `Roster` / `RosterEntry`, `WorkflowRegistry`.
- **Host-facing** — `create_workflow_tool`, `create_workflow_middleware`, `skills_path` / `skill_files`.
- **Primitives** — exposed on the `Ctx` handed to your script: `agent` / `parallel` / `pipeline` / `race` / `phase` / `log` / `budget` / `workflow`. `ctx.race(candidates, *, win, win_tag="")` runs `RaceCandidate` specs concurrently and returns a `RaceResult` for the first whose result satisfies `win`, cancelling the rest; the decision is journaled (content-hash, `win_tag`-keyed) so resume reproduces the winner and dispatches nothing.
- **Cross-leaf reduce** — pure helpers that fold the result lists `parallel` / `pipeline` hand back: `survives` (refute-by-default vote), `dedup`, `reconcile` (dual-blind reviewer reconciliation), `corroborate` (cross-leaf corroboration), plus the `ReviewItem` / `Reconciled` / `Consensus` result types. Also injected into the `run_script` namespace, so a host-authored script calls them by name without an import.
- **Race value types** — `RaceCandidate` (one content-hashable agent-call spec, mirroring an `agent()` call) and `RaceResult` (the winner, its index, and `.won`). Both are injected into the `run_script` namespace, so a host-authored script constructs and reads them by name without an import.
- **Types and errors** — `Budget`, `JournalStore` / `InMemoryJournalStore` / `JournalRecord` / `race_key`, `SandboxManager`, `Span` / `SpanKind` / `SpanSink`, the `BgRunManager` family, and the `Workflow*Error` exceptions (including `WorkflowScriptError`).

Public signatures are stable; new parameters are added keyword-only with defaults. Names prefixed with `_` (modules and members) are internal and may change without notice.

## Development

```bash
uv sync                 # install dependencies + create .venv
uv run pytest           # run tests (coverage gate, line >= 85%)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
uv run lint-imports     # verify the Layer 0/1/2 architecture boundaries
```

## Status

**0.1.0 — architecture locked, public API stable, not yet published to PyPI.** All three layers are implemented, including the Layer 2 meta layer. See [`CHANGELOG.md`](CHANGELOG.md) for the release log.

## License

[MIT](LICENSE)

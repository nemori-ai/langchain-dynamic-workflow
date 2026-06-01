---
name: dynamic-workflow
description: >-
  Author a deterministic orchestration script for the dynamic-workflow engine,
  then launch it with the workflow tool. Use when a task needs control-flow
  inversion — loops, branching, or fan-out owned by deterministic code rather
  than by turn-by-turn model decisions — so that intermediate results live in
  script variables and only the final conclusion reaches your context. Keywords:
  orchestrate, fan-out, parallel, pipeline, multi-agent, deterministic workflow,
  background run, run/status/resume/cancel.
---

# Dynamic Workflow orchestration

This skill teaches you to write a **deterministic orchestration script** and run
it through the `workflow` tool. The control flow lives in code you author, not in
your turn-by-turn decisions: loops, branching, and fan-out are deterministic, the
intermediate results stay in script variables, and only the final result is
returned to you.

A script is an `async def orchestrate(ctx, args)` coroutine. It receives a `ctx`
object exposing the orchestration primitives and an `args` mapping of inputs. It
returns the final result.

## The DSL (`ctx` primitives)

- `await ctx.agent(prompt, *, agent_type, model=None, isolation="shared")` — run
  one leaf subagent in a fresh, discarded context and get back its final text.
  `agent_type` names a registered leaf. This is the only place a model runs.
- `await ctx.parallel(thunks)` — fan out a list of zero-argument thunks
  concurrently with a blocking barrier. Returns results in input order; a thunk
  whose leaf fails lands as `None` (it never aborts the barrier). Filter the
  `None` holes before using the results.
- `await ctx.pipeline(items, *stages)` — stream items through stages with no
  barrier between stages; each item flows independently. A stage is
  `(prev_result, original_item, index) -> next_result`. A stage that raises drops
  that item to `None` and skips its remaining stages. Results come back in input
  order.
- `await ctx.workflow(name, args)` — inline another registered workflow, exactly
  one level deep. The inner workflow shares this run's journal and budget. A
  second nesting level is refused.
- `ctx.phase(title)` / `ctx.log(message)` — narrate progress (grouping marker /
  free-form line). Display-only; safe to repeat in code.
- `ctx.budget` — the shared token pool: `ctx.budget.total`, `ctx.budget.spent()`,
  `ctx.budget.remaining()`. Drive loops with `while ctx.budget.remaining() > T`.

## Determinism rules (the iron law)

The engine replays your script on resume and caches each leaf result by the
content hash of its inputs. Your script's **observable `agent()` call sequence
must be identical run to run**, or the engine fails loud. To stay deterministic:

- Iterate over **ordered** collections. Never iterate a `set` or a `dict` without
  sorting first — use `sorted(...)`.
- Do not branch on wall-clock time, randomness, or any value that varies between
  runs. If you need an identifier, derive it from the inputs, not from `uuid` or
  `time`.
- Build `parallel` thunks with an explicit default-argument capture so each
  closure binds its own value: `[lambda t=t: ctx.agent(..., agent_type="x") for t in items]`.
- Keep the same prompts and the same `agent_type` on every run; changing them
  changes the cache key (which is intended when you mean to).

## Patterns

Sequential refine-until-budget:

```python
async def orchestrate(ctx, args):
    draft = await ctx.agent(args["task"], agent_type="writer")
    while ctx.budget.remaining() > 500:
        critique = await ctx.agent(f"Critique: {draft}", agent_type="critic")
        if "looks good" in critique.lower():
            break
        draft = await ctx.agent(f"Revise per: {critique}\n\n{draft}", agent_type="writer")
    return draft
```

Parallel fan-out then synthesize:

```python
async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    surviving = [f for f in findings if f is not None]
    return await ctx.agent("Synthesize:\n" + "\n".join(surviving), agent_type="writer")
```

No-barrier pipeline:

```python
async def orchestrate(ctx, args):
    async def research(prev, item, i):
        return await ctx.agent(f"Research {item}", agent_type="researcher")

    async def summarize(prev, item, i):
        return await ctx.agent(f"Summarize: {prev}", agent_type="summarizer")

    return await ctx.pipeline(sorted(args["items"]), research, summarize)
```

## Running it with the `workflow` tool

The script is launched **in the background**, so your turn is not blocked:

1. `workflow(command="run", workflow="<name>", args={...})` — returns a `run_id`
   placeholder immediately. The run executes in the background.
2. Continue working. When the run finishes, a `<workflow_notification>` is
   injected before your next reply listing the finished `run_id`(s).
3. `workflow(command="status", run_id="<id>")` — fetch the result. A large result
   is summarized and offloaded behind a handle.
4. `workflow(command="resume", run_id="<id>")` — re-run against the journal so
   completed steps replay at zero cost (use after an interruption).
5. `workflow(command="cancel", run_id="<id>")` — stop an in-flight run.

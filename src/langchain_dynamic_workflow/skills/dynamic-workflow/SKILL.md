---
name: dynamic-workflow
description: >-
  Author or launch a dynamic-workflow orchestration through the workflow tool,
  and understand how these deterministic scripts invert control flow. Use when a
  task needs control-flow inversion — loops, branching, or fan-out owned by
  deterministic code rather than by turn-by-turn model decisions — so that
  intermediate results live in script variables and only the final conclusion
  reaches your context. Keywords: orchestrate, fan-out, parallel, pipeline,
  multi-agent, deterministic workflow, background run, author script,
  run/run_script/status/resume/cancel.
---

# Dynamic Workflow orchestration

This skill explains how **deterministic orchestration scripts** work and how to
run one through the `workflow` tool. The control flow lives in code, not in your
turn-by-turn decisions: loops, branching, and fan-out are deterministic, the
intermediate results stay in script variables, and only the final result is
returned to you.

A script is an `async def orchestrate(ctx, args)` coroutine. There are two ways to
run one:

- **Launch a registered workflow by name** (`run`) — when a task fits a workflow
  someone wired into the roster ahead of time, recognize it and launch it by name.
- **Author an ad-hoc script and submit it** (`run_script`) — when no registered
  workflow fits, write the `orchestrate` coroutine yourself with the DSL below and
  submit the source. A security gate checks it first; if it is rejected, the exact
  violations come back so you fix them and resubmit.

The DSL, determinism rules, and patterns below describe how these scripts are
built — both so you can pick the right registered workflow and so you can author a
correct one yourself.

## The DSL (`ctx` primitives)

- `await ctx.agent(prompt, *, agent_type, schema=None, model=None, isolation="shared")` — run
  one leaf subagent in a fresh, discarded context. Without `schema` it returns the
  leaf's final **text**. With `schema` — a JSON-schema `dict` written inline (no
  imports needed) — it returns a **validated structured object** you read by
  attribute, so the next line is plain Python over typed data. `agent_type` names a
  registered leaf; a schema requires that leaf to be registered with a builder.
  This is the only place a model runs. Pass `isolation="worktree"` only for a leaf
  that **mutates files in parallel** with its siblings (e.g. one fixer per file in a
  fix swarm): it runs in its own copy of a seeded base workspace, isolated from the
  others, and should hand back its change as a structured patch. Read-only and
  synthesis leaves stay on the default `"shared"`.
- `await ctx.parallel(thunks)` — fan out a list of zero-argument thunks
  concurrently with a blocking barrier. Returns results in input order; a thunk
  whose leaf fails lands as `None` (it never aborts the barrier). Filter the
  `None` holes before using the results.
- `await ctx.pipeline(items, *stages)` — stream items through stages with no
  barrier between stages; each item flows independently. A stage is
  `(prev_result, original_item, index) -> next_result`. A stage that raises drops
  that item to `None` and skips its remaining stages. Results come back in input
  order.
- `await ctx.race(candidates, *, win, win_tag="")` — run several `RaceCandidate`
  specs concurrently and return a `RaceResult` for the **first** whose result
  satisfies `win(result)`; the in-flight losers are cancelled. `RaceCandidate(prompt,
  agent_type, schema=None, model=None, isolation="shared")` mirrors an `agent()`
  call; all candidates must be homogeneous (all schema-less, or all the same
  `schema`). Read `result.won` / `result.winner` / `result.winner_index`. Use this
  over `parallel` when you only need the first good-enough answer and want to stop
  the rest (e.g. multi-hypothesis diagnosis). `win_tag` distinguishes the win
  criterion in the resume journal — see the footgun note in the race pattern below.
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
    MAX_REVISIONS = 5  # a hard cap, so a stubborn model can't loop forever
    draft = await ctx.agent(args["task"], agent_type="writer")
    for _ in range(MAX_REVISIONS):
        # Guard the budget check with .total: with no budget, remaining() is inf.
        if ctx.budget.total and ctx.budget.remaining() < 500:
            break
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

Structured output as the handoff between agents (schema):

```python
async def orchestrate(ctx, args):
    verdict = await ctx.agent(
        f"Refute this claim if you can: {args['claim']}",
        agent_type="skeptic",
        schema={
            "type": "object",
            "properties": {
                "refuted": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["refuted", "reason"],
            "additionalProperties": False,
        },
    )
    return "rejected" if verdict.refuted else "stands"
```

## Quality patterns

The basics above are mechanics. These are the **author patterns** that make an
orchestration trustworthy — borrowed from how the best hand-written workflows are
built. Each is a complete `orchestrate` you can adapt. They lean on `schema=` and
on doing the reduce in plain Python (the iron law still applies: iterate ordered
collections, capture loop variables in `parallel` thunks).

**Adversarial verify (refute-by-default).** Don't ask "is this right?" — ask N
independent skeptics to *refute* it, defaulting to refuted unless they can ground
it, and keep only what survives a majority. Catches plausible-but-wrong claims a
single confirmer would wave through.

```python
async def orchestrate(ctx, args):
    claims = sorted(args["claims"])
    confirmed = []
    for claim in claims:
        votes = await ctx.parallel(
            [
                lambda c=claim, v=v: ctx.agent(
                    f"Skeptic #{v + 1}: try to refute this claim, defaulting to refuted unless you can ground it: {c}",
                    agent_type="skeptic",
                    schema={
                        "type": "object",
                        "properties": {"refuted": {"type": "boolean"}, "reason": {"type": "string"}},
                        "required": ["refuted", "reason"],
                        "additionalProperties": False,
                    },
                )
                for v in range(3)  # a voter index keeps the 3 skeptics distinct (resume-safe)
            ]
        )
        if survives(votes, against=lambda v: v.refuted, kill_at=2):
            confirmed.append(claim)
    return confirmed
```

`survives` bakes in the fail-safe — a `None` (failed skeptic) counts as a
refutation, so a claim is never confirmed on absent verification. It is available
by name in a `run_script` script (no import needed).

**Pipeline review → verify (pipeline by default; `parallel` only for a real
barrier).** Stream each dimension through review-then-verify with no barrier, so a
dimension's findings get adversarially checked the moment its review lands instead
of waiting on the slowest reviewer. Reach for `parallel` only when you genuinely
need every result together.

```python
async def orchestrate(ctx, args):
    dimensions = sorted(args["dimensions"])

    async def review(prev, dimension, i):
        return await ctx.agent(
            f"Review the code along the {dimension} dimension; list concrete findings.",
            agent_type="reviewer",
        )

    async def verify(prev, dimension, i):
        return await ctx.agent(
            f"Which of these {dimension} findings are real? Drop the rest:\n{prev}",
            agent_type="skeptic",
            schema={
                "type": "object",
                "properties": {"confirmed": {"type": "array", "items": {"type": "string"}}},
                "required": ["confirmed"],
                "additionalProperties": False,
            },
        )

    verdicts = await ctx.pipeline(dimensions, review, verify)
    return [c for v in verdicts if v is not None for c in v.confirmed]
```

**Fan out → reduce in Python → synthesize.** The intermediate findings live in
script variables, never in a model's context. Dedup and sort with plain Python
before the single synthesis call.

```python
async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    kept = sorted(dedup(f.strip() for f in findings if f))  # dedup() drops None + de-dupes
    return await ctx.agent("Synthesize these findings:\n" + "\n".join(kept), agent_type="writer")
```

**Loop until dry, with a hard `MAX_ROUNDS` and a budget guard.** Keep hunting until
two dry rounds in a row, but always cap the rounds so a model that keeps
"discovering" can't loop forever. Guard the budget check with `ctx.budget.total`:
when no budget was set, `remaining()` is infinite, so the bare check never fires.

```python
async def orchestrate(ctx, args):
    MAX_ROUNDS = 5
    seen = set()
    found = []
    dry_streak = 0
    for round_index in range(MAX_ROUNDS):
        if ctx.budget.total and ctx.budget.remaining() < 1000:
            ctx.log("budget nearly exhausted; stopping the hunt early")
            break
        batch = await ctx.agent(
            f"Find issues not already in this list: {sorted(seen)}",
            agent_type="hunter",
            schema={
                "type": "object",
                "properties": {"issues": {"type": "array", "items": {"type": "string"}}},
                "required": ["issues"],
                "additionalProperties": False,
            },
        )
        fresh = [i for i in batch.issues if i not in seen]
        if not fresh:
            dry_streak += 1
            if dry_streak >= 2:  # two dry rounds in a row -> converged
                break
            continue
        dry_streak = 0
        for issue in fresh:
            seen.add(issue)
            found.append(issue)
    return sorted(found)
```

**Judge panel / multi-modal sweep.** Judge one artifact through several distinct
lenses in parallel and keep it only on a majority. Diverse lenses catch failure
modes a single reviewer (or N identical reviewers) would miss.

```python
async def orchestrate(ctx, args):
    artifact = args["artifact"]
    lenses = ["correctness", "security", "performance"]
    rulings = await ctx.parallel(
        [
            lambda lens=lens: ctx.agent(
                f"Judge this artifact through the {lens} lens. Is it sound?\n{artifact}",
                agent_type="judge",
                schema={
                    "type": "object",
                    "properties": {"sound": {"type": "boolean"}, "note": {"type": "string"}},
                    "required": ["sound", "note"],
                    "additionalProperties": False,
                },
            )
            for lens in lenses
        ]
    )
    return "accepted" if survives(rulings, against=lambda r: not r.sound, kill_at=2) else "rejected"
```

**Per-stage model routing (cost discipline).** Spend a cheap model on bulk triage
and a strong one only on the survivors. `model` is part of the cache key (so it
partitions resume correctly); `label` / `phase` are not. Note: `model=` swaps the
model only for **config-aware** leaves — a leaf with its model bound at
construction ignores it (the override still partitions the cache key, but won't
change which model runs).

```python
async def orchestrate(ctx, args):
    items = sorted(args["items"])
    triaged = await ctx.parallel(
        [lambda x=x: ctx.agent(f"Quick-triage: {x}", agent_type="worker", model="haiku") for x in items]
    )
    interesting = sorted(x for x, t in zip(items, triaged) if t and "interesting" in t.lower())
    return await ctx.agent(
        "Deeply analyze:\n" + "\n".join(interesting), agent_type="worker", model="sonnet"
    )
```

**No silent caps.** If you bound the work — top-N, sampling, a round cap — say what
you dropped with `ctx.log`, so a truncated run never reads as a complete one.

```python
async def orchestrate(ctx, args):
    candidates = sorted(args["candidates"])
    LIMIT = 10
    if len(candidates) > LIMIT:
        ctx.log(f"capping at {LIMIT} of {len(candidates)}; {len(candidates) - LIMIT} dropped")
    chosen = candidates[:LIMIT]
    results = await ctx.parallel(
        [lambda c=c: ctx.agent(f"Evaluate {c}", agent_type="evaluator") for c in chosen]
    )
    return [r for r in results if r is not None]
```

**Cross-leaf corroboration (`corroborate`).** When several leaves research the same
space, keep only what *more than one* of them independently produced. `corroborate`
groups equivalent items by a key and keeps groups with enough support — a far
stronger signal than any single leaf. Available by name in `run_script`.

```python
async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"State one fact about {t}", agent_type="researcher", schema={
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
            "additionalProperties": False,
        }) for t in topics]
    )
    groups = corroborate(findings, key=lambda f: f.fact.strip().lower(), min_support=2)
    return [g.members[0].fact for g in groups]  # one representative per corroborated group
```

**Dual-blind reconciliation (`reconcile`).** Two (or more) independent reviewers
screen each item; the script keeps only what they unanimously include, drops what
they unanimously exclude, and escalates disagreements (or a failed reviewer) as
conflicts. The fan-out stays explicit; `reconcile` just buckets the verdicts.

```python
async def orchestrate(ctx, args):
    records = sorted(args["records"])
    review = []
    for record in records:
        verdicts = await ctx.parallel(
            [lambda r=record, n=n: ctx.agent(
                f"Reviewer #{n + 1}: should this record be INCLUDED in the review? {r}",
                agent_type="screener",
                schema={
                    "type": "object",
                    "properties": {"keep": {"type": "boolean"}},
                    "required": ["keep"],
                    "additionalProperties": False,
                },
            ) for n in range(2)]
        )
        review.append(ReviewItem(item=record, verdicts=verdicts))
    result = reconcile(review, include=lambda v: v.keep)
    ctx.log(f"included {len(result.included)}, conflicts {len(result.conflicts)} to escalate")
    return result.included
```

A judge in any of these patterns ideally cannot edit — separate the agent that
*generates* from the one that *judges* so a hallucinated fix can't land. Register
the judge as a read-only leaf when your roster supports it.

The reduce helpers — `survives`, `dedup`, `reconcile`, `corroborate` (and the
`ReviewItem` / `Reconciled` / `Consensus` types) — are available by name inside a
`run_script` script (injected into the namespace); you do not import them. They are
pure functions over the result list `ctx.parallel` / `ctx.pipeline` hands back, so
the fan-out stays explicit and the reduce stays correct.

**Race to the first good-enough answer (`ctx.race`).** When several independent
attempts could each solve a task and you only need the first that clears a bar,
race them and cancel the rest the moment one wins — far cheaper than waiting for a
`parallel` barrier when the slow attempts are wasted work. The classic case is
multi-hypothesis diagnosis: investigate every hypothesis at once, confirm the root
cause on the first high-confidence result, drop the others.

```python
async def orchestrate(ctx, args):
    hypotheses = sorted(args["hypotheses"])
    result = await ctx.race(
        [
            RaceCandidate(
                prompt=f"Investigate whether the incident root cause is: {h}",
                agent_type="investigator",
                schema={
                    "type": "object",
                    "properties": {
                        "root_cause": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["root_cause", "confidence"],
                    "additionalProperties": False,
                },
            )
            for h in hypotheses
        ],
        win=lambda d: d.confidence >= 0.8,
        win_tag="high-confidence-root-cause",
    )
    if result.won:
        return result.winner.root_cause  # the other hypotheses were cancelled
    ctx.log("no hypothesis reached high confidence")
    return None
```

`race` journals its decision, so a resume reproduces the same winner and re-runs
nothing. **Footgun — always set a distinct `win_tag` when you reuse the same
candidates with a different `win`.** The journal key folds in `win_tag` but not the
predicate, so two races over identical candidates with the same tag share one cached
decision: the second silently replays the first's winner instead of applying your
new criterion. A race that finds **no** winner is not journaled (a resume may retry
it); if you want every result regardless of a bar, use `parallel`, not `race`.

**Real-git worktree fix swarm with a script-owned conflict loop.** When the work
is *editing a real repository* — a fix swarm, a codemod, a migration — give each
file-mutating leaf its own real `git worktree` (an isolation the host wires once).
Each leaf branches from the base repo, edits on its own tree, and cannot see a
sibling's edits; the changeset folded back is the **real `git diff`** of that tree,
not whatever the model claims it wrote — so a leaf that drifts from its own
self-report cannot corrupt the integration. A worktree leaf's `schema` must carry a
`files: dict[str, str]` field for that authoritative diff to land in; the engine
fills it from the on-disk truth and overrides the model's bytes.

The point of control-flow inversion here is that **the script owns the merge**, not
a model. Fan out the fixers in parallel, keep the approved patches, then fold them
into an integration tree in a deterministic Python loop — one journaled merge leaf
per patch running a real three-way `git merge`. On a clean merge, take its merged
tree; on a real conflict, route to a resolver leaf, then fold its resolution into
the working tree (completing the merge as a hand-resolution would). The loop, the
merge order, and the conflict branch are ordinary code over journaled leaves, so a
resume replays every merge from the journal and re-runs no real git. Materializing
the result — opening the pull request, pushing the branch — is a **side effect the
host does once, after the run returns**, never inside the orchestration (a journaled
replay would otherwise skip or duplicate it).

```python
async def orchestrate(ctx, args):
    # Each fixer edits its OWN git worktree; the engine folds the real git diff
    # into `files` (the schema MUST declare files: dict[str, str]).
    patches = await ctx.parallel(
        [
            lambda t=t: ctx.agent(
                f"Fix {t}", agent_type="fixer", isolation="worktree",
                schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "files": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["summary", "files"],
                    "additionalProperties": False,
                },
            )
            for t in sorted(args["targets"])
        ]
    )
    # The SCRIPT owns the merge: fold each patch into the integration tree with a
    # journaled merge leaf; on a real conflict, a resolver leaf resolves the markers.
    integrated = dict(args["base"])
    for patch in [p for p in patches if p is not None]:
        merged = await ctx.agent(_merge_request(args["base"], integrated, patch.files),
                                 agent_type="merge", schema=MERGE_SCHEMA)
        if merged.clean:
            integrated = merged.files
            continue
        resolution = await ctx.agent(_conflict_request(merged.conflicts),
                                     agent_type="resolver", schema=RESOLUTION_SCHEMA)
        integrated = {**merged.files, **resolution.files}
    return integrated  # the host opens the PR once, AFTER this returns
```

## Authoring a script for `run_script`

When no registered workflow fits, write the `orchestrate` coroutine yourself and
submit the source with `run_script`. The source must:

- Define a top-level `async def orchestrate(ctx, args)` coroutine — `args` is the
  mapping you pass alongside the command.
- Use only the `ctx` primitives above plus plain data/iteration builtins (`len`,
  `range`, `enumerate`, `sorted`, `sum`, `min`, `max`, `any`, `all`, `zip`, `map`,
  `filter`, `list`/`dict`/`set`/`tuple`, `str`/`int`/`float`/`bool`, `abs`,
  `round`, `reversed`). String methods like `.lower()` and `.join()` are fine.

A security gate rejects a script that reaches for an escape hatch. Do **not**:

- `import` anything (you have no module access — and so no `time`/`random`, which
  is also why you must not branch on them: see the determinism rules).
- Touch dunder attributes or names (`__class__`, `__builtins__`, ...).
- Call `eval` / `exec` / `open` / `getattr` / `globals` / ... — they are banned.
- Use `str.format` / `format_map` — use an f-string (`f"{x}"`) instead.

If the gate rejects your script, the response lists each violation with its line;
fix them all and resubmit.

> **Security boundary (A1).** This gate plus a restricted-builtins namespace stops
> an accidental slip — it is **not a security sandbox**, and a determined escape
> can still get through. Only submit scripts **you** author; never relay an
> untrusted third party's script through `run_script`.

## Running it with the `workflow` tool

The script is launched **in the background**, so your turn is not blocked:

1. Launch it:
   - `workflow(command="run", workflow="<registered-name>", args={...})` — launch a
     registered workflow by name; or
   - `workflow(command="run_script", script="<source>", args={...})` — launch an
     ad-hoc script you authored.

   Either returns a `run_id` placeholder immediately and runs in the background.
2. Continue working. When the run finishes, a `<workflow_notification>` is
   injected before your next reply listing the finished `run_id`(s).
3. `workflow(command="status", run_id="<id>")` — fetch the result. A large result
   is summarized and offloaded behind a handle.
4. `workflow(command="resume", run_id="<id>")` — re-run against the journal so
   completed steps replay at zero cost (use after an interruption).
5. `workflow(command="cancel", run_id="<id>")` — stop an in-flight run.

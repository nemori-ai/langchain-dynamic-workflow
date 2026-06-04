# Design: Per-leaf live status + runtime event-stream — a framework-native engine capability

> Status: FINALIZED design spec — **DESIGN ONLY, not implemented** (per user, 2026-06-04).
> Committed as a **v0.4.0 roadmap target** — see [`00-roadmap.md`](00-roadmap.md) (M1). Handoff spec
> for the parallel engine-core session to implement the Layer-1 hooks; decisions recorded in §10.
> The demo-app is ONE consumer of this capability.

## 0. The reframe that dissolves the old tension

The earlier framing ("context quarantine ⇒ a leaf's runtime is unobservable") was **wrong**.
Quarantine means a leaf's RESULT and message-trajectory do not enter the *host LLM's context
window* — `agent()` folds out only the final string and discards the rest (`_context.py:442-452`).
It does **not** mean the leaf's runtime events are invisible to the *engine process*.

LangChain/LangGraph already ship a universal observability substrate the engine can tap
**out-of-band**, on the same call stack, without ever feeding leaf messages into the host context:

- **LangChain callbacks** — every runnable fires `on_chain_start/end`, `on_chat_model_start/end`,
  `on_tool_start/end`, and `on_*_error`, each carrying `run_id` + `parent_run_id`
  (`langchain_core/callbacks/base.py:31-32, 48-49, 70-71, 94-95, 113-114, 132-133, 176-177,
  193-194, 210-211, 227`). That is a live **run tree** with a start/end/error edge per node.
- **`astream_events` v2** — same information as a typed stream: `event`, `run_id`, `parent_ids`
  (root→immediate, v2-only), `name`, `tags`, `metadata`, `data`
  (`langchain_core/runnables/schema.py:56-173`). `parent_ids` makes the tree explicit.
- **deepagents forwards `callbacks`/`tags`/`configurable` to its subagents**
  (`deepagents/middleware/subagents.py:560-562`) — so a handler attached to a leaf's config
  propagates through the *entire* leaf subtree, including nested sub-agents. The substrate study
  already calls this out (`research/2026-06-01-langchain-deepagents-substrate.md:306, 440`): the
  engine must replicate this forwarding when it bypasses the LLM-driven `task` tool — which it
  **already does** for the usage callback (`_engine.py:153, 164-167`).

So the per-leaf "running" edge, the live elapsed timer, and the drill-in event stream are all
**genuinely available**: they are the leaf's own callback/`astream_events` subtree, tapped in the
engine process and surfaced out-of-band — *not* the quarantined host context.

---

## 1. What this capability is

A framework-native hook by which a `run_workflow` caller observes, per leaf `agent()`:

1. **Live status** — `idle → running → {complete | error}`, with the running edge arriving the
   instant the leaf's runnable starts (not when it finishes), and a live elapsed time
   (`now − started_at`).
2. **A runtime event subtree** — the leaf's own observability events (its model calls, tool
   calls, nested sub-agent steps), each correlated to *this* leaf's `Span` identity via the
   run-tree `run_id`/`parent_run_id`, streamed live and drillable.

All of this preserves quarantine: none of these events are injected into the host LLM's messages;
they flow to an out-of-band sink the host renders as UI/telemetry.

---

## 2. Where it lands in the 3-layer architecture

| Layer | Role in this capability |
|---|---|
| **Layer 0 — substrate** | The *source*. LangChain callbacks + `astream_events` already produce the run tree. The engine does not reimplement this; it taps it. |
| **Layer 1 — orchestration runtime** | The *correlation + delivery* home. This is where the capability is authored: a per-leaf tap injected at the leaf-invocation site (`_engine.leaf_task`), correlated to the leaf's `Span` (`_observability`), and delivered through new sinks alongside `on_progress` / `on_span`. This is the right layer — `on_progress`/`on_span` already live here (`_engine.py:60-90`), and the leaf-invocation site (`_engine.py:131-200`) is the single chokepoint every leaf passes through. |
| **Layer 2 — meta** | A *consumer*, not the owner. A meta-authored script's leaves get the same observability for free; no meta-specific work. |

**Decision: the capability belongs to Layer 1**, expressed as (a) an enriched `Span` lifecycle
(begin + end, not end-only) and (b) a new per-leaf event sink. The substrate (Layer 0) is the
event source; nothing new is built there. The demo-app sits entirely outside the engine and
consumes the new sinks.

---

## 3. The observability gap today (precise)

- `SpanRecorder.span()` emits **only on completion** — `self._sink(...)` is in the `finally`
  block (`_observability.py:113-147`). There is **no begin signal** and the `Span` is frozen,
  immutable, end-only (`_observability.py:74-91`).
- `agent()` wraps the *entire* leaf lifecycle (determinism check, journal lookup, the whole
  `await self._gate.run(leaf_runner(...))`) in one `with self._spans.span(...)` block
  (`_context.py:382-459`); the span fires only when that block exits.
- The leaf itself runs as a `@task` that calls `runnable.ainvoke(...)` (`_engine.py:163-171`).
  The engine **already** injects a callback (`UsageMetadataCallbackHandler`) into the leaf config
  and relies on deepagents forwarding it down the subtree (`_engine.py:153, 164-167`). **This is
  the exact injection seam a per-leaf event tap reuses.**

So today, against `{idle, running, complete, error}`: only `complete`/`error` are observable
(end-only span). `running`/`idle` and the event subtree are *not surfaced* — but they are
*available* at the leaf-invocation seam. Closing that gap is the capability.

---

## 4. The status lifecycle + elapsed model

```
   idle ───────▶ running ───────────────▶ complete   (Span end, error == None; cached=True ⇒ "replayed")
   (a leaf the         │                └▶ error      (Span end, error != None)
    caller knows is    │
    pending but        └─ running edge fires at leaf-invocation start (Span begin / on_chain_start
    not yet started)      of the leaf's root runnable). Elapsed = now − begin_ts, ticked by the consumer.
```

- **idle**: a leaf the caller has *declared* but the engine hasn't dispatched. The engine cannot
  enumerate future leaves (the script decides them imperatively), so `idle` is **only**
  meaningful if the consumer pre-declares a leaf set (e.g. the demo knows a fan-out's width from
  the script). Recommend: `idle` is an *optional consumer-side* state, not an engine event. The
  engine emits `running`/`complete`/`error`; a consumer that knows the planned set paints the
  rest as `idle`.
- **running**: emitted by a new **span-begin** edge (§5.1), fired before the leaf runnable's
  `ainvoke`. Carries the leaf's stable identity + a wall-clock `started_at`.
- **complete / error**: the existing end-span. `cached=True` is a complete sub-state
  ("replayed from journal, 0 new tokens", `_context.py:402-407`).
- **elapsed**: `now − started_at`; the engine supplies `started_at` (monotonic + wall clock),
  the consumer ticks. A cached/replayed leaf has begin≈end (near-zero), so it shows as instant.

---

## 5. The engine-side design (Layer 1 changes)

Two additions, both at Layer 1, both opt-in (silent no-op when no sink is wired — mirroring the
existing `on_span` default, `_engine.py:230, _observability.py:97-111`).

### 5.1 Span lifecycle: add a begin edge + stable span id

Enrich `SpanRecorder.span()` to emit a **begin** event when the span opens, in addition to the
end event it already emits (`_observability.py:113-147`). Concretely:

- Assign every span a stable `span_id` at open time (a uuid or a deterministic
  `(kind, name, occurrence)` id — see §5.3 for the resume-stable variant).
- Add an optional `on_span_begin: Callable[[SpanBegin], None]` sink to `SpanRecorder` /
  `run_workflow`. `SpanBegin` carries `span_id`, `kind`, `name`, the early attributes already
  known at open (`agent_type` is set first thing inside the `with`, `_context.py:382-383`), and
  `started_at` (wall clock) + a monotonic start.
- The existing end `Span` gains the same `span_id` so begin/end correlate. Keep `Span` otherwise
  unchanged (immutable, end-only fields: `duration_s`, `error`, final attributes).

This is the clean, generic running edge: it fires for **every** primitive (`AGENT` leaves,
`PARALLEL`/`PIPELINE` barriers), every run, with no orchestration-script instrumentation. It is
the framework-native answer the reframe asked for, and it benefits any tracer adapter, not just
the demo.

Determinism/replay note: like the end span, a begin event is **live-only, not replay-suppressed
and not journaled**. On resume, a replayed (cached) leaf re-emits begin+end with begin≈end and
`cached=True` — correct, because the consumer should render it as an instant replayed hit, never
a stuck "running" chip.

### 5.2 Per-leaf event subtree: an `on_leaf_event` tap injected at the leaf seam

Add an optional `on_leaf_event: Callable[[LeafEvent], None]` sink to `run_workflow`. The engine
wires it into each leaf invocation at the existing config-injection seam (`_engine.py:153-167`)
by attaching a small `BaseCallbackHandler` to the leaf's `callbacks` list — exactly where the
`UsageMetadataCallbackHandler` already goes, and which deepagents forwards down the whole subtree
(`subagents.py:560-562`).

The handler translates the leaf subtree's callback edges into `LeafEvent`s:

```
LeafEvent:
  leaf_span_id: str        # the owning leaf's span id (§5.1) — correlation key to the AGENT span
  run_id: str              # the runnable's run_id (run-tree node)
  parent_run_id: str|None  # for nesting inside the leaf subtree
  kind: str                # "chain"|"chat_model"|"tool"  (from on_<type>_start/end/error)
  phase: str               # "start"|"end"|"error"
  name: str                # runnable name (e.g. tool name, model name)
  ts: float                # wall clock
  detail: dict             # bounded, redaction-aware slice (see §7 quarantine)
```

Correlation: the engine knows the leaf's `span_id` (it owns the span, §5.1) and tags the leaf's
root config with it (`config["metadata"]["ldw_leaf_span_id"] = span_id` and/or a tag). Because
tags/metadata are inherited by child runnables (`schema.py:130-148`), every event in the subtree
carries the owning leaf's id, so the consumer files each `LeafEvent` under the right
`agent_span`. The run tree *within* a leaf is reconstructable from `run_id`/`parent_run_id`.

**Two viable taps — pick by ergonomics, equivalent in information:**

- **(a) Callback handler (recommended).** A `BaseCallbackHandler` appended to the leaf config's
  `callbacks`. Stays on the engine's own call stack (synchronous, inline — same model as the
  existing sinks, `ui_bridge.py:20-28`), needs no change to how the leaf is invoked
  (`runnable.ainvoke`, `_engine.py:163-171`), and reuses the proven forwarding path. The handler
  filters to start/end/error of chain/chat_model/tool and forwards `LeafEvent`s.
- **(b) `astream_events(version="v2")` consumption.** Switch the leaf invocation from `ainvoke`
  to draining `astream_events` and re-fold the final state. Richer typed payloads and explicit
  `parent_ids`, but it changes the leaf-invocation contract (folding, structured_response
  extraction, usage metering all assume the `ainvoke` state dict, `_context.py:442-452`,
  `_engine.py:168-171`) and risks the native-cache "hit drops stream data" issue the substrate
  study flags (`research/...substrate.md:322`). **Prefer (a); keep (b) as the option for a future
  richer payload.**

### 5.3 Stable span ids that survive resume (so the consumer can key/dedupe)

The demo's adapter already proves the resume-stable identity scheme: hash the leaf's *stable*
shape (`component + kind + name + agent_type`) and salt with a per-key occurrence ordinal so the
Nth same-type leaf gets ordinal N, and an honest resume (same source order, enforced by the
determinism guard) reproduces the exact id sequence (`ui_adapter.py:84-87, 247-250, 265-285`;
tested at `test_ui_adapter.py:185-240`). The engine should mint the `span_id` with the **same
deterministic, resume-stable scheme** (kind+name+occurrence-ordinal), so begin/end/leaf-events
all share one id that is identical fresh vs. resumed. This moves the proven correlation logic
from the demo into the engine, where every consumer benefits.

### 5.4 Public surface (additive, keyword-only, defaulted)

```python
async def run_workflow(
    orchestrate, *, roster, ...,
    on_progress=None, on_span=None,
    on_span_begin=None,      # NEW: per-span running edge (begin)
    on_leaf_event=None,      # NEW: per-leaf runtime event subtree
    ...,
): ...
```

All default `None` ⇒ silent no-op ⇒ zero cost when unused (the established pattern,
`_engine.py:227-230`, `_observability.py:97-111`). No existing signature changes (stable-interface
rule, AGENTS.md). `Span` gains a `span_id` field — additive — and a new `SpanBegin` + `LeafEvent`
type are exported alongside `Span`/`SpanKind` (the public-import surface listed in
`docs/plans/2026-06-04-phase3-engine-api-notes.md:7`).

---

## 6. Approaches (with recommendation)

### Approach A — span-begin hook only (running + timer, no drill-in stream)

Implement §5.1 only: add `on_span_begin` + `span_id`. Delivers live `running`/`idle` and the
elapsed timer for every leaf with a minimal, fully-generic engine change. No per-leaf event
subtree (drill-in shows only lifecycle + result metadata).

- ✅ Smallest engine change; clean; generic; the running edge benefits any tracer.
- ❌ The drill-in is metadata-only (begin/end/cached/usage/error + correlated phase lines), not a
  true event stream. Satisfies requirement (1) fully, requirement (2) only weakly.

### Approach B — span-begin + `on_leaf_event` callback tap (RECOMMENDED)

§5.1 + §5.2(a) + §5.3. The full capability: live status + timer **and** a real per-leaf runtime
event subtree from the leaf's own callbacks, correlated by `span_id`, surfaced out-of-band.

- ✅ Delivers both requirements as a first-class, framework-native feature on the proven
  callback-forwarding path the engine already uses for usage metering.
- ✅ Preserves quarantine by construction (events go to a sink, never into host messages; §7).
- ✅ Reuses the engine's existing injection seam — low structural risk; the leaf still runs via
  `ainvoke`, folding/usage/structured-output paths untouched.
- ⚠️ Cost: new sink + handler + the begin edge + stable-id move into the engine; a redaction
  policy for `detail` (§7); event volume can be high under deep sub-agent fan-out → needs a
  bounded/sampled `detail` and the consumer should be able to subscribe lazily (only stream
  subtree events for a leaf the user drilled into — see §8).
- ⚠️ Determinism: begin and leaf-events are live-only (documented like spans), so a resumed run
  re-emits them with cached semantics; the consumer must treat them as non-authoritative replay.

### Approach C — `astream_events(v2)` leaf consumption (richest, highest blast radius)

§5.1 + §5.2(b) + §5.3. Replace `ainvoke` with `astream_events` draining inside `leaf_task`.

- ✅ Richest typed payloads, explicit `parent_ids` tree, single mechanism for status + subtree.
- ❌ Changes the leaf-invocation contract (re-fold from the event stream, re-derive
  `structured_response` + usage), touches the most load-bearing path (`_engine.py:163-200`,
  `_context.py:442-452`), and risks the cache-drops-stream-data substrate caveat
  (`research/...substrate.md:322`). Disproportionate vs. (a) for the same correlation result.

**Recommendation: Approach B.** It is the framework-native capability the reframe asks for, built
on the substrate already in place (callbacks + deepagents forwarding), at the engine layer that
already owns observability sinks, with the least disruption to the load-bearing leaf path. Keep A
as a phase-1 milestone (ship the running edge first; it is independently valuable and low-risk),
and C as the documented richer-payload alternative if a future need outgrows callback `detail`.

---

## 7. Quarantine preservation (the honest line, restated correctly)

Quarantine is about the **host LLM's context window**, enforced by `agent()` folding out only the
final result (`_context.py:442-452`). The new sinks are orthogonal to that boundary:

- `on_span_begin` / `on_leaf_event` deliver to an **out-of-band sink** (UI/telemetry), never into
  the host's `messages`. The host LLM's context is byte-for-byte identical whether or not a
  consumer subscribes. So quarantine is **fully preserved**.
- The drill-in stream is therefore genuinely real: it is the leaf's own callback subtree, not the
  quarantined message context.
- **What `detail` should carry (redaction policy):** the *shape* of the run tree (node kind,
  name, timing, nesting) is always safe. Payloads (tool args/outputs, model
  prompts/completions) are leaf-internal and may be sensitive; `detail` must default to
  **bounded, opt-in** content (e.g. tool *names* and durations always; truncated args/outputs
  only when the consumer explicitly opts in via a flag on the sink). This keeps the feature
  honest (you really can see the leaf's steps) without unconditionally streaming raw leaf
  internals.

Honest limitation that remains: a leaf whose backend runs **outside** the LangChain runnable
graph (e.g. a sandbox shell `execute`, `SandboxBackendProtocol.aexecute`,
`research/...substrate.md:404-405`) emits no LangChain callback for that out-of-process work — so
the event subtree shows the *tool call* boundary but not the sandbox's internal process events.
That is a true substrate boundary, not a design miss.

---

## 8. How a consumer (the demo-app, and others) consumes it

The demo-app stays entirely outside the engine; it wires the new sinks the way it already wires
`on_progress`/`on_span` (`host_graph.py:216-221, 273-281`).

- **Backend (`ui_adapter.py` + `ui_bridge.py`):**
  - `on_span_begin` → emit an `agent_span` with `status="running"` + `started_at`, keyed by the
    engine-minted `span_id` (now the engine owns the resume-stable id the adapter used to compute,
    §5.3). The end `Span` re-emits the **same** `span_id` flipping `status` to `complete`/`error`
    + filling `duration_s`/`usage_tokens`/`cached`. The SDK's `uiMessageReducer` **replaces by
    `id`** (verified: `node_modules/@langchain/langgraph-sdk/dist/react-ui/types.js:12-31`) and
    `push_ui_message(id=, merge=)` exists (`langgraph/graph/ui.py:61-126`), and `ui_bridge`
    already threads `event_id → id` (`ui_bridge.py:74-83`) — so the transition updates the chip
    **in place**, live. (`ui_bridge` must learn to forward `merge=True` so the end edge patches
    without clobbering `started_at`; the adapter must exempt the begin/end pair from its `_seen`
    suppression for these deliberate transitions, `ui_adapter.py:251-263`.)
  - `on_leaf_event` → buffer per `leaf_span_id`; emit a `leaf_event_stream` component (or append
    into the existing `agent_span`'s drill-in payload) only for the leaf the user drilled into,
    to bound volume. Each entry: `{run_id, parent_run_id, kind, phase, name, ts, detail}` — the
    run-tree subtree.
- **Frontend:**
  - `AgentSpan.tsx` gains `status` + `started_at`; renders a pulsing chip + a `setInterval`
    elapsed timer while `running` (ordinary React hooks in a Gen-UI component mounted by
    `LoadExternalComponent`, `messages/ai.tsx:37-43`, `registry.ts:22-35`; live custom events
    reduce mid-stream via `onCustomEvent`, `Stream.tsx:97-104`). Drive the timer off prop
    `started_at` (robust to remount).
  - **Drill-in = an expandable inline disclosure inside `AgentSpan`** (a `useState(open)` panel),
    not a modal or route — least invasive, fits agent-chat-ui's inline-under-message model. When
    opened, it renders the leaf's `LeafEvent` subtree as an indented run tree (nest by
    `parent_run_id`), plus the lifecycle (begin→running→end), the journal verdict, and an
    honesty footer: "These are this leaf's own runtime events, observed out-of-band; its result
    is folded away from the host context (quarantine) — only the leaf's run tree and lifecycle
    are surfaced here." Optionally subscribe to subtree events lazily on open.
- **Any other consumer** (a CLI tracer, a LangSmith-style adapter) wires the same three sinks; the
  `span_id`/`run_id`/`parent_run_id` correlation is engine-provided, so no consumer reinvents it.

---

## 9. Honest limitations

- **`idle` is consumer-derived, not an engine event.** The engine cannot enumerate not-yet-started
  leaves (the script chooses them imperatively). A consumer that knows the planned set (the demo
  knows a fan-out's width from its own script) can paint `idle`; the engine only emits
  `running`/`complete`/`error`.
- **Begin/leaf-events are live-only on resume.** A replayed (cached) leaf re-emits begin≈end with
  `cached=True`; the consumer must render it as an instant replayed hit, never a stuck `running`.
- **Out-of-process work is opaque.** Sandbox `execute`/`aexecute` (`research/...substrate.md:404`)
  runs outside the LangChain graph; the subtree shows the tool-call boundary, not the sandbox's
  internal steps. A true substrate boundary.
- **No "X% done."** Status is `running` until the leaf's end edge; sub-step granularity comes only
  from whatever child runnables the leaf actually fires (model/tool start-end), which is real but
  not a percentage.
- **Event volume under deep fan-out.** A leaf with many sub-agents/tools can emit many
  `LeafEvent`s; `detail` must be bounded/redaction-aware (§7) and subtree streaming should be
  lazy (only for a drilled-in leaf) to keep the wire and UI sane.
- **Inline-sink discipline carries over.** The new sinks are called synchronously on the engine's
  stack (like `on_span`/`on_progress`); a consumer must swallow its own exceptions (the demo's
  transport already does, `ui_bridge.py:84-87`, `ui_adapter.py:254-263`) so a UI failure can never
  unwind orchestration.

---

## 10. Resolved decisions (2026-06-04) — design only, not yet implemented

The user reviewed this design and chose **DESIGN ONLY**: finalize the spec, do not implement now.
The decisions below guide whoever implements it later.

1. **Landing: engine-spec handoff.** This doc is the spec for the parallel engine-core session to
   implement the Layer-1 hooks (`on_span_begin` / `on_leaf_event` / `span_id`). No `src/` edits are
   made from the demo worktree (avoids divergence with the engine session). The demo-app consumer
   (status chip + timer + drill-in, `ui_adapter`/`ui_bridge`) is built only once that engine
   surface exists.
2. **Sequencing: phased A → B.** Ship Approach A first (span-begin running edge + stable `span_id`
   + live elapsed timer — small, generic, independently valuable, low risk), then Approach B's
   `on_leaf_event` per-leaf event subtree for the drill-in.
3. **Tap mechanism: callback handler (§5.2a).** A `BaseCallbackHandler` at the existing leaf config
   seam — the leaf `ainvoke` path stays untouched and it reuses the proven usage-callback
   forwarding. `astream_events` (§5.2b) is the documented richer-payload alternative, not chosen.
4. **`span_id`: resume-stable scheme (§5.3).** Mint the id as the demo's proven
   `(kind+name+occurrence-ordinal)` hash so begin/end/leaf-events share one id that is identical
   fresh vs. resumed; the correlation logic moves from the demo into the engine.
5. **`detail`: shape-only by default (§7).** Node kinds/names/timing always; raw tool args / model
   text only behind an explicit opt-in flag on the sink.
6. **`idle`: consumer-derived.** The engine emits only `running`/`complete`/`error`; a consumer
   that knows its planned leaf set paints the rest `idle`. No engine-level planned-leaf-set in scope.
7. **Demo consumer (when built): `ui_bridge` gains `merge` forwarding** plus the two new sinks
   wired into `run_workflow_live` — deferred until the engine hooks land.

**Implementation status: NOT STARTED — design only.** Sequenced after the engine-core session
lands the Layer-1 hooks; the demo-app consumer work follows then.

---

## 11. Key references

Engine / Layer 1:
- End-only span emit (the gap), immutable `Span`: `src/langchain_dynamic_workflow/_observability.py:74-91, 113-147`
- `agent()` wraps the whole leaf lifecycle in one span: `src/langchain_dynamic_workflow/_context.py:382-459`
- Quarantine fold (result only into host): `src/langchain_dynamic_workflow/_context.py:442-452`
- Leaf invocation seam + existing callback injection (the tap point): `src/langchain_dynamic_workflow/_engine.py:131-200`, esp. `:153, 164-171`
- Existing sink pattern (`on_progress`/`on_span`, no-op default): `src/langchain_dynamic_workflow/_engine.py:60-90, 225-231`
- Cached/replayed leaf path: `src/langchain_dynamic_workflow/_context.py:402-407`

Substrate (Layer 0 source):
- `astream_events` v2 schema (event/run_id/parent_ids/name/tags/metadata/data): `.venv/.../langchain_core/runnables/schema.py:56-173`
- `BaseCallbackHandler` hooks with run_id/parent_run_id: `.venv/.../langchain_core/callbacks/base.py:31-32, 48-49, 70-71, 94-95, 113-114, 132-133, 176-194, 210-227`
- deepagents forwards callbacks/tags/configurable to subagents: `.venv/.../deepagents/middleware/subagents.py:542-562`
- Research: callback forwarding + usage substrate: `research/2026-06-01-langchain-deepagents-substrate.md:306, 329, 440`
- Research: streaming/cache caveats, sandbox `execute` boundary: `research/2026-06-01-langchain-deepagents-substrate.md:322, 404-405`
- Engine public-import surface: `docs/plans/2026-06-04-phase3-engine-api-notes.md:7`

Demo-app consumer:
- Adapter mapping + resume-stable id/ordinal (the scheme to lift into the engine): `demo-app/backend/ui_adapter.py:84-87, 154-203, 247-285`; tests `demo-app/backend/tests/test_ui_adapter.py:185-314`
- Transport (event_id→id, merge to add, swallow): `demo-app/backend/ui_bridge.py:46-91`
- Host wiring of sinks: `demo-app/backend/host_graph.py:216-221, 273-281`
- SDK reducer replace-by-id: `demo-app/frontend/node_modules/@langchain/langgraph-sdk/dist/react-ui/types.js:12-31`
- `push_ui_message(id=, merge=)`: `demo-app/backend/.venv/.../langgraph/graph/ui.py:61-126`
- Gen-UI mount + live custom-event reduce: `demo-app/frontend/src/components/thread/messages/ai.tsx:20-46, 185-190`; `demo-app/frontend/src/providers/Stream.tsx:97-104`
- `AgentSpan` (extend with status/timer + drill-in): `demo-app/frontend/src/components/workflow/AgentSpan.tsx:14-70`; registry `demo-app/frontend/src/components/workflow/registry.ts:22-35`

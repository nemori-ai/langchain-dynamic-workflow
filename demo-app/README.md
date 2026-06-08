# Dynamic Workflow — interactive demo

A hands-on UI for the `langchain-dynamic-workflow` engine. You chat with a host
agent; when a task is hard enough to deserve real orchestration, the host runs a
**dynamic workflow** and the chat shows it happening live — phases, a parallel
fan-out of sub-agents, journaled cache hits on resume, and the meta layer
authoring (and gating) a script on the spot.

It is the engine's headline ideas, rendered instead of described:

- **Control-flow inversion** — a deterministic script owns the loops, branching,
  and fan-out; the leaf `agent()` calls delegate to isolated sub-agents whose
  context is discarded, so only the final result comes back. You watch the
  `phase_timeline` advance and the `fanout_graph` widen as the *script* drives.
- **Leaves go live** — each leaf opens as a **running chip with a ticking elapsed
  timer**, then the same card flips in place to its completed state (final
  duration, tokens) the moment the leaf returns — no second card appended. A
  freshly-run leaf also offers a collapsible **drill-in**: expand it to see the
  shape of its interior (the chain / model / tool steps it ran), kinds and names
  only — never the raw tool args or model text, since the leaf's context is
  quarantined.
- **Parallel fan-out** — one researcher per angle, then adversarial skeptics per
  claim, all in flight at once, reduced by plain Python over typed results.
- **Persistence / resume** — ask the host to "pick it back up" and the second run
  replays the first run's leaves from the content-hash journal as zero-cost cache
  hits (each surfaces a `journal_badge`) instead of redoing the work. A replayed
  leaf runs no interior, so it comes back as a **cached chip** — no running timer
  and no drill-in, since there is no fresh work to show.
- **Meta layer** — when no preset fits, the host *writes* an orchestration script,
  submits the source across the AST security gate, and only runs it if admitted.
  Pass or fail, the chat shows the script and the gate verdict.

The whole thing runs locally with two commands, and runs with **zero
credentials** out of the box: with no model key it falls back to a deterministic,
scripted offline host so you can tour every scenario before deciding to plug in a
key.

---

## Prerequisites

| Tool | Why | Check |
|------|-----|-------|
| [**uv**](https://docs.astral.sh/uv/) | Python deps + runs the backend graph | `uv --version` |
| [**Node.js**](https://nodejs.org/) 18+ | Runs the Next.js frontend | `node --version` |
| [**pnpm**](https://pnpm.io/) 10+ | Frontend package manager (pinned `pnpm@10.5.1`) | `pnpm --version` |

No API key is required to try the demo. (See [Bring your own
key](#bring-your-own-key-byo-key) to drive it with a real model.)

---

## Clone and run (two commands)

The demo is two processes: a LangGraph backend serving the host agent graph, and
the chat frontend. Run each in its own terminal, both from this `demo-app/`
directory.

**Terminal 1 — backend** (LangGraph dev server on **http://localhost:2024**):

```bash
cd backend
uv sync          # first run only: installs deps + creates .venv
uv run langgraph dev
```

`langgraph dev` loads the host graph registered under the id **`host`** in
`backend/langgraph.json` and serves it on port **2024**.

**Terminal 2 — frontend** (Next.js dev server on **http://localhost:3000**):

```bash
cd frontend
pnpm install     # first run only
pnpm dev
```

Open **http://localhost:3000**. The frontend ships a committed `frontend/.env` (it
carries only public `NEXT_PUBLIC_*` values — the local backend URL and assistant id
`host`, no secrets), so a fresh clone connects to the backend at
`http://localhost:2024` without a setup form. (The backend needs no `.env` to boot
offline — see [Bring your own key](#bring-your-own-key-byo-key) if you want to supply
an OpenRouter key for a real-model run.)

> The frontend's `NEXT_PUBLIC_ASSISTANT_ID` and the backend graph id must agree —
> both are `host` here. If you rename the backend graph in `langgraph.json`,
> update the frontend env to match.

---

## Bring your own key (BYO-key)

The provider is **locked to [OpenRouter](https://openrouter.ai/)** and the models
are fixed in code — you never pick a model, you only supply **one OpenRouter key**.
There are three ways to run, in priority order:

1. **Paste a key in the UI** (recommended for trying it on your own key). Open the
   provider-key panel in the chat and paste your OpenRouter key (`sk-or-...`). The
   frontend sends it to the backend **per session** on the run config as
   `config.configurable.openrouter_api_key`. The backend reads it from the runtime
   config for that run and builds the OpenRouter model with it. The key lives only
   in your browser and is sent with your runs — it is not written to the
   backend's disk.
2. **Operator / local mode** — set `OPENROUTER_API_KEY` in `backend/.env` (copy
   `backend/.env.example`). Every run uses that key unless a session passes its own
   in the config (the UI key wins).
3. **Offline (no key)** — set nothing. The host falls back to the deterministic
   **offline scripted host** and the workflow leaves swap in deterministic fakes,
   so `langgraph dev` boots and every scenario is fully tourable without
   credentials.

### Fixed models

The two models live as named, swappable constants in `backend/_models.py` so
changing them is a one-line edit:

| Constant | Role | Why this pick |
|----------|------|---------------|
| `HOST_MODEL` | the host agent that drives the multi-step tool calls | the most capable model (`anthropic/claude-opus-4.8`). Our earlier real-model findings showed weak/cheap models (e.g. `gpt-4o-mini`) *cannot* reliably drive the multi-step tool-calling the host needs — they stall or skip the workflow tool. The host must be strong. Matches the engine examples' default. |
| `LEAF_MODEL` | the research fan-out sub-agents | a strong, cheaper model (`anthropic/claude-sonnet-4.6`). Leaves do bounded research / verify work in parallel; sonnet-class is needed to drive the native web-search tool reliably (haiku-class routes the search poorly) while staying below the opus host. |

Both are valid OpenRouter model ids. To swap either, edit the constant — nothing
else changes.

> **OpenRouter-only, Anthropic-locked.** The default, UI path is OpenRouter + one
> key. Every real model is built as a `ChatOpenRouter` (the same client the engine
> examples use) with provider routing **pinned to Anthropic** (no fallback to
> Bedrock / Vertex) — required because the native web search and Anthropic prompt
> caching only work on the Anthropic provider.

### Web search + prompt caching

Real runs are **grounded in live web sources**: the research and verify leaves
carry OpenRouter's native `openrouter:web_search` tool (`engine="native"`, so the
search is Anthropic's own, reached through the provider lock), executed server-side
with citations returned inline. **Anthropic prompt caching** is registered on every
agent (host and all leaves) via a `PromptCachingMiddleware` ported from `omne-next`,
so the growing system prompt and tool-call history are cached across turns. Both are
online-path only; the offline scripted path stays deterministic and credential-free.

---

## The seven scenarios

The chat ships seven preset buttons. Their canonical wording lives in
`scenarios.json`; the frontend's `ScenarioPanel` carries the same seven messages, and
a backend doc-sync test pins the two copies byte-for-byte (and pins the README's
scenario count to the same source) so they cannot drift. Each is phrased as a real
user's request, not a tool instruction — click one, or type your own. Here is what each
one is built to *show*:

### 1. Deep research — a hard, multi-source question
> *"I need a thorough, fact-checked answer on the main trade-offs between
> retrieval-augmented generation and long-context LLMs..."*

The headline workflow. The host decomposes the question into angles and runs a
**parallel research fan-out** (one sub-agent per angle), extracts a falsifiable
claim from each finding through a **no-barrier pipeline**, sends each claim to a
**parallel panel of adversarial skeptics** (a claim dies on a majority refutation),
then synthesizes only the survivors. You watch the `phase_timeline` walk
search → extract → verify → synthesize and the `fanout_graph` widen and narrow as
the script — not the model — drives the control flow.

### 2. Pick it back up — resume a long-running task
> *"Earlier you started looking into that research question for me. Can you pick it
> back up where you left off rather than starting over?"*

Run scenario 1 first, then this on the **same chat thread**. The second run reuses
the first run's content-hash journal: every leaf it already did comes back
**cached**, surfacing a `journal_badge` instead of re-running. This is the engine's
persistence/resume story made visible — the work is replayed from the journal at
zero cost, not redone.

### 3. Novel task — no ready-made procedure
> *"There's no standard playbook for this one, so I'd like you to work out a
> procedure yourself: research a few topics, refine them, have skeptics challenge
> each finding, then synthesize what survives."*

The **meta layer**. With no preset that fits, the host authors an orchestration
script on the spot and submits the *source* across the **AST security gate** before
anything runs. A `meta_script` panel shows the authored script and the gate
verdict; on a pass the admitted script then runs live, exactly like a preset. (Ask
to *see a script get rejected* and the host submits an unsafe, import-bearing
script so you can watch the gate reject it with a line-numbered reason and run
nothing.)

### 4. Delegate heavy work — hand off a multi-step job
> *"This is a heavy, multi-step research job — I don't want to babysit every step.
> Please take it off my hands and run the whole thing in the background; just let
> me know how it went once it's done."*

Background delegation. The host launches the workflow **detached** so the chat turn
returns immediately, then reports the run's lifecycle status and, once it settles,
the final result. This is the "fire it off and tell me how it went" shape — see the
limitation below on what a detached run can and can't show.

### 5. Make it pass — fix code until the tests are green
> *"I've got a small TypeScript module with a couple of failing unit tests. Please
> actually fix the code and keep checking it against the tests until they genuinely
> pass..."*

**In-loop executable verification.** This is the dynamic-workflow thesis at its most
literal: the host launches a **script-owned retry loop** that does not trust the
model's word for "it's fixed" — it *runs the tests* and branches on the **real exit
code**. Each attempt hands a `code_fixer` leaf the current source, the leaf writes the
files into a per-leaf sandbox, runs `bun test` for a TRUE exit code, edits the source,
and returns its work; the script reads the exit code, and on red threads the leaf's
edited files into the next attempt's prompt. The cross-attempt state — the current
source files — lives in a **script variable**, not a persistent workspace, so you watch
the loop accumulate the fix without the model holding it in context. Every real command
streams an `execution_command` event the UI renders as a **TerminalCard** whose header
chip flips from `running` to `exit 0` (green) or `exit N` (red) in place. The loop stops
on the first green exit or returns an honest "still red" when the bounded retry budget is
spent — never a false pass. (Offline-fake caveat: with no key the `code_fixer` is a
deterministic fake that runs no subprocess, so the offline run proves the script-owned
loop and the real exit-code gating logic but emits no live TerminalCard; add a key to see
a real model shell out and the cards stream.)

### 6. Sign off mid-run — pause for human approval
> *"Before you actually run the staging deployment, I want to sign off on the plan
> myself — walk me through the riskiest steps and pause for my approval before you
> proceed."*

**In-run human-in-the-loop sign-off.** The host assesses the plan, then the
**script pauses** at a `ctx.checkpoint` gate and waits for a person — superset over
Claude Code, which takes no input mid-run. The run parks (status `awaiting_signoff`)
and surfaces a **SignoffGate** card with the question and Approve / Reject buttons.
Clicking one (or just typing your answer) feeds the decision back into the *same*
paused run: the pre-gate assessment **replays from the journal at zero cost**, the
script branches on the **real human decision** (not a model guess), and the card flips
in place to `approved` / `rejected` while the run proceeds or holds. The pause and
resume ride the engine's content-hash journal — the same persistence substrate as
"pick it back up" — so the gate is deterministic and resumable, not a fragile
mid-flight interrupt. (Cross-gate state rides the script's own variables.)

### 7. Refactor swarm — parallel fixes merged into a PR
> *"There are a few separate bugs in this little module I'd like fixed all at once.
> Please have several helpers each take a fix in parallel, review the patches, merge
> them together into one change — sorting out any conflicts — and open a pull
> request with the result."*

**A real-git fix swarm with a merge-conflict loop.** The host fans out fixers across a
buggy module in a **parallel barrier**, and each fixer works in its OWN real `git
worktree` on its own `leaf/<id>` branch — fully isolated from its siblings. The engine
folds each leaf's real `git diff` in as the **authoritative changeset** (a model's
self-report can never override the on-disk truth). A two-vote read-only judge reviews
every patch, then a **script-owned integrate loop** folds the approved patches into one
tree with a real three-way `git merge`. Because two fixers rewrite the SAME line two
different ways, the fold hits a **genuine merge conflict**: a resolver leaf flattens
git's `<<<<<<<` markers, and the resolution folds straight in — you watch the real
`git merge` commands stream as TerminalCards and the phases walk fix → verify →
integrate. Opening the PR is a side effect that must NOT live inside the deterministic
replay (a journaled leaf short-circuits on resume), so the workflow returns only the
pure integrated tree plus a **PR intent**, and the **host** opens the pull request once,
*after* the run returns (host finalization), surfacing a **PullRequestCard** with the PR
ref. (Offline-fake caveat: with no key the fixers are deterministic fakes that still
edit a real worktree on disk — so the isolation, the authoritative git-diff collect, and
the real scratch-repo merge conflict are all genuinely exercised — but no real model
authors the fixes; add a key to see a real model drive each fix.)

---

## Honest limitations

This is a demo of real engine behavior, not a polished product. The seams are
intentional and worth knowing:

- **Offline mode is scripted and deterministic.** With no key, the host's
  turn-by-turn decisions and the leaf outputs are hardcoded, not generated. It
  exercises the full control-flow inversion (real phases, real parallel fan-out,
  real journal, real AST gate) with reproducible, canned *content*. It proves the
  plumbing, not model quality. Add a key to see a real model drive it.
- **Resume is in-process journal re-run, not cross-restart.** "Pick it back up"
  works because the journal lives in memory and survives across turns *within one
  running backend process*. Restart `langgraph dev` and the journals are gone —
  there is no on-disk durable store wired up here. It is a true journal replay, but
  not a crash-recovery story.
- **Background surfaces status + result, not live progress.** A detached background
  run executes in a task that does **not** carry the host's UI context, so it
  cannot push live `phase_timeline` / `fanout_graph` updates into the chat. The
  background scenario deliberately shows lifecycle status and the final result
  only. Live streaming is the inline path's job (scenarios 1–3).
- **The drill-in is shape-only, and out-of-process sandbox work stays opaque.** A
  fresh leaf's drill-in shows the *shape* of its interior — the chain / model / tool
  callback steps it ran, kinds and names, never their raw args or text. Anything a
  leaf sub-agent does inside a sandbox backend (file edits, shell, etc.) is not
  surfaced step-by-step beyond those callback edges — you see the leaf's span, the
  shape of its run, and its returned result, not its full internal trajectory. That
  is the context quarantine working as designed.

---

## Deploy notes

The demo is **local-first** — the two-command flow above is the supported path.

If you want to put it somewhere others can reach it:

- **Frontend** deploys as a standard Next.js app (Vercel-style host, or any Node
  host). It is a vendored copy of LangChain's `agent-chat-ui` and includes a
  same-origin API proxy (`src/app/api/[..._path]/route.ts`) for the
  "website + `/api`" deployment shape.
- **Backend** is self-hosted: serve the LangGraph graph (e.g. via LangGraph's
  hosting, a container, or your own host) and point the frontend's
  `NEXT_PUBLIC_API_URL` / assistant id at it.
- **Put it behind HTTPS if users will paste keys.** When the UI collects an
  OpenRouter key per session, that key travels to the backend on the run config —
  only serve it over HTTPS so the key is never in cleartext on the wire. The key
  is not persisted server-side, but it is sent with each run.

---

## Frontend provenance & re-syncing

`frontend/` is a **vendored whole-tree copy** of LangChain's official
[`agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui) (MIT) — not a
submodule or fork. The pinned upstream SHA, the patch policy (all first-party
Gen-UI components live under `src/components/workflow/`), the exact vendored
injection points, and the step-by-step re-sync procedure are documented in
[`frontend/UPSTREAM.md`](frontend/UPSTREAM.md). Read it before touching anything in
`frontend/` outside `src/components/workflow/`, and before pulling a newer upstream.

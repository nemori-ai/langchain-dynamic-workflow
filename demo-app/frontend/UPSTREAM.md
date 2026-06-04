# Vendored upstream: agent-chat-ui

This directory is a **whole-tree copy** (vendored, NOT a git submodule or fork)
of LangChain's official `agent-chat-ui`. The upstream `.git` directory was
removed on copy; the tree is committed directly into this repository.

## Provenance

| Field | Value |
|-------|-------|
| Upstream URL | https://github.com/langchain-ai/agent-chat-ui |
| Commit SHA | `91da3c559daaa0049bf44c1d2100982c0b587ad6` |
| Copied date | 2026-06-04 |
| License | MIT — Copyright (c) 2025 Brace Sproul (see `./LICENSE`) |
| Framework | Next.js 15 (App Router), React 19, TypeScript |
| Package manager | **pnpm** (`packageManager: pnpm@10.5.1`) |
| Lockfile | `pnpm-lock.yaml` |

## Patch policy

- **All local edits live under `src/components/workflow/`.** This keeps our
  Gen-UI workflow components physically separated from the vendored upstream
  tree so future upstream re-syncs (re-copy at a newer SHA) stay mechanical.
- **Touching vendored files is allowed only at the documented injection points
  below** (the Gen-UI local component-map wiring, and the assistant-id default in
  `Stream.tsx`). Keep those edits minimal and greppable; everything else of ours
  goes in `src/components/workflow/`.
- To update upstream: re-clone at a newer SHA, re-copy the tree, re-apply the
  injection-point edits, bump the SHA + date in this file.

## Gen-UI integration facts (locked down for the backend implementer)

These are the exact files/lines the backend wiring depends on. Verified against
the vendored SHA above.

### 1. Backend connection config (deployment URL / assistant id / `useStream`)

**File: `src/providers/Stream.tsx`**

- The typed `useStream` hook is created at **lines 31–41** (`useTypedStream`),
  wrapping `useStream` from `@langchain/langgraph-sdk/react`. State shape is
  `{ messages: Message[]; ui?: UIMessage[] }` — note `ui` carries the Gen-UI
  messages.
- The hook is **called** inside `StreamSession` at **lines 86–111**, taking
  `apiUrl`, `apiKey`, `assistantId`, optional `authScheme` (→ `X-Auth-Scheme`
  header), `threadId`, `fetchStateHistory: true`, plus `onCustomEvent` (lines
  97–104) which runs `uiMessageReducer` to fold UI messages into `prev.ui`.
- Connection values resolve in `StreamProvider` (**lines 143–315**) from, in
  priority order: URL query params (`apiUrl`, `assistantId`, `authScheme`) →
  env vars. Env vars (lines 147–150):
  - `NEXT_PUBLIC_API_URL` (deployment / LangGraph server URL)
  - `NEXT_PUBLIC_ASSISTANT_ID` (graph or assistant id)
  - `NEXT_PUBLIC_AUTH_SCHEME` (optional)
  - API key comes from `localStorage` key `lg:chat:apiKey` via `getApiKey()`.
- Built-in defaults (lines 139–141): `DEFAULT_API_URL = "http://localhost:2024"`,
  `DEFAULT_ASSISTANT_ID` (upstream ships `"agent"`; the demo overrides it to
  `"host"` — see injection point 4 below).
- If `finalApiUrl` or `finalAssistantId` is missing, a setup **form** renders
  (lines 185–303) instead of the chat — to skip it, set the env vars in
  `.env` (see `.env.example`).

`.env.example` keys (copy to `.env`): `NEXT_PUBLIC_API_URL`,
`NEXT_PUBLIC_ASSISTANT_ID`, `NEXT_PUBLIC_AUTH_SCHEME`, `LANGSMITH_API_KEY`
(server-side, NOT exposed to client).

The same-origin API proxy lives at `src/app/api/[..._path]/route.ts` (uses
`langgraph-nextjs-api-passthrough`) — used for the production "website + /api"
deployment mode.

### 2. Gen-UI rendering entry (`LoadExternalComponent`)

**File: `src/components/thread/messages/ai.tsx`**

- Import at **line 8**:
  `import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";`
- The Gen-UI render happens in the local `CustomComponent` function
  (**lines 19–45**):
  - **lines 27–30**: pulls `values.ui` off the stream context and filters UI
    messages whose `metadata.message_id` matches the current AI message id.
  - **lines 36–41**: renders one `<LoadExternalComponent>` per matched UI
    message, passing `stream`, `message={customComponent}`, and
    `meta={{ ui: customComponent, artifact }}`.
- `CustomComponent` is rendered inside the main `AssistantMessage` body further
  down the same file (search for `<CustomComponent`).

### 3. EXACT injection point for a LOCAL `components={...}` map

**File: `src/components/thread/messages/ai.tsx`, the `<LoadExternalComponent>`
element at lines 36–41.**

`LoadExternalComponent` (from `@langchain/langgraph-sdk/react-ui`) accepts an
optional **`components` prop**: a map of `{ [componentName]: ReactComponent }`.
When the server-emitted UI message's `name` matches a key in that map, the SDK
renders the LOCAL component instead of fetching/evaluating remote JS. This is
the supported hook for shipping first-party Gen-UI components.

To wire our local workflow components, define the map (our code lives under
`src/components/workflow/`) and pass it here, e.g.:

```tsx
// at top of ai.tsx
import { workflowComponents } from "@/components/workflow";

// in CustomComponent, the LoadExternalComponent element:
<LoadExternalComponent
  key={customComponent.id}
  stream={thread as unknown as ReturnType<typeof useStream>}
  message={customComponent}
  meta={{ ui: customComponent, artifact }}
  components={workflowComponents}   // <-- LOCAL component map injected here
/>
```

This one-line `components={...}` addition at lines 36–41 is the **only** edit to
a vendored file the Gen-UI integration requires; the component implementations
themselves stay under `src/components/workflow/` per the patch policy above.

### 4. Assistant-id default must match the backend graph id

**File: `src/providers/Stream.tsx`, `DEFAULT_ASSISTANT_ID`.**

Upstream ships `DEFAULT_ASSISTANT_ID = "agent"`, but the demo backend registers
its host graph under **`host`** in `backend/langgraph.json`
(`"host": "./host_graph.py:make_host_graph"`). Connection resolution is
`finalAssistantId = (URL ?assistantId param) || NEXT_PUBLIC_ASSISTANT_ID`, with
`DEFAULT_ASSISTANT_ID` used only as the setup-form's pre-filled default. So two
things must point at `host` for the copy-paste getting-started path to connect to
a graph that exists:

- `NEXT_PUBLIC_ASSISTANT_ID=host` in `.env.example` (demo-owned — the value that
  actually drives the connection once `.env` is copied), and
- `DEFAULT_ASSISTANT_ID = "host"` in `Stream.tsx` (vendored — the form default a
  user sees when they fill the setup form manually with no `.env`).

Both are set in this vendored snapshot. On an upstream re-sync, re-apply the
`Stream.tsx` default so it keeps matching the backend graph id (or rename the
backend graph to `agent` and revert both — pick one and make them agree).

## Dev server

Boots with `pnpm install` then `pnpm dev` (Next.js dev server, default port
3000). Connection target is configured via the env vars / form described above.

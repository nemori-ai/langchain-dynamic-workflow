# `dev-master-orchestrator` — master orchestrator dev-loop workflow

`dev-master-orchestrator.js` is the OMNE master orchestrator's formal **named Claude Code
workflow**. Once on disk it can be invoked from any session with
`Workflow({ name: "dev-master-orchestrator", args })`, completing the orchestrator toolchain
alongside the `.claude/agents/` roster.

---

## 1. Purpose

It runs the master orchestrator's **5-stage dev-loop** over one or more
implementation tasks:

```
Impl  ──▶  Review ∥ Verify  ──▶  convergence test  ──▶  Amend  ──┐
  │            (parallel)                                          │
  └─────────────────────  re-Review ∥ re-Verify  ◀────────────────┘
```

- **Inner loop** (per task): implement → review and gate-verify in parallel →
  test convergence → if not converged, amend → loop back to re-review/re-verify,
  until the task is `ACCEPTED` or `ESCALATED`.
- **Outer loop** (across tasks): tasks are grouped into **waves**; each wave runs
  its tasks either in `parallel` (concurrent) or `serial` (one after another).

The orchestrator never implements or reviews by hand — every Impl / Review /
Verify / Amend step is a dispatched sub-agent.

---

## 2. Prerequisites

- **Roster agents installed.** The workflow dispatches to `omne-impl` (Impl),
  `omne-amender` (Amend), and `general-purpose` (Review and Verify). These — plus
  `omne-explore` — must already live in `.claude/agents/`. If you just authored a
  new agent, run `/reload-plugins`: **the agent registry is a session snapshot**, so
  a newly created agent is not dispatchable until the registry reloads.
- **Worktrees pre-built.** The orchestrator pre-creates each task's git worktree on
  the feature branch (never `main`) and builds the internal `.venv` inside it, so
  `uv run` and `./scripts/run_tests.sh` resolve **this worktree's** source — not the
  main checkout's stale editable install.
- **Spec doc reachable.** `specDoc` is an absolute path the dispatched agents Read;
  each task points at a heading inside it via `anchor` / `reviewAnchor`.

---

## 3. Full `args` schema

`args` may be passed as a JSON string or as an object — the script normalizes both
via `(typeof args === 'string') ? JSON.parse(args) : (args || {})`.

| Field | Type | Required | Default | Meaning |
|-------|------|----------|---------|---------|
| `specDoc` | string (absolute path) | yes | — | Absolute path to the spec doc the agents Read for requirements and acceptance bars. |
| `pkg` | string | no | `'omne_core_v1'` | uv-workspace package passed to `./scripts/run_tests.sh --targets <pkg>` during Verify. |
| `testModeArg` | string | no | `'--test-mode unit'` | Global default test-gate args, used when a task does not set `gateK`. |
| `maxRounds` | number | no | `3` | Max review rounds per task before `ESCALATED`. |
| `waves` | array of wave objects | no | `[]` | Outer-loop wave list (see below). |

### Wave object

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `id` | string / number | yes | Wave identifier (used in logs). |
| `mode` | `'parallel'` \| `'serial'` | no | Defaults to serial when omitted. `parallel` runs the wave's tasks concurrently; any other value (including absent) runs them serially. |
| `tasks` | array of task objects | yes | Tasks belonging to this wave. |

### Task object

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `id` | string | yes | Task identifier (used in agent labels and logs). |
| `anchor` | string | yes | Markdown heading in `specDoc` the **implementer** Reads (`## <anchor>`). |
| `reviewAnchor` | string | no (falls back to `anchor`) | Heading the **reviewer** Reads (`## <reviewAnchor>`); enables the dual-anchor information asymmetry. |
| `worktree` | string (absolute path) | yes | The task's pre-built worktree the agents `cd` into. |
| `gateK` | string | no | Per-task pytest `-k` expression. When set, Verify runs `--test-mode unit -- -k "<gateK>"`, narrowing the gate to this task; when absent, falls back to the global `testModeArg`. |
| `preSeeded` | boolean | no | Debug/demo only — skip Impl and drive the loop from a pre-committed worktree state (see §7). Normal tasks omit it. |
| `seededFiles` | array of strings | no | Only meaningful with `preSeeded`; the file list reported as the (skipped) impl's `filesChanged`. |

---

## 4. Five-stage closed loop and the convergence test

Each round runs Review and Verify in parallel, then evaluates one convergence test:

```
reviewOk  = review is present AND has no blocker/major finding
converged = reviewOk AND verify.allGreen
```

- `converged` → task returns **`ACCEPTED`**.
- not converged and `round >= maxRounds` → task returns **`ESCALATED`**.
- otherwise → **Amend** runs, then the loop re-reviews and re-verifies.

**Gate-green is necessary but not sufficient.** `verify.allGreen` alone does not
accept a task — the review must also be free of blocker/major findings. A missing
review (agent crash/timeout returns `null`) is treated as **not converged**, never as
"zero majors" — this is the no-silent-failure guard in `reviewOk = !!review && ...`.

---

## 5. Dual anchor (information asymmetry)

The implementer and the reviewer read **different** spec sections on purpose:

- **Impl** reads `## <anchor>` — the task's build requirements and acceptance.
- **Review** reads `## <reviewAnchor>` — a fuller acceptance bar (e.g. the complete
  set of input equivalence classes and ordering-dependency contracts the suite must
  cover).

When a task omits `reviewAnchor`, the reviewer falls back to `anchor`
(`const reviewAnchor = task.reviewAnchor || task.anchor`). Giving the reviewer a
fuller bar than the implementer was handed lets the loop catch gaps the implementer
could not have known to close.

---

## 6. Invocation example

```js
Workflow({
  name: "dev-master-orchestrator",
  args: {
    specDoc: "/abs/path/docs/plans/my-feature-spec.md",
    pkg: "omne_veracity",
    testModeArg: "--test-mode unit",
    maxRounds: 3,
    waves: [
      {
        id: "w1",
        mode: "parallel",
        tasks: [
          {
            id: "T1",
            anchor: "TASK-1",
            reviewAnchor: "TASK-1-REVIEW",
            worktree: "/abs/path/.claude/worktrees/feat-t1",
            gateK: "test_evidence or test_provenance"
          },
          {
            id: "T2",
            anchor: "TASK-2",
            worktree: "/abs/path/.claude/worktrees/feat-t2"
          }
        ]
      },
      {
        id: "w2",
        mode: "serial",
        tasks: [
          {
            id: "T3",
            anchor: "TASK-3",
            reviewAnchor: "TASK-3-REVIEW",
            worktree: "/abs/path/.claude/worktrees/feat-t3",
            gateK: "test_aggregate"
          }
        ]
      }
    ]
  }
})
```

`w1` runs `T1` and `T2` concurrently; once both settle, `w2` runs `T3` serially.

---

## 7. `preSeeded` debug usage

`preSeeded` is a **debug/demo** capability, not for normal tasks. With real
implementers the suite usually passes review in one round, so the
review → amend → re-review multi-round path is rarely exercised. To light it up
deterministically:

1. Pre-commit a deliberately **incomplete** state into the task's worktree (e.g. a
   suite missing several equivalence classes the `reviewAnchor` bar requires).
2. Set `preSeeded: true` (and optionally `seededFiles`) on the task.

The workflow then **skips Impl** and enters the loop directly from the committed
state, so Review finds real gaps and Amend has genuine work — driving the inner loop
through multiple rounds on demand.

---

## 8. Return structure

```jsonc
{
  "summary": { "total": 3, "accepted": 2, "escalated": 1 },
  "tasks": [
    {
      "taskId": "T1",
      "status": "ACCEPTED",          // ACCEPTED | ESCALATED | FAILED
      "rounds": 0,
      "impl": { "filesChanged": ["..."], "foundBug": "" },
      "finalReview": { "findings": [], "overallAssessment": "..." },
      "finalVerify": { "allGreen": true, "summary": "..." },
      "amendLog": []
    },
    {
      "taskId": "T3",
      "status": "ESCALATED",
      "rounds": 3,
      "reason": "not converged after 3 review rounds",
      "impl": { "filesChanged": ["..."], "foundBug": "" },
      "lastReview": { "findings": ["..."], "overallAssessment": "..." },
      "lastVerify": { "allGreen": false, "summary": "..." },
      "amendLog": [{ "round": 1, "amend": { "dispositions": ["..."] } }]
    }
  ]
}
```

`summary` aggregates counts across all tasks; each `tasks[]` entry carries its
terminal `status`, the number of review `rounds`, the final (or last) review and
verify payloads, and the per-round `amendLog`. An impl that returns `null` yields a
`FAILED` task (`reason: "impl returned null"`).

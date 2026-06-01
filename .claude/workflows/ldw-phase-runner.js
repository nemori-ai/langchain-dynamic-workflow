// ldw-phase-runner — drive ONE langchain-dynamic-workflow implementation phase.
//
// Adapted from dev-master-orchestrator's 5-stage convergence loop, but retargeted
// from OMNE (worktrees, run_tests.sh, omne-* agent types) to THIS repo's gates
// (uv run ruff/pyright/pytest) and the design_docs/plans/ specs. Works IN-PLACE on
// the current branch (no worktree): tasks within a phase are tightly coupled and
// mutate shared async modules, so they run serially in one place.
//
// Inner loop per phase:  Impl -> (Review ∥ Verify) -> converge? -> Amend -> re-review,
// until review has no blocker/major AND verify is all-green, or maxRounds escalates.
//
// Call: Workflow({ scriptPath: '.claude/workflows/ldw-phase-runner.js',
//                  args: { plan: 'design_docs/plans/02-phase-2-fanout.md', phase: 'Phase 2', maxRounds: 3 } })

export const meta = {
  name: 'ldw-phase-runner',
  description: 'Drive one langchain-dynamic-workflow phase through a convergence loop: an impl agent implements the phase plan with TDD, then a reviewer and a verifier run in parallel; converge on no blocker/major findings plus all-green gates, else amend and re-review up to maxRounds.',
  phases: [{ title: 'Impl' }, { title: 'Review' }, { title: 'Verify' }, { title: 'Amend' }],
}

// ── args ──
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const PLAN = A.plan
const PHASE = A.phase || 'this phase'
const MAX_ROUNDS = A.maxRounds || 3

// ── shared prompt fragments ──
const RULES = 'Project discipline: read AGENTS.md and .claude/rules/{python_general,docstring,testing}.md and obey them. Python 3.12 async-first; complete type hints; Google style; pyright strict; ruff. Tests use pytest + pytest-asyncio (asyncio_mode=auto already set). GREEN tests MUST be driven by a fake model / fake leaf (see tests/conftest.py: CountingFakeModel, make_deep_leaf, make_fake_leaf) — NEVER a real API key. Reuse the existing test scaffolding instead of duplicating it. No filler tests: each test must anti-corrupt the architecture or guard a regression.'

const GATES = 'Quality gates (run from the repo root; ALL must pass): (1) uv run ruff check . — (2) uv run ruff format --check . — (3) uv run pyright — (4) uv run pytest -q > /tmp/ldw-verify.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-verify.log . Every previously-passing test (Phase 1 onward) MUST stay green.'

const GIT_SAFETY = 'Git safety: NEVER run "git add ." or "git add -A" — stage explicit paths only. NEVER stage or print .env (secrets, gitignored). Commit on the current branch. End every commit message body with the trailer: Co-Authored-By: Claude <noreply@anthropic.com>'

function implPrompt() {
  return [
    'You implement ' + PHASE + ' of the langchain-dynamic-workflow project, working IN-PLACE in the current repo (no worktree; you are on the main branch where Phase 1 is already committed).',
    'STEP 1 — Read the spec: the full plan is at ' + PLAN + ' (milestone, acceptance criteria, interface design, file structure, bite-sized TDD task breakdown, demo spec, refactor notes). Also read the design-doc sections it cites under design_docs/ (01-engine-mechanism.md, 02-architecture.md) so your implementation matches the locked architecture.',
    'STEP 2 — Study existing code: read src/langchain_dynamic_workflow/ and tests/ from Phase 1 to match style and REUSE infrastructure. Do not break existing public signatures; new params are keyword-only with defaults.',
    RULES,
    'STEP 3 — Implement EVERY task in the plan with strict TDD: write the failing test, run it red, write the minimal code, run it green, refactor, then COMMIT that task (one commit per task, clear message). Refactors of earlier phases must keep their tests green.',
    'STEP 4 — Run every quality gate and capture RAW output. Do NOT claim green; an independent verifier re-runs the gates.',
    GATES,
    GIT_SAFETY,
    'Return: filesChanged (paths), a per-task summary, gateSelfReport (raw tail of each gate run), and any bug you found (foundBug, or empty string).',
  ].join('\n\n')
}

function reviewPrompt(round) {
  return [
    'Independent code review of ' + PHASE + ' for langchain-dynamic-workflow. Round ' + round + '. You REPORT ONLY — do not modify any file.',
    'STEP 1 — Read the acceptance bar in ' + PLAN + ': the acceptance criteria, the metrics, the interface design, and the "关键设计要点" (key design points). This is the COMPLETE bar the implementation and its tests must meet.',
    'STEP 2 — Read the actual code: the source files and tests listed in the plan\'s "文件结构" (file structure) section — both new modules and modified ones under src/langchain_dynamic_workflow/ and tests/. Use git log/diff on the latest commits if helpful.',
    'STEP 3 — Judge against the bar, focusing on what matters for an async orchestration engine: correctness of concurrency (bounded concurrency actually ENFORCED, no deadlocks, no unbounded fan-out), ordering/保序 guarantees, failure semantics (parallel returns None at that slot and does NOT raise vs pipeline drops the item and skips remaining stages), journal correctness across fan-out AND mid-run resume (completed leaves cache-hit with zero model calls, in-flight re-run live), determinism/budget replay where the phase covers it, and whether each acceptance criterion has a GENUINE test (no filler, real assertions that pin behavior).',
    'STEP 4 — Emit one finding per gap. Severity: a missing acceptance criterion or a real correctness/concurrency/journal bug = blocker or major; a stated guarantee with weak/absent test coverage = major; an assertion that does not pin the claimed behavior, or a docstring/style-rule violation = minor; cosmetic = nit.',
    'Return findings (title, file, severity, rationale) and an overallAssessment stating whether the phase fully meets its bar.',
  ].join('\n\n')
}

function verifyPrompt(round) {
  return [
    'Independent quality-gate verification for ' + PHASE + '. Round ' + round + '. Do NOT trust any prior agent self-report — run the gates yourself from the repo root and report ACTUAL exit codes.',
    GATES,
    'Report: ruffCheck, ruffFormat, pyright, unitTests (each a boolean from the real exit code), allGreen = all four true, and a summary including the pytest pass count and any failure names.',
  ].join('\n\n')
}

function amendPrompt(review, verify, round) {
  return [
    'Amendment task for ' + PHASE + ', round ' + round + '. Working IN-PLACE in the repo.',
    'Review findings to resolve: ' + JSON.stringify((review && review.findings) || []),
    'Current gate status: ' + JSON.stringify(verify || {}),
    'Fix EVERY valid blocker/major finding and any failing gate, surgically (no scope creep): add the missing tests for stated guarantees; fix real concurrency/journal/determinism bugs. For a finding you judge invalid or genuinely out of scope, mark it ESCALATED or FLAGGED_OUT_OF_SCOPE with a note rather than forcing a bad fix.',
    RULES,
    'Re-run ALL gates before finishing; do not stop until they pass (or you have escalated with a clear reason). Commit your fixes (explicit paths only).',
    GIT_SAFETY,
    'Return a per-finding disposition ledger (findingTitle, disposition, note) and a summary.',
  ].join('\n\n')
}

function hasBlockerOrMajor(review) {
  return !!(review && review.findings && review.findings.some(f => f.severity === 'blocker' || f.severity === 'major'))
}

// ── structured-output schemas ──
const IMPL_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    filesChanged: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
    gateSelfReport: { type: 'string' },
    foundBug: { type: 'string' },
  },
  required: ['filesChanged', 'summary'],
}
const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          title: { type: 'string' },
          file: { type: 'string' },
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          rationale: { type: 'string' },
        },
        required: ['title', 'severity', 'rationale'],
      },
    },
    overallAssessment: { type: 'string' },
  },
  required: ['findings', 'overallAssessment'],
}
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    ruffCheck: { type: 'boolean' },
    ruffFormat: { type: 'boolean' },
    pyright: { type: 'boolean' },
    unitTests: { type: 'boolean' },
    allGreen: { type: 'boolean' },
    summary: { type: 'string' },
  },
  required: ['allGreen', 'summary'],
}
const AMEND_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    dispositions: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          findingTitle: { type: 'string' },
          disposition: { type: 'string', enum: ['FIXED', 'ESCALATED', 'FIXED_WITH_RESERVATION', 'FLAGGED_OUT_OF_SCOPE'] },
          note: { type: 'string' },
        },
        required: ['findingTitle', 'disposition'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['dispositions', 'summary'],
}

// ── drive ──
log('ldw-phase-runner start — phase=' + PHASE + ' plan=' + PLAN + ' maxRounds=' + MAX_ROUNDS)

phase('Impl')
const impl = await agent(implPrompt(), { agentType: 'general-purpose', label: 'impl', phase: 'Impl', schema: IMPL_SCHEMA })
if (!impl) return { status: 'FAILED', reason: 'impl returned null' }

const amendLog = []
let round = 0
while (true) {
  phase('Review')
  const [review, verify] = await parallel([
    () => agent(reviewPrompt(round), { agentType: 'general-purpose', label: 'review:r' + round, phase: 'Review', schema: REVIEW_SCHEMA }),
    () => agent(verifyPrompt(round), { agentType: 'general-purpose', label: 'verify:r' + round, phase: 'Verify', schema: VERIFY_SCHEMA }),
  ])
  // A null review means the reviewer crashed — treat as NOT converged (never read a
  // missing review as "zero majors").
  const reviewOk = !!review && !hasBlockerOrMajor(review)
  const converged = reviewOk && !!(verify && verify.allGreen)
  log('round=' + round + ' converged=' + converged + ' green=' + !!(verify && verify.allGreen) + ' reviewOk=' + reviewOk + ' reviewPresent=' + !!review)
  if (converged) {
    return { status: 'ACCEPTED', rounds: round, impl: { filesChanged: impl.filesChanged, foundBug: impl.foundBug || '' }, finalReview: review, finalVerify: verify, amendLog }
  }
  round++
  if (round >= MAX_ROUNDS) {
    return { status: 'ESCALATED', rounds: round, reason: 'not converged after ' + MAX_ROUNDS + ' rounds', lastReview: review, lastVerify: verify, amendLog }
  }
  phase('Amend')
  const amend = await agent(amendPrompt(review, verify, round), { agentType: 'general-purpose', label: 'amend:r' + round, phase: 'Amend', schema: AMEND_SCHEMA })
  amendLog.push({ round, amend })
}

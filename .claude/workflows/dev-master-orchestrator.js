// dev-master-orchestrator — the OMNE master orchestrator's 5-stage dev-loop workflow.
//
// What it is: a named Claude Code workflow that drives one or more implementation
// tasks through a 5-stage inner loop (Impl -> Review ∥ Verify -> convergence test
// -> Amend, repeated until converged or escalated), with an outer loop over waves
// that run their tasks either in parallel or serially.
//
// How to call it (from any session, once this file is on disk):
//   Workflow({ name: 'dev-master-orchestrator', args: { specDoc, pkg, testModeArg, maxRounds, waves } })
// `args` may be passed as a JSON string or an object — see the compatibility header
// below. The full args schema, convergence semantics, dual-anchor rationale and a
// runnable example all live in .claude/workflows/README.md.

export const meta = {
  name: 'dev-master-orchestrator',
  description: 'Master-orchestrator 5-stage dev-loop: implement, then review ∥ verify in parallel, converge on no-blocker/major review plus all-green gates, otherwise amend and re-review; outer loop runs waves of tasks in parallel or serially.',
  phases: [{ title: 'Impl' }, { title: 'Review' }, { title: 'Verify' }, { title: 'Amend' }],
}

// ── Structured-output schemas for each agent role ──
// Each dispatched agent returns JSON conforming to one of these; the orchestrator
// reads structured fields (severity, allGreen, dispositions) rather than prose.
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

// ── args 兼容头 ──
// `args` 既可能是 JSON 字符串(跨进程调用),也可能已是对象(同进程调用),统一归一化。
const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const WAVES = A.waves || []
const SPEC_DOC = A.specDoc
const PKG = A.pkg || 'omne_core_v1'
// 全局缺省 gate。单个 task 可用 task.gateK 覆盖(见 verifyPrompt)。
const TEST_MODE_ARG = A.testModeArg || '--test-mode unit'
const MAX_ROUNDS = A.maxRounds || 3
log('dev-master-orchestrator start — waves=' + WAVES.length + ' pkg=' + PKG + ' maxRounds=' + MAX_ROUNDS)

function implPrompt(WT, anchor) {
  return 'Implementation task. STEP 0: cd to the worktree ' + WT + ' and confirm the branch is NOT main. '
    + 'STEP 1: your full spec is in ' + SPEC_DOC + ' under the markdown heading "## ' + anchor + '" — Read THAT section only for requirements + acceptance. '
    + 'STEP 2: implement with TDD, run every quality gate, return RAW output (do not claim green; verify re-runs independently).'
}

// merge-base 求基线: worktree 的真实 base 是 git merge-base HEAD main,不是 main 的当前 HEAD。
// 会话期间 main 可能前进,裸 git diff main 会把无关的 main 推进喂给 reviewer,所以两处都先求 merge-base。
function reviewPrompt(WT, reviewAnchor) {
  return 'Code review (Claude-based reviewer). cd to ' + WT + '. '
    + 'First compute this worktree\'s true baseline commit: run "git merge-base HEAD main" to get BASE, then "git diff <BASE>" to see ONLY this task\'s net changes — do NOT use "git diff main" (main may have advanced during the session, which would feed you unrelated diff). '
    + 'Your acceptance bar is in ' + SPEC_DOC + ' under "## ' + reviewAnchor + '" — Read that section. It enumerates the COMPLETE set of input equivalence classes the tests MUST cover, plus an ordering-dependency contract. '
    + 'The implementer may have been given a thinner spec or left the suite incomplete, so your job is to catch every missing equivalence class against this fuller bar. First Read the source under test, then diff it against the tests. '
    + 'Emit one finding per gap with severity per the bar (missing equivalence class = major; ordering-dependency contract untested = major; assertion not pinning the error-message text = minor). '
    + 'Return findings with severity (blocker|major|minor|nit) and rationale; overallAssessment summarising whether the suite meets the full bar.'
}

// verifyPrompt 接收整个 task,以便用 task.gateK 把 gate scope 收窄到本 task 的 -k 表达式;
// 缺省时回退全局 TEST_MODE_ARG(向后兼容)。changed-files 框定同样走 merge-base。
function verifyPrompt(WT, task) {
  const gateScope = task && task.gateK
    ? '--test-mode unit -- -k "' + task.gateK + '"'
    : TEST_MODE_ARG
  return 'Independent quality-gate verification — do NOT trust any prior agent self-report; run the gates yourself. '
    + 'cd to ' + WT + ' (confirm branch is NOT main). '
    + 'Frame the changed file set against this worktree\'s true baseline: run "git merge-base HEAD main" to get BASE, then "git diff --name-only <BASE>" — do NOT use "git diff main" (main may have advanced). '
    + 'From inside the worktree run on those changed files: uv run ruff check; uv run ruff format --check; uv run pyright; '
    + 'and ./scripts/run_tests.sh --targets ' + PKG + ' ' + gateScope + ' (redirect to a /tmp log, then tail). '
    + 'Report each gate pass/fail truthfully from ACTUAL exit codes; allGreen = all four passed.'
}

function amendPrompt(WT, review, verify, round) {
  return 'Amendment task (round ' + round + '). cd to ' + WT + '. '
    + 'Review findings to resolve: ' + JSON.stringify((review && review.findings) || []) + '. '
    + 'Current gate status: ' + JSON.stringify(verify) + '. '
    + 'Fix valid findings surgically (no scope creep): add the missing equivalence-class tests and pin the error-message paths the reviewer flagged. '
    + 'For findings that point at out-of-scope source bugs, either pin them with an xfail test or mark ESCALATED. Re-run gates. Return a per-finding disposition ledger.'
}

function hasBlockerOrMajor(review) {
  return !!(review && review.findings && review.findings.some(f => f.severity === 'blocker' || f.severity === 'major'))
}

// ── 内循环: 单 task 收敛 ──
// 标准路径: impl -> (review ∥ verify) -> 收敛判据 -> 不收敛则 amend -> 回 re-review,
//           直到 review 无 blocker/major 且 verify.allGreen,或达 MAX_ROUNDS 升级。
// task.preSeeded(debug/demo 能力): 跳过 impl,直接用 worktree 里已 commit 的(故意残缺的)
//           状态进 review 循环,确定性点亮 review->amend->re-review 多轮路径(真实 impl
//           一次到位率高,该路径平时踩不到)。正常 task 不设此项。
async function runTask(task) {
  const WT = task.worktree
  const reviewAnchor = task.reviewAnchor || task.anchor
  let impl
  if (task.preSeeded) {
    phase('Impl')
    log('task=' + task.id + ' preSeeded — skipping impl, driving loop from committed worktree state')
    impl = { filesChanged: task.seededFiles || [], summary: 'pre-seeded (impl skipped)', foundBug: '' }
  } else {
    phase('Impl')
    impl = await agent(implPrompt(WT, task.anchor), { agentType: 'omne-impl', label: 'impl:' + task.id, phase: 'Impl', schema: IMPL_SCHEMA })
    if (!impl) return { taskId: task.id, status: 'FAILED', reason: 'impl returned null' }
  }

  const amendLog = []
  let lastReview = null
  let lastVerify = null
  let round = 0
  while (true) {
    phase('Review')
    const [review, verify] = await parallel([
      () => agent(reviewPrompt(WT, reviewAnchor), { agentType: 'general-purpose', label: 'review:' + task.id + ':r' + round, phase: 'Review', schema: REVIEW_SCHEMA }),
      () => agent(verifyPrompt(WT, task), { agentType: 'general-purpose', label: 'verify:' + task.id + ':r' + round, phase: 'Verify', schema: VERIFY_SCHEMA }),
    ])
    lastReview = review
    lastVerify = verify
    // no-silent-failure: a null review means the reviewer agent crashed/timed out.
    // Treat that as NOT-converged (never let a missing review read as "zero majors").
    const reviewOk = !!review && !hasBlockerOrMajor(review)
    const converged = reviewOk && !!(verify && verify.allGreen)
    log('task=' + task.id + ' round=' + round + ' converged=' + converged + ' green=' + !!(verify && verify.allGreen) + ' reviewOk=' + reviewOk + ' reviewPresent=' + !!review)
    if (converged) {
      return { taskId: task.id, status: 'ACCEPTED', rounds: round, impl: { filesChanged: impl.filesChanged, foundBug: impl.foundBug || '' }, finalReview: review, finalVerify: verify, amendLog }
    }
    round++
    if (round >= MAX_ROUNDS) {
      return { taskId: task.id, status: 'ESCALATED', rounds: round, reason: 'not converged after ' + MAX_ROUNDS + ' review rounds', impl: { filesChanged: impl.filesChanged, foundBug: impl.foundBug || '' }, lastReview, lastVerify, amendLog }
    }
    phase('Amend')
    const amend = await agent(amendPrompt(WT, review, verify, round), { agentType: 'omne-amender', label: 'amend:' + task.id + ':r' + round, phase: 'Amend', schema: AMEND_SCHEMA })
    amendLog.push({ round, amend })
    // loop back → re-review ∥ re-verify
  }
}

// ── 外循环: 波次/task ──
// 每个 wave 声明 mode=parallel|serial。parallel: 同波 task 并发 runTask;serial: 顺序执行。
const results = []
for (const wave of WAVES) {
  log('=== Wave ' + wave.id + ' [' + wave.mode + '] — ' + wave.tasks.length + ' task(s) ===')
  if (wave.mode === 'parallel') {
    results.push(...(await parallel(wave.tasks.map(t => () => runTask(t)))))
  } else {
    for (const t of wave.tasks) results.push(await runTask(t))
  }
}

const accepted = results.filter(r => r && r.status === 'ACCEPTED').length
const escalated = results.filter(r => r && r.status === 'ESCALATED').length
return { summary: { total: results.length, accepted, escalated }, tasks: results }

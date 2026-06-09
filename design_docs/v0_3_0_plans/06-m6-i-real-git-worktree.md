# M6 · I — 真 git worktree + 分支/PR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每个改文件的 leaf 跑在自己的**真 `git worktree` / 真分支**里(真 build/test,承 M5),改完由脚本拥有的冲突循环把变更**真 `git merge` 折进 integration 分支(非 main)**,最后经 host finalization 开 PR——把内存 worktree 隔离升级成真 git 隔离。

**Architecture:** 方案 1(经 Codex 对抗评审拍板,折入 R1–R10)。三个引擎单元 + 两个范式:① `LocalSubprocessSandbox(root=, on_close=)` 让叶子沙箱根植在真 worktree 目录;② `GitWorktreeProvider`(真 `git worktree add`/`git diff` collect/幂等/异常安全 teardown);③ `PullRequestProvider` 协议 + `LocalPullRequestProvider`(离线默认)。所有 git 物理副作用都在叶子 `@task` 边界内 → resume 时被 content-hash journal 短路、绝不重跑。PR/integration 物化移出确定性 replay 作**幂等 host finalization**;worktree 叶权威变更集 = leaf task 边界内**真 `git diff`**(非模型自报);整合用 merge 叶内**一次性 scratch-repo 真 `git merge`**。

**Tech Stack:** Python 3.12 async-first;现有 `LocalSubprocessSandbox`/`SandboxManager`(`_sandbox.py`/`_local_subprocess.py`,M5)、`WorktreeProvider` seam(`_worktree.py`,G2)、content-hash journal(`_journal.py`/`_context.py`/`_engine.py`);真 `git`(2.50)/`gh`(2.91)在 host PATH;pytest + pytest-asyncio;ruff;pyright(strict);import-linter。

**设计稿(承重墙张力 + Codex 商议 §6 决议):** `docs/plans/2026-06-08-m6-real-git-worktree-design.md`(gitignored)。

---

## 已核实的表面(开工前已证,file:line)

| 事实 | 出处 |
|---|---|
| `LocalSubprocessSandbox.__init__(*, identity, policy, exec_gate)` 构造期 `self._root = tempfile.mkdtemp(...)`;无根植既有目录入口 | `_local_subprocess.py:475-488` |
| `execute` 用 `cwd=self._root`;文件 API 全映射 `self._root` 下;`close()` rmtree `_root`(幂等) | `_local_subprocess.py:848`,`:1214-1225` |
| `SandboxManager._new_sandbox(leaf_id, isolation)` = `self._factory(leaf_id)` + 若 `isolation=="worktree"` 且有 `worktree_provider` → `provider.seed()`+`upload_files()` | `_sandbox.py:554-579` |
| `lease`/`acquire` find-or-create per leaf_id;teardown 路径 `reclaim_idle`/`_evict_one_idle`/`stop` 全调 `_close_backend(slot.sandbox)` → `sandbox.close()` | `_sandbox.py:586-803` |
| `_new_sandbox` 同步、在 `lease` 持 `self._slot_freed` condition 锁内调用(R8 阻塞点) | `_sandbox.py:704-728` |
| `leaf_task` 仅在 journal miss 后进入;`async with sandbox_manager.lease(...) as backend`,`needs_execution` 叶 `leased_execution_leaf_ids.add(leaf_id)`;`_invoke()` 返回 `LeafOutcome(state, usage).to_payload()` | `_engine.py:277-321` |
| `Ctx.agent` 派发前查 journal(hit 直接 return,不进 leaf_task);有 schema → `fold_structured(outcome.state, model)`、journal 其 `model_dump_json`;无 → `fold_result(outcome.state)` | `_context.py:530-567`,`:601-616` |
| 引擎收尾对 `leased_execution_leaf_ids` 逐个 `manager.stop(leaf_id)`(**执行期 pin 确切收尾点**) | `_engine.py`(`leased_execution_leaf_ids` 消费处) |
| 已有 `local_subprocess_factory(policy)` 共享 `exec_gate=BoundedSemaphore`;DANGEROUS opt-in 警告范式 | `_sandbox.py:69-99`,`_local_subprocess.py:451-473` |
| demo 已有 untrusted authored-script 用 reasoning-only roster(`make_reasoning_roster` OMIT `code_fixer`)守边界 | `demo-app/backend/workflows.py:873-890` |

---

## 文件结构(本计划触达)

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/langchain_dynamic_workflow/_local_subprocess.py` | `LocalSubprocessSandbox` 加 `root=`/`on_close=`(根植既有目录、close 不删、调回调) | 修改 |
| `src/langchain_dynamic_workflow/_git_worktree.py` | `GitWorktreeProvider`(真 git worktree/branch/collect/teardown,幂等+异常安全) | **新建** |
| `src/langchain_dynamic_workflow/_pull_request.py` | `PullRequestProvider` 协议 + `PullRequestRef` + `LocalPullRequestProvider`(幂等离线默认) | **新建** |
| `src/langchain_dynamic_workflow/_sandbox.py` | `SandboxManager` 加 `git_worktree_provider=`;worktree 叶走 `open_worktree`;git 阻塞 thread-offload 出锁 | 修改 |
| `src/langchain_dynamic_workflow/_engine.py` | git-worktree 执行叶 `_invoke` 后 lease 内 `provider.collect(leaf_id)` 折进 outcome(真 diff 权威、journaled) | 修改 |
| `src/langchain_dynamic_workflow/__init__.py` | 短路导出 `GitWorktreeProvider`/`PullRequestProvider`/`PullRequestRef`/`LocalPullRequestProvider` | 修改 |
| `tests/unit/test_local_subprocess.py` | `root=`/`on_close=` 单测(不 mkdtemp、不 rmtree、调回调) | 修改/新建 |
| `tests/unit/test_git_worktree.py` | 对真 temp git 仓库:open_worktree 真建/根植、collect=git diff、幂等、异常回滚、teardown、cleanup_all | **新建** |
| `tests/unit/test_pull_request.py` | `LocalPullRequestProvider.open` 落记录 + 幂等 | **新建** |
| `tests/integration/test_git_worktree_swarm.py` | 两并行 worktree fixer 隔离 + collect 权威 + scratch-repo merge 干净/冲突循环 + resume 不重跑 | **新建** |
| `examples/features/git_worktree.py` | 离线 feature demo:真 git provider(deterministic fakes)+ 冲突循环 production-grade | **新建** |
| `examples/AGENTS.md` / 根 `README.md` | demo 索引同步(17 features) | 修改 |
| `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` | "真 git worktree + 冲突循环" 范式段(道) | 修改 |
| `demo-app/backend/workflows.py` | `refactor_swarm` 预设 + git fixer/conflict_resolver roster 角色 + scratch-repo merge builder | 修改 |
| `demo-app/backend/host_graph.py` | PR host-finalization(幂等)+ MergeCard/PullRequestCard 发射 | 修改 |
| `demo-app/backend/ui_adapter.py` | merge/PR Gen-UI 卡 emit | 修改 |
| `demo-app/frontend/src/components/workflow/{MergeCard,PullRequestCard}.tsx` + `registry.ts` | 新卡片 + 注册 | 新建/修改 |
| `demo-app/{scenarios.json,...ScenarioPanel.tsx,README.md}` | "Refactor swarm" 场景(doc-sync 字节一致) | 修改 |
| `demo-app/backend/tests/test_refactor_swarm.py` + `test_m6_refactor_swarm_real.py` | demo 消费离线测试 + gated 真模型 E2E | **新建** |
| `design_docs/{01-engine-mechanism.md,02-architecture.md,uml/*}` + `v0_3_0_plans/00-roadmap.md` | evergreen 同步 + Decision Log D-M6 + roadmap M6 ✅ | 修改 |

---

## 前置:分支(已就绪)

- [x] worktree + 分支已建:`feat/m6-real-git-worktree` @ `../langchain-dynamic-workflow-m6`(off main 630af68)。
- [ ] **首次** `uv sync`(根)+ `uv sync --group example`(真 E2E 前);demo-app 单独 `uv sync`。

---

## Task 1: `LocalSubprocessSandbox` 加 `root=` / `on_close=`(A1 + R2 绑定 teardown + LOW)

**Files:** Modify `src/langchain_dynamic_workflow/_local_subprocess.py`;Test `tests/unit/test_local_subprocess.py`

- [ ] **Step 1(失败测试):** 新增
```python
# tests/unit/test_local_subprocess.py(并入既有/新建)
import os, tempfile, threading
from langchain_dynamic_workflow._local_subprocess import ExecPolicy, LocalSubprocessSandbox

def _gate() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(ExecPolicy().max_concurrent_execs)

def test_root_provided_is_used_and_not_removed_on_close() -> None:
    existing = tempfile.mkdtemp(prefix="ldw-test-root-")
    closed: list[bool] = []
    sb = LocalSubprocessSandbox(
        identity="L1", policy=ExecPolicy(), exec_gate=_gate(),
        root=existing, on_close=lambda: closed.append(True),
    )
    # rooted at the provided dir, did NOT mkdtemp a new one
    assert sb.execute("pwd").output.strip() == os.path.realpath(existing)
    sb.close()
    assert closed == [True]              # on_close fired
    assert os.path.isdir(existing)       # NOT rmtree'd (provider owns lifecycle)
    sb.close()                           # idempotent: on_close fires at most once
    assert closed == [True]
    os.rmdir(existing)

def test_default_root_still_mkdtemps_and_rmtrees() -> None:
    sb = LocalSubprocessSandbox(identity="L2", policy=ExecPolicy(), exec_gate=_gate())
    root = sb.execute("pwd").output.strip()
    assert os.path.isdir(root)
    sb.close()
    assert not os.path.exists(root)      # owns its temp root -> rmtree on close
```
- [ ] **Step 2:** `uv run pytest tests/unit/test_local_subprocess.py -q` → FAIL(`root`/`on_close` 未知参数)。
- [ ] **Step 3(实现):** `__init__` 增 keyword-only `root: str | None = None, on_close: Callable[[], None] | None = None`:
```python
# 替换 self._root = tempfile.mkdtemp(...) 一段:
self._owns_root = root is None
self._root = root if root is not None else tempfile.mkdtemp(prefix=f"ldw-exec-{identity}-")
self._on_close = on_close
self._on_close_fired = False
```
`close()`(`:1214-1225` 附近)改为:先 `if self._on_close is not None and not self._on_close_fired: self._on_close_fired = True; self._on_close()`;**仅当 `self._owns_root`** 才 `shutil.rmtree(self._root, ignore_errors=True)`;`_closed` 幂等守卫保留。docstring 注明 `root`(根植既有目录、close 不删,由调用方管生命周期)与 `on_close`(close 时一次性回调,用于 git worktree remove)。
- [ ] **Step 4:** pytest 通过;`uv run ruff check src/langchain_dynamic_workflow/_local_subprocess.py tests/unit/test_local_subprocess.py` + `uv run pyright src/langchain_dynamic_workflow/_local_subprocess.py`。
- [ ] **Step 5:** commit `feat(sandbox): LocalSubprocessSandbox can root at an existing dir with an on_close hook`。

---

## Task 2: `GitWorktreeProvider`(A2 + R3 异常安全 + R4 幂等 + R2 teardown/cleanup_all)

**Files:** Create `src/langchain_dynamic_workflow/_git_worktree.py`;Test `tests/unit/test_git_worktree.py`

**契约(MergeResult/collect 权威是 R5/R7 关键):**
- `GitWorktreeProvider(*, base_repo, integration_branch="ldw/integration", base_ref="HEAD", workspace_root=None, policy=None, exec_gate=None)`。构造期:校验 `base_repo` 是 git 仓库(`git -C <base_repo> rev-parse --git-dir`,否则 `GitWorktreeError`);`workspace_root` 缺省 `mkdtemp`;`exec_gate` 缺省自建(比照 `local_subprocess_factory`)。
- `open_worktree(leaf_id) -> SandboxBackendProtocol`:**幂等**(先 `_reclaim_stale(leaf_id)`:存在同键 worktree → `git worktree remove --force`,存在 `leaf/<leaf_id>` 分支 → `git branch -D`)→ `git worktree add <workspace_root>/<safe(leaf_id)> -b leaf/<leaf_id> <base_ref>` → 返回 `LocalSubprocessSandbox(identity=leaf_id, policy, exec_gate, root=<path>, on_close=lambda: self.teardown(leaf_id))`。**异常安全**:add 后任何步骤失败 → rollback(remove worktree + del branch)再 raise `GitWorktreeError`(带 stderr)。记录 `self._worktrees[leaf_id] = path`。
- `collect(leaf_id) -> dict[str, str]`:对该 worktree `git add -A` 后 `git -C <path> status --porcelain`(或 `git diff --cached --name-only`)枚举新增/改动 path,逐个读当前内容 → `{path: content}`(删除的 path 值为 `""` 或单列,v1 收新增/改动)。**这是权威变更集**(真磁盘内容,非模型自报)。
- `teardown(leaf_id)`:`git worktree remove --force <path>` + `git branch -D leaf/<leaf_id>`(各自 best-effort,缺失不报错);从 `self._worktrees` 删除。幂等。
- `cleanup_all()`:对所有残留 `self._worktrees` teardown;`shutil.rmtree(workspace_root, ignore_errors=True)` 兜底。run 结束由 host/manager 调。
- 模块 + 类 docstring:**DANGEROUS opt-in**——起真 `git` 子进程、在 host 上以调用者权限运行、非安全沙箱(比照 `LocalSubprocessSandbox`/`local_subprocess_factory`)。

- [ ] **Step 1(失败测试):**
```python
# tests/unit/test_git_worktree.py
import subprocess, pytest
from pathlib import Path
from langchain_dynamic_workflow import GitWorktreeProvider
from langchain_dynamic_workflow._git_worktree import GitWorktreeError

def _git(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)

@pytest.fixture
def base_repo(tmp_path: Path) -> str:
    repo = tmp_path / "base"; repo.mkdir()
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t"); _git(str(repo), "config", "user.name", "t")
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(str(repo), "add", "-A"); _git(str(repo), "commit", "-qm", "seed")
    return str(repo)

def test_open_worktree_seeds_real_repo_and_collect_is_authoritative(base_repo: str) -> None:
    p = GitWorktreeProvider(base_repo=base_repo)
    try:
        sb = p.open_worktree("L1")
        # real git repo: the seeded file is present and on a real branch
        assert "def add" in sb.read("/calc.py").file_data["content"]  # adapt to backend read API
        assert sb.execute("git rev-parse --abbrev-ref HEAD").output.strip() == "leaf/L1"
        sb.write("/calc.py", "def add(a, b):\n    return a + b\n")     # the leaf's real edit
        changeset = p.collect("L1")
        assert changeset == {"/calc.py": "def add(a, b):\n    return a + b\n"}  # authoritative diff
        sb.close()                                                     # on_close -> teardown
        assert "L1" not in p._worktrees
    finally:
        p.cleanup_all()

def test_open_worktree_is_idempotent_for_same_leaf_id(base_repo: str) -> None:
    p = GitWorktreeProvider(base_repo=base_repo)
    try:
        sb1 = p.open_worktree("L1"); sb1.write("/x.py", "1\n")
        sb2 = p.open_worktree("L1")   # stale reclaimed, fresh worktree (no collision)
        assert p.collect("L1") == {}  # fresh: nothing written yet
        sb2.close()
    finally:
        p.cleanup_all()

def test_non_git_base_repo_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(GitWorktreeError):
        GitWorktreeProvider(base_repo=str(tmp_path))
```
> 注:`sb.read(...)` 的确切返回结构按 `SandboxBackendProtocol`/`LocalSubprocessSandbox.read` 现状 pin(执行期对照 `_local_subprocess.py` read 返回的 `ReadResult.file_data`)。
- [ ] **Step 2:** FAIL(模块不存在)。
- [ ] **Step 3:** 实现 `_git_worktree.py`(完整 docstring + `GitWorktreeError` + 上述契约;git 调用走 `subprocess.run([...], capture_output=True)`,非零且非预期 → `GitWorktreeError(f"...: {stderr}")`)。
- [ ] **Step 4:** pytest + ruff + pyright(strict)绿。
- [ ] **Step 5:** commit `feat(worktree): real GitWorktreeProvider — git worktree add/diff/teardown, idempotent + exception-safe`。

---

## Task 3: `PullRequestProvider` 协议 + `LocalPullRequestProvider`(A3 + R1 幂等 host finalization + R9)

**Files:** Create `src/langchain_dynamic_workflow/_pull_request.py`;Test `tests/unit/test_pull_request.py`

**契约:** `PullRequestRef`(frozen dataclass:`number:int`,`branch:str`,`url:str`,`integration_branch:str`,`created:bool`)。`PullRequestProvider`(Protocol):`open(*, branch, title, body, integration_branch) -> PullRequestRef`。`LocalPullRequestProvider`:进程内 `dict[branch, PullRequestRef]`;`open` **幂等**——同 branch 已存在则返回原 ref(`created=False`),否则新建递增 number(`created=True`)、`url=f"local://pr/{number}"`。供测试 + 离线 demo + host finalization。真 `gh` 实现作示例 docstring 轮廓(不在引擎必交,R9)。

- [ ] **Step 1(失败测试):**
```python
# tests/unit/test_pull_request.py
from langchain_dynamic_workflow import LocalPullRequestProvider

def test_open_records_and_is_idempotent() -> None:
    p = LocalPullRequestProvider()
    r1 = p.open(branch="leaf/x", title="t", body="b", integration_branch="ldw/integration")
    assert r1.created and r1.number == 1 and r1.branch == "leaf/x"
    r2 = p.open(branch="leaf/x", title="t2", body="b2", integration_branch="ldw/integration")
    assert not r2.created and r2.number == 1   # same branch -> same PR, no dup
    r3 = p.open(branch="leaf/y", title="t", body="b", integration_branch="ldw/integration")
    assert r3.created and r3.number == 2
```
- [ ] **Step 2:** FAIL。 **Step 3:** 实现。 **Step 4:** pytest + ruff + pyright。 **Step 5:** commit `feat(pr): PullRequestProvider seam + idempotent LocalPullRequestProvider`。

---

## Task 4: `SandboxManager` 接 git provider + git 阻塞 thread-offload 出锁(R2 + R8)

**Files:** Modify `src/langchain_dynamic_workflow/_sandbox.py`;Test `tests/integration/test_git_worktree_swarm.py`(并入)

**契约:** `__init__` 增 keyword-only `git_worktree_provider: GitWorktreeProvider | None = None`(与内存 `worktree_provider` 并存,git 优先用于 worktree 叶)。`_new_sandbox(leaf_id, isolation)`:`isolation=="worktree"` 且 `git_worktree_provider` 非空 → `git_worktree_provider.open_worktree(leaf_id)`(其返回的后端已带 `on_close=teardown`,故现有 `_close_backend→close()` 的所有 teardown 路径**自动**移除 worktree,**无需新 manager 钩子**);否则现有内存分支。

**R8 thread-offload:** `open_worktree` 起 `git` 子进程(阻塞)。**不得**在持 `self._slot_freed` condition 锁时同步调用。`lease` restructure:在 condition 锁内只做 quota/backpressure 决策与"该 leaf 槽是否存在"判断;**真正构造后端(慢)移出锁,经 `await asyncio.to_thread(self._new_sandbox, leaf_id, isolation)`**;用一个 `self._pending: set[str]` 去重并发同-leaf_id 创建(锁内标记 pending+释放→to_thread 创建→重入锁安装 slot/清 pending/notify;若期间他者已装入同键则 close 自己刚建的)。内存路径(快)同样走 to_thread 无害。

- [ ] **Step 1(失败测试,test_git_worktree_swarm.py):** 给 `SandboxManager(git_worktree_provider=GitWorktreeProvider(base_repo=...))`,`async with manager.lease(leaf_id="L1", needs_execution=True, isolation="worktree") as b:` 产出的后端 `b.execute("git rev-parse --abbrev-ref HEAD")` == `leaf/L1`;退出 lease 后 `manager.stop("L1")` → 该 worktree 被移除(provider 不再持有)。两并行 lease(L1/L2)各自分支、互不可见(L1 写文件 L2 `ls` 不含)。
- [ ] **Step 2:** FAIL。 **Step 3:** 实现(含 to_thread restructure + `_pending` 去重)。 **Step 4:** pytest + ruff + pyright;**`uv run lint-imports`**(确认无新违规:`_sandbox` 可依赖 `_git_worktree`)。 **Step 5:** commit `feat(sandbox): wire GitWorktreeProvider for worktree leaves; thread-offload blocking git out of the lease lock`。

---

## Task 5: 引擎 `leaf_task` 对 git-worktree 叶 collect 权威(R5)

**Files:** Modify `src/langchain_dynamic_workflow/_engine.py`;Test `tests/integration/test_git_worktree_swarm.py`(并入)

**契约:** git-worktree 执行叶在 `_invoke()` 后、**仍在 lease 内**(worktree 未 teardown),引擎调 `git_worktree_provider.collect(leaf_id)` 取真 diff,折进 `outcome.state` 的权威字段(键名如 `worktree_changeset`),使其随 `LeafOutcome.to_payload()` **journaled**。`Ctx.agent` 侧:当 schema 绑定且 leaf 走 git-worktree 时,以 collect 的真 diff 为权威文件来源(模型自报的 `files` 仅作 fallback/被覆盖;`summary` 等元数据保留)。**执行期 pin:** `leaf_task` 如何拿到 `git_worktree_provider`(经闭包/engine 装配传入)+ `outcome.state` 折入点 + `_context.py` 的 `fold_structured`/`fold_result` 如何消费 `worktree_changeset`(对照 `_context.py:601-616`、`_leaf`/`LeafOutcome:44-120`)。**此 Task 是全计划最高风险接入点,落地后须 Codex + in-house 双评审重点盯。**

- [ ] **Step 1(失败测试):** 一个 deterministic git-worktree fixer 在沙箱真写文件、但其返回的 schema `Patch.files` 故意**与磁盘不符**(写 `a+b` 但自报 `a*b`)→ 断言 `ctx.agent` 拿到的权威变更集是磁盘真值(`a+b`),非模型自报(`a*b`)。这是 R5 的回归护栏(镜像 M5"据真 exit code")。
- [ ] **Step 2:** FAIL。 **Step 3:** 实现。 **Step 4:** pytest + ruff + pyright。 **Step 5:** commit `feat(engine): git-worktree leaf result uses the real git diff as authoritative changeset`。

---

## Task 6: 包导出 + reasoning-only roster 边界守护(R6)

**Files:** Modify `src/langchain_dynamic_workflow/__init__.py`;Test `tests/unit/test_exports.py`(并入)

- [ ] **Step 1:** `__init__.py` 短路导出 `GitWorktreeProvider`、`PullRequestProvider`、`PullRequestRef`、`LocalPullRequestProvider`(base 安装零新依赖——纯 stdlib subprocess)。
- [ ] **Step 2(测试):** `from langchain_dynamic_workflow import GitWorktreeProvider, PullRequestProvider, PullRequestRef, LocalPullRequestProvider` 成功;`__all__` 含之。
- [ ] **Step 3(R6 守护,文档+断言):** 在 SKILL.md / 安全节明确:真 git 执行叶(`needs_execution=True`、`isolation="worktree"`)**只**在 host-trusted roster;untrusted authored-script(AST-gate)路径用 reasoning-only roster(无此类叶)——比照 demo `make_reasoning_roster`。**不**主张"gated 脚本够不到 git"作安全论据。
- [ ] **Step 4:** pytest + ruff + pyright + `uv run lint-imports`。 **Step 5:** commit `feat: export real-git surface; document the trusted-roster execution boundary`。

---

## Task 7: 集成测试 — swarm 隔离 + scratch-repo 真 merge 冲突循环 + resume(P1 + R7 + 验收门)

**Files:** Test `tests/integration/test_git_worktree_swarm.py`(完成)

**scratch-repo merge 机制(R7,在 merge 叶的 deterministic builder 内,自给自足、journaled):** 给定 `base`(文件内容)、`ours`(integrated_tree)、`theirs`(patch),merge builder:`git init` 一个一次性 temp repo → 写 base/`commit` → 新分支写 ours/`commit` → 回 base 新分支写 theirs/`commit` → `git merge` ours↔theirs → 真冲突(退出非零 + 带标记文件)或干净(merged tree)。返回 `MergeResult{clean: bool, files: dict[str,str], conflicts: dict[str,str]}`。**纯函数式**(从输入重建,无持久态)→ resume-safe。

- [ ] **Step 1(测试):**
  - ① **隔离**:`parallel` 两 worktree fixer(git provider)各改不同文件 → 各自分支、collect 互不含对方改动。
  - ② **merge 干净**:两 patch 改不同文件 → scratch-repo merge clean → integrated_tree 含两改动。
  - ③ **冲突循环**(头条):两 patch 改**同一文件重叠区** → scratch-repo merge 真冲突 → resolver 叶(deterministic fake 解决标记)→ 再 merge 干净 → 断言冲突路径**确被走到**(非 clean 短路)且最终 integrated_tree 是解决后内容。
  - ④ **resume**:同 journal 跨两次 `run_workflow`,worktree/merge 叶第二次命中缓存(journal 短路),无真 git 重跑(注入计数 fake 证 0 次重跑),patch/integrated_tree 还原。
- [ ] **Step 2–4:** 实现脚本(`orchestrate` 含 fix→review→integrate fold 冲突循环)使各测试过;ruff + pyright。 **Step 5:** commit `test(worktree): real-git swarm isolation + scratch-repo merge conflict loop + resume`。

---

## Task 8: 离线 feature demo `examples/features/git_worktree.py`(D + R10 production-grade)

**Files:** Create `examples/features/git_worktree.py`;Modify `examples/AGENTS.md`、根 `README.md`、`SKILL.md`

- [ ] **Step 1:** 写离线 feature demo:`GitWorktreeProvider` 指向一个**测试期现建的真 temp git 仓库**(git 在 PATH,离线零 API key);deterministic fakes 作 fixer(读真 worktree、真改、返回 summary)+ conflict resolver;脚本 fix→review→integrate(scratch-repo merge,**含冲突分支**)→ `LocalPullRequestProvider` host finalization。**结尾断言机制**:每 fixer 隔离、collect 权威、冲突路径被走到并解决、PR 幂等落记录。守 examples/AGENTS.md 道/术线 + skeleton 约定。
- [ ] **Step 2:** `uv run python -m examples.features.git_worktree` → OK + 断言过。
- [ ] **Step 3:** `examples/AGENTS.md` §2 索引加一行(17 features)+ learning path 第 8 组补;根 `README.md` Examples 指针同步;`SKILL.md` 增"真 git worktree + 冲突循环"范式段(道:何时用、控制流反转、脚本拥有冲突循环)。
- [ ] **Step 4:** ruff + pyright。 **Step 5:** commit `docs(examples): offline real-git worktree fix-swarm feature demo + SKILL pattern`。

---

## Task 9: 双轨 demo-app 消费切片 — `refactor_swarm`(E)

**Files:** `demo-app/backend/{workflows.py,host_graph.py,ui_adapter.py,_models.py}`;`demo-app/frontend/src/components/workflow/{MergeCard,PullRequestCard}.tsx,registry.ts`;`demo-app/{scenarios.json,...ScenarioPanel.tsx,README.md}`;`demo-app/backend/tests/test_refactor_swarm.py`

- [ ] **Step 1(后端 workflow,TDD):** `refactor_swarm(ctx, args)`:扇出 git fixer(worktree,真改+真测试)→ 2-vote verify(read-only judge)→ integrate fold(scratch-repo merge 冲突循环,resolver 叶)→ 返回 `integrated_tree` + PR 意图。注册进 `make_workflows()`。roster 加 `git_fixer`(needs_execution worktree)+ `conflict_resolver`(reasoning)。**冲突循环 production-grade**(真断言)。
- [ ] **Step 2(host finalization,R1):** `host_graph` 在 `run_workflow` 返回后**幂等**调 `LocalPullRequestProvider.open(...)`(移出 replay);发 `MergeCard`(per-merge clean/conflict/resolved)+ `PullRequestCard`(PR ref)Gen-UI 卡。**leaf-quarantine 守 UI 层**(`streamSubgraphs:false`,leaf 活动只作 host-channel 卡,见 memory)。复用 M5 `TerminalCard` 显示真命令。
- [ ] **Step 3(前端):** `MergeCard.tsx`/`PullRequestCard.tsx`(3-态:running/clean-or-conflict/resolved;PR ref + 分支)+ `registry.ts` 注册。
- [ ] **Step 4(场景 + doc-sync):** scenarios.json + ScenarioPanel.tsx + README.md 加 "Refactor swarm" 场景(**字节一致**,doc-sync 测试 count +1);更新 doc-sync 测试期望数。
- [ ] **Step 5(测试):** `test_refactor_swarm.py` 离线消费测试(扇出隔离 + 冲突循环 + PR 幂等 finalization);`uvx pyright`(demo-app backend)+ demo-app pytest 绿;前端 build。 **Step 6:** commit `feat(demo): refactor_swarm — real-git fix swarm + merge conflict loop + PR card (M6)`。

---

## Task 10: per-gap 真模型 E2E(gated,**验收时亲手真跑**,守 memory)

**Files:** Create `demo-app/backend/tests/test_m6_refactor_swarm_real.py`(`skipif(not LDW_DEMO_REAL_MODEL)`)

- [ ] **Step 1:** gated 真模型 E2E:测试期现建真 temp git 仓库植 2-3 bug(其中两个 fixer 改同一文件造**真冲突**);真模型 git fixer 在真 worktree 修 + 真跑测试 + collect 权威 diff;真模型 resolver 解决真冲突;本地 merge 进 integration;`LocalPullRequestProvider` finalization。断言:每 fixer 隔离、冲突路径真被走到并由真模型解决、integrated 含全部修复。
- [ ] **Step 2(验收必做):** worktree `uv sync --group example` + 根 `.env` 复制进 worktree;`LDW_DEMO_REAL_MODEL=<model> uv run --group example pytest demo-app/backend/tests/test_m6_refactor_swarm_real.py -q`(**LangSmith tracing 保持开**,守 memory)。设计"smoking gun"使真模型走头条冲突路径(非 fallback)。
- [ ] **Step 3:** commit `test(m6): gated real-model refactor-swarm E2E (conflict path)`。

---

## Task 11: evergreen 同步 + Decision Log

**Files:** `design_docs/01-engine-mechanism.md`、`02-architecture.md`、`uml/{01,02,03}`、`v0_3_0_plans/00-roadmap.md`

- [ ] **Step 1:** `01` sandbox/worktree 节补真 git 语义(`GitWorktreeProvider`/`LocalSubprocessSandbox(root=)`/collect 权威/scratch-repo merge);`02` 接线节补 `git_worktree_provider`/`PullRequestProvider`/host finalization;`uml/02-class` 加新类、`uml/03-sequence` 加 M6 refactor-swarm 时序图;Decision Log 增 **D-M6**(方案 1 + R1–R10 关键:PR 移出 replay、collect 权威、scratch-repo merge、teardown 绑 close)。
- [ ] **Step 2:** roadmap `00-roadmap.md`:M6 行 `待写`→`✅ 已落地`;状态节 + 执行序列更新。
- [ ] **Step 3:** commit `docs(design): sync evergreen + roadmap + Decision Log D-M6`。

---

## Task 12: 全门(独立亲跑,守 memory `independently-verify-gate-claims` + `ruff-format-check-whole-repo-matches-ci`)

- [ ] **根仓库**(worktree root):`uv run ruff check .` → `uv run ruff format --check .`(**整树**)→ `uv run pyright` → `uv run lint-imports` → `uv run pytest -q`(全绿,日志落 `/tmp/m6-*.log`)。
- [ ] **demo-app**:`cd demo-app && uvx pyright`(backend)+ demo-app pytest + 前端 build/lint。
- [ ] **真 E2E 真跑**(Task 10 Step 2)已过、非 skip。
- [ ] **跨模型评审**:落地后代码交 Codex 一轮(service-tier strip 配方)+ in-house 对抗评审;两边 findings 全修 + 回归。
- [ ] commit(如有门相关调整);开 PR(`github-pr` skill,**非** gstack /ship)。

---

## Decision Log(本计划新增)

| # | 决策 | 选择与理由 |
|---|---|---|
| **D-M6** | M6 真 git 落地形态 | **方案 1**(git provider 拥有 worktree-rooted 后端;merge/冲突循环走 journaled 叶 + 脚本变量 fold;PR 作 seam),折入 Codex 评审 R1–R10。关键:① **PR/integration 物化移出确定性 replay 作幂等 host finalization**(BLOCKER R1——副作用须在叶边界或 replay 之外);② **worktree 叶权威变更集 = leaf task 内真 `git diff`**(R5,非模型自报,镜像 M5);③ **整合用 merge 叶内一次性 scratch-repo 真 `git merge`**(R7,比 merge-file 忠于 branch 语义且 resume-safe);④ teardown **绑后端 `close()`**(R2 `on_close`,复用现有所有 teardown 路径,无新 manager 钩子)+ `cleanup_all()` 兜底;⑤ `open_worktree` **幂等**(R4 同键 reclaim)+ **异常安全**(R3 回滚);⑥ git 阻塞 **thread-offload 出 lease 锁**(R8);⑦ 安全据 **reasoning-only roster 边界 + ExecPolicy**(R6,撤"脚本够不到 git"错误论据);⑧ `GhPullRequestProvider` 降级为示例(R9)。否决方案 2(统一生命周期契约——改稳定 seam、共享 integration 工作区与 per-leaf 隔离 + resume 冲突)、方案 3(冲突在单叶内解决——放弃"脚本拥有循环"命门)。 |

---

## 执行序列与编排(master-orchestrator)

```
引擎核心(顺序 TDD,紧耦合,一致落地)  T1 → T2 → T3 → T4 → T5 → T6 → T7
   └─ 跨模型评审①(Codex + in-house)over 引擎核心 → 全修 → 回归
双轨消费 + 示例(承引擎)            T8(离线 demo)∥ T9(demo-app)
验收                               T10(真 E2E 亲跑)→ T11(evergreen)→ T12(全门 + 评审② + PR)
```

**per-gap 交付清单(roadmap §53):** ① 完整 TDD 全门绿;② 真模型 E2E 亲跑(非 skip);③ user-facing 集成示例(`examples/features/git_worktree.py` + demo-app `refactor_swarm`);④ 跨模型 Codex 评审;⑤ evergreen 同步。**独立验证**:不信任何 subagent 自报 green,全门 + 真 E2E 亲跑、findings 亲核。

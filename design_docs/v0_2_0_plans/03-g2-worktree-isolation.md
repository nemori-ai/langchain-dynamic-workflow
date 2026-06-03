# G2 — `isolation='worktree'` 真实语义 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `isolation='worktree'` 不再只是缓存分区标签，而兑现 CC 的真实语义——**每个改文件的 fix 叶跑在自己的、从 base 快照播种的隔离可变工作副本里**，改完**返回 patch**，由编排层 review/合并。把"生成"（fix 叶产 patch）与"应用"分离，镜像社区旗舰用例（Bun 迁移 phase-h：2 hunter/file → worktree fix 返回 patch → 2-vote review）。

**Architecture（决策 D-G2 已锁）：** v1 = **B（内存播种副本）作默认语义** + **A（真 git worktree FS 后端）作同一 `WorktreeProvider` seam 后的可插拔生产实现**。现状：执行叶**本就 per-leaf 隔离**（`SandboxManager` find-or-create 一个 `InMemorySandbox(identity=leaf_id)`，空 `_files`、`execute` 为 no-op echo）——隔离已有，**缺的是**：① 从 base 快照**播种**沙箱（让叶子有文件可改）；② **变更收集**（diff vs seed）/ 叶子复用 G1 `schema=` **返回 `Patch`**；③ 一个 `WorktreeProvider` seam，内存播种为默认、真 git-worktree 为文档化可插拔实现。base 快照经**构造期注入**（引擎/roster 装配），**不进 gated 脚本**（A1 边界）。

**Tech Stack:** Python 3.12（async-first）、现有 `InMemorySandbox`/`SandboxManager`（`_sandbox.py`）、G1 的 `agent(schema=Patch)`、pytest + pytest-asyncio、ruff、pyright(strict)。

**依赖：** 可独立于 G1，但结果回填**复用 G1 `schema=`**（fix 叶返回 `Patch`）。建议在 G1/G3 之后做。

---

## 已核实的 sandbox 表面（开工前已证）

| 事实 | 出处 |
|---|---|
| `InMemorySandbox(SandboxBackendProtocol)`，持 `self._files: dict[str, FileData]`；有 `write/read/edit/ls/grep/glob/upload_files/download_files/execute(no-op echo)` | `_sandbox.py:145-385` |
| `upload_files(list[tuple[str, bytes]])` 可批量注入文件 → **播种入口** | `_sandbox.py:322` |
| `SandboxManager.lease(*, leaf_id, needs_execution) -> AsyncGenerator[BackendProtocol]`；slot 缺失时 `InMemorySandbox(identity=leaf_id)` 新建 | `_sandbox.py:535-585` |
| `leaf_id_from_key(journal_key)` 派生稳定叶身份；`isolation` 已入 journal key（`_context.py`、`_journal.py`） | `_sandbox.py:126`、`_context.py` |
| 当前 `isolation="worktree"` **仅**改缓存键 → 不同 `leaf_id` → 不同空沙箱，**无播种、无变更收集** | （gap 本质） |

## 文件结构（本计划触达）

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/langchain_dynamic_workflow/_worktree.py` | `WorktreeProvider` 协议 + `InMemoryWorktreeProvider`（base 快照播种 + diff 收集） | **新建** |
| `src/langchain_dynamic_workflow/_sandbox.py` | `SandboxManager` 持 `worktree_provider`；`lease(..., isolation=)` 在 `"worktree"` 且 slot 新建时播种 | 修改 |
| `src/langchain_dynamic_workflow/_engine.py` | `leaf_task` 把 `isolation` 透传给 `lease`；（可选）叶后收集 diff 注入结果 | 修改 |
| `src/langchain_dynamic_workflow/_context.py` | `LeafRunner` / `agent()` 把 `isolation` 透传到引擎侧（确认现有透传链） | 修改（按需） |
| `src/langchain_dynamic_workflow/__init__.py` | 短路导出 `WorktreeProvider` / `InMemoryWorktreeProvider` | 修改 |
| `tests/unit/test_worktree.py` | provider 播种 + diff 收集单测 | 新建 |
| `tests/integration/test_worktree_isolation.py` | 两并行 worktree 叶各从同一 base 出发、互不可见；resume 命中缓存不重跑 | 新建 |
| `examples/10_worktree_fix_swarm.py` | **真E2E验收**：2-3 文件小库播种，2 fix 叶并行各在隔离 worktree 改各自文件返回 `Patch`，reviewer 2-vote | 新建 |
| `src/.../skills/dynamic-workflow/SKILL.md` | 文档 `isolation="worktree"`（何时用：并行改文件才用，只读/synth 不用） | 修改 |
| `design_docs/01-engine-mechanism.md` / `02-architecture.md` / `uml/` | evergreen：sandbox 节补 worktree 语义与 `WorktreeProvider` seam；Decision Log 增 D-G2 | 修改 |

---

## 前置：分支

- [ ] **创建特性分支**

```bash
git checkout -b feat/g2-worktree-isolation
```

## Task 1: `WorktreeProvider` seam + 内存播种 provider（先写测试）

**Files:**
- Create: `src/langchain_dynamic_workflow/_worktree.py`
- Test: `tests/unit/test_worktree.py`

- [ ] **Step 1（失败测试）:** 断言 `InMemoryWorktreeProvider(base)` 的 `seed(leaf_id)` 返回 base 快照拷贝；改动后 `collect(leaf_id, files)` 返回相对 seed 的变更集（新增/修改的 path→content，未变的不列）。

```python
# tests/unit/test_worktree.py
"""Unit tests for the in-memory worktree provider (seed + diff collection)."""

from __future__ import annotations

from langchain_dynamic_workflow import InMemoryWorktreeProvider


def test_seed_returns_base_snapshot_copy() -> None:
    base = {"/a.py": "print(1)\n", "/b.py": "x = 2\n"}
    provider = InMemoryWorktreeProvider(base)
    seeded = provider.seed("leaf-1")
    assert seeded == base
    seeded["/a.py"] = "mutated"  # caller mutation must not leak into base
    assert provider.seed("leaf-1")["/a.py"] == "print(1)\n"


def test_collect_returns_changeset_vs_seed() -> None:
    base = {"/a.py": "print(1)\n", "/b.py": "x = 2\n"}
    provider = InMemoryWorktreeProvider(base)
    after = {"/a.py": "print(2)\n", "/b.py": "x = 2\n", "/c.py": "new\n"}
    changes = provider.collect("leaf-1", after)
    # changed + added only; unchanged /b.py excluded.
    assert changes == {"/a.py": "print(2)\n", "/c.py": "new\n"}
```

- [ ] **Step 2:** 运行确认失败（无 `_worktree.py`）。
- [ ] **Step 3: 实现**

```python
# src/langchain_dynamic_workflow/_worktree.py
"""Worktree isolation backends — seed a leaf's mutable working copy and collect its changes.

isolation="worktree" gives each file-mutating leaf its own copy of a base snapshot to
edit; the changeset (relative to the seed) is collected afterward so the orchestration
layer can review and merge patches. The in-memory provider is the offline default; a
real git-worktree FS backend is a pluggable implementation behind the same seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class WorktreeProvider(Protocol):
    """Seeds a leaf's worktree and collects its changeset relative to the seed."""

    def seed(self, leaf_id: str) -> Mapping[str, str]:
        """Files to populate a worktree leaf's sandbox with before it runs."""
        ...

    def collect(self, leaf_id: str, files: Mapping[str, str]) -> dict[str, str]:
        """The leaf's changeset: paths that were added or modified vs the seed."""
        ...


class InMemoryWorktreeProvider:
    """A worktree provider backed by an in-memory base snapshot (offline default)."""

    def __init__(self, base_files: Mapping[str, str]) -> None:
        self._base: dict[str, str] = dict(base_files)

    def seed(self, leaf_id: str) -> Mapping[str, str]:
        return dict(self._base)  # fresh copy per leaf; caller cannot mutate the base

    def collect(self, leaf_id: str, files: Mapping[str, str]) -> dict[str, str]:
        return {
            path: content
            for path, content in sorted(files.items())
            if self._base.get(path) != content
        }
```

- [ ] **Step 4:** 运行确认通过；ruff/pyright；commit `feat(worktree): WorktreeProvider seam + in-memory seeded provider`。

## Task 2: `SandboxManager` 播种 worktree 叶

**Files:**
- Modify: `src/langchain_dynamic_workflow/_sandbox.py`

- [ ] **Step 1（失败测试，并入 `tests/integration/test_worktree_isolation.py`）:** 给 `SandboxManager(worktree_provider=InMemoryWorktreeProvider(base))`，`lease(leaf_id="L1", needs_execution=True, isolation="worktree")` 产出的沙箱 `ls()` 含 base 文件；两个不同 `leaf_id` 各得独立播种副本，一个写入对另一个不可见。
- [ ] **Step 2:** 运行确认失败（`lease` 尚无 `isolation` 参数 / 不播种）。
- [ ] **Step 3: 实现** — `SandboxManager.__init__` 增 `worktree_provider: WorktreeProvider | None = None`；`lease` 签名增 `isolation: str = "shared"`；slot **新建**时若 `isolation == "worktree"` 且 provider 非空，用 `sandbox.upload_files([(p, c.encode("utf-8")) for p, c in provider.seed(leaf_id).items()])` 播种。`"shared"` 维持现状（空沙箱）。**注意**：仅在 slot 新建分支播种（find-or-create 命中已存在沙箱时不重播，保 retry 工作区稳定）。
- [ ] **Step 4:** 运行确认通过；ruff/pyright；commit `feat(sandbox): seed worktree leaves from a base snapshot via WorktreeProvider`。

## Task 3: 引擎透传 `isolation` + 变更收集

**Files:**
- Modify: `src/langchain_dynamic_workflow/_engine.py`（leaf_task）、按需 `_context.py`

- [ ] **Step 1（执行时先核实）:** 读 `_engine.py` 的 `leaf_task` 当前 `lease(leaf_id=..., needs_execution=...)` 调用点，确认 `isolation` 透传链（`agent()` 已把 isolation 折进 journal key；需把原始 `isolation` 值一并传到 `leaf_task` → `lease`）。**此处确切签名在执行时对照源码 pin。**
- [ ] **Step 2（失败测试）:** 一个 worktree fix 叶改了沙箱里两个文件 → 叶后引擎经 `provider.collect(leaf_id, sandbox_files)` 得到含两条 path 的变更集（或叶子经 G1 `schema=Patch` 自报）。
- [ ] **Step 3: 实现** — `leaf_task` 把 `isolation` 传给 `lease`；叶执行完，若 `isolation=="worktree"`，从沙箱 `ls()`/`read()` 取当前 `_files` 快照，经 `provider.collect` 算变更集。**结果回填二选一（执行时定）：** (a) 叶子用 G1 `schema=Patch{files:[{path,diff}], summary}` 自报（推荐，落实"生成与应用分离"）；(b) 引擎侧把 collect 出的变更集注入结果。优先 (a)——更贴 CC 语义且复用 G1。
- [ ] **Step 4:** 运行确认通过；ruff/pyright；commit `feat(engine): thread isolation to lease; collect worktree changeset`。

## Task 4: 确定性 / resume

**Files:**
- Test: `tests/integration/test_worktree_isolation.py`

- [ ] **Step 1（失败测试）:** 同 journal 跨两次 run，worktree 叶第二次命中缓存、不重跑（`leaf_id` 派生不变，`isolation="worktree"` 已入键），patch 还原。
- [ ] **Step 2:** 实现/确认通过（多半已成立——isolation 早入键；本任务是回归护栏）。
- [ ] **Step 3:** commit `test(worktree): resume hits cache; worktree leaf identity stable`。

## Task 5: 生产 seam（A 方案，文档化可插拔，不在 v1 必交）

- [ ] 在 `_worktree.py` 文档化"真 git-worktree FS provider"的实现轮廓（配 base repo 路径时 `git worktree add <tmp>` per 叶、FilesystemBackend 根于此、叶可真跑 git/build/test、完事 `git diff` 收集、worktree 删除），作为 `WorktreeProvider` 的另一实现。**v1 不必交**，留 seam + 文档（与 A1/offline-first 一致；有真 FS/key 的生产环境再启用）。

## Task 6: SKILL.md + 示例 + 真模型 E2E 验收

**Files:**
- Modify: `src/.../skills/dynamic-workflow/SKILL.md`
- Create: `examples/10_worktree_fix_swarm.py`

- [ ] **Step 1:** SKILL.md 文档 `isolation="worktree"`——**何时用**：并行改文件才用，只读/synth 叶不用（社区纪律）。
- [ ] **Step 2:** 写 `examples/10_worktree_fix_swarm.py`：2-3 文件小代码库（各植入 1 bug）经 `InMemoryWorktreeProvider` 播种；2 个 fix 叶**并行、各在隔离 worktree** 改各自文件、返回 `Patch{files:[{path,diff}], summary}`（G1 schema）；reviewer 2-vote（判定叶用 G4 `read_only_builder`）。
- [ ] **Step 3: 主循环真跑**

```bash
LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5 uv run --group example python examples/10_worktree_fix_swarm.py
```

- [ ] **Step 4: 观察 + 断言**：每个 patch 只动自己那个文件（隔离成立）；两 patch 的改动互不可见；reviewer 通过的 patch 确实针对植入 bug。这是 G2 的真实验收门（镜像 Bun phase-h）。

## Task 7: evergreen 同步 + Decision Log

- [ ] `01-engine-mechanism.md` 的 sandbox 节 + `uml/` 补 worktree 语义与 `WorktreeProvider` seam；`02-architecture.md` 注 isolation 行为差异；**Decision Log 增 D-G2**（下方）。commit。

## Task 8: 质量闸门

- [ ] `uv run pytest -q` 全绿；`ruff`/`ruff format --check`/`pyright`/`lint-imports` 全过；commit（如有配置调整）。

---

## Decision Log（本计划新增）

| # | 决策 | 选择与理由 |
|---|---|---|
| **D-G2** | `isolation="worktree"` v1 保真度 | **B（内存播种副本）作默认 + A（真 git-worktree FS）作同一 `WorktreeProvider` seam 后的可插拔实现**。否决"仅文档化 seam"（C，不兑现卖点）与"v1 直接上真 git worktree"（A 单选，与 A1/offline-first 跨度大、重工程）。理由：兑现 worktree 核心语义（隔离副本 + patch 回收）而不被真 FS/git 卡住；与 D4（执行器可替换 seam）同构。结果回填复用 G1 `schema=Patch`，落实"生成与应用分离"。 |

## 待核实（执行时）

1. `_engine.py` `leaf_task` 的 `lease(...)` 确切调用点与 `isolation` 透传链（Task 3 Step 1 pin）。
2. 内存播种副本的 diff 粒度（行级 vs 全文替换）够不够 review 用——v1 用全文变更集（`collect` 返回新内容），行级 diff 留给 fix 叶的 `Patch.files[].diff` 字段（叶自产）。
3. base 快照注入点（引擎装配 vs roster 条目元数据）——v1 走引擎/SandboxManager 构造期注入（不进 gated 脚本，守 A1）。

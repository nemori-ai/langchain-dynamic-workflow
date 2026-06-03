# Phase 4 · Sandbox（per-leaf 隔离与生命周期）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans。步骤用 `- [ ]` 跟踪。

**Goal:** `needs_execution` 叶子获得隔离 backend（身份从 journal key 派生）；并行执行叶子互不串味；纯推理叶子不分配 sandbox；SandboxManager 自管 TTL/池/配额/stop。

**Architecture:** 弃 deprecated `BackendFactory`，改 **per-leaf backend 实例**构造（从 `RunnableConfig` thread_id 读身份、find-or-create 打标 backend、传给 `create_deep_agent`）；SandboxManager 自管生命周期；CompositeBackend `/shared/` hand-off（路径规范化）。详见 [../01-engine-mechanism.md](../01-engine-mechanism.md) §8。

**Tech Stack:** Phase 1–3 栈 + deepagents backends（`StateBackend`/`CompositeBackend`/`FilesystemBackend` 等，测试用 local/state backend，不依赖真实 sandbox 基础设施）。

---

## 里程碑 M4

per-leaf 隔离 + journal-key 身份 + 分层准入 + SandboxManager 生命周期；绿色 e2e demo（隔离产物 + `/shared/` 交接）。

## 接口设计

```python
class SandboxManager:
    def acquire(self, *, leaf_id: str, needs_execution: bool) -> BackendProtocol  # 复用或 lazy-create
    async def stop(self, leaf_id: str) -> None
    # 内部:idle/硬 TTL 回收、最大活跃数配额、池耗尽背压排队
# leaf_id 由 journal key 派生 → retry/resume 稳定、唯一、与 dedup 自洽
```

**关键设计要点**：
- **默认 per-leaf 隔离**；身份从 journal key 派生（retry/resume 稳定）。
- **构造方式**：per-leaf backend **实例**（非 deprecated callable factory，后者 0.7.0 移除）。
- **分层准入**：`needs_execution=True` → 分配隔离 sandbox；`False` → `StateBackend`、**不分配**。
- SandboxManager 自管：lazy-create / idle+硬 TTL / 池化 / 最大活跃数配额 / 池耗尽背压 / `stop()`。
- CompositeBackend：`/shared/` → 共享产物 store（producer 命名空间 + **路由前路径规范化防 `../` 穿越**）；其余 → per-leaf 隔离。
- **#2884 风险**：CompositeBackend route 隔离在共享存储后端间会泄漏 → 并行叶子隔离不能仅靠 routes，须独立验证。

## 验收标准

- [ ] `needs_execution=True` 叶子拿到隔离 backend；两个并行执行叶子写同名文件互不可见（隔离测试，local/state backend）。
- [ ] sandbox 身份从 journal key 派生：retry/resume 时稳定（同叶子 → 同身份 → 同 backend）。
- [ ] `needs_execution=False` 叶子走 StateBackend、不分配 sandbox（断言 manager 未 acquire）。
- [ ] SandboxManager：idle TTL 回收；池耗尽排队（背压）；`stop()` 被调用清理。
- [ ] `/shared/` hand-off：producer 写、consumer 读；`../` 穿越被规范化阻断。
- [ ] ruff/pyright/pytest 全绿。

## 指标

- 隔离测试 ≥1；身份稳定测试 ≥1；分层准入测试 ≥1；TTL/配额测试 ≥1；路径穿越防护测试 ≥1。
- "N 逻辑 agent ≠ N 活跃 sandbox"（纯推理叶子 0 分配）用例验证。

## 文件结构

```
src/langchain_dynamic_workflow/_sandbox.py        # SandboxManager + per-leaf 实例构造 + 身份派生
修改 _leaf.py（分层准入、按 leaf_id acquire backend、传给叶子）、_engine.py（注入 SandboxManager + 生命周期收尾）
tests/unit/test_sandbox.py
tests/integration/test_phase4_sandbox.py
examples/04_sandbox_artifacts.py
```

## 任务分解（bite-sized TDD）

1. **身份派生**：Red（leaf_id 从 journal key 派生、retry/resume 稳定）→ 实现 → Green → commit。
2. **`SandboxManager` 核心**：Red（acquire 复用同 leaf_id、lazy-create、stop 清理）→ 实现 → Green → commit。
3. **TTL/配额/背压**：Red（idle 回收、池耗尽排队、最大活跃数）→ 实现 → Green → commit。
4. **分层准入**：Red（`needs_execution` 决定是否分配；纯推理走 StateBackend 不 acquire）→ 接入 `_leaf` → Green → commit。
5. **CompositeBackend `/shared/`**：Red（producer→consumer 交接、路径规范化防穿越、并行隔离不泄漏）→ 实现 → Green → commit。
6. **集成 + 示例 + 质量门**：`examples/04_sandbox_artifacts.py`；ruff+pyright 清零；里程碑 commit。

## demo 规格（M4）

`examples/04_sandbox_artifacts.py`：两个 `needs_execution` 叶子各在隔离 sandbox 写文件（验证互不可见），再经 `/shared/` 把产物交接给第三个叶子。集成测试用 local/state backend 断言隔离 + 交接 + 身份稳定。

## 对 Phase 1–3 的 refactor

- `_leaf` 调用路径接入 SandboxManager（按 leaf_id acquire/传 backend）。Phase 1–3 全部测试须保持绿（纯推理叶子路径不变）。
- **承接 Phase 2 review（minor #5）**：把 `agent(isolation=...)` 贯通到叶子运行器，让 isolation 模式真正参与 backend 选择（当前它只影响 journal key）。这与本阶段的 per-leaf 隔离/身份派生天然同源，一并闭合 isolation 的 key-vs-execution 口子。

## 交给 Phase 5+ 的点

- sandbox 叶子在 host 后台 workflow 中运行；ResultStore 可复用 sandbox 后端做大结果 offload。

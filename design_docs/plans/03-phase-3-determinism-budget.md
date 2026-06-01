# Phase 3 · 确定性 guard + budget + 进度 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans。步骤用 `- [ ]` 跟踪。

**Goal:** journal-divergence backstop 在重放分叉时 fail-loud；`budget` 共享池可强制、且 `spent()` 重放可重建；`phase()`/`log()` 出进度且重放幂等。

**Architecture:** journal 兼作确定性 oracle（记调用序列，重放不匹配即抛）；budget 由 journal 中每叶子 usage 重建；usage 计量经 `UsageMetadataCallbackHandler` 转发。详见 [../01-engine-mechanism.md](../01-engine-mechanism.md) §5、§10。

**Tech Stack:** Phase 1–2 栈 + `langchain_core.callbacks`（usage 回调）。

---

## 里程碑 M3

确定性 backstop + budget 强制 + 进度叙事，全部重放正确；绿色 e2e demo（loop-until-budget）。

## 接口设计

```python
class Ctx:
    def phase(self, title: str) -> None
    def log(self, message: str) -> None
    @property
    def budget(self) -> Budget          # .total / .spent() / .remaining()

class Budget:
    total: int | None
    def spent(self) -> int              # 由 journal 中累计 usage 重建 → 重放确定
    def remaining(self) -> int          # max(0, total - spent())；无 total → inf

class WorkflowDeterminismError(RuntimeError): ...
class WorkflowBudgetExceededError(RuntimeError): ...
```

**关键设计要点**：
- **确定性重定义**：不禁绝一切非确定性，只在"非确定性改变可观测 `agent()` 调用模式"时炸。journal 记录调用序列；重放时第 k 个 `agent()` 的 key 与记录不符 → `WorkflowDeterminismError`，**绝不喂错位缓存**。
- **budget 重放可重建**：每叶子 usage 连同结果写 journal；`spent()` = 命中+新算的 usage 累计 → resume 时由 journal 重建出与首次相同的累计值（**这是 backstop 不被误触的前提**）。
- `spent() >= total` 后新 `agent()` 抛 `WorkflowBudgetExceededError`；在飞叶子完成并保留结果。
- `budget.total` 守卫：无 total → `remaining()` 为 inf（loop-until-budget 范式可用）。
- `phase`/`log`：输出经引擎收集；重放时已记录的进度不重复投递（幂等）。

## 验收标准

- [ ] 重放时脚本产出的 `agent()` 调用序列与 journal 记录不匹配 → `WorkflowDeterminismError`（清晰信息）。
- [ ] `budget.spent()` 在 resume 时由 journal usage 重建出与首次相同的累计（断言相等）。
- [ ] `spent() >= total` 后新 `agent()` 抛 `WorkflowBudgetExceededError`；在飞结果保留。
- [ ] 无 `total` 时 `remaining()` 为 inf；带 `total` 守卫的 loop-until-budget 正常终止。
- [ ] `phase`/`log` 输出可捕获；重放不重复投递。
- [ ] ruff/pyright/pytest 全绿。

## 指标

- divergence 检出测试 ≥1；budget 重放确定性测试 ≥1；budget-cap 强制测试 ≥1；phase/log 幂等测试 ≥1。
- backstop 把"budget 重放错算"从静默腐坏降级为 loud failure（用例验证）。

## 文件结构

```
src/langchain_dynamic_workflow/_determinism.py   # 调用序列记录 + 重放校验（backstop）
src/langchain_dynamic_workflow/_budget.py         # Budget + usage 重建 + callback 转发
修改 _journal.py（value 增加 usage 字段）、_context.py（phase/log/budget + 序列记录）、_leaf.py（usage 采集 + callback 转发）
tests/unit/test_determinism.py、test_budget.py
tests/integration/test_phase3_loop_until_budget.py
examples/03_loop_until_budget.py
```

## 任务分解（bite-sized TDD）

1. **journal value 扩展**：Red（put/get 带 usage 往返）→ journal value 从纯 result 改为 `{result, usage}` → 迁移 Phase 1/2 测试 → Green → commit。
2. **`_determinism.py` 序列记录**：Red（重放同序列通过、改序列抛 `WorkflowDeterminismError`）→ 实现（journal 记 ordered call-keys；重放比对）→ Green → commit。
3. **`_budget.py`**：Red（`spent()` 由 usage 累计；resume 重建相等；cap 抛 `WorkflowBudgetExceededError`；无 total → inf）→ 实现 + `UsageMetadataCallbackHandler` 转发（**`@task` 直调须复刻 `_build_subagent_config` callbacks 转发**）→ Green → commit。
4. **`phase`/`log`**：Red（输出捕获 + 重放幂等）→ 在 `Ctx` 实现 → Green → commit。
5. **集成 + 示例 + 质量门**：`examples/03_loop_until_budget.py`；ruff+pyright 清零；里程碑 commit。

## demo 规格（M3）

`examples/03_loop_until_budget.py`：`while budget.total and budget.remaining() > THRESHOLD:` 累积式扇出研究叶子，`phase`/`log` 报进度，到顶优雅停。集成测试 fake model 断言：终止、`spent()` 重放相等、cap 强制。

## 对 Phase 1–2 的 refactor

- journal value schema 由 result → `{result, usage}`；更新 Phase 1/2 journal 相关断言。Phase 1/2 全部测试须保持绿。
- **承接 Phase 2 review（minor #5）**：`agent()` 把 `model`/`isolation` 折进 journal key，但 `LeafRunner` 签名只有 `(agent_type, prompt)`，override 从不到达叶子（model= 被静默忽略）。Phase 3 本就要为 usage/budget refactor 叶子路径，顺带把 `LeafRunner`/`leaf_task` 扩成接收并尊重 `model`（把 model override 落到叶子调用上）；`isolation` 的贯通留给 Phase 4（sandbox）。补一条测试：`model=` 不同 → 实际驱动的模型不同（用可区分的 fake leaf 断言），关闭 key-vs-execution 的口子。

## 交给 Phase 4+ 的点

- sandbox 叶子的 usage 同样纳入 budget。
- 确定性 backstop 覆盖 parallel/pipeline 内的调用序列。

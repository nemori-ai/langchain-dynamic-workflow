# Phase 2 · 扇出（parallel + pipeline + 并发闸）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans。步骤用 `- [ ]` 跟踪。

**Goal:** 脚本能 `await ctx.parallel([...])`（barrier）与 `await ctx.pipeline(items, *stages)`（无 barrier 流式）扇出多叶子；并发受显式上限约束；失败语义正确；journal 跨扇出与半途中断仍正确。

**Architecture:** `parallel` = `asyncio.gather` over `@task` future（barrier）；`pipeline` = 每 stage 一个 bounded `asyncio.Queue` + worker 组（无 barrier、背压）；全局 `Semaphore(min(16,cores-2))` 跨 stage 共享 + LangGraph `RunnableConfig.max_concurrency`。详见 [../01-engine-mechanism.md](../01-engine-mechanism.md) §2、§9、§11。

**Tech Stack:** Phase 1 栈 + `asyncio`（Semaphore/Queue/gather）。

---

## 里程碑 M2

`parallel`/`pipeline` 扇出可用、并发有界、失败语义正确、半途中断可 resume；绿色 e2e demo（fake model）。

## 接口设计

```python
class Ctx:
    async def parallel(self, thunks: Sequence[Callable[[], Awaitable[T]]]) -> list[T | None]
    async def pipeline(self, items: Sequence[I], *stages: Stage) -> list[Any | None]
    # Stage = Callable[[prev_result, original_item, index], Awaitable[Any]]
```

**关键设计要点**：
- `parallel` 收 **thunk 列表**（非 promise）；barrier；thunk 抛错 → 该位 `None`，**整体不 raise**（用前 `.filter`）。
- `pipeline` 每 item 独立穿越所有 stage、**stage 间无 barrier**；stage 抛错 → 该 item `None` 跳后续；结果**按输入下标保序**。
- bounded queue 防 item 海啸；全局 semaphore 跨所有 stage 共享并发额度。
- 并发上限**双层显式设**：asyncio `Semaphore` + `RunnableConfig.max_concurrency`（底座默认 None ⇒ 无界，必须显式）。
- journal：parallel/pipeline 内每叶子各自 content-hash 缓存；半途中断 resume → 已完成命中、未完成 live。

## 验收标准

- [ ] `parallel([thunk,...])` 返回按序 list；某 thunk 抛错 → 该位 `None`、调用本身不 raise；`.filter(None)` 后可用。
- [ ] `pipeline` 无 barrier（断言 A 已 stage3 时 B 仍 stage1，用 instrumented 时序）；stage 抛错 → 该 item `None` 跳后续；结果按输入下标保序。
- [ ] 并发上限：instrumented fake 断言最大在飞 ≤ 配置上限。
- [ ] 半途中断后 resume（同 journal）：已完成叶子 0 模型调用、未完成 live。
- [ ] `pipeline` 中途异常/预算耗尽不死锁、队列优雅排空。
- [ ] ruff/pyright/pytest 全绿。

## 指标

- 并发上限断言测试 ≥1；失败语义（parallel `None` / pipeline drop）各 ≥1；无-barrier 交错测试 ≥1；半途 resume 测试 ≥1；不死锁测试 ≥1。
- 墙钟：pipeline 比等价 sequential 明显快（instrumented 计时断言）。

## 文件结构

```
src/langchain_dynamic_workflow/_concurrency.py   # Semaphore 闸 + RunnableConfig 注入
src/langchain_dynamic_workflow/_pipeline.py      # 无 barrier bounded-queue 调度
修改 _context.py（加 parallel/pipeline）、_engine.py（容纳调度器、共享 semaphore）
tests/unit/test_pipeline.py、test_concurrency.py
tests/integration/test_phase2_fanout.py
examples/02_fanout.py
```

## 任务分解（bite-sized TDD）

1. **`_concurrency.py`**：Red（semaphore 限制最大在飞、`max_concurrency` 注入 config）→ 实现 → Green → commit。
2. **`parallel`**：Red（按序返回 / 抛错落 `None` / 不 raise）→ 在 `_context` 实现（`asyncio.gather(*[thunk()...])` + 包 try 落 None）→ Green → commit。
3. **`_pipeline.py` 调度器**：Red（无-barrier 交错 / stage 抛错 drop / 保序 / 不死锁）→ 实现 bounded-queue + worker + 全局 semaphore → Green → commit。
4. **`pipeline` 接入 `Ctx`** + journal 协同：Red（半途 resume 0-call）→ 实现 → Green → commit。
5. **refactor `_engine`**：把 Phase 1 的单叶子 `@task` 调用纳入共享 semaphore/调度器（保持 Phase 1 测试仍绿）→ commit。
6. **示例 + 质量门**：`examples/02_fanout.py`（N 个研究叶子 parallel + 2-stage pipeline）；ruff+pyright 清零；里程碑 commit。

## demo 规格（M2）

`examples/02_fanout.py`：`parallel` 扇出 N 个 `create_deep_agent` 研究叶子，再 `pipeline` 过 2 个 stage（每 item：研究→综述）。集成测试用 fake model 断言结果集 + 并发上限 + 半途 resume。

## 对 Phase 1 的 refactor

- `_engine` 的叶子 `@task` 路径抽成可被 parallel/pipeline 复用的统一 leaf-runner，纳入 semaphore。Phase 1 的 4 个集成测试必须保持绿。

## 交给 Phase 3+ 的点

- 失败语义之上叠加确定性 backstop（重放序列校验）。
- pipeline 内 budget 守卫（耗尽即停、优雅排空）。

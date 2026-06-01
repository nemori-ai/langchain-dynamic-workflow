# Phase 1 · 核心纵切片 实现计划（Implementation Plan）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans。步骤用 `- [ ]` 复选框跟踪。
> **参考实现状态**：Phase 1 已有一份**已验证的参考实现**（全绿：pyright 0 / ruff passed / pytest 17 passed；`examples/01_single_agent.py` → `'Paris'`），用于验证设计可行性。本文件是该阶段的完备设计与任务计划。

**Goal:** `run_workflow(orchestrate, roster=...)` 跑一段调用 `await ctx.agent(...)` 的脚本、命中真实 deepagent 叶子、回填最终结论；二次运行命中 content-hash journal（0 模型调用）。

**Architecture:** 引擎 = LangGraph `@entrypoint`；叶子 = `@task` 调 deepagent；journal 叠在 `@task` 之上做 success-only 记忆化。详见 [../01-engine-mechanism.md](../01-engine-mechanism.md) §1–4、§7。

**Tech Stack:** langgraph 1.2.2 · deepagents 0.6.7 · langchain-core 1.4.0 · pydantic 2 · pytest-asyncio · ruff · pyright(strict)。

---

## 里程碑 M1

单 `agent()` 叶子端到端跑通、journaled、可 resume；绿色 e2e deepagent demo（fake model，无 API key）。

## 接口设计（公共面 + 关键内部）

```python
# 公共
async def run_workflow(
    orchestrate, *, roster: Roster,
    journal: JournalStore | None = None, checkpointer=None,
    thread_id: str = "default", max_concurrency: int | None = None,
) -> Any: ...

class Roster:
    def register(self, name, runnable, *, description="", needs_execution=False, default_model=None) -> Roster
    def resolve(self, name) -> RosterEntry          # 未知名 → KeyError 列可用

class Ctx:
    async def agent(self, prompt, *, agent_type, model=None, isolation="shared") -> str

# 内部
def journal_key(*, prompt, agent_type, model, schema, isolation) -> str   # sha256(canonical json)
class JournalStore(Protocol): ...                # async get(key)->Any|None; async put(key,value)
class InMemoryJournalStore(JournalStore): ...
def fold_result(result: dict) -> str             # 末条非空 AIMessage；缺 messages → ValueError
```

**关键设计要点**：
- journal key 的 5 要素 = `prompt/agent_type/model/schema/isolation`；`label`/`phase` 永不入键。
- **success-only**：`Ctx.agent` 仅在叶子成功返回后 `journal.put`；异常自然不写 → 失败不被缓存。
- `Ctx.agent` 先查 journal（命中即返、0 模型调用），未命中才经 `leaf_runner`（内部 `@task`）调叶子。
- 叶子调用 = `entry.runnable.ainvoke({"messages":[HumanMessage(prompt)]})`，绕开 deepagents LLM-driven `task` 工具。
- resume = 同 journal 实例跨 `run_workflow` 调用复用 → content-hash 命中。

## 验收标准

- [ ] `run_workflow` 跑单 `agent()` 返回回填字符串（fake-model deepagent）。
- [ ] 同 journal 二次运行：content-hash 命中 → 模型调用计数 **0 增量**。
- [ ] 改 `prompt/agent_type/model/isolation/schema` 任一 → journal miss（key 测试覆盖五要素）。
- [ ] success-only：叶子首调抛错 → 不缓存；重试 live 重跑成功。
- [ ] 未知 `agent_type` → `KeyError` 列可用名。
- [ ] `ruff check`、`pyright`(strict) 零告警；`pytest` 全绿。

## 指标

- 测试 ≥15 个（journal/result/roster 单测 + integration ≥4）。
- resume 后 `model.calls` 不增。
- pyright strict 0 error（第三方 partial-unknown 用定点 `# pyright: ignore` + `reportMissingTypeStubs=false`）。

## 文件结构

```
src/langchain_dynamic_workflow/{_journal,_result,_roster,_context,_engine}.py + __init__.py
tests/conftest.py（CountingFakeModel + make_deep_leaf/make_fake_leaf）
tests/unit/{test_journal,test_result,test_roster}.py
tests/integration/test_phase1_single_agent.py
examples/01_single_agent.py
```

## 任务分解（bite-sized TDD）

1. **测试脚手架**：`conftest.py` —— `CountingFakeModel`（`bind_tools`→self + 计数 `_generate`；**关键：`GenericFakeChatModel` 不支持 `bind_tools`、驱动不了 `create_deep_agent`，必须自定义**）；`make_deep_leaf`、`make_fake_leaf(fail_times=)`。
2. **`_journal.py`**：Red（key 稳定/五要素失效/store 往返）→ 实现 → Green → commit。
3. **`_result.py`**：Red（末条/跳尾空/缺 messages/无 AI）→ 实现 `fold_result` → Green → commit。
4. **`_roster.py`**：Red（链式 register/resolve 元数据/contains/未知报错）→ 实现 → Green → commit。
5. **`_context.py`+`_engine.py`**：Red（integration：e2e/resume-0-call/success-only/未知名）→ 实现 `Ctx.agent` + `run_workflow`（`@entrypoint`+`@task`+journal 叠加）→ Green → commit。
6. **公共 API + 示例 + 质量门**：`__init__.py` 短路；`examples/01_single_agent.py`；ruff+pyright 清零；里程碑 commit。

## demo 规格（M1 收尾）

`examples/01_single_agent.py`：一个 `create_deep_agent`（默认离线 `_ScriptedModel`，`LDW_DEMO_REAL_MODEL` 切真实）被 `run_workflow` 编排回答问题，打印结论。集成测试断言结论 + resume 0-call。

## 交给 Phase 2+ refactor 的点

- `@task` 叶子调用抽出、纳入并发闸 + pipeline 调度器。
- `Ctx` 增 `parallel`/`pipeline`/`phase`/`log`/`budget`。
- schema（`ToolStrategy`）路径接入。

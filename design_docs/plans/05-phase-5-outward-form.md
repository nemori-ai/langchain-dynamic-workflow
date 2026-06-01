# Phase 5 · 对外形态（tool + middleware + async 后台 + skills）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans。步骤用 `- [ ]` 跟踪。

**Goal:** 一个 host deepagent 通过 `workflow_tool(run, script)` **后台**启动 workflow、立即拿 run_id 占位、完成后经 `abefore_model` 注入通知、再 `status`/`get_result` 取最终结论；skills 注入 host prompt；`workflow()` 一级嵌套。

**Architecture:** 自建轻量后台机制（蓝本 = omne-next `BackgroundToolCallMiddleware` + deepagents `AsyncSubAgentMiddleware` API 形态），跑在自有 registry + LangGraph checkpointer、**无 server**；middleware 贡献 tool + `abefore_model` 注入通知。详见 [../02-architecture.md](../02-architecture.md) §2–5。

**Tech Stack:** Phase 1–4 栈 + langchain `AgentMiddleware`（`.tools`/`abefore_model`/`wrap_model_call`）+ deepagents `SkillsMiddleware`（`create_deep_agent(skills=...)`）+ `asyncio.create_task`。

---

## 里程碑 M5（旗舰）

host deepagent 后台驱动 workflow、poll/notify 取回结论；skills 教 host 写脚本；`workflow()` 一级嵌套；绿色 e2e demo。

## 接口设计

```python
def create_workflow_tool(roster: Roster, *, ...) -> BaseTool       # 多命令 run/status/resume/cancel
def create_workflow_middleware(roster: Roster, *, skills_dir=None, ...) -> AgentMiddleware
    # .tools = [workflow_tool]；abefore_model 注入 <workflow_notification>；持 BgRunManager

class BgRunManager:
    def start(self, coro, *, run_id, thread_id) -> BgRunSlot       # asyncio.create_task 脱离
    def poll(self, run_id) -> Status                                # pending/running/done/failed
    def drain_notifications(self, thread_id) -> list[Notice]

class Ctx:
    async def workflow(self, name, args=None) -> Any               # 一级嵌套；二级 → raise
```

**关键设计要点**：
- **workflow tool 多命令**：`run`（`asyncio.create_task` 启 `run_workflow`，**即返** run_id 占位 ToolMessage）/ `status`（轮询）/ `resume`（journal-backed）/ `cancel`。
- **完成投递**：done callback 入队 → `abefore_model` 在 host 下一轮 model call 前注入 `<workflow_notification>`（in-band，无 harness）。**poll + notify 双支持**。
- registry 按 `{thread_id}__{run_id}` 复合键隔离；idle/硬 TTL 清扫；完成结果存 completed-index 供稍后取。
- **大结果 offload**：`ResultStore`（memory，可选 sandbox）；`status` 回摘要 + 句柄。
- bg run 状态存**专用 state channel**（仿 deepagents `async_tasks`，扛 context 压缩）。
- **scope 边界**：middleware 横切作用于 **host 回合**；workflow run 内部 leaf 的 budget/journal 在引擎内部 scope（别混）。
- **skills = L2-as-skill**：一套 SKILL.md（教写脚本 + DSL + 确定性铁律 + 范式），`create_deep_agent(skills=[...])` 原生加载；`skills_metadata` 不传子 agent。
- `workflow()` 一级嵌套：内层 workflow **inline 跑在父 entrypoint body 内、共享父 `Ctx`**（journal/budget/gate/progress 一并共享），其叶子 `agent()` 仍各自走 durable `@task`、journal 正常，故可 resume；二级嵌套抛 `WorkflowNestingError`。（实现取 inline 共享而非为内层单建 subgraph——一级嵌套下二者等效，inline 更简洁。）

## 验收标准

- [ ] host deepagent 调 `workflow_tool(run, script=...)` → 即返 run_id 占位（不阻塞 host 回合）。
- [ ] 后台 `asyncio.create_task` 跑 `run_workflow`；done → 下一轮 `abefore_model` 注入 `<workflow_notification>`（断言注入）。
- [ ] `status`/`get_result(run_id)` 返回结果；大结果 offload + 摘要/句柄。
- [ ] `resume`/`cancel` 命令生效。
- [ ] skills 经 `create_deep_agent(skills=[...])` 进 host prompt（断言 skill 元数据出现）。
- [ ] `workflow()` 一级嵌套通过；二级嵌套抛错。
- [ ] ruff/pyright/pytest 全绿。

## 指标

- e2e「host agent 驱动后台 workflow」测试 ≥1；通知注入测试 ≥1；status/resume/cancel 各 ≥1；skills 加载测试 ≥1；嵌套守卫测试 ≥1。
- 占位即返（host 在 workflow 未完成前可继续回合）用例验证。

## 文件结构

```
src/langchain_dynamic_workflow/tool.py            # create_workflow_tool（多命令）
src/langchain_dynamic_workflow/_background.py      # BgRunManager / BgRunSlot / ResultStore(memory)
src/langchain_dynamic_workflow/middleware.py       # create_workflow_middleware（.tools + abefore_model）
src/langchain_dynamic_workflow/skills/*.md         # SKILL.md 编排教学集
修改 _context.py（加 workflow()）、__init__.py（短路 tool/middleware）
tests/unit/test_background.py、test_tool.py、test_middleware.py
tests/integration/test_phase5_host_agent_workflow.py
examples/05_host_agent_workflow.py
```

## 任务分解（bite-sized TDD）

1. **`_background.py`**：Red（start 脱离即返 run_id / poll 状态机 / done 入队 / TTL）→ 实现 BgRunManager/BgRunSlot/ResultStore → Green → commit。
2. **`tool.py` 多命令**：Red（run 即返占位 / status / resume / cancel）→ 实现（委托 BgRunManager + run_workflow）→ Green → commit。
3. **`middleware.py`**：Red（`.tools` 贡献 tool / `abefore_model` 注入通知 / 专用 state channel）→ 实现 → Green → commit。
4. **`workflow()` 嵌套**：Red（一级通过 / 二级 raise）→ 在 `Ctx` 实现（内层 `@task`）→ Green → commit。
5. **skills**：Red（`create_deep_agent(skills=[dir])` 后 skill 元数据进 prompt）→ 写 SKILL.md 编排教学集 + 接线 → Green → commit。
6. **e2e 集成 + 示例 + 质量门**：`examples/05_host_agent_workflow.py`（旗舰）；ruff+pyright 清零；里程碑 commit。

## demo 规格（M5，旗舰）

`examples/05_host_agent_workflow.py`：一个 host `create_deep_agent`（fake model 脚本化：先调 `workflow_tool(run,...)`、收到通知后调 `status`）driving 一次后台 workflow（内部 parallel 扇出），最终把结论纳入 host 回答。集成测试 fake model 全绿。

## 对 Phase 1–4 的 refactor

- `run_workflow` 被 BgRunManager 以 `asyncio.create_task` 包裹后台化；同步直调路径仍保留（库 core）。Phase 1–4 测试须保持绿。

## 交给 Phase 6 的点

- 把旗舰 demo 升级为多特性 capstone；import-linter、可观测性、覆盖率门、文档收口。

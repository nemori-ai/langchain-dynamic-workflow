# 02 · 架构设计（Architecture）

> **范围**：软件长什么样、怎么接入 agent——三层架构、对外软件形态、五个消费面、自建 async 后台 tool 执行机制、L2-as-skill、build-vs-buy 账本、v1 范围。
> **引擎内部机制**（journal / 确定性 / sandbox / 叶子契约）见 [01-engine-mechanism.md](01-engine-mechanism.md)；图见 [uml/](uml/)。
> **日期**：2026-06-01　**状态**：对外形态已锁。

---

## 1. 三层架构

```
┌─ Layer 2 · Meta 层(v1 紧跟,落法 = skill)：教 agent 写编排脚本 ─────────────┐
│ 一套 SKILL.md(DSL + 确定性铁律 + 范式)→ host agent 读 → 自己写脚本 → 调 tool │
├─ Layer 1 · 编排运行时(v1 核心)：八原语 + 两补丁 ─────────────────────────┤
│ agent/parallel/pipeline/race/phase/log/budget/workflow                     │
│ 补丁① content-hash journal(success-only) 补丁② 确定性 fail-loud guard       │
├─ Layer 0 · 底座：LangGraph durable execution ──────────────────────────────┤
│ @entrypoint + @task + checkpointer + 三档 durability                        │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       ▼ agent() 叶子(R1 roster 解析)
        deepagents：create_deep_agent → 直接 ainvoke(context quarantine + per-leaf sandbox)
```

Layer 0/1 = 引擎本体（见 01）。Layer 2 在 v1 以 **skill** 形态落地（见 §4）。

## 2. 对外软件形态（核心）

### 2.1 第一性原理：消费者是 AI agent，只能 tool call

本库的运行时消费者是 **AI agent**;agent 对外界的唯一动作是 **tool call**。因此:

> **唯一运行时公共面 = 一个 workflow tool(一次 tool call)。** 库 API、primitives 都是 **build-time / 开发者面**,不是 agent 面。tool 是承重墙,库 API 是次要(接线)。

**它不是 middleware**:`wrap_tool_call` 单粒度、扇不出并发 subagent(实证),编排本质是 fan-out——middleware 撑不起引擎本体。middleware 只当**横切配角 + async 交付载体**(见 §3、§5)。

### 2.2 五个消费面

| 面 | 形态 | 性质 |
|---|---|---|
| **① 库 core** | `run_workflow(script_or_callable, *, roster, config) -> result`(keyword-only、单函数、**无** operation-plane) | 开发者 / build-time;真相与地基 |
| **② tool adapter** | `create_workflow_tool(roster, ...)` → 挂 host deepagent 的**多命令** tool(`run`/`status`/`resume`/`cancel`) | **agent 唯一运行时面** |
| **③ skills** | 一套 SKILL.md(教 host agent 写脚本 + DSL + 确定性铁律 + 范式),`create_deep_agent(skills=[...])` 原生加载 | agent 行为塑形(prompt);也是 L2 落法 |
| **④ primitives** | 注入 `agent/parallel/pipeline/race/phase/log/budget/workflow`(手写脚本用);`agent` 支持 `schema=`(pydantic 类或 JSON-schema dict)产校验过的结构化对象;`race` 是 best-of-N 早退原语(首个令 `win` 为真者胜、在飞 loser 全 cancel、决策内容哈希 journal 故 resume 复现胜者且零派发) | 开发者 / build-time |
| **⑤ middleware** | `create_workflow_middleware(roster, ...)` 打包 ②+③ + 承载 async 后台机制 | **async fire-and-notify 的交付载体**(非纯可选) |

> "tool + skills"是两种不同性质:**tool 是唯一可调用面;skills 是 prompt 行为塑形**。闭环 = skills 教 → agent 写出 script → `workflow_tool(run, script)` 执行 → 只回结论。

> **跨叶归约 helper(F)随 primitives 同属开发者面**:`survives` / `dedup` / `reconcile` / `corroborate`(+ `ReviewItem` / `Reconciled` / `Consensus`)是折叠 `parallel` / `pipeline` 结果列表的纯函数,从包根导出供开发者 `import`,并由引擎注入 `run_script` 命名空间(L2 脚本禁 import,按名直调)——详见 [01 §2](01-engine-mechanism.md)。

> **race 值类型(B)同属开发者面**:`RaceCandidate` / `RaceResult`(+ `race_key`)随 `ctx.race` 原语成对——两个 frozen dataclass 从包根导出供开发者 `import`,并注入 `run_script` 命名空间(L2 脚本禁 import,按名直调);`race_key` 仅导出(脚本走 `ctx.race`、不直接碰 key)——详见 [01 §2](01-engine-mechanism.md)。

### 2.3 build-time 接线（开发者干,agent 不参与）

```python
roster = Roster()
# 预构造 runnable:仅服务 schema-less 叶子
roster.register("researcher", create_deep_agent(...), needs_execution=False)
# builder 工厂:schema-capable 叶子,引擎按需以 response_format 构造结构化变体(供 agent(schema=))
roster.register(
    "skeptic",
    builder=lambda *, response_format=None: create_deep_agent(
        model=..., response_format=response_format
    ),
)

# 纯库:开发者自己编排(无 host agent)
result = await run_workflow(orchestrate, roster=roster, config=cfg)

# 接入 host agent:tool + skills(+ 可选 middleware 承载 async 通知)
host = create_deep_agent(
    model,
    middleware=[create_workflow_middleware(roster)],   # 贡献 workflow tool + 后台机制 + 注入通知
    skills=["…/workflow-skills/"],                       # 教 host 写脚本(L2)
)
```

## 3. 自建 async 后台 tool 执行机制

### 3.1 为何自建（三路核实结论）

| 路径 | 结论 |
|---|---|
| langchain/langgraph/deepagents 原生 | **无**:tool 执行严格同步;只有积木(`interrupt+checkpointer`、`@task`、`asyncio.create_task`) |
| omne-next 自建 | **有,纯 middleware**:placeholder `Command` 即返 + 脱离 `asyncio.Task` + Registry + `abefore_model` 注入通知 + 轮询 meta-tool。现成蓝本 |
| 社区生态 | **无干净可加依赖**:LangGraph Platform background-run = Elastic 许可 server + Postgres(出局);deepagents `AsyncSubAgentMiddleware`(MIT)形态最对但 subagent-scoped + 需 Agent Protocol server;Celery/Dramatiq 太重;langchain-runner 太嫩 → **BUILD** |

**决定(D20)**:**自建**轻量后台机制,license 干净(MIT-only)、零额外重依赖、无 server,原生嵌进我们的 primitives + journal。蓝本 = omne-next `BackgroundToolCallMiddleware` 实现 + deepagents `AsyncSubAgentMiddleware` 的 API 形态。

### 3.2 机制

- **workflow tool 多命令**:`run`(`asyncio.create_task` 脱离启动 `run_workflow`,**即返** run_id 占位 ToolMessage)/ `status`(轮询:pending/running/done+结果)/ `resume`(journal-backed,见 01 §4)/ `cancel`。
- **完成投递**:done callback → 入队 → `abefore_model` 在 host agent **下一轮 model call 前注入** `<workflow_notification>`(in-band,无需 harness)。**poll + notify 双支持**。
- **registry + 生命周期**:`BgRunSlot`(RACING→BACKGROUND→COMPLETED→delivered),按 `{thread_id}__{run_id}` 复合键隔离;idle/硬 TTL 清扫;完成结果存活于 completed-index 供稍后取。
- **大结果 offload**:`ResultStore`(memory / sandbox 两后端),`status` 回摘要 + 句柄。
- **状态存专用 state channel**(抄 deepagents `async_tasks` 那招,扛 context 压缩)。

### 3.3 两层 scope 边界（务必钉死，易混淆）

- **host 面后台 tool 包装**(本节):让 host agent 不被 `run_workflow` 阻塞;作用于 host agent 自己的回合;`abefore_model` 注入完成通知。
- **引擎内部 durable execution**(见 01):`@task`/`parallel`/journal/sandbox,在 `run_workflow` 内部、与 host middleware **不同 scope**。
- middleware 的 budget/guard 横切作用于 **host agent 回合**;workflow run 内部的 leaf agents budget/journal/guard 是**引擎内部 scope**。两者不是一回事。

## 4. L2-as-skill（meta 层落法）

deepagents 自带轻量 `SkillsMiddleware`(skill = 目录 + SKILL.md,渐进披露:frontmatter 进 prompt、body 靠已有 `read_file` 按需读)。→ **meta 层 = 一套 SKILL.md**(教 host agent 写编排脚本 + DSL + 确定性铁律 + 范式),`create_deep_agent(skills=[...])` 原生加载。host agent 读 skill → 自己写脚本 → 调 workflow tool。

- **skills 侧零自建**:不学 omne-next 的 capability marketplace / 双 BM25 索引(过重);v1 skill 集小,prompt-injection + read_file 够用。
- **不必引擎内置 codegen agent**:host agent 自身即"meta 层 LLM",正是 Claude Code 的真实形态。
- `skills_metadata` 是 PrivateStateAttr **不传子 agent** → skills 只塑形 host/编排 agent,不泄进 leaf context(合"skills 教 host、leaf 只干活"分层)。

## 5. middleware 角色

- **能力(实证)**:`AgentMiddleware` 可同时 (a) `.tools` 贡献 workflow tool、(b) `wrap_model_call`/`abefore_model` 注入 prompt/通知、(c) `after_model` 做横切记账/guard。单个 middleware 打包三者是 idiomatic langchain。
- **为何升级为"交付载体"而非纯可选**:async fire-and-notify 的完成通知靠 `abefore_model` 注入——**必须由 middleware 承载**。纯轮询(status)可不靠 middleware,但 notify 体验必须它。
- **硬天花板**:`wrap_tool_call` 单粒度扇不出并发 → `parallel()` 落在引擎 `@task` 层,绝不靠 middleware。

## 6. build-vs-buy 账本（据底座研报，逐项实证）

| 引擎需求 | 底座支持 | level | 自建什么 |
|---|---|---|---|
| `@entrypoint`/`@task` durable body | `langgraph.func` | provided | 无 |
| resume / replay / cached-skip | checkpointer + positional `task_id` | provided | 无(task 顺序须稳定) |
| journal 存储底座 | `BaseStore`(实测往返通过) | provided | 无 |
| `agent()` 叶子调用 | `create_deep_agent` → `ainvoke` | provided | 结果回填(镜像 `subagents.py`) |
| context quarantine | 直调天然隔离 | provided | 无 |
| `parallel()`(barrier) | `asyncio.gather` over `@task` | provided | 薄封装 |
| skills | deepagents `SkillsMiddleware` | provided | 仅提供 SKILL.md 内容 |
| `pipeline()`(无 barrier 流式) | 无原语 | **gap** | 自建调度器(见 01 §9) |
| content-hash journal | 内容寻址但 qualname-scoped;sync put_writes 无 ERROR 守卫(#7589) | **partial→自建** | success-only + per-node scoping |
| 确定性 guard | 仅 `-O` 可剥离裸 assert | **gap** | divergence backstop |
| per-leaf sandbox identity + SandboxManager | backend 无 lifecycle 方法 | **gap** | 自建(见 01 §8) |
| **async 后台 tool 执行** | **无(原生/社区皆无干净依赖)** | **gap** | **自建(本文 §3)** |
| budget(token) | `usage_metadata` + callback | partial | enforcement + `@task` 层复刻 callback 转发 |
| `max_concurrency` 设界 | 默认 None ⇒ 无界 | partial | 显式设值 + 资源守卫 |

**净结论**:provided 占多数;自建集中在 **6 项**——pipeline、journal(success-only)、确定性 guard、per-leaf identity、SandboxManager、**async 后台 tool 执行机制**——加 budget enforcement 与 `max_concurrency` 显式设界两个收口。

## 7. v1 范围与交付切分

- **v1 主形态①**:库 core `run_workflow()`(真相/地基,可独立测/手写)+ tool adapter `create_workflow_tool()` / middleware(agent 接入主形态);两者一等。
- **v1 含**:八原语(含 race 早退/取消)+ roster + 叶子契约 + 双维 guard + journal + 确定性 backstop + pipeline + SandboxManager + **async 后台 tool 机制(run/status/resume/cancel + 通知注入)** + budget + 显式并发。
- **L2 紧跟(v1 即可起步)**:skills 形态(SKILL.md),不需重组件——host agent 写脚本传入即可跑。
- **v2 backlog**:A2/A3 安全硬化、codegen prompt 工程深化、可观测性(phase/log → LangSmith trace、独立 `*-tracing` 子包、token rollup)、跨进程持久化(加 `langgraph-checkpoint-sqlite/-postgres`)、skill 语义检索、`agent(isolation="worktree")`。

## 8. 与 Claude Code 的差异点

| | Claude Code | 本项目 |
|---|---|---|
| 脚本语言 | JS | Python(callable 本体 + 源码前门) |
| 执行底座 | 官方未披露(社区信号指向 Node `vm` 式;"V8 isolate" 实为 Cloudflare 另一产品,**勿混**) | 骑 LangGraph functional API |
| resume 作用域 | 仅同 session | 可跨 session(需加持久 checkpointer 依赖) |
| 确定性 | 运行时 fail-loud | journal-divergence backstop(更普适,不 monkeypatch) |
| content-hash journal | 单源逆向、未证实键推导 | 自主工程取舍(success-only),非"复刻官方" |
| 叶子 agent | Anthropic 内部 agent | deepagent(context quarantine + per-leaf sandbox 实例) |
| async 体验 | 宿主 harness 主动注入 | 自建 middleware(placeholder + 脱离 task + `abefore_model` 注入) |
| meta 层 | 主 agent 写脚本塞给 Workflow 工具 | L2-as-skill:host agent 读 SKILL.md → 写脚本 → 调 tool |

## 9. 借鉴 promptflow 的工程实践（落地清单）

- `agent()` @task wrapper 照 `invoke_tool` bracket:prepare → journal-lookup → trace → **`finally` 持久化(杜绝 silent failure)**。
- journal entry schema 参照 `RunInfo/LineResult`(parent_run_id / status / cached / content-hash)。
- **observability-by-default**:primitives 自动 emit span/journal;规划独立 `langchain-dynamic-workflow-tracing` 子包;token rollup 喂 budget。
- **import-linter `forbidden` contracts** 现在就上,机械守 Layer 0/1/2 边界(禁 AST-gate/roster 直接 import LangGraph 内部)。
- **规避**:string-keyed config patching(promptflow `FlowContext.overrides` footgun)、OS 进程池、外部 scheduler + 文件 handoff、控制流入 config(静态 DAG)。

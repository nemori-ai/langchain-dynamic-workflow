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
| **② tool adapter** | `create_workflow_tool(roster, ...)` → 挂 host deepagent 的**多命令** tool(`run`/`run_script`/`status`/`resume`/`cancel`/`runs`) | **agent 唯一运行时面** |
| **③ skills** | 一套 SKILL.md(教 host agent 写脚本 + DSL + 确定性铁律 + 范式),`create_deep_agent(skills=[...])` 原生加载 | agent 行为塑形(prompt);也是 L2 落法 |
| **④ primitives** | 注入 `agent/parallel/pipeline/race/phase/log/budget/workflow`(手写脚本用);`agent` 支持 `schema=`(pydantic 类或 JSON-schema dict)产校验过的结构化对象;`race` 是 best-of-N 早退原语(首个令 `win` 为真者胜、在飞 loser 全 cancel、决策内容哈希 journal 故 resume 复现胜者且零派发) | 开发者 / build-time |
| **⑤ middleware** | `create_workflow_middleware(roster, ...)` 打包 ②+③ + 承载 async 后台机制 | **async fire-and-notify 的交付载体**(非纯可选) |

> "tool + skills"是两种不同性质:**tool 是唯一可调用面;skills 是 prompt 行为塑形**。闭环 = skills 教 → agent 写出 script → `workflow_tool(run, script)` 执行 → 只回结论。

> **跨叶归约 helper(F)随 primitives 同属开发者面**:`survives` / `dedup` / `reconcile` / `corroborate`(+ `ReviewItem` / `Reconciled` / `Consensus`)是折叠 `parallel` / `pipeline` 结果列表的纯函数,从包根导出供开发者 `import`,并由引擎注入 `run_script` 命名空间(L2 脚本禁 import,按名直调)——详见 [01 §2](01-engine-mechanism.md)。

> **race 值类型(B)同属开发者面**:`RaceCandidate` / `RaceResult`(+ `race_key`)随 `ctx.race` 原语成对——两个 frozen dataclass 从包根导出供开发者 `import`,并注入 `run_script` 命名空间(L2 脚本禁 import,按名直调);`race_key` 仅导出(脚本走 `ctx.race`、不直接碰 key)——详见 [01 §2](01-engine-mechanism.md)。

> **逐叶 live 可观测性 sink(Layer-1,挂 ① 库 core)**:`run_workflow` 在既有 `on_progress`(进度叙事)/ `on_span`(完成 span)之外,提供三个 keyword-only、默认 `None` no-op 的带外 sink——`on_span_begin`(每个原语 span **打开**即发的 running 边,携引擎铸的 resume-稳定 `span_id` + `started_at`,供 live elapsed timer)、`on_leaf_event`(把叶子自己的回调子树 normalize 成 `LeafEvent`,经 `leaf_span_id` 关联到所属叶、经 `run_id`/`parent_run_id` 重建叶内 run tree)与 `on_command`(把执行叶子跑的真 shell `execute` normalize 成成对 `CommandEvent`,在 subprocess 边界前后发 `"start"`/`"end"` 两条边,共享 resume-稳定 `command_id`、携所属叶子的 `leaf_span_id`),配 `leaf_event_include_payloads: bool` / `command_include_payloads: bool` 控制各自 `detail` / `output` 是否带原始 payload(默认 shape-only / 截尾)。值类型 `SpanBegin` / `LeafEvent` / `CommandEvent`(+ `SpanBeginSink` / `LeafEventSink` / `CommandSink` 别名)从包根导出供宿主消费;三条 sink 皆走带外、不进宿主 LLM 上下文,故接不接 sink 宿主上下文逐字节相同——机制见 [01 §2b](01-engine-mechanism.md)。`on_command` 与另两条同属 miss-only(journal 命中的 cached 叶不 lease 不跑 subprocess,故重放叶零 command 事件;且仅当注入的 `sandbox_manager` 工厂产 `LocalSubprocessSandbox` 真后端时才有真 execute 边界可观测,离线 `InMemorySandbox` 不发)。这些正是交互式 demo app 画 live trace(running chip + elapsed timer + 可折叠嵌套子会话 + 真命令的 terminal card)所消费的引擎面。

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

# 可选:跨进程持久化(M3)——接一个持久 store + checkpointer(见 §10)
store = await SqliteWorkflowStore.open("workflows.db")   # async 工厂,宿主持久 loop 内构造
host = create_deep_agent(
    model,
    middleware=[create_workflow_middleware(
        roster, store=store, checkpointer=store.checkpointer,
    )],
    skills=["…/workflow-skills/"],
)
# store= 省略时回落 InMemoryRunStore(零依赖、同会话语义)
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

- **workflow tool 多命令**:`run`(`asyncio.create_task` 脱离启动 `run_workflow`,**即返** run_id 占位 ToolMessage)/ `status`(轮询:pending/running/done+结果)/ `resume`(journal-backed,见 01 §4)/ `cancel` / `runs`(**聚合视图**:`BgRunManager.list_runs(thread_id)` 一次列出本 thread 全部 run 的 run_id + workflow label + 实时状态 + 落定后的短摘要,免逐个 `run_id` 轮询;每条 `RunSnapshot` 只读快照、不漏可变 slot)。
- **完成投递**:done callback → 入队 → `abefore_model` 在 host agent **下一轮 model call 前注入** `<workflow_notification>`(in-band,无需 harness)。**poll + notify 双支持**。
- **registry + 生命周期**:`BgRunSlot`(RACING→BACKGROUND→COMPLETED→delivered),按 `{thread_id}__{run_id}` 复合键隔离;idle/硬 TTL 清扫;完成结果存活于 completed-index 供稍后取。
- **多并行 run + quota**:同一 session 可后台并发任意多 run(每 run 独立 journal/budget/gate/确定性 guard,按 `run_id` 取消/恢复互不干扰);可选 `BgRunManager.max_concurrent_runs` 上限,满则 `run` loud 拒(`BgRunQuotaExceededError`),默认无界。quota **归 `BgRunManager`**——`create_workflow_middleware` 的 `max_concurrent_runs` 仅作用于其自建的默认 manager;与显式 `manager` 同传即冲突,**抛 `ValueError`**(杜绝静默忽略)。
- **大结果 offload**:`ResultStore`(memory / sandbox 两后端),`status` 回摘要 + 句柄。
- **状态存专用 state channel**(抄 deepagents `async_tasks` 那招,扛 context 压缩):`workflow_runs` 用 `merge_workflow_runs` reducer **按 `run_id` upsert**——launch 写 `running`、落定时 `abefore_model` 同步改写为终态,故 channel 反映实时状态而非停在 launch 时的 `running`。

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
| journal 存储底座 | `BaseStore`(实测往返通过) | provided | 默认 in-memory;**跨进程持久 store(M3)自建**(`SqliteWorkflowStore`:aiosqlite autocommit + run_id 命名空间化四表,见 §10) |
| 跨进程 checkpointer | `langgraph-checkpoint-sqlite` `AsyncSqliteSaver` | provided | 接线 + event-loop-绑定纪律(`[sqlite]` extra) |
| `agent()` 叶子调用 | `create_deep_agent` → `ainvoke` | provided | 结果回填(镜像 `subagents.py`) |
| context quarantine | 直调天然隔离 | provided | 无 |
| `parallel()`(barrier) | `asyncio.gather` over `@task` | provided | 薄封装 |
| skills | deepagents `SkillsMiddleware` | provided | 仅提供 SKILL.md 内容 |
| `pipeline()`(无 barrier 流式) | 无原语 | **gap** | 自建调度器(见 01 §9) |
| content-hash journal | 内容寻址但 qualname-scoped;sync put_writes 无 ERROR 守卫(#7589) | **partial→自建** | success-only + per-node scoping |
| 确定性 guard | 仅 `-O` 可剥离裸 assert | **gap** | divergence backstop |
| per-leaf sandbox identity + SandboxManager | backend 无 lifecycle 方法 | **gap** | 自建(见 01 §8) |
| 真本地执行后端(local subprocess) | `SandboxBackendProtocol.execute` 是 no-op 契约位、无真实现 | **gap** | 自建(M5)`LocalSubprocessSandbox`(stdlib-only 全协议 + per-leaf 临时根 + 有界抽干/超时组杀/`ExecGate`/`before_execute`/POSIX rlimit),经可插拔 `sandbox_factory` 注入,默认仍 `InMemorySandbox`(见 01 §8b) |
| **async 后台 tool 执行** | **无(原生/社区皆无干净依赖)** | **gap** | **自建(本文 §3)** |
| budget(token) | `usage_metadata` + callback | partial | enforcement + `@task` 层复刻 callback 转发 |
| `max_concurrency` 设界 | 默认 None ⇒ 无界 | partial | 显式设值 + 资源守卫 |

**净结论**:provided 占多数;自建集中在 **7 项**——pipeline、journal(success-only)、确定性 guard、per-leaf identity、SandboxManager、**真本地执行后端(`LocalSubprocessSandbox`,M5)**、**async 后台 tool 执行机制**——加 budget enforcement 与 `max_concurrency` 显式设界两个收口。

## 7. v1 范围与交付切分

- **v1 主形态①**:库 core `run_workflow()`(真相/地基,可独立测/手写)+ tool adapter `create_workflow_tool()` / middleware(agent 接入主形态);两者一等。
- **v1 含**:八原语(含 race 早退/取消)+ roster + 叶子契约 + 双维 guard + journal + 确定性 backstop + pipeline + SandboxManager + **async 后台 tool 机制(run/run_script/status/resume/cancel/runs + 通知注入 + 多并行 run quota + 落定刷新的 workflow_runs)** + budget + 显式并发。
- **L2 紧跟(v1 即可起步)**:skills 形态(SKILL.md),不需重组件——host agent 写脚本传入即可跑。
- **v2 backlog**:A2/A3 安全硬化、codegen prompt 工程深化、可观测性的**外接适配**(phase/log/span → LangSmith trace、独立 `*-tracing` 子包、token rollup)、skill 语义检索、`agent(isolation="worktree")` 真 git 后端。**逐叶 live 可观测性 substrate 已落地(M1,Layer-1 能力)**:span begin 边 + 引擎铸的 resume-稳定 `span_id` + `on_leaf_event` 逐叶回调子树 tap(见 §2.2、[01 §2b](01-engine-mechanism.md));剩在 backlog 的是上述 **LangSmith-adapter / 独立 tracing 子包**(把这些带外 sink 桥到外部 tracer)。**跨进程持久化已落地(M3,`SqliteWorkflowStore` 经 `[sqlite]` extra,见 §10)**;postgres 后端是同 `WorkflowRunStore` 协议后的可插拔扩展点。**真本地执行后端已落地(M5,Layer-1 能力)**:`LocalSubprocessSandbox`(per-leaf 临时根 + 有界输出抽干 + 超时进程组杀 + `ExecGate` + `before_execute` 准入 + POSIX best-effort rlimit + 命令可观测性 `on_command`/`CommandEvent` + 诚实非安全-sandbox posture)作可插拔 Layer-1 后端经 `SandboxManager` 的 `sandbox_factory` 接缝交付,默认仍离线 `InMemorySandbox`、stdlib-only 零依赖(见 [01 §8b](01-engine-mechanism.md));剩在 backlog 的 **A2/A3 级硬化**——container / cgroup / 网络隔离与真 git-worktree 执行后端——其中 container 后端是同工厂接缝后 `[sqlite]` 之后的**下一个 opt-in extra**。

## 8. 与 Claude Code 的差异点

| | Claude Code | 本项目 |
|---|---|---|
| 脚本语言 | JS | Python(callable 本体 + 源码前门) |
| 执行底座 | 官方未披露(社区信号指向 Node `vm` 式;"V8 isolate" 实为 Cloudflare 另一产品,**勿混**) | 骑 LangGraph functional API |
| resume 作用域 | 仅同 session | **可跨 session / 跨进程(M3 已落地)**:接 `SqliteWorkflowStore`(`[sqlite]` extra),零成本重放由持久 journal 交付、checkpointer 作 durable add-on |
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

## 10. 跨会话持久化接线（M3,host 面）

**头条**:全新进程(或在同一 db 文件上拆掉再重开的 store)按 `run_id` resume 一个 run,完成过的叶子从持久内容哈希 journal **零模型成本重放**——超越 Claude Code(仅同会话)。机制不变量(两连接 / event-loop 绑定 / 规范 id 谱系 / schema-version guard)见 [01 §13b](01-engine-mechanism.md);本节是 host 接线契约。

- **公共面**:`from langchain_dynamic_workflow import SqliteWorkflowStore`(包根 lazy 导出,需 `[sqlite]` extra;`WorkflowRunStore` / `RunSpec` / `InMemoryRunStore` 直接导出、零依赖)。
- **构造**:`store = await SqliteWorkflowStore.open(db_path)`——**async 工厂,必须在 host 的单一持久 event loop 内 await**(`AsyncSqliteSaver` 构造期绑 loop,**绝不**跨 loop 复用)。亦可 `async with await SqliteWorkflowStore.open(db_path) as store: ...`(退出关两条连接)。否则 host shutdown 调 `await store.aclose()`。
- **接线**:`create_workflow_middleware(roster, workflows=wf, store=store, checkpointer=store.checkpointer, ...)`(或 `create_workflow_tool` 同 `store=`/`checkpointer=` kwargs)。`store=` 省略时回落零依赖的 `InMemoryRunStore`(同会话语义、行为不变)。
- **谁交付零成本重放**:**journal**,不是 checkpointer——`store.checkpointer` 是 durable add-on(durable `@task` cache + 单 run 内 interrupt/resume + 跨进程按 thread_id resume)。resume 侧即便 `checkpointer=None`,journal 仍独立交付零成本重放。
- **已知边界**:`args` 须 JSON-可序列化(持久 store 经 JSON round-trip);`race()` 无胜者不 journal,跨进程 resume 会重派候选(新成本);跨进程 resume 须用同一 db 文件。

```python
from langchain_dynamic_workflow import SqliteWorkflowStore

# 进程 A:开 store、起后台 run、落定、退出
store = await SqliteWorkflowStore.open("workflows.db")
host = create_deep_agent(
    model,
    middleware=[create_workflow_middleware(
        roster, workflows=wf, store=store, checkpointer=store.checkpointer,
    )],
    skills=["…/workflow-skills/"],
)
# ... host 经 workflow tool 起 run(得 run_id)、run 完成、journal 落盘 ...
await store.aclose()

# 进程 B(全新进程,同一 db 文件):重开 store、按 run_id resume → 完成叶零成本重放
store = await SqliteWorkflowStore.open("workflows.db")
# 重建同样接线的 host,host 经 workflow tool 的 resume(run_id)续跑
```

## 11. 运行中 HITL 签核接线（M4,host 面）

脚本以 `ctx.checkpoint(ask, *, tag="")` 暂停等人工签核;host 经 workflow tool 观测并续跑。机制(journal 驱动的门、为何弃 LangGraph interrupt、载重不变量)见 [01 §14](01-engine-mechanism.md);本节是 host 面接线。

```python
# 一个会停下等签核的工作流(脚本拥有暂停点,门前叶重放免费)
async def gated(ctx, args):
    assessment = await ctx.agent("评估部署计划风险", agent_type="auditor")
    decision = await ctx.checkpoint({"ask": "批准部署?", "summary": assessment}, tag="deploy")
    if not decision.get("approved"):
        return f"held: {decision.get('note', '未批准')}"
    return f"proceeding: {assessment}"

# host 经 workflow tool:run → 停在门(status=awaiting_signoff,带 ask)→ approve(注入决策)→ 续跑
# - status <run_id>  : awaiting_signoff 时回 ask + 如何 approve
# - approve <run_id> : args 携人工决策(如 {"approved": true, "note": "..."}),喂回暂停的 run
```

- **新态 `BgStatus.AWAITING_SIGNOFF`**(非终态、计入 active):parked run 占一个 slot 直到 approve/cancel;`get_signoff(run_id)` 取 ask;`runs`/`status` 暴露它。abandoned 的 park 由 `sweep` 经 `park_ttl_seconds` 过期回收(防永久占 quota)。
- **`approve` 命令(本进程在活的 parked run)**:复用 parked slot、**同 run_id** 就地续跑(`BgRunManager.approve` 先同步翻 `RUNNING` 杜绝双批竞争),demo 卡片据 id 跨暂停原地更新。**只批准本进程在活的 parked run**——parked 态(哪道门、ask)只在内存 manager、未持久化,故 swept/跨进程的 `UNKNOWN` run 拒批(以免把非 parked run 推过人没看过的门);跨会话 HITL 待持久 park 态,列后续里程碑。与 `resume`(崩溃重放、无值注入)是不同动词。
- **决策载体**:`approve` 复用 tool 的 `args` 作人工决策 dict(脚本 `ctx.checkpoint` 的返回值,须 JSON-可序列化);术(命令名/参数形状)只入 tool description / `help` / SKILL.md,绝不入 demo prompt(道/术线)。
- **安全/防御**:authored(meta)脚本经 AST gate **禁调 `ctx.checkpoint`**(签核是注册工作流能力);引擎对"注入决策却无门消费"**fail-loud**;门入确定性序列,漂移 fail-loud。

demo 消费(`run_workflow_live` 内联捕 `WorkflowSignoffRequired` → 经 `_ResumeLane` 跨轮持久 journal 续跑 + `signoff_gate` Gen-UI 卡片原地翻面)见 demo-app;集成示例见 [`examples/features/signoff.py`](../examples/features/signoff.py);时序见 [uml/03-sequence.md](uml/03-sequence.md) G 图。

# 01 · 引擎机制设计（Engine Mechanism Design）

> **范围**：引擎内部（Layer 0 底座 + Layer 1 编排运行时）"怎么算对"的机制设计——控制流反转、八原语、骑 LangGraph 的两块补丁、脚本执行模型、叶子调用契约、sandbox、pipeline、race、budget、确定性。
> **对外软件形态**（怎么接入 agent、tool/skills/middleware）见 [02-architecture.md](02-architecture.md)；图见 [uml/](uml/)。
> **日期**：2026-06-01　**状态**：机制已锁。
> **勘误**：官方 Claude Code 编排语言是 JS，Anthropic 只文档化行为契约、从未发布原语级 API。本库 Python 原语镜像社区逆向出的 JS 表面，**非**官方 API；本文档才是本端口预期行为的权威。

---

## 1. 核心范式：控制流反转

| | 普通 agent | Dynamic Workflow |
|---|---|---|
| 谁决定下一步 | LLM 逐回合 | **确定性脚本（代码）** |
| 中间结果存哪 | LLM 的 context window | **脚本变量里** |
| 最终进主 context 的 | 全过程 | **只有结论** |

循环 / 分支 / 扇出写成确定性代码；LLM 只在叶子 `agent()` 出现,每个 subagent 跑在**全新、用完即弃**的 context 里、只吐结果。

## 2. 八个原语

| 原语 | 语义 |
|---|---|
| `agent(prompt, *, schema, agent_type, model, label, isolation)` | 起全新 context 的 subagent;带 `schema` 强制结构化输出 + 校验 + 不匹配重试。`schema` 可为 pydantic 类或 **JSON-schema dict(L2 脚本禁 import 时的内联形态,引擎经 `to_pydantic_model` 归一为 pydantic)**;脚本下一行直接属性访问 |
| `parallel(thunks)` | 并发 + **阻塞 barrier**;thunk 抛错 → 该位 `null`,整体**永不 reject**(用前 `.filter`) |
| `pipeline(items, *stages)` | 多 stage **无 barrier** 流水线;stage 签名 `(prev_result, original_item, index)`;抛错 → 该 item `null` 跳后续 |
| `race(candidates, *, win, win_tag="")` | **best-of-N 早退**:N 个 `RaceCandidate`(镜像 `agent()` 入参)经 `agent()` 并发,第一个令 `win(result)` 为真者胜,在飞 loser 全数 cancel;决策**内容哈希 journal**(`win_tag` 折进 key)——resume 复现胜者、**零派发**;无胜者**不** journal(resume 可重试)。返回 `RaceResult`(`won`/`winner`/`winner_index`);候选须同构(全无 schema 或全同一 schema)。靠两补丁:race-key 用 content-hash journal、确定性 guard 只在深度 0 observe race-key(候选 `agent()` 在深度 > 0、不入序列,同 `parallel`/`pipeline` 叶) |
| `phase(title)` / `log(msg)` | 进度分组 / 叙事日志 |
| `budget` | `{total, spent(), remaining()}`,**共享池**,到顶 `agent()` 抛 |
| `workflow(name, args)` | 内联调另一 workflow,**仅一层嵌套**(内层 = `@task`) |

失败语义照搬 Claude Code:`parallel` 永不 reject、`pipeline` 抛错落 `null`;`race` 单个候选叶失败仅淘汰该候选、其余继续,引擎控制流信号(budget/确定性)或 `win` 谓词抛错则在拆除 loser 后**失声而抛**(fail-loud)。

**跨叶归约 helper(`_reduce`,纯函数,F)**:折叠 `parallel` / `pipeline` 交回的结果列表(失败叶=`None`)的一等公民——`survives`(refute-by-default 投票,`None` 恒计反对的 fail-safe,覆盖 adversarial-verify 与 judge-panel)、`dedup`(丢 `None` + 按 key 去重,保首见序)、`reconcile`(双盲复核分桶 included/excluded/conflicts,`None`/空裁决恒落 conflict)、`corroborate`(按 key 分组、≥`min_support` 才留的跨叶相互印证),配 `ReviewItem` / `Reconciled` / `Consensus` 三个 frozen dataclass。它们**无 `agent()` 调用、无引擎状态**,故天然 replay-safe、不碰 journal/确定性 guard;由包根导出供开发者 workflow `import`,并由 `_codegen` 注入 `run_script` 命名空间(L2 脚本禁 import,故按名直调)。

**race 公共面(`_race_types`,纯值类型 + `race_key`,B)**:`ctx.race`(原语)的开发者面是两个 frozen dataclass——`RaceCandidate`(`prompt`/`agent_type`/`schema`/`model`/`isolation`,镜像 `agent()` 入参,故候选 journal-key 与直接 `agent()` 同源)与 `RaceResult[T]`(`winner`/`winner_index` + `.won` 属性),配 `_journal` 内的 `race_key`(对候选叶 key 序列 + `win_tag` 取 SHA-256、`"race"` 命名空间隔离,绝不与叶 key 撞)。两个值类型**无 `agent()` 调用、无引擎状态**,与 `_reduce` 同级:由包根导出供开发者 workflow `import`,并由 `_codegen` 注入 `run_script` 命名空间(L2 脚本禁 import,故按名直调);`race_key` 仅导出(脚本走 `ctx.race`、不直接碰 key)。`SpanKind.RACE` 标注 race 扇出 / journaled-decision replay 的 span。

这些是"能写什么"；用好它们的**作者模式库**（adversarial-verify、pipeline-by-default、loop-until-dry + 硬 MAX_ROUNDS、judge-panel、model-routing…）及其确定性适配见 [03-authoring-patterns.md](03-authoring-patterns.md)，可运行投影在 `skills/dynamic-workflow/SKILL.md`。

## 3. 底座同构：LangGraph Functional API + 两偏差（实证）

LangGraph functional API 与 Claude Code Workflow 是同一范式,durable execution 白送大半:

| 维度 | LangGraph 1.2.2（实读源码） |
|---|---|
| 控制流归属 | `@entrypoint` 函数体(普通 Python/asyncio) |
| 工作单元 | `@task`(返回 `SyncAsyncFuture`,可 await 可 `.result()`) |
| resume | 重放 entrypoint body,已完成 `@task` 凭 `task_id` 取缓存不重跑(`_loop.py:724-737` / `_runner.py:745-759`) |
| barrier 并行 | 多 future 先发起、await 处隐式 barrier(`asyncio.gather`) |
| 持久化 | checkpointer + 三档 `durability`(默认 `"async"`,`main.py:2574`) |

**两偏差 → 两补丁(必需,非可选)**。根因:LangGraph 假设 entrypoint body 是**人写可信代码**,本项目脚本是 **LLM 现写的不可信代码**。

### 偏差①(实证改写):缓存键是两套机制,原设计混为一谈

- **结果缓存(CachePolicy)是内容寻址,不是 index-based**:`default_cache_key = pickle.dumps((_freeze(args),_freeze(kwargs)))` 再 xxh3-128(`_internal/_cache.py:26-31`、`_algo.py:858-870`),且 **opt-in**。
- **真正 positional 的是 `task_id`**:编码 step + 节点名 + write 索引(`_algo.py:834-842`),驱动 resume replay-skip。

### 偏差②(实证):确定性零强制

全底座唯一确定性检查是一句 `assert task_id == task_id_checksum`(`_algo.py:662,855`),被窄守卫包裹且 `python -O` 下**整句剥离**;`errors.py` 无任何确定性异常类;resume 仅按 `task_id` 贴 writes、零比对。

## 4. 补丁① · content-hash journal（success-only）

引擎在 `@task` 之上自建内容哈希 journal:

```
schema(dict 来源)先经 to_pydantic_model 归一为 pydantic 模型,再取其 model_json_schema() 入 key:
key = sha256(canonical_json({prompt, agent_type, model,
        schema: model.model_json_schema() if schema else None, isolation}))
命中(且 success) → 反序列化缓存结果(连 @task 都不进,0 模型调用)
        ├─ 有 schema → model_validate_json(缓存 JSON) 还原结构化对象
        └─ 无 schema → 直接返缓存文本
未命中 → 起 @task 跑 deepagent → 校验 → 写 journal(有 schema 存 model_dump_json,连同 usage)
```

**正当理由(实证三条,替代原"native 是 index-based"的错误论证)**:

1. **success-only 语义**:bug `#7589`——同步 `SyncPregelLoop.put_writes`(`_loop.py:1586`)缓存结果**无 INTERRUPT/ERROR 守卫**(async 路径有),失败/中断的 task 会被缓存并 replay 成 success。journal 必须显式只写 success。
2. **per-node content scoping**:原生 ns 仅按函数 qualname 命名,有跨调用点串用风险。
3. **positional resume identity**:`task_id` 含 step+write_idx,脚本顺序漂移即静默失配。

（附:原生 cache 命中还会丢自定义 stream 数据 `#6265`。）

- `JournalStore` Protocol:v1 交付 **in-memory 实现（默认，进程内）**;LangGraph `BaseStore`-backed 持久化(跨会话 resume 的底座,namespaced KV)是 Protocol 之后的**文档化扩展点,未交付**。故默认 resume 是同进程语义,跨会话需接一个持久 `JournalStore`。
- 命中后若有 `schema`,以 `model_validate_json` 把缓存 JSON 重新校验回归一后的 pydantic 模型实例(`to_pydantic_model` 的等值-dict 同类缓存保证 `model_json_schema()` 逐字节稳定 → resume 不静默重跑)。

## 5. 补丁② · 确定性 fail-loud guard（三段式）

确定性**重定义**:不禁绝一切非确定性,只在"非确定性改变了编排的可观测 `agent()` 调用模式"时炸。journal 即确定性 oracle。

| 段 | 机制 | 覆盖 |
|---|---|---|
| 预防(便宜) | AST 禁 `import` + 受限 builtins | 仅 L2(不可信源) |
| 普适 backstop | journal 记调用序列,重放不匹配即 **fail-loud** | 所有源(含手写) |
| 引擎自持不变量 | `budget.spent()` 重放可重建、`phase`/`log` 幂等 | 引擎自己 |

它顺带把 budget 重放分叉从"静默腐坏"降级成 loud failure。`python -O` 会蒸发底座那句唯一 assert——又一条 guard 必自建的理由。

## 6. 脚本执行模型（接缝① · 方案丙）

执行核统一为"跑一个 async callable",两道前门:

- **手写(可信)**:直接传 `async def orchestrate(ctx)` → 安全维度关、确定性维度仍开。
- **L2/不可信**:源码字符串 → AST gate → 受限 globals 下 `compile` + **单点** `exec` 成 callable → 完整 guard。

两条路都 checkpoint **源码/注册键**当 entrypoint 输入,resume 时重铸 callable——**callable 临时、源码持久**。callable 是 exec 模型的严格超集,trust 边界显式分级。

**guard 两正交维度**:安全维度(防 exec 逃逸/读文件网络;仅 L2)、确定性维度(防影响编排的非确定性;所有源)。

## 7. 叶子调用契约（接缝②D，verified-in-source）

- **roster 条目**(Builder-roster,D-G1a) = `RosterEntry{name, description, needs_execution, default_model?}` + **`runnable` / `builder` 二选一**:`runnable` 是预构造的 `CompiledStateGraph`(`create_deep_agent(...)` 懒构造一次、作 `@task` 直调,**绕开** deepagents 的 LLM-driven `task` 工具),**仅服务 schema-less**;`builder = (*, response_format) -> Runnable` 是工厂,按需以 `response_format=ToolStrategy(...)` 构造结构化变体,使 `agent(schema=)` 可用。`register(...)` 互斥校验:不给或都给即 fail-loud。不碰私有 `_SubagentSpec`。
- **`runnable_for(name, *, response_format)` 解析 + 构建缓存**(D-G1b):`response_format=None` 取 schema-less 变体(builder 条目调一次 builder、`runnable` 条目直接返预构造体);带 `response_format` 则要求 builder 条目(预构造 `runnable` 条目 fail-loud),按 `(agent_type, response_format identity)` 缓存绑定变体——`identity` 取被绑 pydantic 模型 `model_json_schema()` 的 sha256(与 journal key 同源)。缓存归 Roster(进程级、并发安全),因编译图跨 run 无状态,resume 不重建。
- **调用** = `runnable.ainvoke({"messages":[HumanMessage(prompt)]}, config=..., context=...)`。
- **结果回填**(镜像 `subagents.py:494-532`,即 context-quarantine 边界):有 `structured_response` → 序列化(pydantic `model_dump_json` / dataclass `asdict`+`json.dumps` / 否则 `json.dumps`);否则**逆序**扫 `messages` 取第一条 `.text.rstrip()` 非空的 `AIMessage`(避开 Anthropic 尾部空 `end_turn`);`messages` 缺失抛 `ValueError`。`agent(schema=)` 路径走 `fold_structured(state, model)`——取 `structured_response`,缺失**或类型不匹配**(`isinstance` 校验,防 builder 绑错 `response_format`)均 fail-loud;回填序列化用 `model_dump_json(by_alias=True, round_trip=True)`,保带 field alias 的 schema 经 resume 往返不裂(`model_validate_json` 默认按 alias 校验)。
- **schema** = 脚本传 pydantic 类或内联 JSON-schema `dict`,后者经 `to_pydantic_model` 归一为 pydantic 模型(进程级缓存,等值 dict → 同一类 → 同一 `model_json_schema()`;对 `$ref`/`allOf`/`anyOf`/`oneOf`/`patternProperties`/`not`、非 bool 的 `additionalProperties`、不在 `properties` 中的 `required`、未支持的约束关键字(`pattern`/`minimum`/`format`/`const`…)、值相等坍缩的枚举(`True==1`/`1==1.0`)一律 **fail-loud,不静默降级**,并设递归深度/字段数/枚举规模/缓存条目数护栏防资源耗尽);引擎以 `response_format=ToolStrategy(model, handle_errors=True)`(in-loop 纠错重试)构造叶子,journal 只缓存校验过的 `model_dump_json`、命中以 `model_validate_json` 还原。`ProviderStrategy` 无 in-loop retry,不作默认。
- **budget 管线**:`UsageMetadataCallbackHandler` 跨嵌套聚合 token;但绕开 task 工具直调时**必须自己复刻 `_build_subagent_config` 的 callbacks 转发**,否则共享 budget 漏算;**每叶子 usage 入 journal** → 保 `spent()` 重放可重建。
- **state schema**:自定义须继承 `DeepAgentState`(其 `DeltaChannel(snapshot_frequency=50)` 把 checkpoint 增长压到 O(N))。
- **只读裁判叶(D-G4)**:库级 `read_only_leaf` / `read_only_builder` 以 `create_deep_agent(permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")])` 构造**工具面只读**的叶——可 read/grep/glob/ls,写/编辑在工具边界被拒(实测:fake 模型调 `write_file` → 无文件落地)。deepagents 无 execute 权限维度,故只读裁判同时 `needs_execution=False`(走 `StateBackend`、无 execute 工具,该默认后端遵守 deny-write)。adversarial-verify / judge-panel 的裁判注册为只读叶 → 幻觉修复落不了地("生成"与"判定"分离);builder 形态支持 `schema=`(裁判返 `Verdict`)。只读是工具面属性、归宿主侧装配,引擎不参与。

## 8. sandbox 机制（接缝②E，verified-in-source）

- **默认隔离粒度 = per-leaf**(每个 `agent()` 叶子一个隔离 sandbox);协同工作区(多叶子共享可变工作区)做 opt-in 风险模式。
- **身份从 journal key 派生**:一举满足 retry 稳定 / resume 稳定 / 唯一性 / 与 journal dedup 自洽。
- **构造方式**:弃用 `BackendFactory`(deprecated 0.5.0、移除 0.7.0),改 docs 推荐的 **per-leaf backend 实例**——从 `runtime.config["configurable"]["thread_id"]` 读身份、find-or-create 打标 sandbox、包成实例传给 `create_deep_agent`。
- **SandboxManager 自建**(底座零生命周期方法实证:`BackendProtocol`/`SandboxBackendProtocol` 仅文件操作 + `id`/`execute`,lifecycle grep 零命中):lazy-create / idle+硬 TTL / 池化 / 配额(最大活跃数、per-sandbox 工具调用上限)/ 池耗尽背压 / `sandbox.stop()`。
- **分层准入**:roster `needs_execution` 元数据;纯推理 agent 走 StateBackend **不分配 sandbox**。
- **CompositeBackend**:`/shared/` 路由共享产物 store(显式 hand-off、版本化、producer 命名空间、路由前路径规范化防穿越);**但 `#2884`(OPEN)route 隔离会在共享存储后端间泄漏 → 并行叶子隔离不能仅靠 routes,须独立验证**。
- **并发安全 stance**:默认假设单叶子内 deepagents 可能并发调工具 → per-leaf sandbox 访问默认串行化,实测安全再放开。
- **`isolation="worktree"`(D-G2)**:并行改文件的 fix 叶各跑在**从 base 快照播种的隔离可变副本**里——`WorktreeProvider.seed(leaf_id)` 给一份隔离拷贝,`SandboxManager._new_sandbox` 在 slot **新建时**用 `upload_files` 播种(retry 复用不重播),`isolation` 经 `agent → leaf_task → lease` 透传。叶子改完**返回 `Patch`**(G1 `schema=`,落实"生成"与"应用"分离)。v1 默认语义 = 内存播种副本(`InMemoryWorktreeProvider`);真 git-worktree 后端(`git worktree add` per 叶 + `git diff` 作 `collect`)是同 `WorktreeProvider` 协议后的可插拔生产实现,未在 v1 交付。`"shared"`(默认)维持空 sandbox 的既有 per-leaf 隔离。

## 9. pipeline 调度器（无 barrier，自建——LangGraph 结构盲区）

```
每 stage 一个 bounded asyncio.Queue(背压,防 item 海啸打爆内存)
每 stage 一组 worker:pull → 跑 stage fn(内部调 agent())→ push 下级 queue
全局 semaphore = min(16, cores-2),跨所有 stage 共享
item 各自独立穿越 → A 在 stage3 时 B 还在 stage1
stage 抛错 → 该 item 掉 null 跳后续;结果按输入下标回收保序
```

底座无任何无-barrier 流式原语(`Send` 是 map-reduce barrier);完全自建。中途异常/预算耗尽须保证队列优雅排空、不死锁。

## 10. budget

共享 token 池 `{total, spent(), remaining()}`,到顶 `agent()` 抛。`spent()` 由 journal 中每叶子 usage 重建 → 重放确定。计量底座:`usage_metadata`(AIMessage)+ `UsageMetadataCallbackHandler`;`ModelCallLimitMiddleware` 可作只计次的粗兜底。

## 11. 并发上限 & 硬上限

- **双层都显式设**:asyncio `Semaphore(min(16, cores-2))` + LangGraph `RunnableConfig.max_concurrency`(底座**默认 None ⇒ 无界**,`_executor.py:135-140`,必须显式设界)。
- 总量硬顶 `1000`(防失控)。

## 12. Decision Log

| # | 决策 | 选定 |
|---|---|---|
| D1 | 项目形态 | 独立开源库,面向 deepagents 社区 |
| D2 | dynamic 边界 | 含完整 meta 层(LLM 写脚本) |
| D3 | 脚本语言/执行模型 | Python 原生,骑 LangGraph |
| D4 | 安全边界 | A1 进程内受限 exec 起步,执行器抽可替换 seam,预留 A2/A3 |
| D5 | `pipeline()` | 进 v1 |
| D6 | `workflow()` 嵌套 | `@task`/subgraph 实现 |
| D7 | `agent()` 解析 | R1 纯命名 roster |
| D8 | journal 存储 | `JournalStore` Protocol;v1 交付 in-memory 实现(默认、进程内);`BaseStore` 持久化(跨会话)是文档化扩展点,未交付 |
| D9 | 确定性实现 | import-ban + 受限 builtins + 教 LLM `sorted()`/忌迭代 set |
| D10 | pipeline 背压 | bounded `asyncio.Queue` |
| D11 | Layer 2 codegen | 一次过:AST 校验通过即执行(违规重试),不做 dry-run |
| D12 | journal 哈希 | 不纳入 agent 定义/版本哈希 |
| D13 | sandbox 生命周期 | per-leaf 隔离默认;协同工作区 opt-in |
| D14 | 接缝① 脚本执行模型 | 方案丙:callable 本体 + 源码前门 + exec 收敛 L2 单点 |
| D15 | guard 维度切分 | 安全维度(仅 L2)/ 确定性维度(所有源)两正交 |
| D16 | 确定性强制 | 三段式:AST 预防 + journal-divergence backstop + 引擎自持不变量;不 monkeypatch |
| D17 | sandbox 身份/构造 | per-leaf 隔离 + journal-key 派生身份 + per-leaf backend **实例**(弃 deprecated factory) + 自建 SandboxManager |
| D18 | 接缝②D schema 强制 | `ToolStrategy(schema, handle_errors=True)` in-loop 重试 |
| D-G1a | roster 注册形态(schema 落地) | **callable builder**(`(*, response_format) -> Runnable`),否决"roster 持 deep-agent kwargs 自建":依赖倒置(引擎不耦合 deepagents 构造签名)+ roster 通用性(任意 `Runnable` 工厂皆可)+ 宿主稳定性(`response_format` 是构造期参数、预构造 runnable 无法事后改)。`runnable` / `builder` 互斥,前者仅 schema-less |
| D-G1b | 构建缓存归属 | **缓存归 Roster**(`(agent_type, schema) -> Runnable` 进程级、并发安全):内聚(紧邻持有 builder 的 roster)+ 生命周期匹配(编译图跨 run 无状态,进程级正合)+ 构建期无外求 + resume 不重建。二者在"预构造 runnable 无法事后绑 `response_format`"约束下,落实了 D18 的逐次 schema 绑定 |
| D-G2 | `isolation="worktree"` 保真度 | **v1 默认 = 内存播种副本**(`InMemoryWorktreeProvider`:`seed` 给隔离 base 快照拷贝、`collect` 算相对 seed 变更集)+ **真 git-worktree 后端作同 `WorktreeProvider` 协议后的可插拔生产实现**(未在 v1 交付)。否决"仅文档化 seam"(不兑现卖点)与"v1 直接上真 git worktree"(与 offline-first 跨度大)。`SandboxManager._new_sandbox` 仅 slot 新建时播种;`isolation` 经 `agent → leaf_task → lease` 透传;fix 叶复用 G1 `schema=Patch` 自报变更(生成/应用分离) |
| D-G4 | read-only judge 形态 | **库级辅助**(`read_only_leaf` / `read_only_builder`)+ deny-write `FilesystemPermission` + `needs_execution=False`,否决"引擎内建只读 agentType":引擎不构造叶(宿主构造),只读是工具面属性归宿主侧;deepagents 无 execute 权限维度,靠不分配 sandbox 禁执行 + deny-write 禁写,叠加才是真只读;builder 形态复用 G1,只读 + 结构化裁决一行可得 |
| D19 | 接缝③ L2 交付节奏 | v1 = L0/L1 先行,L2 架构预留紧跟(L2-as-skill,见 02) |
| D20 | async 后台 tool 执行 | 自建轻量后台机制(无 server / 无重依赖);v1 即含;蓝本 = omne-next 实现 + deepagents async-task API 形态。详见 [02 §3](02-architecture.md) |

## 13. 实现待核实清单（开工前/中逐条钉测试）

1. **journal × 原生 cache 交互**:引擎统一走 async(避 `#7589` sync error-caching);显式决定是否关原生 CachePolicy、让 journal 成唯一记忆化源。
2. **`task_id` 顺序敏感性**:加"脚本编辑后 resume"集成测试——顺序漂移会静默失配重跑。
3. **`max_concurrency` 嵌套语义**:确认叶子 fan-out 是共享 entrypoint 层 semaphore,还是 deepagents 子调用另开无界 executor。
4. **CompositeBackend 隔离泄漏(#2884)**:并行叶子隔离独立验证。
5. **callback 转发**:`@task` 层直调须复刻 `_build_subagent_config` callbacks 转发,否则共享 budget 漏算。
6. **`-O` 风险**:生产开 `PYTHONOPTIMIZE` 时底座唯一 determinism assert 蒸发——再证 guard 必自建。
7. **单叶子内 deepagents 是否并发调工具**:决定 per-leaf sandbox 是否需内部串行化。
8. **硬契约逐条钉测试**:journal-key 派生身份 / retry 时 thread_id 稳定 / 路径规范化防穿越 / pipeline 异常不死锁。

---

> 信源(版本锚定 langgraph 1.2.2 / langchain 1.3.2 / langchain-core 1.4.0 / deepagents 0.6.7):见 `research/`。

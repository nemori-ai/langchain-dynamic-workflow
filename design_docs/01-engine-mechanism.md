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
| `batch_map(items, fn, *, max_in_flight=None, total=None, label="batch_map")` | **流式准入扇出**:把一个异步 `fn` map 到 `Iterable`/`AsyncIterable` 每个 item,经**有界准入窗口**懒消费(N 千 item 永不一次性物化 N 千 task)、结果按输入序回收(`list[T \| null]`,失败叶落 `null`、永不 abort barrier);自动发 transient count/ETA 进度;见 §9b |
| `race(candidates, *, win, win_tag="")` | **best-of-N 早退**:N 个 `RaceCandidate`(镜像 `agent()` 入参)经 `agent()` 并发,第一个令 `win(result)` 为真者胜,在飞 loser 全数 cancel;决策**内容哈希 journal**(`win_tag` 折进 key)——resume 复现胜者、**零派发**;无胜者**不** journal(resume 可重试)。返回 `RaceResult`(`won`/`winner`/`winner_index`);候选须同构(全无 schema 或全同一 schema)。靠两补丁:race-key 用 content-hash journal、确定性 guard 只在深度 0 observe race-key(候选 `agent()` 在深度 > 0、不入序列,同 `parallel`/`pipeline` 叶) |
| `dag(nodes)` | **依赖序（拓扑序）fan-out**——第四种扇出帧:见 §2c |
| `loop_until(body, *, done, max_iters)` | **有终止保证的顺序循环**:见 §2d |
| `phase(title)` / `log(msg)` | 进度分组 / 叙事日志 |
| `budget` | `{total, spent(), remaining()}`,**共享池**,到顶 `agent()` 抛 |
| `workflow(name, args)` | 内联调另一 workflow,深度上限可配置(默认 8 层,含循环检测);见 §2e |

失败语义照搬 Claude Code:`parallel` 永不 reject、`pipeline` 抛错落 `null`、`batch_map` 失败 `fn` 落 `null` 不 abort(共享 `pipeline` 的 drop-to-`null` 引擎,见 §9b);`race` 单个候选叶失败仅淘汰该候选、其余继续,引擎控制流信号(budget/确定性)或 `win` 谓词抛错则在拆除 loser 后**失声而抛**(fail-loud)。**内部 `CancelledError`(thunk/stage 的子任务被取消、而非整 run 被外部拆除)按普通叶失败掉 `null`**,不混作控制流信号抛——内/外取消之分与拆除 await 见 §9。

**跨叶归约 helper(`_reduce`,纯函数,F)**:折叠 `parallel` / `pipeline` 交回的结果列表(失败叶=`None`)的一等公民——`survives`(refute-by-default 投票,`None` 恒计反对的 fail-safe,覆盖 adversarial-verify 与 judge-panel)、`dedup`(丢 `None` + 按 key 去重,保首见序)、`reconcile`(双盲复核分桶 included/excluded/conflicts,`None`/空裁决恒落 conflict)、`corroborate`(按 key 分组、≥`min_support` 才留的跨叶相互印证),配 `ReviewItem` / `Reconciled` / `Consensus` 三个 frozen dataclass。它们**无 `agent()` 调用、无引擎状态**,故天然 replay-safe、不碰 journal/确定性 guard;由包根导出供开发者 workflow `import`,并由 `_codegen` 注入 `run_script` 命名空间(L2 脚本禁 import,故按名直调)。

**race 公共面(`_race_types`,纯值类型 + `race_key`,B)**:`ctx.race`(原语)的开发者面是两个 frozen dataclass——`RaceCandidate`(`prompt`/`agent_type`/`schema`/`model`/`isolation`,镜像 `agent()` 入参,故候选 journal-key 与直接 `agent()` 同源)与 `RaceResult[T]`(`winner`/`winner_index` + `.won` 属性),配 `_journal` 内的 `race_key`(对候选叶 key 序列 + `win_tag` 取 SHA-256、`"race"` 命名空间隔离,绝不与叶 key 撞)。两个值类型**无 `agent()` 调用、无引擎状态**,与 `_reduce` 同级:由包根导出供开发者 workflow `import`,并由 `_codegen` 注入 `run_script` 命名空间(L2 脚本禁 import,故按名直调);`race_key` 仅导出(脚本走 `ctx.race`、不直接碰 key)。`SpanKind.RACE` 标注 race 扇出 / journaled-decision replay 的 span。

## 2c. DAG / 依赖序（拓扑序）fan-out（第四种扇出帧）

`ctx.dag(nodes)` 是继 `parallel`/`pipeline`/`race` 之后的**第四种扇出帧**:给每个节点声明前置依赖,调度器确保节点只在全部前置节点都完成后才启动;有依赖关系的节点可任意拓扑深度、独立分支照常并发。

### 值类型:`DagNode`

```
DagNode(id: str, deps: Sequence[str], run: Callable[[dict], Coroutine])
```

- `id`:节点在图内的唯一名称;结果 dict 以此为 key 返回。
- `deps`:此节点依赖的其它节点 id 列表;根节点传空列表(`deps=[]`)。
- `run`:接收 `{dep_id: result}` 映射的异步工厂,通常是一个闭包内调 `agent()`。

`DagNode` 是一个纯值类型(`@dataclass(slots=True)`)——无引擎状态、无 `agent()` 调用——由 `_codegen` 注入 `run_script` 命名空间(L2 脚本禁 import,按名直构造)。

### 调度器(`_dag.run_dag`)

1. **急切合法性校验**:在任何节点开始运行之前一次性验证——重复 id、依赖未知 id、自依赖、依赖环——均 fail-loud 抛 `WorkflowDagError`。宁在 O(N) 校验时炸,不在调度过程中死锁。
2. **Kahn 入度调度**:维护每个节点的剩余前置数;某节点最后一个前置节点一完成,该节点**立即**启动——无层级 barrier,快分支超前于慢分支(同 `pipeline` 的 no-barrier 精神)。
3. **失败与跳过的区分**:一个节点的 `run` 抛出普通异常 → 该节点 `failed`,结果置 `None`;所有**传递依赖**于它的节点均被 **skip** 至 `None`(不会启动)。但节点**合法返回 `None`** 不会触发 skip——失败状态和 `None` 返回值用**独立的 `failed: set`** 跟踪。
4. **控制流信号穿透**:budget / determinism / 格式非法的嵌套 dag(`WorkflowDagError`/`WorkflowCycleError`)属于 `WORKFLOW_CONTROL_FLOW_SIGNALS`;任一节点抛出时,记录信号、停止发射新节点、等在飞节点排空(drain),然后**原样重新抛出**——绝不被 mask 为 `None` 空洞。
5. **并发闸归属于叶子**:调度器本身不持有任何并发 gate slot;`run` 函数内的 `agent()` 调用才获取共享闸——故一个 `dag` 嵌套在另一 `dag` 节点的 `run` 内不会死锁池。

### 确定性与 resume 安全

`ctx.dag` 是扇出帧(递增 `_FANOUT_DEPTH`),其内部节点的 `agent()` 调用不记入顺序序列 guard(完成顺序随墙钟变化)。每个叶子仍由内容哈希 journal 守护——resume 时完成节点零模型成本重放。DAG 图的拓扑结构由脚本代码定义(确定性),故**无需 dag 级 journal 条目**;resume 重新执行调度器,命中的叶子直接短路,未命中的继续运行。**journal 是纯成功记录**（与 `parallel`/`pipeline`/`race` 引擎一致）：失败或被 skip 的节点不入 journal，crash 重试时这些节点会重新运行——完成（成功）节点仍从内容哈希 journal 零成本重放，不重跑模型。

## 2d. `loop_until` — 有终止保证的顺序循环

`ctx.loop_until(body, *, done, max_iters)` 是一个把两条作者纪律固化进 API 的**顺序(深度-0)循环 helper**:

- **停止谓词覆盖全部累积结果**:`done(accumulated)` 每次迭代后以**完整列表**调用,而非仅看最新一轮——保证 dedup / 收敛判断针对所有轮次的见过的内容。
- **强制 `max_iters` 硬上限**:必须传入正整数;到达上限不抛异常——改发一条 replay 幂等的 `log(...)` 行后返回已积累的列表(graceful、非 silent)。

签名:`body(iter_index, accumulated_so_far) -> result`;每轮调用一次,结果 append 进 `accumulated`,然后检查 `done(accumulated)`,满足即提前返回。

顺序运行意味着 `body` 内的直接 `agent()` 调用记入确定性序列;resume 时循环次数由 journal 结果自然恢复(相同的 `done` 条件在相同的累积结果下产生相同的停止轮次),完成叶零成本重放。

## 2e. 命名工作流嵌套:深度上限 + 循环检测

`ctx.workflow(name, args)` 可内联另一注册工作流,V0.3.0 M7 将 1 层硬限**放开为可配置深度上限**(`DEFAULT_MAX_WORKFLOW_DEPTH = 8`),并加上独立的**循环检测**。

### 深度上限

- 每次 `ctx.workflow()` 进入时递增 `_WORKFLOW_DEPTH` contextvar;退出时在 `finally` 重置。
- 若即将超过 `max_workflow_depth`(默认 8)时抛 `WorkflowNestingError`——是"跑飞的递归"的最后防线,**并入 `WORKFLOW_CONTROL_FLOW_SIGNALS`**（深度上限越界是结构性/跑飞递归错误,同 `WorkflowDagError`/`WorkflowCycleError`——fan-out 帧内必须 fail-loud,不能被 mask 为 `None` 空洞;Codex 跨模型评审驱动此修正）。

### 循环检测

- `_WORKFLOW_NAME_STACK` contextvar 记录当前正在内联的工作流名集合(`frozenset`)。
- 若目标名已在集合中(直接递归 `A→A` 或相互 `A→B→A`)立即抛 `WorkflowCycleError`——比等到深度上限触发给出更精确的诊断。`WorkflowCycleError` **并入** `WORKFLOW_CONTROL_FLOW_SIGNALS`(结构错误,同 `WorkflowDagError`),在 fan-out 帧内 fail-loud。

### 确定性与 resume 安全

内层工作流的 `orchestrate` 以 `inline` 方式在父 `@entrypoint` body 内运行,共享同一 journal / budget / concurrency gate / progress log——叶子 call-key 和结果由内容哈希守护,resume 同样零成本重放。嵌套结构由脚本代码定义(确定性),故不需要嵌套层本身的 journal 条目。

这些是"能写什么"；用好它们的**作者模式库**（adversarial-verify、pipeline-by-default、loop-until-dry + 硬 MAX_ROUNDS、judge-panel、model-routing…）及其确定性适配见 [03-authoring-patterns.md](03-authoring-patterns.md)，可运行投影在 `skills/dynamic-workflow/SKILL.md`。

## 2b. Span 生命周期与逐叶 live 可观测性（带外、隔离不变量保持）

每个原语调用(`agent` / `parallel` / `pipeline` / `race`)经 `SpanRecorder.span()` 开一个 span,期间产**两条**带外边——**打开**即发的 `SpanBegin`(running 边)与**关闭**时发的既有完成 `Span`。两条边共享一个 `span_id`,故消费者据此把 running 与 done 对上。

- **两条边的载荷分工**:`SpanBegin` 携 `started_at`(墙钟,供 `now - started_at` 算 live elapsed timer)+ `monotonic_start`(无漂移内部时长基准)+ 打开时已知的 `attributes`;完成 `Span` 携 `duration_s`(monotonic 差)/ `error` / cache 结果(`cached`)/ usage 等关闭时才知道的字段。
- **`span_id` 由谁铸**:引擎铸,消费者只读。`span_id` = `(kind + name + 出现序号)` 经 `json.dumps(sort_keys=True)` 后 SHA-256 截 16 hex;出现序号按 `SpanRecorder` 实例(即每 run)对每个 `(kind, name)` 计数(第 N 个同名同 kind 的 span 取序号 N,故同名叶子彼此区分)。**顺序路径(深度 0)resume 稳定**——脚本同源序重放(确定性 guard 背书)、序号每 run 重置,故 fresh 与诚实 resume 铸出逐字节相同的 id 序列。**扇出 span 的打开顺序随墙钟变化**,其 `span_id` 不保证 fresh/resume 一致(同"扇出叶不入确定性序列"同因);故 resume 稳定性仅对顺序深度-0 span 担保。
- **三个带外 sink(皆 keyword-only、默认 `None` no-op、不入 journal、live-only)**:
  - `on_span_begin`——**全 span 类型**,打开即发。它**不**被 replay 抑制:resume 时为每个被重放(cached)的叶子重发一条 begin,且其完成 `Span` 标 `cached=True`、`duration_s` 近零——故缓存叶子渲染成"即时命中"而非卡住的 running chip(begin 发于 span 打开、早于 journal 查询,故天然"先发后定")。
  - `on_leaf_event`——把**叶子自己的回调子树**(其 model/tool/chain 的 `on_*_start`/`on_*_end`/`on_*_error` 边)normalize 成 `LeafEvent`,经 `run_id`/`parent_run_id` 可重建叶内 run tree。handler 挂在既有 `leaf_config["callbacks"]` 列表上(见 §7 budget 管线复刻的同一条 callbacks 转发路径);**deepagents 把 `callbacks`/`tags`/`configurable` 转发给 subagent,但不转发 `metadata`**,故关联**不**靠 metadata 继承,而靠 handler 实例在构造时闭包持有的 `leaf_span_id`(一叶一 handler,它收到的每条边按构造即属本叶子)。它**仅真执行触发**:handler 只在 journal **未命中**走真叶子时挂上,journal 命中走缓存路径根本不进叶子 runner,故重放叶子**零** interior 事件。
  - `on_command`——把执行叶子跑的**真 shell `execute`** normalize 成成对的 `CommandEvent`。它是 `on_leaf_event` 的执行面同构体,但触发点不在 LangChain 回调,而在**真 subprocess 执行边界**:一条 `"start"` 边在 `subprocess.Popen` **之前**发(`command` / `started_at`,`exit_code=None`),一条 `"end"` 边在子进程被 reap **之后**发(`exit_code` / 有界 `output` / `duration_s`)——故消费者命令一起就画一张 terminal card、完成即原地翻 pass/fail。两条边共享一个 **resume 稳定的 `command_id`**(对 `(leaf_span_id, command, 叶内出现序号)` 取 SHA-256 截 16 hex,故同一叶子里重复的同名命令——如 fix-loop 里堆叠的 `bun test`——得各自不同的 id,而单条命令的 begin/end 始终同 id),并携所属叶子的 `leaf_span_id`(其 AGENT span id)以归位到正确 span。引擎在 lease 真执行叶子时,把 sink 经 `set_command_sink` 接到该叶租到的 `LocalSubprocessSandbox`(命令观测性的事件源就在后端的 execute 体内,非新增一条编排层岔口)。它**仅真执行触发**,且**仅命中真后端时触发**:journal **命中**的(cached)叶子根本不 lease、不跑 subprocess,故重放叶子**零** command 事件;离线默认的 `InMemorySandbox` 无真 execute 边界,亦不发。一旦 begin 边已画出"running"card,**terminal end 边必随**——纵使 `Popen` 失败(`OSError` EMFILE/ENOMEM)或 preexec 钩子在 fork 与 exec 间夭折,引擎也以同一 `command_id`、`exit_code=127`(`EXIT_SPAWN_ERROR`)补发一条诚实的 spawn-error end 边再原样抛原异常,故 card 绝不卡死在"running"。
- **隔离不变量**:三个 sink 皆走带外,事件只进 sink、**绝不**注入宿主 LLM 的 message context;`agent()` 仍只 fold 最终结论。故接不接 sink,宿主上下文逐字节相同(quarantine 保持)。`LeafEvent.detail` 默认 **shape-only**(节点 kind/name/timing);原始 tool 入参/输出、模型文本仅在显式 opt-in(`leaf_event_include_payloads=True`)时带上,且截断有界。同理 `CommandEvent`:`command`/`exit_code`/`duration_s` 恒可安全 surface,`output` 默认仅截一小段诚实尾段(`truncated` 据实标),全量捕获输出仅在 `command_include_payloads=True` 时随 end 边带上(且仍受后端自身 output cap 约束)。对 **detached 后台 run**,这几条带外 sink(+ `on_span` / `on_progress`)经 `BgRunManager.event_sinks(run_id)` 缓冲到 `BgRunSlot` 的有界 buffer(`BufferedEvent`,transient、不入 journal/replay、随 slot sweep、跨线程锁),供活的一轮全量重放进 fresh adapter——隔离不变量同样保持(事件只进 buffer、绝不进宿主 LLM 上下文),机制见 [02 §3.2](02-architecture.md)。

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

- `JournalStore` Protocol:默认 **in-memory 实现（进程内、零依赖）**;**跨会话/跨进程持久化已落地(M3)**——`SqliteWorkflowStore` 经可选 `[sqlite]` extra 给出 run_id 命名空间化的持久 journal（详见 §13b）。故默认 resume 是同进程语义,接一个持久 store 即可跨进程 resume。
- 命中后若有 `schema`,以 `model_validate_json` 把缓存 JSON 重新校验回归一后的 pydantic 模型实例(`to_pydantic_model` 的等值-dict 同类缓存保证 `model_json_schema()` 逐字节稳定 → resume 不静默重跑)。
- **journal-key 跨进程稳定**:`journal_key` 哈希 `model_json_schema()` + `json.dumps(sort_keys=True)`,对 pydantic 模型与 L2 dict-schema 在不同 `PYTHONHASHSEED` 下逐字节不变,故 A 进程写的叶子键 B 进程逐字节命中(实证,跨子进程回归测试钉死)。

## 5. 补丁② · 确定性 fail-loud guard（三段式）

确定性**重定义**:不禁绝一切非确定性,只在"非确定性改变了编排的可观测 `agent()` 调用模式"时炸。journal 即确定性 oracle。

| 段 | 机制 | 覆盖 |
|---|---|---|
| 预防(便宜) | AST 禁 `import` + 受限 builtins | 仅 L2(不可信源) |
| 普适 backstop | journal 记调用序列,重放不匹配即 **fail-loud** | 所有源(含手写) |
| 引擎自持不变量 | `budget.spent()` 重放可重建、`phase`/`log` 幂等 | 引擎自己 |

它顺带把 budget 重放分叉从"静默腐坏"降级成 loud failure。`python -O` 会蒸发底座那句唯一 assert——又一条 guard 必自建的理由。

**只在深度 0 记序列**:`agent()`/`race()`/`checkpoint()` 三处都仅在 `_FANOUT_DEPTH==0`(顺序编排路径)`observe` 进**同一条**有序序列;`parallel`/`pipeline`/`race`/`dag`/`batch_map` 帧内的叶子在深度 > 0,其 observe 顺序随墙钟变化,被刻意排除出序列(叶子仍由内容哈希 journal 守护,只是**顺序**不入 guard)。由此推论:**深度 0 并发 observe 不受支持、且 fail-loud**——手写编排若在顶层裸 `await asyncio.gather(branch_a(ctx), branch_b(ctx))`(每支调 `ctx.agent` 或 `ctx.race`),两处在深度 0 并发 observe,顺序随墙钟翻转、逐 run 漂移,一次逻辑确定的 resume(同叶、全 journal 命中)会观测到不同顺序而**误**抛 `WorkflowDeterminismError`、令正确 workflow 永久不可 resume。引擎以一个**进程内/run 本地的"深度-0 在飞 observe 计数器"**(`Ctx._depth0_inflight` 实例属性,非 ContextVar——gather 出的兄弟 task 各复制一份 context,故须用共享 `ctx` 对象上的属性才能彼此看见对方的自增)在首跑即检测此并发并抛 `WorkflowConcurrencyError`。**三处共用单一收口 `Ctx._observe_depth0(key)`**(检查→observe→自增同步无 await 间隙,首支在首个 await 前已自增、次支的同步检查即见计数 ≥ 1;返回是否自增,调用方在 `finally` 按此在**每条**退出路径自减——成功/抛错/cache 命中/`checkpoint` park 皆然,故出错的深度-0 调用也不会把后续顺序调用误判为并发):只守 `agent()` 不够,`race()`/`checkpoint()` 否则仍会竞写共享序列。**并发扇出必须走 `ctx.parallel()`/`ctx.dag()`/`ctx.race()`**(它们标记扇出帧,故 guard 正确排除其叶子顺序)。`WorkflowConcurrencyError` 因仅在深度 0 触发(永不在扇出帧内)故 mask 非顾虑,仍为一致并入 `WORKFLOW_CONTROL_FLOW_SIGNALS`。

**best-effort 设计(刻意,非正确性 bug)**:guard 在两个深度-0 observe 真正**重叠**时触发。唯一漏检的情形是**两个 `agent()`/`race()` 皆 journal cache 命中**——深度-0 每个 `await`(含 cache 命中的 `journal.get`)都是调度点,故兄弟支会交错、在首支在飞时被观测到,**除非**两次命中之间事件循环未偏好兄弟支,此时二者按实参序**顺序**执行:记下**确定**的顺序、resume 稳定无翻转无误报(结果靠内容哈希仍正确),故残留漏检恰是良性情形;任何跑真叶子的调用都重叠且**被捕获**。**并发 `checkpoint()` park 不构成问题,但其根由是 durable executor、非 `gather`**:裸 `asyncio.gather`(默认 `return_exceptions=False`)一支抛错时**不取消**其兄弟支(CPython 文档:其余 awaitable"不会被取消、将继续运行"——本运行时已直接验证,兄弟支跑到完成)。无工作越过 park 的保证来自引擎路径:`orchestrate` 跑在 LangGraph `@entrypoint`(pregel executor 驱动)内,该节点抛错时(`checkpoint` park 抛 `WorkflowSignoffRequired`)**durable executor 拆除节点并取消该 run 仍 pending 的子任务**,故被孤立的 gathered `agent()` 在其下一个 `await`(`journal.get`/叶派发)处被取消、叶子从不运行,且发生在 `run_workflow` 自身 unwind 期间——故即便宿主事件循环在 park 后仍保持开启也不会让叶子运行(已经验证,含"park 后保持事件循环存活 100ms"的探针:agent 被取消、叶子从不运行)。"并发扇出应走 `parallel`/`dag`/`race`(其帧确定性地管理兄弟拆除,而非依赖 executor 的 unwind 时机)"仍成立。

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

### 8b. sandbox 工厂接缝与真执行后端（D-M5）

- **sandbox 工厂接缝**:`SandboxManager` 现接受可注入的 `sandbox_factory: SandboxFactory`(`Callable[[str], SandboxBackendProtocol]`,接 `leaf_id` 出一只全新隔离后端)。默认 = 既有 `InMemorySandbox`-播种行为,**零依赖、离线默认逐字节不变**;`_new_sandbox` 先走工厂再叠加既有 worktree 播种(`upload_files`,协议方法)。`_SandboxSlot.sandbox` 与 `_new_sandbox` 返回类型从 `InMemorySandbox` **放宽到 `SandboxBackendProtocol`**,容下非内存后端而不让 slot 假设具体类型。
- **`LocalSubprocessSandbox`(M5 真后端,全协议)**:每叶子一只 `tempfile.mkdtemp()` 临时根(`root_path`),`execute` 经 `Popen(cwd=临时根, stdout=PIPE, stderr=STDOUT, start_new_session=(POSIX))` 真跑命令;文件操作(`ls`/`read`/`write`/`edit`/`grep`/`glob`/`upload_files`/`download_files`)读写同一临时根下的真文件,故 **`execute` 与文件工具共享同一文件系统**(协议绝对路径 `/x` ↦ `临时根/x`,`..` 穿越硬拒)。它**原样**叠在既有 `_GuardedBackend` / `build_leaf_backend` 之下——引擎 lease/teardown 路径不变(`_GuardedBackend.execute` 已委派给被租隔离后端、`isinstance(backend, SandboxBackendProtocol)` 闸已纳任意全协议后端),**故执行变真无需改引擎**,只需在构造 manager 时注入工厂。
- **韧性(M5 头条 = 有界资源耗尽防护)**:
  - **有界合并输出抽干**:分块读 `proc.stdout`,过 cap 即停止累积(标 `truncated`)但继续读丢到 EOF,故 buffer 永不无界增长、chatty 子进程也不会塞满管道把自己卡死。
  - **超时进程组升级杀**:抽干跑在 worker 线程,调用线程 `proc.wait(超时)`;超时则升级——POSIX `os.killpg(SIGTERM)` → grace 窗 → `os.killpg(SIGKILL)`(因 `start_new_session` 整组连同子孙一并杀),非 POSIX `terminate`→grace→`kill`。超时码 **124**;准入拒绝码 **126**。每条路 `finally` reap 子进程 + join 抽干线程,**无僵尸/孤儿/漏线程/漏 fd**。
  - **`ExecGate`**:`threading.BoundedSemaphore`,与叶级 `ConcurrencyGate`(asyncio,绕事件循环,逐 `agent()` 叶计数)**正交**——`execute` 在 deepagents 的 `aexecute → to_thread(self.execute)` worker 线程上跑,且一叶可发多次 `execute`、多 run 共享宿主,故须跨线程信号量,在同步 `execute` 体内 `acquire`/`release`(`finally` 必还闸位)。一工厂一闸 ⇒ 一 run 的并发 exec 上限全局。
  - **`before_execute(request) -> decision` 准入**:`ExecRequest(command, timeout, leaf_id)` 在 spawn 前(已持闸位后)过钩子,`ExecDecision` 可 allow / reject(不 spawn、返 126)/ 缩超时(只能收紧、不可放宽)/ 降 output cap / 选 `RLimitProfile`。**准入控制而非可观测性 sink**——命令文本/输出的可观测性走另一条专门通道 `on_command`(下一条)。
  - **命令可观测性 `on_command`(真 execute 边界的带外 sink)**:execute 体内在 spawn 前后各发一条 `CommandEvent`(`"start"` / `"end"`),成对、共享 resume 稳定的 `command_id`、携所属叶子的 `leaf_span_id`,故消费者能在 subprocess 真启动一刻画 terminal card、reap 后原地翻 pass/fail。它是 `on_leaf_event` 在执行面的同构 sink——同为 keyword-only、默认 no-op、不入 journal、**仅真执行(miss-only)触发**(journal 命中叶不 lease 也不跑 subprocess、零 command 事件;离线 `InMemorySandbox` 无真 execute 边界亦不发),`output` 默认仅截诚实尾段、`command_include_payloads=True` 才带全量。引擎在租到真后端时经 `set_command_sink` 把 sink 接进该叶的 `LocalSubprocessSandbox`;sink 只在 lease(真执行)路接上,故 miss-only 重放策略由构造天然保持。详见 §2b。
  - **POSIX `rlimit`s(默认 ON、generous-but-bounded)**:经**最小化 `preexec_fn`** 在 fork 与 exec 间于子进程里 `resource.setrlimit()` 逐项设——soft/hard 值已在父进程预算好(`_build_rlimit_setters`),故子侧钩子只迭代成品列表调 syscall、不分配/不 import/不取锁,在 `to_thread` 多线程运行时下 fork-then-exec 窗口内安全;不另起 `python -c` wrapper 故每条命令不背一次完整解释器启动。命令直接 `shell=True` 跑,子进程仍 `start_new_session` 自成会话故超时组杀仍及。默认 profile:CPU 60s、地址空间 2 GiB、文件大小 256 MiB、open files 1024;**`RLIMIT_NPROC` 默认 unset**(它计宿主用户**全部**进程而非本命令子树,固定值反映环境负载、会在繁忙宿主上误伤 `fork`,宿主要 fork-bomb 守卫须显式按基线留余量设)。内核拒设的限(如 Darwin 的 `RLIMIT_AS`)best-effort 跳过、不中断命令。
- **资源清理**:`close()` 删临时根、幂等;子进程清理无需在 `close` 做——每次 `execute` 已在返回前 reap 自身子进程(及超时时其进程组)。`InMemorySandbox` 也得了 no-op `close()`,故 `SandboxManager.stop`/`reclaim_idle`/`_evict_one_idle` 经 `_close_backend`(`isinstance(_Closeable)`)**统一调 `close()`** 释放后端,无需逐次 `getattr` 探测。
- **零依赖底座保持**:真本地后端只用 stdlib(`subprocess`/`resource`/`signal`/`tempfile`/`threading`),`pyproject.toml` 依赖**不变**;base 安装离线默认仍是 `InMemorySandbox`,真执行是宿主显式 `SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy(...)))` 的危险 opt-in。
- **诚实非目标(明列为 sharp edge,非安全 sandbox)**:per-leaf 临时根只界定**默认工作目录**,命令仍能经绝对路径读写任意宿主路径、以用户权限用网络、超 best-effort rlimit 耗资源。**不提供**:硬文件系统封闭(container / chroot / nsjail / firejail)、网络出口阻断、cgroup 级 CPU/内存/进程封闭、守护/逃组进程的强孤儿清理、真 git-worktree 执行后端。**最大风险 = false security**(看着隔离实则不是);对不可信/对抗性命令须跑在进程外隔离后端之后。container 后端是同工厂接缝后的**延后 opt-in extra**(仿 `[sqlite]`),M5 不交付。Windows 仅 best-effort(无 rlimit、无进程组语义、posture 弱),诚实 skip 与文档明示。

### 8c. 真 git worktree + 分支/PR（M6,已落地——超集 Claude Code，配对 M5 成 Bun 级 epic）

把 D-G2 留的"真 git-worktree 后端"接缝兑现:每个改文件的 leaf 跑在自己的**真 `git worktree` / 真分支**里(真跑 git/build/test,承 M5),改完由**脚本拥有的冲突循环**把变更**真 `git merge` 折进 integration 分支(非 main)**,最后经 host finalization 开 PR。经 Codex + in-house 跨模型对抗评审拍板(折入 R1–R10),脊柱(方案 1)= "git provider 拥有 worktree-rooted 后端 + merge/冲突循环走 journaled 叶 + 脚本变量 fold + PR 作 host finalization"。

- **`LocalSubprocessSandbox(root=, on_close=)`(承重接缝)**:M5 的后端总是 `mkdtemp` 自建临时根、`close()` 删根;M6 加两个 keyword-only 参数——`root` 传入则**根植该既有目录**(不 mkdtemp、`_owns_root=False`、`close()` 不 rmtree),`on_close` 是 close 时**一次性**回调。这是"让叶子执行沙箱根植在 `git worktree add` 那个真目录"成立的唯一缺口;默认路径(都不传)逐字节不变。
- **`GitWorktreeProvider`(真 git 服务,装配期注入,危险 opt-in)**:`open_worktree(leaf_id)` = `git worktree add <workspace_root>/<safe_id> -b leaf/<leaf_id> <base_ref>`,返回根植该真目录、且 `on_close=lambda: teardown(leaf_id)` 的 `LocalSubprocessSandbox`。**幂等**(R4:`add` 前先 reclaim 同键陈旧 worktree+分支,故 crash-after-add-before-journal 的 resume 重试不撞)+ **异常安全**(R3:任一步失败回滚半成品 worktree/分支再 raise)。`safe_dirname` 含 leaf_id 的 sha256 短缀,杜绝非 hash leaf_id 的路径碰撞。`collect(leaf_id)` = `git add -A` 后真 `git diff` 枚举新增/改动并读当前内容——**这是权威变更集**(真磁盘内容,非模型自报),删除在 v1 **fail-loud**(`dict[str,str]` 无 tombstone 表达,宁炸不静默不完整)。`teardown` = `worktree remove --force` + `branch -D`(best-effort、幂等);`cleanup_all()` 扫 `workspace_root` 兜底——**host 拥有 provider 生命周期、须在 `finally` 调它**(引擎刻意不调,因 host 可能跨 run 复用 provider)。构造期校验 `base_repo` 是 git 仓库(否则 fail-loud)。
- **`SandboxManager(git_worktree_provider=)` 接线**:`_new_sandbox` 对 `isolation="worktree"` 叶,有 git provider 时走 `open_worktree`(真目录后端),否则保 D-G2 内存 `seed`+`upload_files` 分支——**两份契约各一条 manager 分支,离线默认逐字节不变**。worktree 移除**绑后端 `close()` 的 `on_close` 钩子**:既有所有 teardown 路径(`reclaim_idle`/`_evict_one_idle`/`stop`/引擎 run-end `stop`)都已 `_close_backend→close()`,故 worktree 在每条路自动移除、**无需新 manager 钩子**。
- **R8 阻塞 git 不在 lock/loop 上**:`git worktree add`(开)与 `worktree remove`/`branch -D`(teardown)都是阻塞子进程,**绝不在持 `_slot_freed` condition 锁时或在 event loop 上同步跑**。`_admit_slot` 重构:锁内只做 quota 决策 + 把 reclaim/evict 的牺牲 slot **POP**(立即释放配额、backend 暂不关)+ 标 `_pending` 占位;然后**释放锁、经 `asyncio.to_thread` off-loop 关牺牲后端 + off-loop 建新后端**,再重入锁装 slot。`_pending` 去重并发同-leaf 创建(`_would_exceed_quota` 计 pending、第二个同-leaf lease 停泊),race 败者**off-loop** 关掉自己刚建的冗余后端。**`_admit_slot` 整个 post-claim 流(自 `_pending.add` 起)取消安全**:叶子 lease 中途被取消(`race()` 败者、被取消的后台 run)时,`CancelledError`(`BaseException`)可落在三个真挂起点——① 建前 off-loop 关牺牲 victim、② off-loop 建新后端(worker 可能已产后端而 `to_thread` 取消时丢弃)、③ 装 slot 前的锁重入。故从 `_pending.add` 之后整段裹一个 `try/except BaseException`:victim 批量关经 `_close_backends_off_loop`(`asyncio.gather(return_exceptions=True)` + `_shielded_drain` driven-to-completion,故 mid-batch 取消不漏关后续 victim——victim 已出池故漏关即纯泄漏;关完把观察到的取消上报由 caller re-raise 而非吞掉)、build 经 `_shielded_drain` 保证 worker 已产后端必到手,任一窗口的异常/取消都走统一 reclaim:把尚未关的 victim 与 built-but-uninstalled 后端**全收进一个 `asyncio.gather(return_exceptions=True)`**(每个 close 皆 best-effort——某个 close 抛错绝不中断 reclaim),随后 `discard` `_pending` + `notify_all` 唤醒同-leaf 停泊者,再 re-raise。故 `_pending` 的 discard **无条件执行**:即便 built 后端 close 自己抛错,leaf 也绝不永久搁浅在 `_pending`(否则永占一个 `max_active` 位、同-leaf 后续 lease 永泊)。`handed_off`/`victims_closed` 标记保证 happy 与 race-败者路后端不被双关。**race 败者路的 rebalance-on-failure 不变量**:败者在锁内对赢家 slot **乐观 `in_use += 1`**(防并发 evict 把赢家从脚下回收),再 off-loop 关自己的冗余后端;但若该 close **抛错**或关闭期间**观察到取消**,`_admit_slot` 会在 `lease()` 的 `finally` 之前就异常退出(caller 拿不到 slot、`finally` 的 in_use 减不掉),故败者路在这两种非常态退出里都先经 `_release_lease(installed)`(同样 `_shielded_drain` driven,后续取消也压不掉)把那次乐观 bump 还回去,再 propagate——否则赢家(别人的活 slot)会留一个永不归还的 phantom `in_use`,卡死其 evict/reclaim 并使后续 lease 在配额上永泊;且观察到的取消必须**真 propagate**(race 败者 / 被取消后台 run 须中止 leaf body,绝不吞掉继续跑)。`_new_sandbox` 须**有界**(git worktree add / 内存构造)——线程不可取消,无界工厂无论有无 shield 都会挂死 lease(故不加 build 超时)。同步 `acquire()` 路保留内联(阻塞)reclaim/evict 作文档化 escape hatch。
- **R5 collect 权威化(确定性命门之一)**:git-worktree 执行叶在 `_invoke()` 后、**仍在 lease 内**(worktree 未 teardown),引擎经 `await asyncio.to_thread(provider.collect, leaf_id)` 取真 diff,折进 `outcome.state` 的保留键 `WORKTREE_CHANGESET_KEY`,随 `LeafOutcome.to_payload()` **journaled**。`ctx.agent` 折叠时:changeset 在则**经 `model_validate` 覆写 schema 的 `files` 字段**(非 `model_copy`——后者跳校验、错类型会污染 journal 致 resume 崩),且 schema-less / 缺 `files` 字段 / `files` 非 `dict[str,str]` 三种皆**fold 期 fail-loud**(首跑即炸,绝不 journal 坏 payload)。镜像 M5"据真 exit code 非模型布尔"——**模型自报的文件字节永远赢不过磁盘真值**。
- **整合 = 脚本拥有 over journaled 叶,真 `git merge`(R7)**:integration 状态走脚本变量 `integrated_tree`(**不**做长命物理 integration worktree 累积 merge——那不重跑 git 就无法 resume-safe)。merge 叶在一个**一次性 scratch repo** 内跑真 branch-level `git merge`(commit base、branch ours=integrated_tree、branch theirs=patch、`git merge`),自给自足(从 journaled 输入重建)→ resume-safe 且忠于真 branch 合并(含真冲突标记/真 exit code);冲突 → resolver 叶(纯 LLM,冲突内容入/解决内容出)→ 折进 `integrated_tree`。
- **PR = host finalization 移出 replay(R1)**:workflow 只返回纯 `integrated_tree` + PR 意图;`run_workflow` 返回后由 **host 幂等调** `PullRequestProvider.open(...)`(check-existing-then-create)。`PullRequestProvider` 协议 + `LocalPullRequestProvider`(离线默认、幂等)在引擎;真 `gh` 实现作文档化示例,不进 CI/E2E。**PR 不是编排里的一步**(否则 resume 会重开)。
- **确定性命门**:每个 git 物理变更(worktree 创建在 `open_worktree`/leaf `@task` 内、merge 在 merge 叶 `execute`/scratch repo 内)都在叶子 `@task` 边界内;`ctx.agent` 派发前查 journal,命中即 return、**不进 `leaf_task`、不建 worktree、不跑 git**——resume 重放 journaled 结果、零真 git 重跑(经计数 fake 实证)。
- **安全(R6)**:`GitWorktreeProvider` + 真 `gh` provider 起真子进程、**非安全沙箱**(警告比照 `local_subprocess_factory`),装配期注入、不进 gated 脚本。真边界 = **untrusted authored-script(AST-gate)路径用 reasoning-only roster**(无 worktree/`needs_execution` 叶,即 `make_reasoning_roster` 对 code_fixer 已做的)+ ExecPolicy/ExecGate admission;**不**以"脚本够不到 git 二进制"作安全论据(假命题:一旦能调 `needs_execution` worktree 叶就够得到)。
- **零依赖底座保持**:新代码纯 stdlib `subprocess`,无新依赖(`deepagents`/`LocalSubprocessSandbox` 本就 base+eager);真 git 隔离是 host 显式 `SandboxManager(git_worktree_provider=GitWorktreeProvider(...))` 的危险 opt-in。

## 9. pipeline 调度器（无 barrier，自建——LangGraph 结构盲区）

```
每 stage 一个 bounded asyncio.Queue(背压,防 item 海啸打爆内存)
每 stage 一组 worker:pull → 跑 stage fn(内部调 agent())→ push 下级 queue
全局 semaphore = min(16, cores-2),跨所有 stage 共享
item 各自独立穿越 → A 在 stage3 时 B 还在 stage1
stage 抛错 → 该 item 掉 null 跳后续;结果按输入下标回收保序
```

底座无任何无-barrier 流式原语(`Send` 是 map-reduce barrier);完全自建。中途异常/预算耗尽须保证队列优雅排空、不死锁。

**CancelledError 内/外之分(并发×取消命门)**:`CancelledError` 是 `BaseException`(非 `Exception`),既是 asyncio 外部拆除一个 task 的手段,也可能由 stage/leaf 自身内部冒出(它 await 的子任务被取消)。stage_worker 用 `asyncio.current_task().cancelling() > 0` 区分二者:① **`cancelling()>0`**——本 worker 的 task 正被**外部**取消(整个 run 在拆除),直接 re-raise 让 `run_pipeline` 的 `finally`(cancel + `await gather`)干净拆除;② **`cancelling()==0` 的 `CancelledError`**——**内部**取消(stage 的子任务被取消、本 worker 并未被拆),按文档化 stage-抛错契约**掉 `None`**、其余 item 存活、绝不令整条流水线 reject(同 `parallel` thunk 内部 `CancelledError` 掉 `None`:`parallel` 用 `gather(return_exceptions=True)` 只把**子任务内部** `CancelledError` 收进 settled 列表——外部取消会从 `await gather` 自身抛出、不进列表,故列表里的 `CancelledError` 必为内部);③ **`cancelling()==0` 的其它 `BaseException`**(`KeyboardInterrupt`/`SystemExit`)——既不能静默吞(失声而抛),又不能裸 re-raise 杀死独 worker 致 feeder 永挂 `join()`,故镜像控制流-信号 abort 路:记入 `aborted`、掉本 item、worker 续活排空+吃毒丸,`run_pipeline` 在干净拆除后再 `raise aborted[0]`(fail-loud、不死锁)。

**取消即拆除须 await(无泄漏 task)**:`run_pipeline` 的 `finally` 在外部取消时不只 `cancel` worker/feeder,还 `await asyncio.gather(*pending, return_exceptions=True)`——同 `dag`/`race`/`parallel`,故 `run_pipeline` 抛回 `CancelledError` 前 worker/feeder(及源的 `aclosing` finally)的拆除已**完成**,无 "Task was destroyed but it is pending"、无在飞 leaf/源清理被甩为 detached。**前提是取消协作**:一个抑制 `CancelledError` 或在 await 中永不返回的病态 stage/源 cleanup 会令此 teardown await 滞留(属病态用户代码,v1 不加 teardown 超时——宁滞留可诊断,不静默泄漏)。另:feeder 阻塞在 async 源 `__anext__` 时内部 abort 不唤醒它(须等源 yield),为既有特性,新 BaseException abort 路同样继承。

## 9b. batch_map — 流式准入扇出（E:大规模 fan-out 人体工学）

`ctx.batch_map(items, fn)` 是面向**大规模扇出**(数千叶)的薄 map 原语:把一个异步 `fn`(典型为单个 `agent()`)map 到一个 `Iterable` 或 `AsyncIterable` 的每个 item,结果按**输入序**回收为 `list[T | None]`。它是 `parallel`(吃预物化 thunk 列表)的大扇出对位面——`parallel` = 小已知集 + barrier,`batch_map` = 大流式集 + 进度。**它复用 §9 的 `pipeline` 调度器**,不另起一套:`batch_map(items, fn)` 即"单 stage 的广义 `pipeline` 运行"(stage = `lambda _prev, item, _idx: fn(item)`),白拿失败隔离(抛错落 `None`)、控制流 drain-then-reraise、poison-pill 拆除,以及 `_FANOUT_DEPTH` 帧(内部 `agent()` 跳确定性序列 guard,同 `parallel`/`pipeline`/`dag` 叶)。

### 流式准入不变量（载重)

把 `run_pipeline` 从"`list` 专用"广义化为"吃任意 `Iterable | AsyncIterable`"——只需拆掉三处 `len` 依赖:① feeder 经一个 `_drain(items)` 适配器消费(`AsyncIterable` 走 `async for`、`Iterable` 走 `enumerate`,故 feeder 是 `async for index, item in _drain(items)`,先探 async 协议——`list` 是 `Iterable` 非 `AsyncIterable`、异步生成器二者皆是,先探 async 故各走对支);② 结果回收弃 `[None] * count` 预分配,改 `dict[int, T | None]` 按下标 key,末了 flatten 成 `list`(空 run 回 `[]`,缺位补 `None`)——**保序而无需提前知道长度**;③ worker 数:`Sized` 输入 → `min(gate.limit, len, max_in_flight)`,未知长度 → `min(gate.limit, max_in_flight)`(默认 `gate.limit`)。

**流式准入不变量**:有界 queue(`maxsize = max_in_flight`)已天然提供背压——feeder 在满 queue 上 `put` 阻塞,故任一时刻**在飞 envelope/task 数 ≈ `worker_count + queue_maxsize`、与 N 解耦**。N 千 item 永不一次性物化 N 千 task,内存被窗口而非总量绑定。这是与"急切物化(`list(items)`)"的真正分水岭:急切实现会在任何 stage 跑之前把源**全抽干**(N 次 pull),流式实现在 barrier 处只 pull 到 `~worker_count + window` 就被背压挡住、绝不到 N(headline 不变量测试度量的正是**源 pull 数**,而非 in-`fn` 并发——后者无论准入策略如何都被 `worker_count` 循环界住,度量它是空的)。`asyncio.wait_for` 守背压死锁。

### 流式-IN / barrier-OUT（诚实非目标)

`batch_map` 是**流式准入、barrier 收口**:item 懒进,但结果**收齐才一次性返回**(对齐输入序),不随完成边流式 yield。真流式输出与确定性 replay 冲突(已是 race 的非目标),故不做。ETA 是 `remaining / mean rate` 的朴素线性估计,不建模方差/尾重,是提示非保证。

### count/ETA 进度 — transient 投递的确定性边界（载重)

`batch_map` 随推进自动发**实时 count/ETA 进度**,无需脚本自插桩。新增 `ProgressKind.BATCH` + 一个 frozen `BatchMetrics(completed, elapsed_seconds, rate, total=None, eta_seconds=None)`,经 `ProgressEntry` 的可选 `metrics: BatchMetrics | None = None` 字段搭载(`PHASE`/`LOG` 条目留 `None`,向后兼容)。`rate = completed / elapsed`;`total` 已知(`Sized` 输入取 `len`,或 `total=` 提示)则算 `eta = (total - completed) / rate`,未知 `total` 只发 `completed`、`eta=None`(graceful degradation)。**节流**:每 K item 或每 T 秒发一次而非逐 item(N 千次发射是噪音),`K = max(1, total // 100)`(未知 total 用固定步 64),且**总发一条最终 settled 条目**故末态精确。

**transient 投递是这个里程碑唯一的载重不变量**:`BATCH` 条目是**带外、瞬时、非确定**的,绝不走 `ProgressLog` 的 append-only 记录路径,而经 `ProgressLog.emit_transient` 投递:

- **投递到 sink 但不记录**(不入 `_entries`、不计入 `delivered_count`)。fire-and-forget 刷新,消费者覆写上一条来渲染(进度条),正如 M3.5 upsert `workflow_runs`、M1 带外 tap 叶事件。
- **绝不进 journal / 确定性 guard / replay 结果**。它携的时间戳(`elapsed`/`eta`)是非确定的,绝不能 key 一条 journal 条目。
- **resume 时 re-emitted、非 replayed**:resume 重执行脚本,`batch_map` 重跑、实时 BATCH 进度**被重新发出**——作为实时视图重新生成,而非从记录回放(`emit_transient` 刻意永不被 replay 抑制,即便 `delivered_count > 0`)。这不动 journal 分毫:完成叶仍零模型成本重放,且因 BATCH 条目从不入 `_entries`/`delivered_count`/journal/确定性 guard,重发的进度**不携确定性重量**。载重不变量是 **not-recorded**,**不是 not-re-emitted**——进度是实时视图,每跑重生,从不被回放。

### parallel 不动、pipeline 的 Sequence 契约保持

**`parallel` 完全不动**——它仍急切物化、要求 `Sequence`、span 记 `thunk_count`;`batch_map` 是独立的第三条路而非 `parallel` 的改造(改 `parallel` 吃迭代器会牺牲其稳定契约,违反 preserve-public-signatures)。**`pipeline` 的公共契约逐字节保持**:`Sequence` 输入仍走 `len` 快速路径(`length = len(items) if isinstance(items, Sized) else None`),广义化后的同一个 `run_pipeline` 同时服务 `pipeline` 与 `batch_map`,既有 10 个 list-输入 pipeline 测试一字未改照绿。新增公共面全为 additive、keyword-only、有默认:`Ctx.batch_map` · `ProgressKind.BATCH` + `BatchMetrics`(包根导出) · `SpanKind.BATCH`(记 `total`/`admitted_count`/`surviving_count`) · `ProgressEntry.metrics`(可选、默认 `None`)。`batch_map` 是 `Ctx` 方法 → 不在 AST gate 禁用名单、无需 `run_script` 注入(`BatchMetrics` 引擎产、脚本从不构造)。详见 [v0_4_0_plans/02-e-batch-ergonomics.md](v0_4_0_plans/02-e-batch-ergonomics.md)。

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
| D8 | journal 存储 | `JournalStore` Protocol;默认 in-memory 实现(进程内、零依赖);**跨会话/跨进程持久化已落地(M3)**——`SqliteWorkflowStore`（经 `[sqlite]` extra,统一 sqlite db、run_id 命名空间化 journal + 第二连接上的持久 `AsyncSqliteSaver`,见 §13b） |
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
| D-G2 | `isolation="worktree"` 保真度 | **v1 默认 = 内存播种副本**(`InMemoryWorktreeProvider`:`seed` 给隔离 base 快照拷贝、`collect` 算相对 seed 变更集)+ **真 git-worktree 后端作可插拔生产实现**(**M6 已交付** `GitWorktreeProvider`,见 §8c / D-M6)。否决"仅文档化 seam"(不兑现卖点)与"v1 直接上真 git worktree"(与 offline-first 跨度大)。`SandboxManager._new_sandbox` 仅 slot 新建时播种;`isolation` 经 `agent → leaf_task → lease` 透传;fix 叶复用 G1 `schema=Patch` 自报变更(生成/应用分离) |
| D-G4 | read-only judge 形态 | **库级辅助**(`read_only_leaf` / `read_only_builder`)+ deny-write `FilesystemPermission` + `needs_execution=False`,否决"引擎内建只读 agentType":引擎不构造叶(宿主构造),只读是工具面属性归宿主侧;deepagents 无 execute 权限维度,靠不分配 sandbox 禁执行 + deny-write 禁写,叠加才是真只读;builder 形态复用 G1,只读 + 结构化裁决一行可得 |
| D19 | 接缝③ L2 交付节奏 | v1 = L0/L1 先行,L2 架构预留紧跟(L2-as-skill,见 02) |
| D20 | async 后台 tool 执行 | 自建轻量后台机制(无 server / 无重依赖);v1 即含;蓝本 = omne-next 实现 + deepagents async-task API 形态。详见 [02 §3](02-architecture.md) |
| D21 | 跨会话持久化形态(M3) | **一个统一 sqlite db 文件**,run_id 命名空间化四表(registry + journal),**两条连接**(autocommit store + explicit-commit `AsyncSqliteSaver`,皆 WAL)。否决"分文件 / 复用单连接":隔离 regime 不兼容须分连接、同文件保单一持久单元。**journal(非 checkpointer)交付零成本重放**,checkpointer 是 durable add-on。`[sqlite]` 可选 extra 把守,base 安装零依赖。per-run 规范 id(`journal_run_id`)同 key journal 谱系与 checkpoint thread;host thread 仅 key manager slot。schema-version guard(`PRAGMA user_version`)fail-loud。详见 §13b |
| D22 | 逐叶 live 可观测性 tap 形态(M1) | **选 callback-handler tap**(把 `BaseCallbackHandler` 挂上叶子既有 `leaf_config["callbacks"]`),否决 `astream_events`:叶 `ainvoke` 调用路径不变(folding / 结构化输出 / usage 计量逐字节不动),复用 budget 已走的同一条 callbacks 转发路径。**`metadata` 关联法被否**——deepagents subagent 边界丢弃 `metadata`(只转发 `callbacks`/`tags`/`configurable`),故 `leaf_span_id` 关联改由 per-leaf handler 实例在构造时**闭包**承担(一叶一 handler,它见的每条边按构造即属本叶子)。`SpanBegin` 立为**独立值类型**(非把 begin 字段塞进统一 event),与完成 `Span` 共享 `span_id`。`detail` opt-in 取**单一 bool**(`leaf_event_include_payloads`),不引入更重的 sink-config 对象。replay 策略**随控制流自落**:begin 发于 span 打开、早于 journal 查询故 cached 叶照常重发(配 cached 完成);`on_leaf_event` 只在 miss 路挂 handler,故 journal 命中**零** interior 事件——无需额外开关 |
| D-M5 | 真本地执行后端形态(M5) | **可插拔 sandbox 工厂 + `LocalSubprocessSandbox`(stdlib-only 全协议后端,per-leaf 临时根)**,默认仍 `InMemorySandbox`(零依赖、离线默认逐字节不变),宿主经 `SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy(...)))` 危险 opt-in。选 `Popen`(非 `subprocess.run`)以做进程树清理 + 有界流式抽干(过 cap 标 `truncated` 不无界缓冲、续读到 EOF 防满管道死锁);超时杀**进程组**(`start_new_session` + `os.killpg` SIGTERM→grace→SIGKILL,码 124),准入拒绝码 126。`ExecGate`(`threading.BoundedSemaphore`)与叶级 `ConcurrencyGate`(asyncio)**正交**——`execute` 在 `aexecute→to_thread` worker 上跑,须跨线程闸、一工厂一闸。`before_execute(request)->decision` 为**准入控制非 sink**(allow/reject/缩超时/降 cap/选 profile)。命令可观测性走**专门的带外 sink `on_command`**——execute 体内在 spawn 前后发成对 `CommandEvent`(`"start"`/`"end"`,共享 resume 稳定 `command_id`、携 `leaf_span_id`),是 `on_leaf_event` 在执行面的同构 sink:keyword-only、默认 no-op、不入 journal、miss-only(journal 命中叶不 lease 不跑 subprocess 故零 command 事件)、`output` 默认截尾、`command_include_payloads` 才带全量;引擎经 `set_command_sink` 仅在 lease 真后端时接上(`InMemorySandbox` 无真 execute 边界不发)。rlimit 走**最小化 `preexec_fn`**(父进程预算 soft/hard、子侧只 `setrlimit` 不分配/不 import/不取锁,故 fork-then-exec 窗口内安全;不另起 `python -c` wrapper 省每命令一次解释器启动),默认 generous-but-bounded profile ON(CPU 60s / AS 2 GiB / FSIZE 256 MiB / NOFILE 1024;NPROC 默认 unset——per-user 计数使固定 cap 不可靠)。`ExecDecision.rlimits` 内联 `RLimitProfile` 覆写即"选 profile",否决命名 profile 注册表(v1 过度抽象)。per-leaf 临时根保持**私有**(`/shared/` hand-off 仍走内存 `SharedArtifactStore` 复合,不耦合真 FS 到内存 store)、`close()` 删根幂等(子进程已在每次 `execute` reap)。否决 `run_workflow` 级 `execution=` 旋钮(危险 opt-in 该是审慎的 manager 构造期决定)。诚实非目标明列为 sharp edge:非安全 sandbox、不封闭 FS/网络/cgroup、Windows best-effort。container 后端是同接缝后延后 opt-in extra(仿 `[sqlite]`),M5 不交付。详见 §8b |
| D-M7a | DAG 调度失败语义 | **独立 `failed: set` + 传递 skip**:节点 `run` 抛 → 该节点 failed、结果 `None`、传递依赖全 skip 到 `None`;合法 `return None` **不触发** skip——二者须可区分,否则"无结论"与"失败"语义混淆。控制流信号穿透:drain 后原样重抛,从不 mask 为 `None` 空洞。 |
| D-M7b | `WorkflowDagError` / `WorkflowCycleError` / `WorkflowNestingError` 入信号元组 | 三者均**并入 `WORKFLOW_CONTROL_FLOW_SIGNALS`**——dag 格式非法、命名循环、以及深度上限越界均是结构性作者 bug、非叶子失败;fan-out 帧内 fail-loud 而非被 mask。初版设计有意将 `WorkflowNestingError` 排除（"命名边界非引擎信号"），Codex 跨模型评审指出此为 BLOCKER：fan-out 帧内深度越界会被静默 mask 为 `None` 空洞，遮蔽结构错误——已修正，三个结构性错误全部入元组。 |
| D-M7c | DAG scheduler 不持 gate slot | 调度器不持有并发 gate;`run` 函数内的 `agent()` 才获取。故 dag 嵌套在 dag 节点的 `run` 里不会 deadlock 池,且并发界仍全局有效。 |
| D-M7d | dag 无需 journal 条目 | DAG 结构由确定性脚本定义;resume 重跑调度器、已完成叶走 journal 短路。无需 dag 级 journal 条目;与 `parallel`/`pipeline`/`race` 的统一处理:结构元数据不入 journal,只有叶子结果入。 |
| D-M7e | `workflow()` 嵌套:循环检测先于深度检测 | `WorkflowCycleError`(名字已在栈上)优先于 `WorkflowNestingError`(深度超限):循环检测代价低、诊断更精确;深度上限是兜底防线。两个 contextvar(`_WORKFLOW_DEPTH` + `_WORKFLOW_NAME_STACK`)在 `finally` 中各自独立 reset——栈和深度各自恢复,避免其中一个失败导致另一个泄漏。 |
| D-M6 | 真 git worktree + 分支/PR 形态(M6) | **方案 1**(经 Codex + in-house 跨模型评审拍板,折入 R1–R10):**git provider 拥有 worktree-rooted 后端 + merge/冲突循环走 journaled 叶 + 脚本变量 fold + PR 作 host finalization**。`LocalSubprocessSandbox(root=, on_close=)` 让叶沙箱根植真 worktree(默认逐字节不变);`GitWorktreeProvider`(`git worktree add -b leaf/<id>`、真 `git diff` 作 collect、幂等 R4 + 异常安全 R3、teardown 绑 `on_close`、`cleanup_all` 由 host 在 `finally` 调 R-host-contract);`SandboxManager(git_worktree_provider=)` 两契约各一分支、离线默认零改动。**三处承重决策**:① **PR/integration 物化移出确定性 replay 作幂等 host finalization**(R1——副作用须在叶边界或 replay 之外,否则 resume 重开);② **worktree 叶权威变更集 = leaf task 内真 `git diff`**(R5,经 `model_validate` 覆写 schema `files`、schema-less/缺字段/错类型皆 fold 期 fail-loud,非模型自报);③ **整合用 merge 叶内一次性 scratch-repo 真 `git merge`**(R7,比 merge-file 忠于 branch 语义且自给自足 resume-safe)。R8:阻塞 git(add + teardown)thread-offload 出 slot 锁(`_admit_slot` 锁内 POP 牺牲 slot + `_pending` 占位、off-loop 关/建)。R6 安全:真 git 执行叶只在 host-trusted roster,untrusted authored-script 走 reasoning-only roster(`make_reasoning_roster`)+ ExecPolicy admission,撤"脚本够不到 git"错误论据。collect 删除 v1 fail-loud(无 tombstone)。真 `gh` provider 降级为示例(R9)。否决方案 2(统一 open/collect/close 生命周期契约——改稳定 seam、共享 integration 工作区与 per-leaf 隔离 + resume 冲突)、方案 3(冲突在单叶内解决——放弃"脚本拥有循环"命门)。详见 §8c |
| D-E | 批处理人体工学(`batch_map` + 流式准入 + count/ETA 进度,E) | **流式准入落在新 `batch_map`、`parallel` 不动、内部广义化既有 `run_pipeline`**(路线图原写"修 parallel"——改 `parallel` 吃迭代器会牺牲其 `len`/index 预分配/`thunk_count` span 的稳定契约,违反 preserve-public-signatures;分工:`parallel` = 小已知集 + barrier,`batch_map` = 大流式集 + 进度)。`batch_map` 是**薄 map**(单 `fn`),非广义多 stage pipeline 亦非 map+内建 reduce——多 stage 链是 `pipeline` 的活、reduce 留作结果列表上的独立 M1-helper 调用(避两个近同 API + 避耦合,YAGNI)。**进度复用 `ProgressSink` + 一个 transient `BATCH` 条目**,非新 `on_batch_progress` 钩子(零新 API 面;transient/not-recorded 语义已隔离确定性)。**输入从一开始就支持 `Iterable` + `AsyncIterable`**(异步源——paged API/流式行读——正是流式准入的意义,feeder 本就 async)。**载重不变量 = transient 投递的确定性边界**:BATCH 条目 delivered-but-not-recorded(不入 `_entries`/`delivered_count`/journal/确定性 guard),故 resume 时 re-emitted-not-replayed——not-recorded 是命门,not-re-emitted **不是**。流式准入不变量:在飞 task ≈ `worker_count + queue_maxsize`、与 N 解耦。详见 §9b 与 [v0_4_0_plans/02-e-batch-ergonomics.md](v0_4_0_plans/02-e-batch-ergonomics.md) |

## 13. 实现待核实清单（开工前/中逐条钉测试）

1. **journal × 原生 cache 交互**:引擎统一走 async(避 `#7589` sync error-caching);显式决定是否关原生 CachePolicy、让 journal 成唯一记忆化源。
2. **`task_id` 顺序敏感性**:加"脚本编辑后 resume"集成测试——顺序漂移会静默失配重跑。
3. **`max_concurrency` 嵌套语义**:确认叶子 fan-out 是共享 entrypoint 层 semaphore,还是 deepagents 子调用另开无界 executor。
4. **CompositeBackend 隔离泄漏(#2884)**:并行叶子隔离独立验证。
5. **callback 转发**:`@task` 层直调须复刻 `_build_subagent_config` callbacks 转发,否则共享 budget 漏算。
6. **`-O` 风险**:生产开 `PYTHONOPTIMIZE` 时底座唯一 determinism assert 蒸发——再证 guard 必自建。
7. **单叶子内 deepagents 是否并发调工具**:决定 per-leaf sandbox 是否需内部串行化。
8. **硬契约逐条钉测试**:journal-key 派生身份 / retry 时 thread_id 稳定 / 路径规范化防穿越 / pipeline 异常不死锁。

## 13b. 跨会话持久化（M3,已落地——超集 Claude Code）

跨进程 resume 是 D（跨会话持久）里程碑的交付:**一个全新进程指向同一 sqlite 文件,按 `run_id` resume 一个 run,完成过的叶子从持久 journal 零模型成本重放**。Claude Code 仅同会话,本端口经可选 `[sqlite]` extra 跨进程存活。

### 三个公共面（皆 Layer 2 host-wiring,不碰 L0/L1 内部）

- **`WorkflowRunStore` Protocol**(`_run_store`):workflow tool 的 run 注册表持久化边界——`save_spec` / `delete_spec` / `load_spec`(async)+ `journal_for(run_id) -> JournalStore`(sync,launch 前同步接线)。`RunSpec`(frozen+slots)携 `kind`("name"|"script")/ `name_or_source` / `args`(须 JSON-可序列化)/ `label` / **`journal_run_id`**(规范来源 id 谱系,见下)。
- **`InMemoryRunStore`**(默认,零依赖):specs 进 dict、每个 `run_id` 缓存恰一个 `InMemoryJournalStore`(repeated `journal_for` 返同一实例 → 同会话 resume 复用原 journal)。base 安装行为不变。
- **`SqliteWorkflowStore`**(`_persistence`,经 `[sqlite]` extra):一个统一 sqlite db 文件、按 `run_id` 命名空间化的四表(`run_specs` / `journal_records` / `journal_sequence` / `journal_progress`)+ 一个跑在**第二条连接**上的持久 `AsyncSqliteSaver` checkpointer。async 工厂 `await SqliteWorkflowStore.open(db_path)` 构造;`store.checkpointer` 取 saver;`await store.aclose()`(或 `async with`)收口。

### 载重不变量（非显然、经评审硬化——逐条违反会静默砸碎卖点或挂死宿主）

| # | 不变量 | 为何载重 |
|---|---|---|
| (a) | **journal(非 checkpointer)交付零成本重放** | 原生 checkpointer 是 **index-based**、同 thread 重调会重跑叶子;LangGraph 每次 `.ainvoke` 把 `@entrypoint` body 整体重执行。是**内容哈希 journal** 让完成的叶子重放免费。"全新进程零模型成本 resume"由 journal **独立**交付(resume 侧 `checkpointer=None` 亦可证),checkpointer 是**鲁棒性 add-on**(durable `@task` cache + 单 run 内 interrupt/resume + 跨进程按 thread_id resume)。 |
| (b) | **两条连接、一个 db 文件** | store 连接(autocommit,`isolation_level=None`,WAL,busy_timeout)与 `AsyncSqliteSaver`(explicit-commit + 自有 WAL regime)**隔离 regime 不兼容**,必须**分两条** `aiosqlite.Connection` 指同一文件;WAL 下跨连接读见已提交写。autocommit 让每个 `put()` 返回即持久、零显式 `commit`(default deferred 模式会在 close/crash 回滚未提交 DML → 丢光每条已 journal 的叶子)。 |
| (c) | **`AsyncSqliteSaver` 绑定 event loop** | 其 `__init__` 调 `asyncio.get_running_loop()` 绑定;**循环外构造**抛 `RuntimeError('no running event loop')`,跨**不同**循环复用一个实例(如两次 `asyncio.run()`)**挂死**。宿主须在其**单一持久 loop 内**构造、并在该 loop 上跨所有后台 run / thread_id **复用同一实例**(实证:3 并发 + 2 顺序于一实例全对)。直接 `AsyncSqliteSaver(conn)` 构造,**绝不**用 `from_conn_string`(它是 `@asynccontextmanager`,`__aexit__` 关连接,毁掉跨进程 resume)。 |
| (d) | **per-run 规范 id 同时 key journal 谱系与 checkpoint thread** | 每个 run 一个规范来源 id(`RunSpec.journal_run_id`,fresh launch 采自身 `run_id`、resume 从 spec 继承),它**既** key per-run journal(零成本重放谱系)**又** key per-run LangGraph checkpoint thread。**host thread 是另一回事**——只 key BgRunManager 的 manager slot,让发起 launch 的 caller 能 poll。这条切分让一个 host thread 上的多个 run 不塌进同一 checkpoint thread,且 resume 鲁棒地重接原 run 的 thread。 |
| (e) | **`[sqlite]` 可选 extra 把守持久化** | base 安装零依赖、行为不变(回落 `InMemoryRunStore`)。`SqliteWorkflowStore` 经包根 lazy `__getattr__` 暴露——`import langchain_dynamic_workflow` 不触发 sqlite import;缺 extra 时模块顶 `try/except ImportError` 抛清晰"装 `[sqlite]`"消息。`_persistence` / `_run_store` 入 import-linter Contract 1 `source_modules`,只从 `._engine`(公共墙)import `JournalStore`/`JournalRecord`,**绝不**碰 `._journal`。 |
| (f) | **schema-version guard(fail-loud)** | `PRAGMA user_version`:`0`(fresh/未追踪)→ 跑 DDL + stamp `_SCHEMA_VERSION`;等于当前版本 → 幂等 proceed;其它非零值 = 不兼容 schema → 立即抛 `IncompatibleSchemaError`,把静默 shape-drift 升成 loud、可诉的失败。 |

旁注:**save-before-start**——`_launch` 先 `save_spec`(spec 携 stamped `journal_run_id`)**再** `manager.start`,故被准入的 run 总有可 resume 的 spec;quota 拒入(`BgRunQuotaExceededError`)则 `delete_spec` 回滚,refused launch 不留 unresumable 孤儿。**strict-msgpack 诚实**:`AsyncSqliteSaver` 对**每个** `@task` 返回值 msgpack 序列化,叶子状态须保持 msgpack-friendly 形状(经早期 spike 钉死,见下游 plan)。`race()` 无胜者**不** journal → 跨进程 resume 会重派候选(新成本),是记录在案的已知边界。

接线:`create_workflow_middleware(roster, workflows=wf, store=store, checkpointer=store.checkpointer, ...)`(或 `create_workflow_tool` 同 `store=`/`checkpointer=` kwargs);`store=` 省略时回落 `InMemoryRunStore`。详细 host 接线见 [02 §10](02-architecture.md),时序见 [uml/03-sequence.md](uml/03-sequence.md) D 图。

## 14. 运行中 HITL 签核（M4,已落地——超集 Claude Code）

脚本可在阶段间**暂停等人工签核**再续:`ctx.checkpoint(ask, *, tag="")` 把运行停在一道门,host 拿到决策后以 `run_workflow(..., resume=value)` 续跑,该 value 即 `checkpoint()` 的返回值。Claude Code 运行中不接受输入,本端口超集之。

### 关键决策:签核门走 journal,**不**用 LangGraph 原生 `interrupt`（C8 spike 否决）

spike(`@entrypoint`+`@task` 真跑 interrupt/resume,langgraph 1.2.2)证实 interrupt 路线有**致命缺陷**:checkpointer 的 `@task` 重放缓存是 **index-based**(第 N 次 `leaf_task` 调用 → 缓存槽 N),而 `agent()` 在 journal 命中时**短路、不调 `leaf_task`**;于是 resume 时前置叶 A 走 journal 短路、不调 task,叶 B 的 task 调用落到槽 0,被 checkpointer 喂回 **A 的缓存**(实测复现 `b == A 的结果`)。M3 崩溃-resume 靠 resume 侧 `checkpointer=None` 规避,但 interrupt-resume 本要复用同一 checkpointer,矛盾炸出。**故弃 interrupt,改 journal 驱动**:与"journal 是真相源、checkpointer 是 add-on"哲学(§13b 不变量 a)一致,且无需 checkpointer(H7 消失)。

### 机制（journal 驱动的签核门）

`ctx.checkpoint`:门按 `signoff_key(position, tag)` 取键(position = 本 run 第几个 checkpoint 调用,namespaced `"signoff"`,与 leaf/race 键不撞);查 journal——① 有记录决策 → 重放(零成本);② 无记录但有待注入的 `pending_signoff`(来自 approve) → 写入决策并消费、返回;③ 否则 raise `WorkflowSignoffRequired(ask, tag, gate_key)`。park 与 approve 是**两次独立 `run_workflow` 调用**,各自一次 `.ainvoke`(各持自己的 per-call `InMemorySaver`),journal 是唯一跨调用缓存——`@task` index 缓存绝不跨两次,故不会错位。`run_workflow(resume=)` 经 `Ctx(pending_signoff=)` 注入,首个未决门消费;`UNSET`(常量命名的共享哨兵,非 `None`)区分"未注入"与"注入 None"。

### 载重不变量

| # | 不变量 | 为何载重 |
|---|---|---|
| (a) | **签核是 depth-0 原语(fan-out 守卫)** | 门由序号(checkpoint 调用顺序)定身份;在 `parallel`/`pipeline`/`race` 帧内,序号会被并发 thunk 竞争 → 键非确定 → 重放崩,且并发到达的多门无序可供顺序人审。`_FANOUT_DEPTH>0` 时抛 `WorkflowCheckpointError`,并将其并入共享 `WORKFLOW_CONTROL_FLOW_SIGNALS`,故 fan-out 内触发会 fail-loud(绝不被 mask 成 `None` 空洞)。 |
| (b) | **脚本体在 approve 时从头重执行(D4 边界)** | journal 重放门的**决策**(零成本),但脚本**代码**重跑:门前 `agent()` 走 journal 缓存,但脚本里的非幂等副作用(如外部 append)会重复。最终**结果**正确;副作用重复是已记录边界(同 §13b 失败-重试语义)。**进度叙述不再重发**——签核 park 被当作 designed-stop(非崩溃),引擎在 park 时持久 `progress_count`(仅它、不持久 sequence,以免 approve 多出的调用触发 determinism guard 的 extra-calls 检查),故 approve 不重叙门前 phase/log(评审 M4 硬化)。 |
| (c) | **无 checkpointer 要求(对比 interrupt)** | 决策走 journal 注入,非 checkpointer 的 interrupt 状态。`checkpointer=None` 即可工作;同会话 approve 复用内存 journal,持久 journal(§13b)让门前工作跨进程零成本重放。 |
| (d) | **/shared/ artifact 不跨签核门存活** | per-run `SharedArtifactStore` 在 approve 重启时重建为空;门前叶重放其**结果**但不重填 `/shared/`——同崩溃-resume 边界。跨门状态须走脚本变量(M5 fix_loop 范式),真 worktree 持久是 M6。 |
| (e) | **门入确定性序列(漂移 fail-loud)** | `checkpoint` 把门键 `signoff_key(position, tag)` 经 `sequence_guard.observe` 记入与叶调用同一条有序序列,故全量 resume 时门顺序/身份漂移(前面插/删了叶或门、或同位换了 tag)**失声而抛**而非把决策静默绑错门(叶漂移 fail-loud,门也须)。门身份是 `(position, tag)`——`ask` 因可能携非确定内容(如模型评估)被排除出键,故**不同门必须用不同 tag**,且编辑脚本门/叶结构会作废 parked journal(同叶键边界)。评审 M5 硬化。 |
| (f) | **决策必 JSON-可序列化 + 消费前序列化** | 决策被 `json.dumps` 记进 journal;`checkpoint` 在**消费 pending 之前**序列化,故非 JSON 决策抛清晰 `WorkflowCheckpointError` 且门保持未决(仍可重批),不丢值不留半门。评审 Codex#4 硬化。 |

### host 面(BgRunManager + workflow tool)

`BgStatus.AWAITING_SIGNOFF`(**非终态**,计入 `active_run_count`)。`_run_wrapped` 在 `CancelledError` 之后、泛 `Exception` 之前捕 `WorkflowSignoffRequired` → `_park`(存 ask、记 `parked_at`、入一条 notice)。`BgRunManager.approve(coro, run_id, thread_id)` 就地复用 parked slot(**同 run_id**,host 据 id 跨暂停追踪),且**先同步把 slot 翻 `RUNNING` 再排续跑**——杜绝双 approve 竞争(否则第二次 approve 仍见 `AWAITING_SIGNOFF`、过守卫、孤儿化第一个续跑,两个续跑撞同一 journal,评审 H1)。workflow tool 的 `approve` **只批准本进程在活的 parked run**:parked 态(在哪道门、ask)只在内存 manager、未持久化,故 swept/跨进程的 `UNKNOWN` run 无法核实其确在门上——从持久 spec 重启会有把**非 parked run 推过人没看过的门**的授权缺口(评审 M1/Codex#3),故拒批;跨会话 HITL 待持久 park 态,列后续里程碑。引擎兜底:注入了 `resume` 决策却无门消费(run 完成时 `pending_signoff` 仍在)→ **fail-loud** `WorkflowCheckpointError`(签核决策绝不静默丢)。abandoned 的 park 由 `sweep` 经 `park_ttl_seconds` 过期成 `CANCELLED`(防永久占 quota,评审 M2;活的签核仍能 approve)。AST gate **禁 authored 脚本调 `ctx.checkpoint`**(签核是注册工作流能力,authored run 无 resume lane、且会占 quota——评审 M3,防御纵深叠在 reasoning-only roster 上)。`status`/`runs` 暴露 `awaiting_signoff` + ask;`resume`(崩溃重放,无值注入)与 `approve`(注入人工决策)是不同动词。详见 [02 §11](02-architecture.md),时序见 [uml/03-sequence.md](uml/03-sequence.md) G 图。

---

> 信源(版本锚定 langgraph 1.2.2 / langchain 1.3.2 / langchain-core 1.4.0 / deepagents 0.6.7):见 `research/`。

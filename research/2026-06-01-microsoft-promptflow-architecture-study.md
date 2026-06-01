# Microsoft promptflow 架构研究：对 langchain-dynamic-workflow 对外形态设计的启发

**日期**: 2026-06-01
**Scope**: 本报告研究 Microsoft `promptflow`（已退役并迁移至 Microsoft Agent Framework）的对外软件形态——flow 定义模型、执行器/运行时、SDK 集成形态、tooling/connections/tracing 子系统、monorepo 包切分——目的**唯一且明确**：为 `langchain-dynamic-workflow`（下称 LDW，一个"代码拥有控制流"、构建于 LangChain deepagents 之上的 dynamic-workflow 引擎）的**对外形态选型**提供有据可依的启发。我们正在三种对外形态之间做权衡：library core 的 `run_workflow()`、面向 deepagents 的 `create_workflow_tool()` adapter、以及供手写脚本注入的 primitives（`agent()` / `parallel()` / `pipeline()` / `phase()` / `log()` / `budget`）。报告的 **payload 是第 7 节（启发账本）**。

**Evidence note**: 本报告绝大多数论断为 **verified-in-source**——来自克隆于 `/tmp/promptflow-study` 的真实代码独立复核。验证轮次确认 8 个高风险论断中 7 个 `confirmed-in-source`（DAGManager 的 NODE_REFERENCE 解析、`ThreadPoolExecutor` 的 `DEFAULT_CONCURRENCY_FLOW=16`、`ScriptExecutor` 的 importlib+getattr+直接 `self._func()` 调用、`PFClient` 的 operation namespaces、`Flow.__call__` 的 keyword-only flow-as-function、`@trace`→`_traced`→OpenTelemetry、`@tool`→`_traced` metadata stamping、`/score` serving endpoint）。**唯一降级为 `partially-confirmed` 的论断是"DAG-yaml→flex 的历史 pivot"**：flex code-first 模型已在源码完全证实，但"从 DAG 主动转向"是 docs 对比 + 退役迁移轨迹的**诠释性 framing**，仓库内并无显式的"pivot away"叙事（仅 `migration-guide/` 记录退役至 MAF）。文中所有路径与 URL 均来自 findings/verification，未杜撰。

---

## 1. promptflow 是什么 / 定位，与我们的根本范式差异

promptflow 是 Microsoft 的 LLM 应用编排框架，原始设计哲学（`docs/concepts/design-principles.md`）**为"可见性"（visibility）而生**：把工具拼成一张可视化的静态 DAG，支持图形化、单节点 run/debug、low-code 拖拽 UI。其设计原则 #3（line 51）**刻意拒绝让 Flow 图灵完备**，并显式把"完全动态、LLM 引导的 agent"重定向到 Semantic Kernel。这是一句对 LDW 极有价值的自白：promptflow 把**控制流归属交给框架**（engine-owns-control-flow），用静态配置换取可视化与可调试性。

LDW 的范式与之**根本对立**：控制流归属是**代码**（script-owns-control-flow / control-flow inversion）。循环、分支、fan-out 写在普通 Python 语句里（`if`/`for`/`await`），`agent()` 是隔离上下文的 leaf 调用，中间结果存于 script 变量而非 model context，只有 final result 回到 caller。

| 维度 | promptflow（DAG 模式） | LDW |
|---|---|---|
| 谁决定下一步 | engine（DAGManager 从声明的 topology 推导） | script（确定性代码） |
| 中间结果位置 | DAGManager 的 `_completed_nodes_outputs` dict | script 变量 |
| 分支表达 | 声明式 `activate: {when, is}`（图剪枝） | 真实 Python `if` |
| 设计取向 | 可见性优先，刻意非图灵完备 | 控制流表达力优先 |

**关键**：promptflow 自己也意识到静态 DAG 的天花板，于是引入 **flex flow（code-first）**——这恰是 LDW 范式的"同侧"先例，下一节展开。`relevance: borrow（结论） + contrast（DAG 模式）`，`evidence_basis: verified-in-source`。

---

## 2. flow 定义模型：DAG flow / flex flow / prompty —— 三种范式与控制流归属

promptflow 有且仅有三种 flow 定义模型，全部继承同一契约基类 `FlowBase`（`promptflow-core/promptflow/contracts/flow.py`），但它们落在控制流归属轴的**两端**。

**(a) DAG flow（`flow.dag.yaml`）—— 控制流是静态声明的 DAG。** YAML 列出 `nodes`，每个 node 的 `inputs` 通过 `${node.output}` 引用其它 node、通过 `${inputs.x}` 引用 flow 输入。`InputValueType` 枚举区分 `LITERAL` / `FLOW_INPUT` / `NODE_REFERENCE`；`InputAssignment.deserialize` 把 `${...}` 解析为 `NODE_REFERENCE`。引擎（**非用户代码**）据此重建执行图。分支同样声明式：`activate: {when: ${n.output}, is: <value>}`——conditional-flow-for-if-else / -switch 示例**完全在 YAML 里**实现 if/else 和 switch，没有一行 Python 决定走哪条分支。`node_variants` 提供按 id 选择的可替换 node 实现（A/B prompt 调优），同样不靠运行时代码分支。契约 dataclass `Flow(FlowBase)` 持有 `nodes: List[Node]`、`tools`、`node_variants`。`relevance: contrast`。

**(b) flex flow（`flow.flex.yaml`）—— 控制流归代码所有。** 整个 YAML 本质就一行：`entry: module:callable`（如 `programmer:write_simple_program`、`flow:ChatFlow`）。**没有 nodes、没有 edges、没有 `${...}` 引用**——YAML 只声明 entry 指针，加可选的 `inputs`/`outputs`/`init`/`sample`/`environment` 元数据。Python 函数或 callable class **即是** flow；所有循环/分支/fan-out 活在普通 Python 里（chat-basic 的 `ChatFlow.__call__` 有真实的 `while` token-trimming 循环，chat-minimal 的 `chat()` 有 `if`）。契约 dataclass `FlexFlow(FlowBase)` 有 `init`（class entry 的构造 kwargs）但**故意没有 `nodes` 字段**。**没有 scheduler**——`ScriptExecutor` 通过 `importlib` 导入 `module:func`，若 entry 是 class 则用 `init` kwargs 实例化，然后直接调用。签名由**运行时反射**恢复（`inspect.signature` via `function_to_interface`），在隔离子进程里生成（`entry_meta_generator.generate_flow_meta`），函数体**不做 AST 分析、是不透明的**。可选的 `__aggregate__(self, line_results)` 是 batch eval 的 map/reduce barrier 约定。promptflow 官方 docs（`concept-flows.md`）**逐字**给出选 flex 的理由："Users can write complex flow with Python built-in control operators (if-else, foreach) or other 3rd party / open-source library." 这与 LDW 的 control-flow-inversion 立场**完全同侧**。`relevance: borrow`。

**(c) prompty（`.prompty`）—— 根本不是编排模型，是单次 LLM 调用单元。** markdown 文件：`---` 之间是 YAML front-matter（model config、inputs、sample），之后是 jinja2 模板化的 prompt body。`Prompty.__call__` 渲染模板并发起**一次** LLM 请求（内置 exponential-backoff retry）。`PromptyFlow(FlowBase)` 既无 nodes 也无 entry。它是**可组合的 leaf 单元**（flex 代码在自己的循环里 load 并调用它）。这是与 LDW 的 leaf `agent()` 调用**最接近的类比**。`relevance: borrow`。

**判别靠 file shape，不靠 mode flag**（`get_flow_type` in `_utils/flow_utils.py`）：`.prompty` 扩展名 → PROMPTY；YAML dict 含 `entry` 键 → FLEX；否则 DAG。一个 bit（`entry` 的有无）就分开了"代码拥有"与"声明 DAG"。

**关于"pivot 的教训"（诚实标注）**：源码确认 flex 是 code-first，docs 对比了 DAG vs flex 的取舍，退役迁移轨迹（`migration-guide/PromptFlow-to-MAF/`）也佐证方向——但**仓库内无显式的"主动转离 DAG"历史叙事**，"pivot"是诠释性 framing（`partially-confirmed`）。教训本身仍成立且对 LDW 极有分量：**visibility-first 的静态 DAG 是一个局部最优，一旦编排必须动态（data-dependent loop、动态 fan-out 宽度、任意分支）就会崩塌**——而那正是 LDW 瞄准的 regime。

**两点须注意的分歧**：(1) promptflow 用子进程**运行时反射**恢复 flex 签名（**对函数体不设 AST gate**），LDW 的 Layer 2 AST gate 更严、出于安全动机；(2) promptflow 的 node 缓存（`CacheManager`）是**按 tool-call 内容哈希**（`hashlib` over args/kwargs），反而比 LangGraph 的 index-based native cache **更接近** LDW 的 content-hash journal——是部分先例而非反例。

---

## 3. 执行器 / 运行时架构

promptflow 有**两套正交的执行模型**，二者的对比是对 LDW 最重要的一课。

**DAG 运行时（engine 拥有控制流，contrast）。** `FlowExecutor`（`executor/flow_executor.py`）是单 line 入口类与 scheduler dispatcher：`FlowExecutor.create(...)` 工厂按 file type 分派到 `PromptyExecutor` / `ScriptExecutor` / DAG `FlowExecutor`；`exec_line(...)` / `exec_line_async(...)` 是 per-line 入口，最终落到 `_traverse_nodes` → `_submit_to_scheduler` → `_extract_outputs`（按 flow.outputs 的引用映射出 final dict）。控制流核心是 `DAGManager`（`executor/_dag_manager.py`）——一个 **pull-based 拓扑调度器**：`pop_ready_nodes()` 返回所有 NODE_REFERENCE 依赖（+ `activate.condition`）已满足的 node；`pop_bypassable_nodes()` 把 activate 条件不满足或依赖被 bypass 的 node 停用；`complete_nodes(outputs)` 把结果并回 `_completed_nodes_outputs` dict；`completed()` 判全完成。**中间结果完整存于这个 dict**——数据传递通道是 script 变量，不是 context window。

两个 scheduler 驱动同一 DAGManager：
- `FlowNodesScheduler`（sync，`ThreadPoolExecutor`，`max_workers=min(node_concurrency, 16)`，`DEFAULT_CONCURRENCY_FLOW=16`）：loop 提交 ready node → `futures.wait(FIRST_COMPLETED)` → complete → 续提，直至完成；注册 SIGINT/SIGTERM handler，必要时 `os._exit` 逃离滞留的 worker 线程。
- `AsyncNodesScheduler`（asyncio `Semaphore(node_concurrency)` + `ThreadPoolExecutor` 混合）：async tool 包成 coroutine，sync tool 经 `loop.run_in_executor` 桥接；`asyncio.wait(FIRST_COMPLETED)` under `wait_for(timeout)`；刻意不用 `with ThreadPoolExecutor`（手动 shutdown）以免取消时被运行中的线程阻塞。当任一 tool 为 async 或 `PF_USE_ASYNC=true` 时选用。

`FlowExecutionContext.invoke_tool(node, f, kwargs)` 是 scheduler 调用的 **leaf-call wrapper**：`_prepare_node_run`（生成 RunInfo）→ 可选 cache lookup（命中则设 `cached_run_id` 并跳过 `f`）→ `Tracer.start/end_tracing` → `run_tracker.end_run` → `finally: persist_node_run`（**即便异常也在 finally 持久化 run record**，no-silent-failures）。`relevance: borrow`——这正是 LDW `agent()` @task wrapper 需要的"隔离一个工作单元 + 记录 start/result/error + content-hash skip"边界。

**flex 运行时（代码拥有控制流，borrow）。** `ScriptExecutor`（`FlowExecutor` 子类）**没有 DAGManager、没有 scheduler、没有 node graph**。`_exec_line` 就一句 `output = self._func(**inputs)`，包在 tracing span 里，返回 `node_run_infos` 为空的 `LineResult`。class entry 用 `init` kwargs 实例化（构造期 config 与 per-invocation inputs 干净分离），`__call__` 为 flow body，可选 `__aggregate__` 为一次性 reduce。**值得深挖的不对称**：promptflow 的 flex 模式放弃了函数**内部**的 batch/aggregation/run-record 脚手架（函数对 runtime 是黑盒）；而 **LDW 想要 code-owns-control-flow 的同时保留结构化 primitives（agent/parallel/pipeline/phase）与 durable journaling——即 instrument 控制流而非把它当不透明黑盒。这是 LDW 相对 promptflow flex 的核心增量**。

**统一的 run/line 记录模型（borrow）。** 跨两种模式统一：`LineResult(output, aggregation_inputs, run_info, node_run_infos)` 是 per-line 记录；`AggregationResult(output, metrics, node_run_infos)` 是 reduce 记录；`RunInfo`（`contracts/run_info.py`）是 per-node 记录（`node`、`run_id`、`status: Status`、`parent_run_id`、`api_calls`、`inputs`/`output`、`cached_run_id`）。`RunTracker` 持有 `_flow_runs` / `_node_runs` dict 并经 `AbstractRunStorage` 持久化。**这是一套以 run_id 为键、parent 链接、status 标注的可序列化 journal**——是 LDW journal entry 的强参照（尤其 content-hash + parent-run 链接）。

**Batch 是 devkit 的独立层（不在 core）。** `BatchEngine`（`promptflow-devkit/batch/_batch_engine.py`）跨 line fan-out：async 路径用 `asyncio.Semaphore(worker_count)` + `asyncio.wait(FIRST_COMPLETED)`；sync Python flow 则走 `LineExecutionProcessPool`（`Manager().Queue()` 任务/输出队列、fork-vs-spawn 处理、基于内存的 worker-count 启发式）。**resume 是粗粒度的**：跳过 index 已有结果的 line，剩余重跑——**没有细粒度 per-step replay 或 checkpointer**。`LineExecutionProcessPool` 这套 OS 进程机器（fork/spawn 陷阱、queue-based IPC）是 **LDW 应当 AVOID 自建**的——LangGraph `@task` + async leaf agents on durable substrate 是更轻的路径，进程池仅作 fork/spawn 踩坑的警示参照。Aggregation node 是 scatter→gather/map-reduce：per-line 产出 `aggregation_inputs`，batch 后 transpose 成列再一次性 reduce（`borrow` 思路，但 LDW 可用普通 Python over script-held results 实现，不需 transpose/collect_lines——那些 helper 只因 line 跑在隔离进程才需要）。

**`promptflow-parallel`（contrast/avoid）** 是 Azure ML Parallel-Run-Step 适配器，非 in-process orchestrator：minibatch → map row → `executor.execute(row)` → 序列化到 JSONL → `finalize()` reduce，并发/分布由 Azure ML PRS host 拥有，靠 storage 层解耦。这是"把并行委托给外部 scheduler + 文件系统 handoff"的极端，与 LDW 的 in-process、code-driven fan-out 对立。

---

## 4. SDK & 集成形态（重点章，对照我们缺失的"对外形态"）

这是与 LDW 三种对外形态选型最直接对话的一节。promptflow **同时铺了三种 shape**，分层于不同 package——这本身就是一个值得我们权衡的"surface area"信号。

**(i) `PFClient` —— hub-and-spoke 控制平面 client（borrow）。** `_sdk/_pf_client.py` 的 `class PFClient(**kwargs)` 在 `__init__` 里构建 operation namespaces 并惰性暴露为 property：`.runs`（RunOperations）、`.flows`、`.connections`（lazy，延迟 Azure 凭据）、`.tools`、`.traces`。高层动词挂在 client 上：`run(flow: Union[str, PathLike, Callable], *, data=, run=, column_mapping=, init=, resume_from=, ...) -> Run`、`test(...)`、`stream`、`get_details`、`get_metrics`、`visualize`，委托给 namespace。`client/__init__.py` 把 `PFClient` 列在 public surface 首位（`__all__ = ['PFClient', 'load_run', 'load_flow']`）。**对 LDW**：`run_workflow()` 是 `PFClient.run()` 的精神类比——但我们**不应**抄整套 operation-namespace 控制平面（runs/connections/traces 是 promptflow 卖的 experimentation 产品形态，LDW 不卖这个）。建议：`run_workflow()` 作为 **library core 的一等入口**，保持单函数 + 关键 kwargs（keyword-only、有默认），把 `resume_from` 这样的 durable 语义留给 LangGraph substrate 承接。

**(ii) flow-as-function（混合：borrow 调用形态，contrast 生命周期）。** `core/_flow.py` 的 `Flow.__call__(self, *args, **kwargs)` **keyword-only**（传位置参直接 `raise UserErrorException`），内部 `self.invoke(inputs=kwargs)` 返回 `result.output`；`AsyncFlow` 有 async `__call__`。`load_flow(source)` 返回可直接调用的 Flow；调用前可设 `f.context = FlowContext(connections=, overrides=, streaming=)` 在内存里改写连接/输入/streaming 以复用。**重要注脚**（verification）：flex flow 的 `Flow._invoke` 会 `raise 'Please call entry directly for flex flow'`——flow-as-function 经 `Flow.__call__` 仅适用 DAG/prompty，flex entry 由 `ScriptExecutor` 直接调用。**对 LDW**：keyword-only 的 flow-as-callable 调用形态值得借鉴（清晰、可复用）；但 `FlowContext.overrides` 是**对 `flow.dag.yaml` 做字面 YAML 字符串路径替换**（`overrides={'nodes.X.inputs.url': ...}`，且与 `connections` 组合是"undefined behavior"）——这是 config-DAG 模型催生的 leaky abstraction，**LDW 必须 AVOID**：参数化只走普通 Python kwargs / script 变量，绝不引入 string-keyed config patching（包括对 roster/agent override 的诱惑）。

**(iii) `@tool` + 反射 + deferred discovery（borrow）。** `_core/tool.py` 的 `@tool` **只 stamp 元数据并无条件包 `_traced(func, trace_type=TraceType.FUNCTION)`**（每个 tool 调用自动出 span，observability-by-default），并把 `__tool/__name/__description/__type/__input_settings` 挂到 wrapped function 上供后续生成 tool-definition。tool **不在 import 时注册**，而是经 entry-point group（`package_tools`）后续发现，`ToolsManager` 是 dict registry。签名由 `function_to_interface(f, ...)` 反射（annotation → `FlowInputDefinition`）。**对 LDW roster/registry**：这套"装饰器 stamp 元数据 + 反射出 typed contract + 延迟发现"是 roster 注册的好模板。

**(iv) serve/deploy + CLI（borrow + neutral）。** serve 是一等故事：`pf flow serve --source <flow> --port 8080 --host localhost` 启动 Flask（v1）/FastAPI（v2）app，暴露 `POST /score`（`v1/app.py` `@app.route('/score')`、`v2/routers/score.py` `@router.post('/score')`）+ 内置测试页，连接经 env var `{CONNECTION_NAME}_{KEY_NAME}` override；同一 flow artifact 可不变部署到 dev-server / Docker / K8s / Azure App Service / PyInstaller executable。`pf` console-script 是 15 行 shim（`promptflow._cli.pf:main`），CLI **整层活在 devkit**（import-linter 禁止 core/tracing 反向 import），保持 runtime CLI-free。**对 LDW**：HTTP serve 是重投资（Flask/FastAPI/Docker/K8s/monitoring）；考虑到 LDW 跑在 deepagent host 内，独立 server **可能冗余**——这是开放问题，建议 v1 不投，把 `create_workflow_tool()`（attach 到 deepagent）作为主集成形态。CLI 分层到独立 layer 的纪律值得借鉴。

**三形态决策（对 LDW 直接主张）**：`create_workflow_tool()` ≈ flex 的 class-entry（把 code-owned workflow 包成 deepagent 可调用的单元），`run_workflow()` ≈ `PFClient.run()`（library core 入口）。**不要**复制 promptflow 的三形态全铺——我们不卖 experimentation 平台，surface-area sprawl 是成本。

---

## 5. tooling / connections / tracing 子系统

**tools（borrow）**：见第 4 节 `@tool`。`ToolsManager` dict registry + entry-point 发现 → 直接映射 LDW roster/registry。

**connections（borrow）**：`core/_connection_provider/_connection_provider.py` 的 `ConnectionProvider` ABC——configs/secrets 分离、ABC 单例按 env 切换后端、按 name 引用、resolver 断言解析出的 class 与声明的 param type 匹配。**对 LDW**：是 roster + config/secret provider 的模板。注意 promptflow connections **无 TTL/budget/quota**——LDW 的 `budget`（成本上限）需自建，无先例可抄。

**tracing（borrow，observability-first）**：`promptflow-tracing` 是独立最小依赖包（仅 OTel+openai+tiktoken）。`@trace`（`_trace.py`，sync/async 透明，delegate `_traced(func, TraceType.FUNCTION)`，`otel_trace.get_tracer('promptflow')` + `start_as_current_span`）、`start_trace()` 一行 opt-in 自动注入 OpenAI instrumentation 并设 OTel TracerProvider（装了 devkit 才接本地 trace UI/exporter——分层使 core 无需 UI 即可工作）。`OperationContext` + `ThreadPoolExecutorWithContext`（contextvar-copy pool，使 fan-out 保持 span lineage）+ `TokenCollector`（child-token 向 parent rollup 为 `__computed__.cumulative_token_count.*`）。**对 LDW**：(1) `agent()`/`parallel()`/`pipeline()` 应像 `@tool` 那样**自动 emit span/journal entry**（observability-by-default）；(2) 强烈建议设一个 `langchain-dynamic-workflow-tracing` 最小依赖子包，镜像 `promptflow-tracing`；(3) token rollup 模型直接服务 LDW 的 leaf-agent tracing + `budget` 汇总。开放问题：`cumulative_token_count` 是否经 OTLP export 存活，待确认。

**evals（borrow，parallel 的近亲）**：`promptflow-evals` 的 `evaluate(evaluators=Dict[str,Callable])` 是 **code-owned 确定性编排 + aggregation**；`QAEvaluator` 用 `as_completed` barrier fan-out——是 LDW `parallel()` 的概念近亲。

**两点 contrast**：OpenAI instrumentation 走 **monkey-patch + primitive-allowlist headers**（header 模式可借，monkey-patch 整体规避）；meta-gen 在 fresh `ModuleType` + multiprocessing+timeout 子进程里执行（path 隔离），**弱于** LDW 的 AST gate 但**互补**——可考虑在 AST gate 之上叠加 subprocess+timeout 作纵深防御（开放问题）。

---

## 6. monorepo 包切分与可扩展性

promptflow 是分层 monorepo：`promptflow-tracing` → `promptflow-core` → `promptflow-devkit` → `promptflow-azure`，顶层 `promptflow` 是薄 facade（`__getattr__` 惰性解析 + deprecation routing）。每个 package 单独 pip-installable：装 `promptflow-tracing` 只要 observability，装 `promptflow-core` 要 runtime 但避开 devkit 的重依赖（sqlalchemy/pandas/streamlit），版本 lockstep（1.18.0）。

**层边界是机械强制的（borrow，最具操作性的一招）**：每个 package 的 `pyproject.toml` 有 `[tool.importlinter]` 的 `type = "forbidden"` contracts——如 `promptflow-tracing` 禁止 `promptflow.tracing` import core/_sdk/azure；`promptflow-core` 禁止 public 模块 import `_cli/_sdk/batch/client/azure`。**架构分层从文档变成 CI 时校验的不变量**。`relevance: borrow`，`evidence_basis: verified-in-source`。

**对 LDW**：LDW 目前是单 package，尚非 monorepo。但 import-linter `forbidden` contracts 可**立即采用**来机械守护 Layer 0（substrate）/ Layer 1（runtime）/ Layer 2（meta/AST-gate）边界——例如禁止 AST gate 与 roster 层直接 import LangGraph 内部。tracing 子包的拆分（见第 5 节）是日后 monorepo 化的第一刀。

---

## 7. 对我们对外形态的启发账本（payload）

| promptflow 的做法（evidence） | 借鉴 / 对照 / 规避 | 落到 LDW 哪一处（具体主张） |
|---|---|---|
| **flex flow**：`flow.flex.yaml` 仅 `entry: module:callable`，无 nodes/edges，代码拥有全部控制流（`FlexFlow` 无 `nodes` 字段；`ScriptExecutor` 直接 `self._func(**inputs)`）`[verified-in-source]` | **借鉴**（同侧验证） | **library-core / primitives**：确认 LDW 的 outward form 走"薄 manifest 命名 code entry + 代码拥有编排"。`create_workflow_tool()` ≈ flex class-entry；不要为编排引入任何 nodes/edges 声明。 |
| **DAG flow + `activate(when/is)` + `node_variants` + `DAGManager`**：声明式分支、engine 拥有 topology `[verified-in-source]` | **对照（精确反模式）** | **全局定位**：这是 LDW 存在的理由的反面。文档/示例里明确 position against 声明式分支——LDW 用真实 `if`/`for`。**不**实现 activate-when DSL、**不**实现 node_variants（变体由普通 Python branching/config 处理）。 |
| **prompty**：`.prompty` 单次 LLM 调用 leaf，无控制流，被 flex 代码在循环里组合 `[verified-in-source]` | **借鉴** | **primitives / roster**：prompty ≈ LDW 的 leaf `agent()`。确认 leaf 应是"无控制流的可组合单元"，控制流只在脚本里。 |
| **`FlowExecutionContext.invoke_tool`**：bracket 一次 leaf call = prepare run-record → cache lookup（命中跳过）→ tracing → `finally: persist`（异常也持久化）`[verified-in-source]` | **借鉴** | **primitives（`agent()` @task wrapper）+ tracing**：照搬这个 quarantine+record 边界——隔离 context、记录 start/result/error、content-hash skip、**`finally` 持久化以杜绝 silent failure**。 |
| **统一 run/line 记录模型**：`LineResult`/`RunInfo`（run_id、`parent_run_id`、`status`、`api_calls`、`cached_run_id`），dataclass 化、parent 链接、可序列化 `[verified-in-source]` | **借鉴** | **journal（content-hash journal entries）**：直接作为 LDW journal entry schema 参照，尤其 parent-run 链接 + content-hash + status 标注。 |
| **`CacheManager`**：content-hash（sha1 of tool-computed cache_string，qualified by flow_id+tool_name）`[verified-in-source]`；但默认 `DummyCacheManager`（no-op）、opt-in per-tool、tool 自定义如何哈希 | **借鉴思路 + 对照其负担** | **journal（content-hash journal）**：内容哈希方向对（比 LangGraph index-based 更接近我们要的），但 LDW 应让**runtime 自动哈希 call 内容**（而非 per-leaf 手工注册）。难点：`agent()` 输入含大 prompt + roster identity，哈希需确定性/稳定（开放问题）。 |
| **`PFClient` hub-and-spoke + operation namespaces（.runs/.flows/.connections/.traces）+ `run()`/`test()` 动词**`[verified-in-source]` | **借鉴调用入口，规避全套控制平面** | **library-core（`run_workflow()`）**：`run_workflow()` ≈ `PFClient.run()`，单函数 + keyword-only kwargs。**不**复制 runs/connections/traces operation 平面（那是 experimentation 产品形态，LDW 不卖）。`resume_from` 语义交给 LangGraph substrate。 |
| **flow-as-callable**：`load_flow(source)()`，`Flow.__call__` keyword-only（传位置参 raise）`[verified-in-source]` | **借鉴调用形态** | **library-core / tool-adapter**：workflow 可作 keyword-only callable 复用；与 `create_workflow_tool()` 的 schema 推导对齐。 |
| **`FlowContext.overrides`**：对 `flow.dag.yaml` 做 string-path YAML 替换，叠加 connections 为 "undefined behavior"`[verified-in-source]` | **规避（documented footgun）** | **roster / primitives**：参数化**只走** Python kwargs / script 变量；**禁止**任何 string-keyed config patching（包括 roster/agent override 的诱惑）。 |
| **`@tool`**：stamp 元数据 + **无条件 auto-`_traced`**；不在 import 时注册、经 entry-point 延迟发现；`ToolsManager` dict registry；`function_to_interface` 反射签名 `[verified-in-source]` | **借鉴** | **roster / tracing**：roster 注册采"装饰器 stamp 元数据 + 反射 typed contract + 延迟发现"；leaf 调用 **observability-by-default**（像 `@tool` 那样自动出 span/journal）。 |
| **`@trace` + `start_trace()` + 独立最小依赖 `promptflow-tracing` 包**；`OperationContext`/contextvar-copy pool 保 fan-out span lineage；token child→parent rollup `[verified-in-source]` | **借鉴** | **tracing（独立子包）**：设 `langchain-dynamic-workflow-tracing` 最小依赖子包；`agent()/parallel()/pipeline()` 自动 emit span；token rollup 直接喂 `budget`。 |
| **import-linter `forbidden` contracts** 在每个 pyproject 机械强制层边界 `[verified-in-source]` | **借鉴（立即可用）** | **全局（Layer 0/1/2 边界）**：现在单包就引入 import-linter contracts——禁止 AST-gate / roster 层直接 import LangGraph 内部；为日后 monorepo 化铺路。 |
| **monorepo 分层 + 各包独立 pip-install + 薄 facade（`__getattr__` 路由）**`[verified-in-source]` | **借鉴（渐进）** | **packaging**：先拆 `*-tracing` 子包验证分层；core（runtime）与 devkit-style 工具分离的方向值得规划，但不必一步到位。 |
| **flex 签名靠子进程运行时反射，函数体不设 AST gate**（仅 path 隔离）`[verified-in-source]` | **对照（LDW 更严）** | **Layer 2 AST gate**：promptflow 对编排代码**无静态门禁**，无可借的"校验不可信编排代码"先例——LDW 的 AST gate（no imports/dunders/banned names）须**自研**；可选叠加 subprocess+timeout 作纵深防御。 |
| **`BatchEngine` 跨 line fan-out（Semaphore + FIRST_COMPLETED）；resume 是粗粒度 line-skip，无 checkpointer**`[verified-in-source]` | **借鉴 fan-out 模式 / 对照 resume** | **primitives（`parallel()` barrier）+ substrate**：Semaphore-bounded `create_task` + FIRST_COMPLETED drain 是 `parallel()` barrier 的干净模式；**fine-grained replay/resume 完全靠 LangGraph substrate**（promptflow 无此先例）。 |
| **`LineExecutionProcessPool`**：multiprocessing + Manager queue + fork/spawn 处理（sync flow 真并行）`[verified-in-source]` | **规避（自建过重）** | **substrate**：不自建 OS 进程池；走 LangGraph `@task` + async leaf agents。仅作 fork/spawn 踩坑警示。 |
| **`promptflow-parallel`**：Azure ML PRS 适配，scatter-to-files → gather map-reduce，并行委托外部 scheduler + 文件 handoff `[verified-in-source]` | **对照 / 规避** | **primitives**：LDW 的 fan-out 必须 in-process、code-driven（`parallel()`/`pipeline()` 返回值），不向外部 scheduler + 文件系统 handoff。 |
| **aggregation node**：scatter→gather 一次性 reduce，transpose per-line inputs（因 line 跑隔离进程才需 transpose）`[verified-in-source]` | **借鉴思路，简化实现** | **primitives（`phase()` + reduce）**：reduce 用普通 Python over script-held results 即可；LDW in-process 模型**无需** transpose/collect_lines。是否要专门 reduce primitive 为开放问题——control flow 已 code-owned，普通代码或已足够。 |
| **`__aggregate__(self, line_results)`**：class entry 上的 map/reduce barrier 约定（method 而非 engine primitive）`[verified-in-source]` | **中性** | **primitives**：概念类比 `parallel()` barrier / reduce，但 LDW 以**显式 primitive**表达更清晰，不必抄 method-convention。 |
| **MAF 后继**：`WorkflowBuilder().add_edge(a,b,condition=fn)` / `add_fan_out_edges` / `add_fan_in_edges`——code-first 但控制流仍是 **graph DSL**`[verified-in-source]` | **对照（定位差异化）** | **全局定位**：连 promptflow 自己的后继都把控制流留在 graph builder。LDW 用**真实 Python 语句 + 返回值 primitives** 更进一步——应写一篇 doc/example 显式对照 `parallel([...])`（返回值调用）vs MAF 的 edge-declaration ceremony。 |
| **`design-principles.md`**：DAG 为可见性而生、刻意非图灵完备、把动态 agent 推给 Semantic Kernel `[verified-in-source]` | **借鉴结论 + 对照取向** | **全局**：可见性是真需求——但 LDW **用 tracing/journal 提供可见性，而非用 config graph**。保留 promptflow 的可见性收益，丢掉静态 DAG 的刚性。 |

---

## 8. 关键对照表：promptflow vs langchain-dynamic-workflow

| 维度 | promptflow | langchain-dynamic-workflow |
|---|---|---|
| **控制流归属** | DAG: engine（`DAGManager` 从声明 topology 推导）；flex: code（`ScriptExecutor` 直调 `self._func`）。后继 MAF 仍为 graph DSL | **code**（脚本里真实 `if`/`for`/`await`；`parallel()`/`pipeline()` 为返回值调用） |
| **定义形态** | DAG: `flow.dag.yaml`（nodes + `${...}` refs + `activate` + `node_variants`）；flex: `flow.flex.yaml`（`entry: module:callable`）；prompty: `.prompty` | 手写 Python 脚本（注入 primitives）+ roster；leaf = `agent()`（≈ prompty/flex class-entry） |
| **执行模型** | DAG: thread/async scheduler over DAGManager（concurrency=16）；flex: 直接函数调用；batch: devkit 的 async/process-pool fan-out | LangGraph `@entrypoint`+`@task`+checkpointer substrate；`agent()` 为 @task；`parallel()` Semaphore barrier，in-process |
| **集成形态** | `PFClient`（operation namespaces）+ flow-as-callable（`load_flow()()`）+ `@tool`/`@trace` + `pf` CLI + `pf flow serve`→`/score` | `run_workflow()`（library core）+ `create_workflow_tool()`（deepagents adapter）+ 注入 primitives（手写脚本）；serve 暂不投 |
| **可观测性** | observability-first：`@tool` auto-`_traced`、`start_trace()` 一行、OTel-native、独立 `promptflow-tracing` 包、token rollup | 主张 observability-by-default：primitives 自动 emit span/journal entry；规划独立 `*-tracing` 子包；token rollup 喂 `budget` |
| **状态 / 持久化** | `RunInfo`/`LineResult` 经 `AbstractRunStorage` 持久化；content-hash cache 但默认 no-op + opt-in；**resume 粗粒度 line-skip，无 checkpointer，无确定性保证** | 中间结果存 script 变量；**content-hash journal（自动、runtime-owned）** + **fail-loud 确定性 guard** + LangGraph **fine-grained replay/resume**（promptflow 无此先例，须靠 substrate + AST gate） |

---

## 9. 信源清单（带标注）

**契约与定义模型**
- `/tmp/promptflow-study/src/promptflow-core/promptflow/contracts/flow.py` — `Flow`/`FlexFlow`/`PromptyFlow`、`Node`、`ActivateCondition`、`NodeVariants`、`InputAssignment`、`InputValueType` `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/core/_flow.py` — `Flow`/`AsyncFlow`/`Prompty`/`AsyncPrompty` 的 load+call surface `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_utils/flow_utils.py` — `get_flow_type`/`is_flex_flow`/`is_prompty_flow`/`resolve_flow_path` `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_utils/tool_utils.py` — `function_to_interface` 反射 `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/entry_meta_generator.py` — `generate_flow_meta` 子进程反射 `[local-source]`

**执行器 / 运行时**
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/flow_executor.py` — `FlowExecutor` 工厂/dispatch/output 抽取 `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/_dag_manager.py` — `DAGManager` pull-based 拓扑调度 `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/_flow_nodes_scheduler.py` — sync `ThreadPoolExecutor` scheduler（`DEFAULT_CONCURRENCY_FLOW=16`）`[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/_async_nodes_scheduler.py` — asyncio Semaphore + ThreadPool 混合 scheduler `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/_script_executor.py` — flex `ScriptExecutor`（importlib + 直调 `self._func` + `__aggregate__`）`[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/flow_execution_context.py` — `invoke_tool` per-node bracket（cache/trace/run-record + `finally` 持久化）`[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/_result.py` — `LineResult`/`AggregationResult` `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/contracts/run_info.py` — `RunInfo`/`FlowRunInfo`/`Status` `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/run_tracker.py` — `RunTracker` + storage 持久化 `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/cache_manager.py` — content-hash `CacheManager`/`enable_cache` `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/executor/_line_execution_process_pool.py` — multiprocessing worker pool（fork/spawn、Manager queue）`[local-source]`
- `/tmp/promptflow-study/src/promptflow-devkit/promptflow/batch/_batch_engine.py` — `BatchEngine`（async/process fan-out、line-skip resume、aggregation）`[local-source]`
- `/tmp/promptflow-study/src/promptflow-parallel/promptflow/parallel/_processor/base.py` — Azure ML PRS map-reduce 适配 `[local-source]`

**SDK / tooling / connections / tracing / evals**
- `/tmp/promptflow-study/src/promptflow-devkit/promptflow/_sdk/_pf_client.py` — `PFClient` operation namespaces `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-devkit/promptflow/client/__init__.py` — public surface（`PFClient`/`load_flow`/`load_run`）`[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/tool.py` — `@tool`（auto-`_traced` + metadata stamp）`[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/tools_manager.py` — `ToolsManager` registry + entry-point 发现 `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/core/_connection_provider/_connection_provider.py` — `ConnectionProvider` ABC `[local-source]`
- `/tmp/promptflow-study/src/promptflow-tracing/promptflow/tracing/_trace.py` — `@trace`/`_traced`/OTel `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-tracing/promptflow/tracing/_start_trace.py` — `start_trace()` 一行自动 instrument `[local-source]`
- `/tmp/promptflow-study/src/promptflow-evals/promptflow/evals/evaluate/_evaluate.py` — `evaluate()` code-owned 编排 + aggregation `[local-source]`
- `/tmp/promptflow-study/src/promptflow-evals/promptflow/evals/evaluators/_qa/_qa.py` — `QAEvaluator` as_completed barrier fan-out `[local-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/_core/tool_meta_generator.py` — meta-gen 子进程隔离（对照 AST gate）`[local-source]`

**serving / 部署**
- `/tmp/promptflow-study/src/promptflow-core/promptflow/core/_serving/app.py` — `create_app` 工厂 `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-core/promptflow/core/_serving/v1/app.py` — Flask `/score` `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-devkit/promptflow/_cli/_pf/_flow.py` — `pf flow serve` 子命令 `[local-source, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow-devkit/promptflow/_cli/pf.py` — `pf` console-script shim `[local-source]`

**packaging / 分层**
- `/tmp/promptflow-study/src/promptflow-core/pyproject.toml`、`promptflow-tracing/pyproject.toml`、`promptflow-devkit/pyproject.toml`、`promptflow-azure/pyproject.toml` — 分层 + import-linter `forbidden` contracts `[github, verified-in-source]`
- `/tmp/promptflow-study/src/promptflow/promptflow/__init__.py` — 薄 facade `__getattr__` deprecation routing `[github]`

**docs / 示例 / 迁移**
- `/tmp/promptflow-study/docs/concepts/concept-flows.md` — DAG vs flex；逐字"why flex"（control-operator framing）`[official-docs, verified-in-source]`；hosted: https://microsoft.github.io/promptflow/concepts/concept-flows.html
- `/tmp/promptflow-study/docs/concepts/design-principles.md` — DAG-for-visibility + 刻意非图灵完备（line 51 重定向 Semantic Kernel）`[official-docs, verified-in-source]`
- `/tmp/promptflow-study/docs/how-to-guides/develop-a-flex-flow/class-based-flow.md` — class entry（`__init__`/`__call__`/`__aggregate__`）`[official-docs]`
- `/tmp/promptflow-study/docs/how-to-guides/execute-flow-as-a-function.md` — `load_flow` callable + `FlowContext.overrides`（footgun）`[official-docs]`
- `/tmp/promptflow-study/docs/how-to-guides/deploy-a-flow/deploy-using-dev-server.md` — `pf flow serve`→`/score` `[official-docs]`
- `/tmp/promptflow-study/docs/how-to-guides/tracing/index.md` — observability-first `[official-docs]`
- `/tmp/promptflow-study/migration-guide/PromptFlow-to-MAF/README.md` — 退役 + MAF `WorkflowBuilder` 后继 `[github]`
- `/tmp/promptflow-study/migration-guide/PromptFlow-to-MAF/phase-2-rebuild/03_conditional_flow.py`、`04_parallel_flow.py` — MAF control-flow-as-graph-edges（对照 LDW `parallel()`）`[example]`
- 示例：`examples/flows/standard/basic/flow.dag.yaml`、`web-classification/flow.dag.yaml`（node_variants）、`conditional-flow-for-if-else/`、`conditional-flow-for-switch/`；`examples/flex-flows/basic/`（function entry + `flow.flex.yaml`）、`chat-basic/flow.py`（class entry + `while`）、`chat-minimal/flow.py`（`if` + load prompty）、`eval-checklist/check_list.py`（`__aggregate__`）；`examples/prompty/basic/basic.prompty` `[example, verified-in-source]`
- 官方文档站：https://microsoft.github.io/promptflow `[official-docs]`

**待澄清（open questions，影响后续设计决策）**：(1) LDW 的 fine-grained replay/resume 与 fail-loud 确定性 guard **无 promptflow 先例**，须完全由 LangGraph substrate + AST gate 承接；(2) content-hash journal 应 runtime 自动哈希（含大 prompt + roster identity 的稳定哈希待设计）；(3) `cumulative_token_count` 是否经 OTLP export 存活待确认；(4) 是否需要专门 reduce/aggregate primitive，抑或普通 Python over script-held results 足够；(5) HTTP serve 是否在 scope 内（deepagent host 内可能冗余）；(6) connections 无 TTL/budget/quota，LDW 的 `budget` 成本上限须自建。

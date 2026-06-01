# LangChain / LangGraph / deepagents 底座能力调研：支撑 dynamic-workflow 引擎的 build-vs-buy 账本

**日期**：2026-06-01
**适用读者**：langchain-dynamic-workflow 引擎的实现工程师
**范围（Scope）**：以钉死的依赖版本为锚，逐层核实 LangGraph durable execution（Layer 0）、LangGraph 持久化/缓存/确定性/并发、deepagents 叶子调用与 backends/sandbox、LangChain v1 middleware/结构化输出/token 计量，对引擎的两个自建补丁（content-hash journal、fail-loud determinism guard）和 per-leaf sandbox identity / SandboxManager 需求给出 build-vs-buy 裁决。本报告**不**覆盖 LangGraph Platform/SDK 远程图、AsyncSubAgent 远程叶子、向量检索语义搜索的深入实现。

## 钉死版本表（Pinned Versions）

| 包 | 版本 | 本调研验证方式 |
|---|---|---|
| langgraph | 1.2.2 | 源码 + 运行时 `importlib.metadata` 确认 |
| langgraph-checkpoint | 4.1.1 | 源码 + 运行时确认 |
| langchain | 1.3.2 | 源码 + 运行时确认 |
| langchain-core | 1.4.0 | 源码 + 运行时确认 |
| deepagents | 0.6.7 | 源码 + 运行时确认 |
| langchain-anthropic | 1.4.4 | 运行时确认（未深读其结构化输出 strategy 选择） |
| Python | 3.12.11 | 运行时确认 |

所有版本通过本机 `.venv` 下 `importlib.metadata` 当场打印核对，与 pin 集完全一致。

**证据强度说明（Evidence Note）**：本报告的核心论断绝大多数为 **verified-in-source**——直接阅读 `.venv` 下安装包源码，而非依赖上游 agent 摘要或文档。九条关键 claim 经独立二次核验：其中 6 条 `confirmed-in-source`（确定性不强制、`max_concurrency` 无核数默认、resume 跳过语义、backend 无生命周期方法、`wrap_tool_call` 单粒度、BaseStore 可作 journal）、1 条由实测运行佐证（叶子 `ainvoke` 返回形状）、2 条 `partially-confirmed`（native cache 是否 index-based、ToolRuntime 是否暴露 thread_id——两者原始表述都把不同机制混为一谈，已在下文逐条拆清）。另有 3 处 live empirical probe：ToolRuntime dataclass 字段反射、真实 `create_agent.ainvoke` 输出形状、InMemoryStore journal 完整往返。凡是只有文档佐证（非源码核实）之处，文中明确标注 `evidence_basis: official-docs`。**唯一需要架构吸收的纠偏**：设计文档"native cache 是 index-based"对**结果缓存**不成立——它是内容寻址的；index/positional 的是 `task_id`（驱动 resume replay-skip），详见第 3、7、8 节。

---

## 1. 总览：底座能力地图

引擎分三层。Layer 0 直接骑 LangGraph durable execution。Layer 1 是编排原语，其中绝大多数能用底座现成机制拼出，但有两块必须自建。Layer 2（AST gate + LLM 作者）与本调研无关，不展开。

下表是全局能力地图（`support_level` ∈ provided/partial/gap，`evidence_basis` 优先 verified-in-source）：

| 引擎能力 | 底座出处 | support_level | evidence_basis |
|---|---|---|---|
| `@entrypoint`/`@task` 控制流反转 | `langgraph.func` | provided | verified-in-source |
| resume / replay / cached-skip | checkpointer + `task_id` | provided | verified-in-source |
| `agent()` 叶子调用 | `deepagents.create_deep_agent` → 直接 `ainvoke` | provided | verified-in-source |
| `parallel()`（barrier） | `asyncio.gather` over `@task` futures | provided | verified-in-source |
| `pipeline()`（no-barrier streaming） | 无原语 | **gap** | verified-in-source |
| `phase()` / `log()` | 脚本层 + `astream` | partial | verified-in-source |
| `budget`（token 计量） | `usage_metadata` + `UsageMetadataCallbackHandler` | partial | verified-in-source |
| `budget`（call 计数兜底） | `ModelCallLimitMiddleware` | partial | verified-in-source |
| content-hash journal | native cache 内容寻址但 qualname-scoped + 含 bug | **partial→自建** | verified-in-source |
| fail-loud determinism guard | 仅 `-O` 可剥离的裸 assert | **gap** | verified-in-source |
| per-leaf sandbox identity | 仅 `id` 属性，factory 路径 deprecated | **gap** | verified-in-source |
| SandboxManager（生命周期） | backend 无任何 lifecycle 方法 | **gap** | verified-in-source |
| context quarantine | 直接 `ainvoke` 时天然成立 | provided | verified-in-source |

**一句话结论**：底座把 Layer 0 几乎完整地给了你；Layer 1 的并发屏障、叶子调用、context 隔离也是 provided；真正要自建的只有四样——**no-barrier streaming pipeline**、**content-hash journal（success-only 语义）**、**fail-loud determinism guard**、**SandboxManager + per-leaf identity**。

---

## 2. Layer 0 — LangGraph Functional API

### `@entrypoint` 与 `@task`

`langgraph.func.__all__ == ('task', 'entrypoint')`——整个包只导出这两个名字（`func/__init__.py:56`）。真实签名（verified-in-source）：

```python
entrypoint(
    checkpointer=None, store=None, cache=None,
    context_schema=None, cache_policy=None, retry_policy=None, timeout=None,
) -> Pregel                                # func/__init__.py:262；__call__ -> Pregel @516
entrypoint.final(*, value: R, save: S)     # 区分"返回给调用方的值"与"写入 checkpoint 的值"

task(__func_or_none__=None, *, name=None,
     retry_policy=None, cache_policy=None, timeout=None)
```

`@entrypoint` 把一个单输入的 body 编译成**单节点 Pregel 图**——这就是控制流反转的落点：body 里的 Python 控制流（循环、分支、fan-out）由脚本拥有，LLM 不参与。`@task` 装饰的函数调用后返回一个 `SyncAsyncFuture`（`concurrent.futures.Future` 的子类，`pregel/_call.py:253`），它**既可 await 也可 `.result()`**——这是 barrier 语义的基础。

### Resume / Replay 语义（confirmed-in-source）

这是 Layer 0 最关键的语义，引擎的 cached-skip 完全依赖它。resume 时（`pregel/_loop.py`）：

1. body **整体重跑**（@entrypoint 是确定性脚本，必须能重新发出同样的 task 序列）。
2. `_reapply_writes_to_succeeded_nodes`（`_loop.py:724-737`）把 checkpoint 里保存的 writes，**仅凭 `task_id`** 贴回重新发出的内存 task：`if task := tasks.get(tid): task.writes.append((k, v))`，并跳过 ERROR/INTERRUPT/RESUME 控制信号——所以失败/被中断的 task 保持空 writes，会重跑。
3. runner（`_runner.py:745-759`）：`elif next_task.writes:` 直接 `fut.set_result(ret)` 返回缓存的 RETURN，**不重新执行**；只有 `not t.writes` 的 task 才被 submit 重跑。带 restored writes 的 task 以 `cached=True` emit（`_loop.py:670-672`）。

**对引擎的含义**：resume-skip 的正确性绑定在"脚本以稳定顺序/step 重新产出 task"上——因为 `task_id` 是 positional（见第 3 节）。脚本若在两次运行间产生不同的 task 顺序，saved writes 会**静默失配**而重跑。这正是 content-hash journal 要堵的缝。

### Durability 模式

`Durability = Literal["sync", "async", "exit"]`，默认 `"async"`（`pregel/main.py:2574`，**该默认值未出现在文档页**）。sync=每步前持久化，async=步内持久化（默认），exit=仅退出时。`ainvoke(..., durability=...)` 可逐次覆写。

### interrupt + Command(resume)

`interrupt(value)` 抛 `GraphInterrupt`；`Command(*, graph=None, update=None, resume=None, goto=...)` 恢复。resume 按位置索引匹配，已完成 task 仍走缓存。

---

## 3. LangGraph 持久化 / 缓存 / 确定性 / 并发

本节直接裁决引擎的「偏差①②」假设。**核心发现：存在两套互相独立的 keying 机制，设计文档把它们混为一谈了。**

### 偏差① — native cache 是否 index-based？（partially-confirmed，需改写 rationale）

**两套机制，必须分清：**

**(A) CachePolicy 结果缓存——内容寻址，不是 index-based。** opt-in，需要通过 `entrypoint(cache=...)` 显式接入一个 `BaseCache`。`pregel/_algo.py:858-870` 构造：
```python
CacheKey(
    ns=(CACHE_NS_WRITES, identifier(call.func)),                  # 函数 module.qualname
    key=xxh3_128_hexdigest(cache_policy.key_func(*args, **kwargs)),
    ttl=...,
)
```
默认 `key_func = default_cache_key`（`_internal/_cache.py:26-31`）= `pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5)`——递归冻结、dict 键排序、再做 128-bit xxh3。查找（`match_cached_writes`，`_loop.py:1526`）**只比对 `(ns, key)`**。对结果缓存而言，「native cache 是 index-based」这句话**被源码证伪**。

```python
@dataclass
class CachePolicy(Generic[KeyFuncT]):              # types.py:508
    key_func: KeyFuncT = default_cache_key
    ttl: int | None = None
```

**(B) `task_id`——positional，驱动 resume replay-skip（不需要任何 CachePolicy）。** `_algo.py:834-842`：
```python
task_id = task_id_func(
    checkpoint_id_bytes, checkpoint_ns, str(step), name,
    PUSH, task_path_str(task_path[1]), str(task_path[2]),   # step号 + 节点名 + push标记 + write索引
)   # task_id_func = _xxhash_str (checkpoint v>1) else _uuid5_str
```
它编码的是 step 号 + 节点名 + write 索引，**不含输入内容**。对 resume 身份而言，「index-based」这句话**成立**。

**裁决**：引擎的 content-hash journal 补丁仍然正当，但 rationale 必须改写为三条真实理由，而非"native cache 是 index-based"：
1. **success-only 语义**：已确认 bug `#7589`——同步路径 `SyncPregelLoop.put_writes`（`_loop.py:1586`）缓存结果时**无 INTERRUPT/ERROR 守卫**（async 路径有），失败/中断的 task 会被缓存并 replay 成 success。journal 必须显式 success-only。
2. **per-node content scoping**：native ns 仅按函数 qualname 命名，存在跨调用点复用风险、无 per-workflow-node 隔离。
3. **positional resume identity**：`task_id` 含 step+write_idx，脚本顺序漂移即静默失配。

附带：`#6265`——缓存命中会丢失自定义 stream 数据。

### 偏差② — 确定性是否不强制？（confirmed-in-source，gap 确认）

LangGraph **完全不强制确定性**。全底座唯一的"确定性检查"是一句裸断言：
```python
assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"   # _algo.py:662 与 855
```
它被 `if task_id_checksum is not None` 守卫（仅窄路径非 None），且在 `python -O` / `PYTHONOPTIMIZE` 下**被整句剥离**。`errors.py` 里 grep `determin/nondeterm` 零命中——**不存在任何确定性专属异常类**（只有 `InvalidUpdateError` 管并发 channel 写、`GraphRecursionError`、`NodeTimeoutError` 等）。resume 时 `_reapply_writes_to_succeeded_nodes` 仅按 `task_id` 贴回 writes，**零输入/输出比对、零违规检测**。底座不拦截用户的 time/random/IO——`func/__init__.py:321` 的 `import time` 在 docstring 示例里，`_call.py:66` 对 `sys.modules` 的遍历是为解析函数 `__module__`（identity），不是 monkeypatch；`_runner.py:280-338` 唯一的 `time.monotonic` 用于 LangGraph 自己的 timeout 记账。**fail-loud determinism guard 填的是真实空白，必须自建。**

### checkpointer 与 BaseStore

`langgraph-checkpoint 4.1.1` **只装了 `InMemorySaver`（别名 `MemorySaver`）**——SQLite/Postgres checkpointer 在独立的、未安装的包里（`import langgraph.checkpoint.sqlite/.postgres` → `ModuleNotFoundError`，已实测确认）。`BaseCheckpointSaver` API（`checkpoint/base/__init__.py`）：`put/aput`、`put_writes/aput_writes`、`get_tuple/aget_tuple`、`list/alist`、`get_next_version`（默认整数自增，`current+1`，`None→1`，`str` 抛 `NotImplementedError`，`:692-714`）。

`BaseStore`（`store/base/__init__.py:700`）**可直接作为 journal 底座**——已用 InMemoryStore 实测完整往返（put→get→search→list_namespaces→delete→get=None 全部成功）：
```python
BaseStore.put(namespace: tuple[str,...], key: str, value: dict[str,Any],
              index: Literal[False]|list[str]|None=None, *,
              ttl: float|None|NotProvided=NOT_GIVEN) -> None        # :848
get(namespace, key, *, refresh_ttl=None) -> Item | None            # :748
search(namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, ...) -> list[SearchItem]
list_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0) -> list[tuple[str,...]]
batch(ops: Iterable[Op]) -> list[Result]
```
`@entrypoint(store=...)` 自动注入。namespace=`tuple[str,...]`、key=`str`、value=`dict`，原生支持 TTL 与可选向量索引——content-hashed、namespaced journal 的完美底座。

### 偏差 — `max_concurrency` 是否无核数默认？（confirmed-in-source）

`RunnableConfig.max_concurrency: int | None`，**无默认值**（`langchain_core/runnables/config.py:95`）。LangGraph 异步执行器：
```python
if max_concurrency := config.get('max_concurrency'):
    self.semaphore = asyncio.Semaphore(max_concurrency)
else:
    self.semaphore = None                    # pregel/_executor.py:135-140 → 无界
```
None/falsy ⇒ **不创建 semaphore ⇒ 无界并发**。唯一让 CPU 核数进场的是 langchain-core **同步**路径 `get_executor_for_config`（`config.py:632-634`）：`ContextThreadPoolExecutor(max_workers=config.get('max_concurrency'))`——None 时落到 stdlib `ThreadPoolExecutor` 默认 `min(32, os.cpu_count()+4)`。**那是 stdlib 线程池兜底，绝非 LangGraph 自己推导核数。** 引擎必须显式设置 `max_concurrency` 才能给 fan-out 设界——这是 bounded-queue / 资源耗尽守卫的前置条件。

### Send / Pregel fan-out

```python
Send(node: str, arg: Any, *, timeout: float|timedelta|TimeoutPolicy|None=None)   # types.py:654
```
`Send` 触发 PUSH map-reduce fan-out（`prepare_push_task`，`_algo.py:806`），在下一个 BSP superstep 处理，**带 reduce barrier**。Functional-API 的并行 = 在 `@entrypoint` 里并发启动多个 `@task` future 再 await——每个 `@task` 是一个 Pregel call task。

---

## 4. deepagents 叶子调用契约

**核心发现**：编译后的 deep agent 就是 `create_deep_agent(...)` 返回的一个 `CompiledStateGraph`（Pregel 实例）。引擎**完全不需要** deepagents 那个由 LLM 驱动的 `task` 工具——直接把编译后的 runnable 当 `@task`，用 `.ainvoke(input, config, *, context=...)` 调用即可。契约从源码完整可重建。

### `create_deep_agent` —— 叶子工厂

```python
create_deep_agent(
    model=None, tools=None, *,
    system_prompt=None, middleware=(), subagents=None,
    skills=None, memory=None, permissions=None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on=None, response_format=None,
    state_schema=None, context_schema=None,
    checkpointer=None, store=None, debug=False, name=None, cache=None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]
# graph.py:217；内部委托 create_agent（graph.py:806）
```
返回的图带 `.with_config({recursion_limit: 9999, metadata: {...}})`。引擎每个 roster 条目调一次，持有 runnable，作为 `@task` 直接 invoke——复用 deepagents 的 filesystem/subagent/summarization middleware 栈与 backend，但 LLM 不决定控制流。想要更薄的叶子可直接用 `create_agent`（同样的编译图返回类型与 I/O 契约）。

### INPUT / OUTPUT 形状（verified-in-source + 实测）

```python
class _InputAgentState(TypedDict):
    messages: Required[Annotated[list[AnyMessage | dict[str, Any]], add_messages]]
class _OutputAgentState(TypedDict, Generic[ResponseT]):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
```
叶子调用的标准输入是 `{"messages": [HumanMessage(content="...")]}`（dict 形 `{"role":"user","content":"..."}` 也接受；`StateBackend` 下可带 `"files"`）。实测 `create_agent(...).ainvoke(...)` 在无 `response_format` 时返回 `{"messages": [Human, AI]}` 且**无** `structured_response`；设了 `response_format` 才暴露两键。

### 结果回填（"只有最终结果进入调用方上下文"）—— 引擎须复刻

deepagents 在 `_return_command_with_state_update`（`subagents.py:494-532`）实现了该折叠逻辑，引擎应镜像：
1. 若 `result["structured_response"] is not None` → 序列化（pydantic `model_dump_json()` / dataclass `json.dumps(dataclasses.asdict(...))` / 否则 `json.dumps(...)`）。
2. 否则**逆序**遍历 `result["messages"]`，取第一条 `.text.rstrip()` 非空的 `AIMessage`（规避 Anthropic 尾部空 `end_turn` AIMessage）。`AIMessage.text` 已验证能处理 str 与 content-block list，空时返回 `""`。
3. `messages` 键缺失则抛 `ValueError`。

这条提取出的字符串是父上下文**唯一**能看到的东西——即 context-quarantine 边界。

### context quarantine 机制

deepagents 的 `task` 工具用 `_EXCLUDED_STATE_KEYS = {messages, todos, structured_response, skills_metadata, skills_load_errors, memory_contents}`（`subagents.py:240`），传给子 agent 时剔除这些键再把 `messages` 覆写成 `[HumanMessage(description)]`，返回时再剔除同样的键、只转发单条 `ToolMessage`。**引擎直接调 runnable 时隔离更强**：只传 `{"messages":[HumanMessage(...)]}`、只读提取出的最终字符串——子 agent 的中间 tool 调用永不进入编排器变量，除非引擎主动读 `result["messages"]`。运行时 `context` 会传播到子 agent（docs 确认）。

### registry / roster

- **`CompiledSubAgent`**（公开 TypedDict）：`{name, description, runnable}`，硬要求 runnable 的 state schema 含 `messages` 键。这是引擎最干净的注册路径——自建图、包成 `CompiledSubAgent`、直接 `ainvoke`。
- **`SubAgent`**（声明式）：`{name, description, system_prompt, tools?, model?, middleware?, ...}`，`system_prompt` 不继承主 agent，`tools` 默认继承。
- **避坑**：`SubAgentMiddleware._get_subagents` / `_SubagentSpec` 是私有（下划线），不要伸进去；走 `CompiledSubAgent` 公开路径。
- **`DeepAgentState`** 默认 state schema 用 `DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)` 把 checkpoint 增长压到 O(N)——自定义 state_schema 须继承它以保留该 reducer。

---

## 5. deepagents backends & sandbox（D13 实证裁决）

逐条裁决 D13 的六个子假设（全部 verified-in-source，最后一条为 official-docs）。

### (a) BackendFactory 类型与 per-call 调用 —— CONFIRMED（带 deprecation 警告）

```python
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]   # protocol.py:851
BACKEND_TYPES = BackendProtocol | BackendFactory                       # protocol.py:852
```
factory 在**每次**文件/exec 工具调用时被调：每个 filesystem 工具闭包首行 `resolved_backend = self._get_backend(runtime)`，`FilesystemMiddleware._get_backend`（`filesystem.py:727-749`）`if callable(self.backend): warn_deprecated(...); return self.backend(runtime)`。memory/skills/summarization middleware 同模式。**关键 caveat**：callable factory 自 deepagents 0.5.0 起 **deprecated，0.7.0 移除**，每次 resolve 都发 `LangChainDeprecationWarning`。

### (b) ToolRuntime 字段与 thread_id —— CONFIRMED（带 nuance）

`ToolRuntime`（`langgraph/prebuilt/tool_node.py:1663` dataclass）字段（实测 `dataclasses.fields` 反射）：`state, context, config: RunnableConfig, stream_writer, tool_call_id: str|None, store: BaseStore|None, tools: list[BaseTool], execution_info, server_info`。**没有直接的 `thread_id` 属性（`hasattr == False`）**。thread_id 仅可经 `runtime.config["configurable"]["thread_id"]` 取到（`CONFIG_KEY_THREAD_ID = "thread_id"`，`_internal/_constants.py:52`）。ToolNode 每次调用都用 live 的 per-call `RunnableConfig` 构造 ToolRuntime，所以 factory 调用点能拿到 thread_id。**注意**：裸 graph-node `Runtime`（`langgraph.runtime`）**没有 `config` 字段**——thread_id 在 ToolRuntime 之外（如 before_agent middleware hook）不可达。

### (c) 生命周期方法有无 —— CONFIRMED：NONE（gap）

`BackendProtocol`（`protocol.py:319`）只暴露文件操作：`ls/read/grep/glob/write/edit`（+async a* 变体）+ `upload_files/download_files`。`SandboxBackendProtocol`（`protocol.py:770`）只多加 `id` 属性（`:782`）与 `execute/aexecute`（`:787,812`）。在 `deepagents/backends/` 全仓 regex grep `def (close|start|stop|create|setup|teardown|shutdown|connect|disconnect|__enter__|__exit__|__aenter__|__aexit__|cleanup|dispose|release)` **零命中**（`wc -l == 0`）。backend 仅经 `__init__` 构造，cleanup 隐式（GC）。`BaseSandbox.__abstractmethods__ == {execute, id, upload_files, download_files}`。**sandbox 生命周期（`sandbox.stop()`）完全是调用方责任。**

### (d) Composite / 各 backend 路由与签名 —— CONFIRMED

- `CompositeBackend(default, routes, *, artifacts_root="/")`（`composite.py:140`）：文件操作按**最长前缀**路由；根 `ls/grep/glob` 聚合 default + 全 routes 并 remap 路径；`execute()` **不可路由**——总是委托 `self.default`，default 非 `SandboxBackendProtocol` 则抛 `NotImplementedError`（`composite.py:537-573`）。**已知 issue `#2884`（OPEN）：CompositeBackend route 隔离在共享存储后端间泄漏**——并行叶子的隔离不能仅靠 routes 保证。
- `StateBackend(runtime=None 已弃用并忽略, *, file_format="v2")`（`state.py:50`）：文件存 LangGraph state，thread-scoped、ephemeral，经 `get_config()` + 私有 `CONFIG_KEY_READ/SEND`。
- `FilesystemBackend(root_dir=None, virtual_mode=None, max_file_size_mb=10)`（`filesystem.py:114`）：真实 FS，`virtual_mode` 只做路径护栏、**非隔离**。
- `BaseSandbox`（abstract `execute/id/upload_files/download_files`）：所有文件操作派生自 `execute()`。`LocalShellBackend(root_dir, *, virtual_mode, timeout=120, max_output_bytes=100000, env, inherit_env)`（`local_shell.py:104`）= FilesystemBackend + subprocess，**无隔离**。`LangSmithSandbox(sandbox: Sandbox)`（`langsmith.py:51`）**包裹**外部已建 Sandbox，`id == sandbox.name`——生命周期在外部 Sandbox 对象上，不在 backend。

### (e) thread_id-memoized factory 可行性 —— FEASIBLE 但有硬 caveat

机械上可行：factory 收 ToolRuntime，读 `runtime.config["configurable"]["thread_id"]`，按 thread_id memoize backend。但：(1) factory 路径 deprecated（0.7.0 移除），每次工具调用仍发警告；(2) 无任何 lifecycle hook，memoized sandbox 没有 deepagents 驱动的 teardown，cleanup 全靠调用方。**docs 推荐的非弃用替代（official-docs）**：per-thread 构造 agent——从 `RunnableConfig` 读 thread_id，find-or-create 以 thread_id 打标的外部 sandbox，包成 backend **实例**（非 factory）传给 `create_deep_agent`。

**D13 裁决**：引擎应放弃 BackendFactory，采用 per-thread/per-leaf **实例**构造；per-leaf identity（roster name + workflow node id + 输入内容 hash）与 sandbox 生命周期（lazy-create / TTL / pool / quota / `sandbox.stop()`）由引擎的 **SandboxManager** 自建——底座零支持。

---

## 6. LangChain v1 middleware & 结构化输出 & token 计量

### Middleware hook 粒度

`AgentMiddleware(Generic[StateT, ContextT, ResponseT])`（`langchain/agents/middleware/types.py:383`）暴露状态变更 hook `before_agent/after_agent/before_model/after_model(self, state, runtime) -> dict|None`（+ `a`-前缀异步变体），返回值经 graph reducer 合并；`@hook_config(can_jump_to=[...])` 支持短路 jump（'tools'/'model'/'end'）。两个 wrap hook：`wrap_model_call`、`wrap_tool_call`。列表中第一个 middleware 最外层。

### `wrap_tool_call` 单粒度天花板 —— gap（confirmed-in-source）

```python
def wrap_tool_call(self, request: ToolCallRequest,
                   handler) -> ToolMessage | Command[Any]      # types.py:662
```
`ToolCallRequest` 是 dataclass，**只携带一个** `tool_call: ToolCall`（`tool_node.py:146-147`）。多 tool-call 的 fan-out 由 LangGraph **ToolNode** 拥有，不在 hook：异步 `_afunc` `for call in tool_calls: coros.append(self._arun_one(call,...))` 再 `asyncio.gather(*coros)`（`tool_node.py:856-858`）；同步 `executor.map(self._run_one, ...)`（`:823`）。每个 `_run_one/_arun_one` 构造一个 `ToolCallRequest`、调一次 wrapper。**结论：一次 `wrap_tool_call` 只见一个 tool call，N 路并发子 agent fan-out 无法塞进单个 hook——必须活在 `@task` / ToolNode 编排层。** 这从底座层面强制了引擎 `parallel()` 落在 `@task` 层。deepagents 子 agent 被编译进 `subagent_graphs: dict[str, Runnable]`，引擎可绕过 hook 直接 invoke 并 `asyncio.gather`。

### `wrap_model_call`（per-call 预算记账）

```python
def wrap_model_call(self, request: ModelRequest[ContextT],
                    handler) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]
```
包一次 `model.invoke`。`ModelResponse.result[0]` 是携带 `usage_metadata` 的 `AIMessage`——middleware 可在此读 per-call token。`ModelRequest` 用 `.override(...)` 改 model/messages/tools/response_format。

### `ModelCallLimitMiddleware`（预算兜底）

```python
ModelCallLimitMiddleware(*, thread_limit: int|None=None, run_limit: int|None=None,
                         exit_behavior: Literal['end','error']='end')   # model_call_limit.py:94
```
`before_model`（`can_jump_to=['end']`）超限则 jump_to='end' 或抛 `ModelCallLimitExceededError`；`after_model` 自增计数。**只计 model 调用次数、不计 token**——粗粒度兜底。兄弟 `ToolCallLimitMiddleware` 类似限工具调用。

### 结构化输出

`response_format` 接受三种 strategy（`structured_output.py:194-463`）：`ToolStrategy(schema, *, handle_errors=True)`（仿真 tool call，TypeAdapter 解析，**支持 in-loop retry**——失败发一条纠错 `ToolMessage` 回环）、`ProviderStrategy(schema, *, strict=None)`（原生 `response_format` JSON schema，从 AIMessage text 解析，**无 in-loop retry**，解析失败总抛 `StructuredOutputValidationError`）、`AutoStrategy(schema)`（按模型能力选，setup 默认 ToolStrategy）。错误类：`StructuredOutputValidationError`、`MultipleStructuredOutputsError`。结果落 state `structured_response`。

### token 计量 —— 驱动共享 budget

```python
class UsageMetadata(TypedDict):                    # core/messages/ai.py:104
    input_tokens: int; output_tokens: int; total_tokens: int
    input_token_details: NotRequired[...]; output_token_details: NotRequired[...]
AIMessage.usage_metadata: UsageMetadata | None     # :176
add_usage(left, right) / subtract_usage(left, right)   # :721-783
```
`UsageMetadataCallbackHandler`（`core/callbacks/usage.py:18`，线程安全）在 `on_llm_end` 按 model name 累加到 `.usage_metadata: dict[str, UsageMetadata]`；`get_usage_metadata_callback()` context manager 注册可继承 ContextVar 自动向嵌套调用传播。**因为 deepagents 的 `_build_subagent_config` 把父 `callbacks` 转发给子 agent**，父 config 上挂一个 handler 即可跨所有子 agent 聚合 usage——直接用来驱动引擎共享 budget。**注意**：引擎绕过 LLM-driven `task` 工具、在 `@task` 层直接 invoke 子 agent 时，必须自己复刻这个 callback 转发，否则共享 budget 记账漏算。

---

## 7. BUILD-vs-BUY 账本

| # | 引擎需求 | 底座支持 | support_level | 我们必须自建什么 |
|---|---|---|---|---|
| 1 | `@entrypoint`/`@task` durable body | `langgraph.func`（仅这两名字） | provided | 无（直接骑） |
| 2 | resume / replay / cached-skip | checkpointer + positional `task_id`（`_loop.py:724-737`） | provided | 无（但 task 顺序须稳定，见 #9） |
| 3 | checkpointer | 仅 `InMemorySaver`（SQLite/PG 未装） | partial | 跨进程持久化需自加 `langgraph-checkpoint-sqlite/-postgres` 并重验 |
| 4 | journal 存储底座 | `BaseStore`（实测往返通过） | provided | 无（直接用作 namespaced KV journal） |
| 5 | `agent()` 叶子调用 | `create_deep_agent` → 直接 `ainvoke` | provided | 结果回填逻辑（镜像 `subagents.py:494-532`） |
| 6 | context quarantine | 直接 `ainvoke` 天然隔离 | provided | 无（只读提取的最终字符串） |
| 7 | `parallel()`（barrier） | `asyncio.gather` over `@task` futures | provided | 薄封装 |
| 8 | `pipeline()`（no-barrier streaming + 背压） | **无任何原语**；Send 是 map-reduce barrier | **gap** | 自建：`asyncio.Semaphore` + `asyncio.Queue` over `@task` futures，含 bounded-queue 背压 + 部分排空的 resume |
| 9 | content-hash journal | native cache 内容寻址但 qualname-scoped；同步 `put_writes` 无 ERROR 守卫（bug #7589）；命中丢 stream 数据（#6265） | **partial → 自建** | 自建：**success-only 语义** + per-node content scoping over `BaseStore` |
| 10 | fail-loud determinism guard | 仅 `-O` 可剥离的裸 assert；无确定性异常类 | **gap** | 自建：记录并在 leaf-call **顺序/参数内容**漂移时 raise |
| 11 | `budget`（token） | `usage_metadata` + `UsageMetadataCallbackHandler` | partial | 自建 enforcement + `@task` 层复刻 callback 转发 |
| 12 | `budget`（call 计数兜底） | `ModelCallLimitMiddleware` | partial | 可选复用，但只计次不计 token |
| 13 | `max_concurrency` 设界 | 默认 None ⇒ **无界** | partial | 引擎必须显式设值 + 资源耗尽守卫 |
| 14 | per-leaf sandbox identity | 仅 `id` 属性；factory 路径 deprecated（0.7.0 移除） | **gap** | 自建：identity 派生（roster name + node id + content hash）+ 传 backend **实例** |
| 15 | SandboxManager（生命周期） | backend **无任何** lifecycle 方法 | **gap** | 自建：lazy-create / TTL / pool / quota / `sandbox.stop()` |
| 16 | `phase()` / `log()` | 脚本层 + `astream` | partial | 脚本层编排 + 叶子进度透传 |

**净结论**：底座 provided 的占多数（#1/2/4/5/6/7）；真正的自建工作集中在 **5 个 gap/必自建项**——pipeline、journal、determinism guard、per-leaf identity、SandboxManager——加上 budget enforcement 与 `max_concurrency` 显式设界两个 partial 收口。

---

## 8. 设计假设 vs 实证 核对表

| 假设 | 裁决 | 证据要点 |
|---|---|---|
| native cache 是 index-based | **partially-confirmed（需改写）** | 结果缓存内容寻址（`_cache.py:26-31` pickle+xxh3，`_algo.py:858-870`）；index 的是 `task_id`（`_algo.py:834-842`）驱动 resume。journal 应以 success-only + per-node scoping + positional resume identity 为由，而非"index-based" |
| determinism 不强制 | **confirmed-in-source** | 唯一检查是 `-O` 可剥离的裸 `assert task_id == task_id_checksum`（`_algo.py:662,855`）；无确定性异常类；resume 仅按 task_id 贴 writes，零比对 |
| `max_concurrency` 无核数默认 | **confirmed-in-source** | `RunnableConfig.max_concurrency` 无默认；async None ⇒ 无 semaphore ⇒ 无界（`_executor.py:135-140`）；核数仅 stdlib 线程池兜底 |
| `BackendFactory` per-call 调用 | **confirmed-in-source** | 每个文件/exec 工具首行 `_get_backend(runtime)`（`filesystem.py:727-749`），但路径 deprecated 0.5.0、移除 0.7.0 |
| ToolRuntime 暴露 thread_id | **partially-confirmed** | 无直接 `thread_id` 字段（实测 `hasattr==False`）；仅经 `runtime.config["configurable"]["thread_id"]` 可达 |
| backend 无生命周期方法 | **confirmed-in-source** | `BackendProtocol`/`SandboxBackendProtocol` 仅文件操作 + `id`/`execute`；lifecycle grep 零命中 |
| `wrap_tool_call` 不能并发扇出 | **confirmed-in-source** | `ToolCallRequest` 仅一个 tool_call；fan-out 在 ToolNode（`tool_node.py:823,856-858`），非 hook |
| 叶子 `ainvoke` 返回 last AIMessage / structured_response | **confirmed-in-source + 实测** | `_OutputAgentState`（`types.py:367-371`）；实测无 response_format 时仅 `messages`、无 structured_response |
| BaseStore 可作 journal | **confirmed-in-source + 实测** | InMemoryStore put→get→search→list_namespaces→delete 完整往返通过 |

---

## 9. 真实 API 签名附录（从源码原样抓取）

```python
# langgraph 1.2.2 — func/__init__.py
entrypoint(checkpointer=None, store=None, cache=None, context_schema=None,
           cache_policy=None, retry_policy=None, timeout=None) -> Pregel
entrypoint.final(*, value: R, save: S)
task(__func_or_none__=None, *, name=None, retry_policy=None, cache_policy=None, timeout=None)

# langgraph 1.2.2 — types.py
@dataclass class CachePolicy(Generic[KeyFuncT]): key_func: KeyFuncT = default_cache_key; ttl: int|None = None
class CacheKey(NamedTuple): ns: tuple[str,...]; key: str; ttl: int|None
Send(node: str, arg: Any, *, timeout: float|timedelta|TimeoutPolicy|None=None)
Command(*, graph=None, update=None, resume=None, goto: Send|Sequence[Send|N]|N=())
Durability = Literal["sync","async","exit"]   # 默认 "async"（main.py:2574）

# langgraph 1.2.2 — _internal/_cache.py
default_cache_key(*args, **kwargs) -> str|bytes  # pickle.dumps((_freeze(args),_freeze(kwargs)), protocol=5)

# langgraph 1.2.2 — pregel/_algo.py（positional task_id）
task_id = task_id_func(checkpoint_id_bytes, checkpoint_ns, str(step), name,
                       PUSH, task_path_str(task_path[1]), str(task_path[2]))
assert task_id == task_id_checksum   # :662, :855（-O strippable）

# langgraph-checkpoint 4.1.1 — checkpoint/base
BaseCheckpointSaver.put(self, config, checkpoint, metadata, new_versions) -> RunnableConfig
put_writes(self, config, writes: Sequence[tuple[str,Any]], task_id: str, task_path: str='') -> None
get_tuple(self, config) -> CheckpointTuple | None
get_next_version(self, current: V|None, channel: None) -> V
InMemorySaver(*, serde=None, factory=defaultdict)   # 别名 MemorySaver

# langgraph 1.2.2 — store/base
BaseStore.put(namespace: tuple[str,...], key: str, value: dict[str,Any],
              index: Literal[False]|list[str]|None=None, *, ttl: float|None|NotProvided=NOT_GIVEN) -> None
BaseStore.search(namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None) -> list[SearchItem]

# langchain-core 1.4.0 — runnables/config.py
RunnableConfig.max_concurrency: int | None      # 无默认
get_executor_for_config(...) -> ContextThreadPoolExecutor(max_workers=config.get('max_concurrency'))

# langgraph 1.2.2 — pregel/_executor.py
if max_concurrency := config.get('max_concurrency'): self.semaphore = asyncio.Semaphore(max_concurrency)
else: self.semaphore = None

# deepagents 0.6.7 — graph.py
create_deep_agent(model=None, tools=None, *, system_prompt=None, middleware=(), subagents=None,
    backend: BackendProtocol|BackendFactory|None=None, response_format=None, state_schema=None,
    context_schema=None, checkpointer=None, store=None, name=None, cache=None) -> CompiledStateGraph[...]

# deepagents 0.6.7 — backends/protocol.py
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]   # :851（DEPRECATED 0.5.0，移除 0.7.0）
class SandboxBackendProtocol(BackendProtocol):
    @property def id(self) -> str
    def execute(self, command: str, *, timeout: int|None=None) -> ExecuteResponse
    def aexecute(...) -> ExecuteResponse                # 无 close/start/stop

# deepagents 0.6.7 — middleware/subagents.py
class CompiledSubAgent(TypedDict): name: str; description: str; runnable: Runnable   # runnable state 须含 'messages'
_EXCLUDED_STATE_KEYS = {'messages','todos','structured_response','skills_metadata','skills_load_errors','memory_contents'}

# langgraph 1.2.2 — pregel/main.py（叶子调用）
async def ainvoke(self, input, config=None, *, context=None, stream_mode='values',
    output_keys=None, durability=None, version='v1', **kwargs) -> dict[str,Any] | Any

# langchain 1.3.2 — agents/middleware/types.py
def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command[Any]   # 单 tool_call

# langchain 1.3.2 — agents/middleware/model_call_limit.py
ModelCallLimitMiddleware(*, thread_limit=None, run_limit=None, exit_behavior: Literal['end','error']='end')

# langchain-core 1.4.0 — messages/ai.py & callbacks/usage.py
class UsageMetadata(TypedDict): input_tokens: int; output_tokens: int; total_tokens: int; ...
class UsageMetadataCallbackHandler(BaseCallbackHandler): usage_metadata: dict[str, UsageMetadata]
@contextmanager get_usage_metadata_callback(name='usage_metadata_callback')
```

---

## 10. Open Questions / 版本风险

1. **设计文档纠偏（最高优先级）**：把"native cache 是 index-based"改写——结果缓存内容寻址，positional 的是 `task_id`（驱动 resume）。journal 的 rationale 改为 success-only + per-node scoping + positional resume identity。
2. **bug #7589（OPEN）**：同步 `put_writes` 缓存 INTERRUPT/ERROR 无守卫，async 路径有。**确认引擎用 sync `invoke/stream` 还是 async `ainvoke/astream`**——若 sync 路径用 CachePolicy，error-caching 是 live 风险。建议引擎统一走 async + 自建 success-only journal。
3. **`task_id` 顺序敏感性**：resume-skip 正确性依赖脚本跨运行产出**相同 task 顺序/step**。加一个"脚本编辑后 resume"的集成测试——顺序漂移会静默失配重跑。
4. **checkpointer 持久化**：仅装 `InMemorySaver`。需跨进程持久化则加 `langgraph-checkpoint-sqlite/-postgres` 并对 1.2.2 重验 `get_next_version`/`put_writes`。
5. **`max_concurrency` 嵌套语义**：确认 `agent()` 叶子 fan-out 是共享 entrypoint 层 semaphore，还是 deepagents 子调用另开无界 executor——关乎 bounded-queue 守卫。
6. **BackendFactory 弃用时钟**：0.7.0 移除。per-leaf identity **不要**依赖 factory callable——改 per-thread/per-leaf 实例构造。
7. **CompositeBackend 隔离泄漏（#2884，OPEN）**：并行叶子隔离不能仅靠 routes 保证，需独立验证。
8. **structured_response 非 pydantic 序列化**：`json.dumps` fallback 是否适配所有 strategy（ToolStrategy/ProviderStrategy/AutoStrategy）——strategy 决定 `structured_response` 是 pydantic 实例还是 plain dict。
9. **ProviderStrategy 无 in-loop retry**：若引擎要原生结构化输出 retry，须自建（如 `wrap_model_call`）；并确认 langchain-anthropic 1.4.4 下 AutoStrategy 对 Anthropic 是否实际选 ToolStrategy。
10. **callback 转发**：`@task` 层直接 invoke 子 agent 须复刻 `_build_subagent_config` 的 `callbacks` 转发，否则共享 budget 漏算。
11. **determinism guard 范围**：确认校验 leaf-call **顺序**、**参数内容**还是两者——底座唯一信号（positional `task_id` 裸 assert）两者都覆盖不牢且 `-O` 可剥离。
12. **`-O` 风险**：生产若开 `PYTHONOPTIMIZE`，底座那句唯一的 determinism assert 整句消失——又一条 fail-loud guard 必须自建的理由。

---

## 11. 带标注信源清单

**github-source（installed source，verified-in-source 优先）**
- langgraph 1.2.2 — `func/__init__.py`：`file:///Users/panqiwei/Dev/repos/nemori-ai/langchain-dynamic-workflow/.venv/lib/python3.12/site-packages/langgraph/func/__init__.py`
- langgraph 1.2.2 — `types.py`：`.../langgraph/types.py`
- langgraph 1.2.2 — `_internal/_cache.py`：`.../langgraph/_internal/_cache.py`
- langgraph 1.2.2 — `pregel/_algo.py`（cache key + task_id + determinism assert）：`.../langgraph/pregel/_algo.py`
- langgraph 1.2.2 — `pregel/_loop.py`（match_cached_writes、_reapply_writes_to_succeeded_nodes、sync put_writes）：`.../langgraph/pregel/_loop.py`
- langgraph 1.2.2 — `pregel/_runner.py`（cached RETURN skip、time.monotonic timeout）：`.../langgraph/pregel/_runner.py`
- langgraph 1.2.2 — `pregel/_executor.py`（async semaphore）：`.../langgraph/pregel/_executor.py`
- langgraph 1.2.2 — `pregel/_call.py`（SyncAsyncFuture）：`.../langgraph/pregel/_call.py`
- langgraph 1.2.2 — `pregel/main.py`（ainvoke 签名 + CONFIG_KEY_THREAD_ID 落点）：`.../langgraph/pregel/main.py`
- langgraph 1.2.2 — `errors.py`（无确定性异常类）：`.../langgraph/errors.py`
- langgraph 1.2.2 — `_internal/_replay.py`、`_internal/_constants.py`：`.../langgraph/_internal/`
- langgraph 1.2.2 — `prebuilt/tool_node.py`（ToolRuntime、ToolNode fan-out、wrapper per-call）：`.../langgraph/prebuilt/tool_node.py`
- langgraph 1.2.2 — `store/base/__init__.py`（BaseStore）：`.../langgraph/store/base/__init__.py`
- langgraph-checkpoint 4.1.1 — `checkpoint/base/__init__.py`、`checkpoint/memory/__init__.py`：`.../langgraph/checkpoint/`
- langchain 1.3.2 — `agents/factory.py`（create_agent、_handle_model_output）：`.../langchain/agents/factory.py`
- langchain 1.3.2 — `agents/middleware/types.py`、`model_call_limit.py`、`structured_output.py`：`.../langchain/agents/middleware/`、`.../langchain/agents/structured_output.py`
- langchain-core 1.4.0 — `messages/ai.py`、`callbacks/usage.py`、`runnables/config.py`：`.../langchain_core/`
- deepagents 0.6.7 — `graph.py`、`middleware/subagents.py`、`backends/protocol.py`、`backends/composite.py`、`backends/sandbox.py`、`backends/state.py`、`backends/filesystem.py`、`backends/local_shell.py`、`backends/langsmith.py`、`middleware/filesystem.py`、`__init__.py`：`.../deepagents/`
- Issue #7589（sync put_writes 缓存 INTERRUPT/ERROR 无守卫）：`https://github.com/langchain-ai/langgraph/issues/7589`
- Issue #6265（缓存命中丢 stream 数据）：`https://github.com/langchain-ai/langgraph/issues/6265`
- Issue #6491（无校验写入 checkpoint）：`https://github.com/langchain-ai/langgraph/issues/6491`
- deepagents Issue #2884（CompositeBackend 隔离泄漏）：`https://github.com/langchain-ai/deepagents/issues/2884`
- deepagents Issue #2882 / #3128（首方 sandbox 仍在添加，无生命周期管理）：`https://github.com/langchain-ai/deepagents/issues/2882`

**reference-api**
- CachePolicy 参考（docs 确认 key_func 默认 pickle-hashing，内容寻址）：`https://reference.langchain.com/python/langgraph/types/CachePolicy`
- langgraph 1.2.2 — `cache/base/__init__.py`（BaseCache）：`.../langgraph/cache/base/__init__.py`
- deepagents Python API 参考：`https://reference.langchain.com/python/deepagents`

**official-docs**
- LangGraph Functional API：`https://docs.langchain.com/oss/python/langgraph/functional-api`
- LangGraph Durable Execution（v1.x，**无显式确定性/副作用警告章节**）：`https://docs.langchain.com/oss/python/langgraph/durable-execution`
- LangGraph Pregel / BSP：`https://docs.langchain.com/oss/python/langgraph/pregel`
- Functional API entrypoint 参考：`https://langchain-ai.github.io/langgraph/reference/func/`
- deepagents Subagents（context quarantine、task tool 返回）：`https://docs.langchain.com/oss/python/deepagents/subagents`
- deepagents Customization：`https://docs.langchain.com/oss/python/deepagents/customization`
- deepagents Backends（factory 弃用、Composite/State/Filesystem 路由）：`https://docs.langchain.com/oss/python/deepagents/backends`
- deepagents Sandboxes（backend 包外部 sandbox、无生命周期、per-thread 模式，**支撑 §5(e) 的唯一 official-docs 证据**）：`https://docs.langchain.com/oss/python/deepagents/sandboxes`
- LangChain middleware 概览（仅 overview，细节转 API 参考子页）：`https://docs.langchain.com/oss/python/langchain/middleware`

---

## 补充 A — deepagents 0.6.7 原生 skills 机制（轻量，直接骑；实读源码）

> 动机：本库对外形态拟用「tool + skills」。skills 侧**不自建**（不学 omne-next 的 capability marketplace / 双 BM25 索引 / 多层 gating 那套重型方案），**直接骑 deepagents 原生 `SkillsMiddleware`**。本节为该机制的源码实证补录。

### 机制（verified-in-source，`deepagents/middleware/skills.py`）

- **类 / 挂载**：`SkillsMiddleware`（extends `AgentMiddleware[SkillsState, ...]`，`skills.py:748`）；`create_deep_agent(..., skills=...)` 非空时条件挂载（`graph.py:715-716`）。
- **skill 格式 = 目录 + `SKILL.md`**（YAML frontmatter + markdown body，可带 `helper.py`/`scripts/`/`assets/`）。`SkillMetadata` TypedDict（`skills.py:232-289`）：`path / name / description / license / compatibility / metadata / allowed_tools`；`name` 1–64 字符、kebab、须与目录名一致，`description` 1–1024。**与 Claude Code / omne-next 同款 SKILL.md**。
- **`skills=` 参数** = `list[str] | None`（`graph.py:224`）：每项是**源路径**（POSIX）；也接受 `(path, label)` 元组（`SkillSource = str | tuple[str,str]`，`skills.py:150`）。加载经 backend：`backend.ls(source)` 列子目录 → 每子目录一个 skill → `download_files` 取 `SKILL.md` → 解析+校验 frontmatter；**last-one-wins** 同名覆盖（`skills.py:948-975`）。
- **渐进式披露（关键，天然契合）**：`before_agent()`（`skills.py:941-985`）只加载 **frontmatter** 进 state `skills_metadata`（`PrivateStateAttr`，**不传子 agent**），错误进 `skills_load_errors`；`modify_request()`（`skills.py:913-939`）把 skill 列表（name + description + "Read path for full instructions"）注入 system prompt；**body 按需**——agent 用**已有的 `read_file` 工具**读全文，**没有**专门的 read_skill / load_skill 工具。
- **与 memory / summarization 同模式**：load-once → private state → `append_to_system_message()` 注入 system prompt。
- **最小用法**：`create_deep_agent(model=..., skills=["/skills/user/", "/skills/project/"])`；backend 默认 `StateBackend`（ephemeral，经 `invoke(files=...)` 喂）或 `FilesystemBackend(root_dir=...)`。

### 对本库（LDW）的含义

1. **skills 侧 = provided（deepagents 原生），零自建**。我们只需**提供 SKILL.md 内容**（教 host agent 写编排脚本 + DSL + 确定性铁律 + 范式：parallel / pipeline / loop-until-budget / 对抗验证），放进一个 skills 目录，`create_deep_agent(skills=[...])` 即可。
2. **「L2-as-skill」落地坐实**：host agent 经 skill（渐进披露）学会写脚本 → 调 workflow tool。**无需引擎内置 codegen agent**（呼应接缝③、让 D2「含 meta 层」更轻）。
3. 渐进披露用的是 deepagents **自己的 backend + `read_file`**，与本库 per-leaf sandbox/backend 体系同源，**无新依赖**。
4. `skills_metadata` 是 **PrivateStateAttr 不传子 agent** → skills 天然只给**host / 编排 agent**，不泄进 leaf agent context——正合「skills 教 host 写脚本、leaf 只干活」的分层。
5. **取舍**：deepagents skills 走 prompt-injection + `read_file`，**不是**结构化检索（无 BM25 / capability index）。skill 数量大时无语义检索——但我们 v1 的 skill 集很小（一套编排教学），**够用**；要检索是 v2 再说，别学 omne-next 提前上重型索引。

---

## 补充 B — langchain agentic middleware 机制（实读源码，langchain 1.3.2）

> 动机：本库消费者是 AI agent（只能 tool call）；要决定怎么把 workflow tool + skills + 横切（budget/guard）交付给 deepagent，须先吃透 middleware 能力面。

### 机制（verified-in-source，`langchain/agents/middleware/types.py` + `factory.py`）

- **`AgentMiddleware[StateT, ContextT, ResponseT]`**（`types.py:383`）：贯穿 agent 生命周期的扩展点基类，由 `create_agent`（`factory.py:696`）编进 LangGraph agent 图的多个节点：START → before_agent → before_model → model → (tools|end) → after_model → after_agent → END。
- **能力面（每个扩展点能干什么）**：

| 扩展点 | 位置 | 能力 / 限制 |
|---|---|---|
| `before_agent`/`after_agent` | types.py:419/638 | 会话首尾各一次，返回 state 合并 |
| `before_model`/`after_model`（+`a*`） | types.py:443/467 | 每次 model call 前后；`@hook_config(can_jump_to=['end','tools','model'])` 可跳转控制流 |
| `wrap_model_call` | types.py:491 | 包**一次 model 调用**：retry/fallback/短路/改 request(含 `request.override(system_message=...)`/`request.tools`) |
| `wrap_tool_call` | types.py:662 | 包**单个 tool 调用**（单粒度）：retry/改参/短路。**硬天花板:一个 hook 只见一个 tool call,扇不出 N 个 subagent** |
| `.tools` 属性 | types.py:398 | **middleware 可贡献工具**:`create_agent` 收集 `[t for m in middleware for t in m.tools]`、与用户 tools 合并(扁平、同名 last-wins)。**创建期静态注册**,不能运行期动态加 |
| `state_schema` | types.py:353 | 扩展私有/持久 state 字段(`PrivateStateAttr`),多 middleware + 用户 schema 合并 |
| `dynamic_prompt`/`wrap_model_call` | types.py:1603 | 注入/改写 system prompt(读全量 `ModelRequest`) |

- **挂载**：`create_agent(..., middleware=[...])` / `create_deep_agent(..., middleware=...)`；**列表首个 = 最外层**；middleware tools 与用户 tools 合并；state schema 合并。
- **CAN**：贡献工具、注入 prompt、拦改 model/tool call、扩展/管理 state、跳转控制流、限额（`ModelCallLimitMiddleware`）、横切记账/可观测。
- **CANNOT（硬天花板）**：`wrap_tool_call` 扇不出并发 subagent；不能运行期动态加 tool 到 model 列表（创建期静态；workaround = `wrap_model_call` 改 `request.tools`）；不能注入任意计算 graph 节点。
- **内置范例**：`ModelCallLimitMiddleware`（state+limit）、`SummarizationMiddleware`（wrap_model_call 上下文管理）、`FilesystemFileSearchMiddleware`（`.tools` 贡献 glob/grep）、deepagents `SkillsMiddleware`（prompt 注入 skills）。

### 裁决与对 LDW 的含义

- **单个 middleware 可同时**：(a) `.tools` 贡献 workflow tool、(b) `wrap_model_call`/`before_model` 注入 skills、(c) `after_model`/`before_model` 做 budget/guard 横切。**这是 idiomatic langchain**——omne-next 的 middleware-as-bundle 被实证为正统。→ 我们可选的 `create_workflow_middleware()` 站得住。
- **关键 scope 边界（易混淆，务必钉死）**：middleware 的 budget/guard 横切作用于 **host agent 自己的 model/tool 回合**；而 workflow **run 内部的 leaf agents** 跑在 `run_workflow`（独立 `@entrypoint`）里、**不在 host agent 的 middleware scope**。→ **引擎内部 budget/journal/determinism guard 是另一套 scope，与 host middleware 横切不是同一回事，别混。**
- **workflow tool 经 `.tools` 静态注册**（就一个工具、无 fan-out 诉求）完全 OK；`wrap_tool_call` 单粒度天花板对"工具本身"无碍——fan-out 发生在工具内部的 `run_workflow` @task 层。
- **skills 两条路**：deepagents 原生 `skills=[...]`（最轻）；或由我们的 middleware 一并注入——取决于是否要 middleware 打包。

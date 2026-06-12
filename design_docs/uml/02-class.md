# UML · 类图（Class）

```mermaid
classDiagram
  class run_workflow {
    <<facade>>
    +run_workflow(script, *, roster, config) Any
    +on_progress / on_span / on_span_begin / on_leaf_event / on_command sinks
    +leaf_event_include_payloads: bool
    +command_include_payloads: bool
  }
  class WorkflowEngine {
    +run(script, thread_id, config) Any
    -build_entrypoint() Pregel
  }
  class Ctx {
    <<injected primitives>>
    +agent(prompt, *, schema, agent_type, ...) Any
    +parallel(thunks) list
    +pipeline(items, *stages) list
    +race(candidates, *, win, win_tag) RaceResult
    +dag(nodes) dict (DAG 拓扑序 fan-out; DagNode 依赖边; Kahn 调度; 独立失败跟踪)
    +loop_until(body, *, done, max_iters) list (有终止保证的顺序循环; 全量 done 谓词)
    +workflow(name, args) Any (内联嵌套, depth/cycle 守卫)
    +checkpoint(ask, *, tag) Any (HITL 签核门:journal 记决策/注入 pending/否则 raise WorkflowSignoffRequired;depth-0 守卫)
    +phase(title)
    +log(msg)
    +budget
    +max_workflow_depth: int (默认 DEFAULT_MAX_WORKFLOW_DEPTH=8; run_workflow 线传)
  }
  class Roster {
    -_built: dict~tuple, Runnable~
    +register(name, runnable, *, builder, description, needs_execution, default_model)
    +resolve(name) RosterEntry
    +runnable_for(name, *, response_format) Runnable
    +list_agents() list~RosterEntry~ (按名排序目录; 供 agents/catalog 发现)
  }
  class RosterEntry {
    +name
    +description
    +runnable: CompiledStateGraph | None
    +builder: Callable | None
    +needs_execution: bool
    +default_model
  }
  class SchemaConverter {
    <<_schema.to_pydantic_model>>
    +to_pydantic_model(schema) type~BaseModel~
  }
  class Journal {
    <<content-hash, success-only>>
    +lookup(key) Result
    +put(key, result, usage)
    +journal_key(*, prompt, agent_type, ...) str
    +race_key(*, candidate_keys, win_tag) str
  }
  class JournalStore { <<protocol>> }
  class DeterminismGuard { +check_replay(seq) }
  class PipelineScheduler { +run(items, stages) list }
  class SandboxManager {
    +acquire(leaf_id, isolation) Backend
    +lease(leaf_id, needs_execution, isolation) Backend
    +stop(id) (调 close)
    +git_worktree_provider property
    -worktree_provider: WorktreeProvider?
    -git_worktree_provider: GitWorktreeProvider? (M6, isolation=worktree 优先)
    -factory: SandboxFactory (默认 InMemorySandbox)
    -_admit_slot(leaf_id, isolation) (M6/R8: 阻塞 git off-loop, 出 slot 锁)
    -_pending: set (M6/R8: 并发同-leaf 去重 + 配额计 pending)
  }
  class WorktreeProvider {
    <<protocol>>
    +seed(leaf_id) Mapping
    +collect(leaf_id, files) dict
  }
  class InMemoryWorktreeProvider {
    +seed(leaf_id) Mapping
    +collect(leaf_id, files) dict
  }
  class GitWorktreeProvider {
    <<M6 真 git 服务: 装配期注入 — DANGEROUS OPT-IN 非安全 sandbox>>
    +open_worktree(leaf_id) SandboxBackendProtocol (git worktree add -b leaf/<id>, 根植真目录 + on_close→teardown; 幂等 R4 + 异常安全 R3)
    +collect(leaf_id) dict (真 git diff = 权威变更集; 删除 v1 fail-loud)
    +teardown(leaf_id) (worktree remove --force + branch -D, best-effort 幂等)
    +cleanup_all() (扫 workspace_root; host 须在 finally 调)
    -base_repo / integration_branch / base_ref / workspace_root
    -exec_gate: threading.BoundedSemaphore
  }
  class PullRequestProvider {
    <<protocol, M6>>
    +open(*, branch, title, body, integration_branch) PullRequestRef
  }
  class LocalPullRequestProvider {
    <<M6 离线默认, 幂等 per branch — host finalization (R1, 移出确定性 replay)>>
    +open(...) PullRequestRef
  }
  class PullRequestRef {
    <<frozen>>
    +number / branch / url / integration_branch / created
  }
  class SandboxBackendProtocol {
    <<deepagents protocol: 全文件操作 + id + execute>>
    +id property
    +execute(command, *, timeout) ExecuteResponse
    +ls / read / write / edit / grep / glob / upload_files / download_files
    +close() (_Closeable, 幂等)
  }
  class InMemorySandbox {
    <<离线默认, 零依赖, per-instance dict>>
    +execute(...) no-op ExecuteResponse
    +close() no-op
  }
  class LocalSubprocessSandbox {
    <<M5 真后端: 全协议 + per-leaf 临时根 + stdlib-only — DANGEROUS OPT-IN 非安全 sandbox>>
    +execute(command, *, timeout) ExecuteResponse (Popen, 有界抽干, 超时组杀, spawn 前后发 CommandEvent)
    +ls / read / write / edit / grep / glob / upload_files / download_files (真文件 @ 临时根)
    +id property
    +root_path property
    +set_command_sink(*, sink, leaf_span_id, include_payloads) (引擎接 on_command)
    +close() (删临时根, 幂等)
    -policy: ExecPolicy
    -exec_gate: threading.BoundedSemaphore (一工厂一闸)
    -root: str (默认 tempfile.mkdtemp; M6 可传既有目录根植真 worktree, 不删)
    -on_close: Callable? (M6 一次性回调; git provider 绑 worktree teardown)
    -_command_sink: CommandSink or None (默认 no-op)
  }
  class ExecPolicy {
    <<_local_subprocess: frozen+slots, 韧性+准入策略>>
    +default_timeout: int (30s)
    +output_cap_bytes: int (1MB)
    +grace_seconds: float (2.0)
    +max_concurrent_execs: int (8)
    +rlimits: RLimitProfile
    +before_execute: BeforeExecuteHook or None
  }
  class RLimitProfile {
    <<_local_subprocess: frozen+slots, POSIX best-effort 资源上限>>
    +cpu_seconds: int or None (60)
    +address_space_bytes: int or None (2 GiB)
    +file_size_bytes: int or None (256 MiB)
    +open_files: int or None (1024)
    +processes: int or None (None — per-user 计数不可靠)
  }
  class ExecRequest {
    <<_local_subprocess: frozen+slots, spawn 前准入请求>>
    +command: str
    +timeout: int or None
    +leaf_id: str
  }
  class ExecDecision {
    <<_local_subprocess: frozen+slots, before_execute 返回>>
    +outcome: allow or reject
    +timeout: int or None (仅可收紧)
    +output_cap_bytes: int or None
    +rlimits: RLimitProfile or None
    +reason: str
  }
  class SandboxFactory {
    <<_sandbox: 类型别名 leaf_id to SandboxBackendProtocol>>
    +local_subprocess_factory(policy) SandboxFactory
  }
  class WorkflowTool {
    <<BaseTool, multi-command>>
    +run / run_script / status / resume / cancel / runs / approve
    +catalog / agents (只读: 渲染注册 workflow / leaf 目录, 同样内联进 tool description)
  }
  class WorkflowMiddleware {
    <<AgentMiddleware>>
    +tools: list~WorkflowTool~
    +abefore_model() inject_notice
  }
  class BgRunManager {
    +start(coro, run_id, thread_id) BgRunSlot
    +poll(run_id) Status
    +approve(coro, *, run_id, thread_id) BgRunSlot (就地续跑 parked slot,同 run_id)
    +get_signoff(run_id) ask or None
    +drain_notifications(thread_id)
  }
  class BgRunSlot {
    +run_id
    +status (含非终态 AWAITING_SIGNOFF:计入 active、不被 TTL sweep)
    +task: asyncio.Task
    +result
    +ask (AWAITING_SIGNOFF 时的签核 ask,否则 None)
  }
  class ResultStore { <<protocol: memory|sandbox>> }
  class WorkflowRunStore {
    <<protocol: _run_store, run 注册表持久化边界>>
    +save_spec(run_id, spec) async
    +delete_spec(run_id) async
    +load_spec(run_id) RunSpec or None async
    +journal_for(run_id) JournalStore
  }
  class RunSpec {
    <<frozen+slots: 可 resume 的 launch 描述>>
    +kind: str (name or script)
    +name_or_source: str
    +args: dict (JSON-可序列化)
    +label: str
    +journal_run_id: str or None
  }
  class InMemoryRunStore {
    <<默认, 零依赖>>
    -_specs: dict
    -_journals: dict~run_id, JournalStore~
  }
  class SqliteWorkflowStore {
    <<M3, [sqlite] extra: 统一 db 文件 + 两连接>>
    +open(db_path)$ SqliteWorkflowStore async
    +checkpointer: AsyncSqliteSaver
    +aclose() async
    -_store_conn: Connection (autocommit, WAL)
    -_checkpointer_conn: Connection (第二连接)
  }
  class RunScopedJournal {
    <<_persistence: 一个 run_id 的 JournalStore 视图>>
    +get / put / get_sequence / put_sequence
    +get_progress_count / put_progress_count
  }
  class Reduce {
    <<_reduce: pure cross-leaf helpers>>
    +survives(votes, *, against, kill_at) bool
    +dedup(items, *, key) list
    +reconcile(review_items, *, include) Reconciled
    +corroborate(items, *, key, min_support) list~Consensus~
  }
  class ReviewItem~T,V~ {
    +item: T
    +verdicts: Sequence of V or None
  }
  class Reconciled~T~ {
    +included: list~T~
    +excluded: list~T~
    +conflicts: list~T~
  }
  class Consensus~K,T~ {
    +key: K
    +members: list~T~
  }
  class RaceCandidate {
    <<_race_types: content-hashable agent-call spec>>
    +prompt: str
    +agent_type: str
    +schema: type or dict or None
    +model: str or None
    +isolation: str
  }
  class RaceResult~T~ {
    +winner: T or None
    +winner_index: int or None
    +won: bool
  }
  class DagNode {
    <<_dag: pure value type, slots>>
    +id: str (图内唯一名称; 结果 dict key)
    +deps: Sequence~str~ (前置节点 id; 根节点传空列表)
    +run: Callable~dict, Coroutine~ (接 {dep_id: result} 映射的异步工厂)
  }
  class WorkflowDagError {
    <<_errors: control-flow signal — 入 WORKFLOW_CONTROL_FLOW_SIGNALS>>
    graph structurally invalid before scheduling
    (duplicate id / unknown dep / self-dep / cycle)
  }
  class WorkflowCycleError {
    <<_errors: control-flow signal — 入 WORKFLOW_CONTROL_FLOW_SIGNALS>>
    workflow name already on inlining stack
    (direct recursion or mutual A→B→A)
  }
  class Codegen {
    <<_codegen: L2 AST gate + exec>>
    +compile_workflow_source(source) Callable
  }
  class SpanKind {
    <<_observability: StrEnum — span 类型>>
    AGENT
    PARALLEL
    PIPELINE
    RACE
    DAG
  }
  class SpanRecorder {
    <<_observability: 开 span, 发 begin+end 两条带外边>>
    +span(kind, name) ActiveSpan
    -_mint_span_id(kind, name) str
    -_sink: SpanSink
    -_begin_sink: SpanBeginSink
  }
  class SpanBegin {
    <<_observability: span 打开即发的 running 边>>
    +span_id: str
    +kind: SpanKind
    +name: str
    +attributes: dict
    +started_at: float
    +monotonic_start: float
  }
  class Span {
    <<_observability: span 关闭时发的完成边>>
    +span_id: str
    +kind: SpanKind
    +name: str
    +attributes: dict
    +duration_s: float
    +error: str or None
  }
  class LeafEvent {
    <<_leaf_events: 叶子回调子树的一条 normalize 边>>
    +leaf_span_id: str
    +run_id: str
    +parent_run_id: str or None
    +kind: str (chain|chat_model|llm|tool)
    +phase: str (start|end|error)
    +name: str
    +ts: float
    +detail: dict (默认 shape-only)
  }
  class LeafEventHandler {
    <<_leaf_events: BaseCallbackHandler, 引擎内部, 一叶一实例>>
    +on_chain_start / on_chat_model_start / on_tool_start ...
    -_leaf_span_id: str (构造时闭包持有 → 关联)
    -_include_payloads: bool
  }
  class CommandEvent {
    <<_observability: 真 execute 边界的一条命令生命周期边 (start|end)>>
    +leaf_span_id: str (所属叶 AGENT span id → 关联)
    +command_id: str (resume 稳定, begin/end 共享)
    +command: str
    +phase: str (start|end)
    +exit_code: int or None (start 为 None)
    +output: str or None (start 为 None; end 默认截尾)
    +truncated: bool
    +duration_s: float or None (start 为 None)
    +started_at: float
  }

  run_workflow --> WorkflowEngine
  WorkflowEngine --> Ctx
  WorkflowEngine ..> EntryPoint : LangGraph @entrypoint
  Ctx --> Journal
  Ctx --> DeterminismGuard
  Ctx --> PipelineScheduler
  Ctx --> SandboxManager
  Ctx --> Roster
  Ctx ..> SchemaConverter : agent(schema=) 归一
  Journal ..> JournalStore
  Roster *-- RosterEntry
  SandboxManager ..> Backend : deepagents backend 实例
  SandboxManager ..> SandboxFactory : _new_sandbox 经工厂构造 (默认 InMemorySandbox)
  SandboxManager ..> WorktreeProvider : isolation=worktree 内存播种
  WorktreeProvider <|.. InMemoryWorktreeProvider
  SandboxManager ..> GitWorktreeProvider : isolation=worktree 优先 (M6 真 git)
  GitWorktreeProvider ..> LocalSubprocessSandbox : open_worktree 返回 root= 的真后端 (on_close→teardown)
  PullRequestProvider <|.. LocalPullRequestProvider
  LocalPullRequestProvider ..> PullRequestRef : open 返回 (host finalization)
  SandboxBackendProtocol <|.. InMemorySandbox
  SandboxBackendProtocol <|.. LocalSubprocessSandbox
  SandboxFactory ..> LocalSubprocessSandbox : local_subprocess_factory 产 (共享 exec_gate)
  LocalSubprocessSandbox *-- ExecPolicy
  ExecPolicy *-- RLimitProfile
  LocalSubprocessSandbox ..> ExecRequest : before_execute 入参
  LocalSubprocessSandbox ..> ExecDecision : before_execute 返回 (准入)
  ExecDecision ..> RLimitProfile : rlimits 覆写 (内联 = 选 profile)
  WorkflowMiddleware *-- WorkflowTool
  WorkflowMiddleware *-- BgRunManager
  WorkflowTool --> BgRunManager
  BgRunManager *-- BgRunSlot
  BgRunManager ..> ResultStore
  BgRunManager ..> run_workflow : asyncio.create_task
  WorkflowTool ..> WorkflowRunStore : save/load spec + journal_for(canonical)
  WorkflowRunStore ..> RunSpec : save_spec / load_spec
  WorkflowRunStore ..> JournalStore : journal_for() returns
  WorkflowRunStore <|.. InMemoryRunStore
  WorkflowRunStore <|.. SqliteWorkflowStore
  SqliteWorkflowStore ..> RunScopedJournal : journal_for() view (run_id-scoped)
  SqliteWorkflowStore ..> EntryPoint : checkpointer = AsyncSqliteSaver (第二连接)
  RunScopedJournal ..|> JournalStore
  Reduce ..> ReviewItem : reconcile() input
  Reduce ..> Reconciled : reconcile() output
  Reduce ..> Consensus : corroborate() output
  Codegen ..> Reduce : inject into run_script namespace
  Codegen ..> RaceCandidate : inject into run_script namespace
  Codegen ..> RaceResult : inject into run_script namespace
  Codegen ..> DagNode : inject into run_script namespace (_SCRIPT_DAG_API)
  Ctx ..> RaceCandidate : race() candidate spec
  Ctx ..> RaceResult : race() return
  Ctx ..> Journal : race_key() journals the decision
  Ctx ..> DagNode : dag() fan-out input
  Ctx ..> WorkflowDagError : dag() 格式非法 re-raise
  Ctx ..> WorkflowCycleError : workflow() 重入检测
  SpanKind ..> SpanRecorder : span(kind, ...) 参数类型
  WorkflowEngine --> SpanRecorder : SpanRecorder(sink=on_span, begin_sink=on_span_begin)
  Ctx --> SpanRecorder : span() per primitive (mints span_id, threads into leaf_runner)
  SpanRecorder ..> SpanBegin : 打开即 emit (begin sink)
  SpanRecorder ..> Span : 关闭时 emit (end sink)
  WorkflowEngine ..> LeafEventHandler : 叶调用接缝 append 到 leaf_config.callbacks (miss-only, 闭包持 leaf_span_id)
  LeafEventHandler ..> LeafEvent : normalize 回调边 → on_leaf_event sink
  WorkflowEngine ..> LocalSubprocessSandbox : lease 真后端时 set_command_sink(on_command, leaf_span_id) (miss-only)
  LocalSubprocessSandbox ..> CommandEvent : execute spawn 前后 emit (start 然后 end) → on_command sink
```

## 分层归属

- **公共面(开发者)**：`run_workflow`、`Roster`/`RosterEntry`(+ `Roster.list_agents()` 目录)、`WorkflowRegistry`/`WorkflowEntry`(命名 workflow 注册表 + `list_workflows()` 目录条目;`register(description=)` 缺省回退 callable docstring、经 `_one_line_summary` 归一为单行有界)、`create_workflow_tool`(产 `WorkflowTool`;新增只读 `catalog`/`agents` 命令 + 两套目录 build 时渲染进 tool description,使真实 host 不靠 prompt 写死名字即可发现注册的 workflow/`agent_type`——解「道 vs 术」死结)、`create_workflow_middleware`(产 `WorkflowMiddleware`)、`InMemoryWorktreeProvider`/`WorktreeProvider`(worktree 隔离 seam)、`read_only_leaf`/`read_only_builder`(只读裁判叶,deny-write permission,D-G4)、跨叶归约 helper `survives`/`dedup`/`reconcile`/`corroborate`(+ `ReviewItem`/`Reconciled`/`Consensus`,`_reduce` 纯函数,F)、race 值类型 `RaceCandidate`/`RaceResult`(+ `race_key`,`_race_types` 纯类型,B,配 `ctx.race` 原语)、**DAG 值类型 `DagNode`(`_dag`,M7,配 `ctx.dag` 原语;`_codegen` 注入 `run_script` 命名空间)** + **`WorkflowDagError`/`WorkflowCycleError`(M7)/`WorkflowConcurrencyError`(v0.4.0 M2:depth-0 并发 fail-loud),从包根导出,入 `WORKFLOW_CONTROL_FLOW_SIGNALS`**、可观测性值类型与 sink 别名 `Span`/`SpanBegin`/`SpanKind`/`LeafEvent`/`CommandEvent`(+ `SpanSink`/`SpanBeginSink`/`LeafEventSink`/`CommandSink`,供宿主消费 `run_workflow` 的 `on_span`/`on_span_begin`/`on_leaf_event`/`on_command` 带外边——`CommandEvent`/`on_command` 是执行面同构 sink,在真 subprocess 边界发成对命令生命周期边;`SpanKind.DAG` 标注 dag fan-out span)、**真执行 opt-in 面 `LocalSubprocessSandbox`/`SandboxFactory`/`local_subprocess_factory` + 策略值类型 `ExecPolicy`/`ExecRequest`/`ExecDecision`/`RLimitProfile`(M5,`_local_subprocess`/`_sandbox`,供宿主 `SandboxManager(sandbox_factory=local_subprocess_factory(ExecPolicy(...)))` 注入真本地执行——危险 opt-in,非安全 sandbox)**。`LeafEventHandler` 是引擎内部(不导出)的回调 normalizer,一叶一实例、闭包持 `leaf_span_id` 关联(见 [01 §2b](../01-engine-mechanism.md))。
- **agent 面(运行时)**：`WorkflowTool`(多命令)。
- **host 后台机制**：`WorkflowMiddleware` + `BgRunManager` + `BgRunSlot` + `ResultStore`。
- **跨会话持久化(M3,Layer 2 host-wiring)**：`WorkflowRunStore`(协议)+ `RunSpec`(可 resume 的 launch 描述,携规范 `journal_run_id` 谱系)+ `InMemoryRunStore`(默认、零依赖)+ `SqliteWorkflowStore`(`[sqlite]` extra:统一 sqlite db 文件 + 两条连接——autocommit store 背 registry + `RunScopedJournal` per-run journal、第二连接背持久 `AsyncSqliteSaver` checkpointer)。`_persistence` / `_run_store` 只从 `._engine`(公共墙)import `JournalStore`/`JournalRecord`,故 import-linter Contract 1 把这两模块列入 `source_modules`。零成本重放由持久 journal 交付,checkpointer 是 durable add-on。
- **引擎核心(不可见)**：`WorkflowEngine`、`Ctx`、`Journal`(+`JournalStore`)、`DeterminismGuard`、`PipelineScheduler`、**`_dag.run_dag`(DagScheduler,M7:Kahn 入度调度 + 独立失败跟踪 + 控制流信号穿透)**、`SandboxManager`、`SchemaConverter`(`_schema.to_pydantic_model`,把 JSON-schema dict 归一为 pydantic 模型)、`Codegen`(`_codegen`,L2 AST gate + 受限 `exec`,**依赖 `_reduce` / `_race_types` / `_dag`** 把归约 helper / race 值类型 / `DagNode` 注入 `run_script` 命名空间——故 import-linter "L2 不得触 L0/L1 内部" 契约的 `forbidden_modules` 既不含 `_reduce` / `_race_types` 也不含 `_dag`)。`Roster` 经 `runnable_for(response_format)` 按 `(agent_type, schema)` 缓存绑定变体,builder 条目供 `agent(schema=)`。
- **底座**：LangGraph `@entrypoint`/`@task`/checkpointer/`BaseStore`；deepagents `CompiledSubAgent`/`AgentMiddleware`/`SkillsMiddleware`/backend。

# UML · 类图（Class）

```mermaid
classDiagram
  class run_workflow {
    <<facade>>
    +run_workflow(script, *, roster, config) Any
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
    +phase(title)
    +log(msg)
    +budget
  }
  class Roster {
    -_built: dict~tuple, Runnable~
    +register(name, runnable, *, builder, needs_execution, default_model)
    +resolve(name) RosterEntry
    +runnable_for(name, *, response_format) Runnable
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
  }
  class JournalStore { <<protocol>> }
  class DeterminismGuard { +check_replay(seq) }
  class PipelineScheduler { +run(items, stages) list }
  class SandboxManager {
    +acquire(leaf_id, isolation) Backend
    +lease(leaf_id, needs_execution, isolation) Backend
    +stop(id)
    -worktree_provider: WorktreeProvider?
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
  class WorkflowTool {
    <<BaseTool, multi-command>>
    +run / status / resume / cancel
  }
  class WorkflowMiddleware {
    <<AgentMiddleware>>
    +tools: list~WorkflowTool~
    +abefore_model() inject_notice
  }
  class BgRunManager {
    +start(coro, run_id, thread_id) BgRunSlot
    +poll(run_id) Status
    +drain_notifications(thread_id)
  }
  class BgRunSlot {
    +run_id
    +status
    +task: asyncio.Task
    +result
  }
  class ResultStore { <<protocol: memory|sandbox>> }

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
  SandboxManager ..> WorktreeProvider : isolation=worktree 播种
  WorktreeProvider <|.. InMemoryWorktreeProvider
  WorkflowMiddleware *-- WorkflowTool
  WorkflowMiddleware *-- BgRunManager
  WorkflowTool --> BgRunManager
  BgRunManager *-- BgRunSlot
  BgRunManager ..> ResultStore
  BgRunManager ..> run_workflow : asyncio.create_task
```

## 分层归属

- **公共面(开发者)**：`run_workflow`、`Roster`/`RosterEntry`、`create_workflow_tool`(产 `WorkflowTool`)、`create_workflow_middleware`(产 `WorkflowMiddleware`)。
- **agent 面(运行时)**：`WorkflowTool`(多命令)。
- **host 后台机制**：`WorkflowMiddleware` + `BgRunManager` + `BgRunSlot` + `ResultStore`。
- **引擎核心(不可见)**：`WorkflowEngine`、`Ctx`、`Journal`(+`JournalStore`)、`DeterminismGuard`、`PipelineScheduler`、`SandboxManager`、`SchemaConverter`(`_schema.to_pydantic_model`,把 JSON-schema dict 归一为 pydantic 模型)。`Roster` 经 `runnable_for(response_format)` 按 `(agent_type, schema)` 缓存绑定变体,builder 条目供 `agent(schema=)`。
- **底座**：LangGraph `@entrypoint`/`@task`/checkpointer/`BaseStore`；deepagents `CompiledSubAgent`/`AgentMiddleware`/`SkillsMiddleware`/backend。

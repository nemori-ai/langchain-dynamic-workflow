# UML · 组件图（Component）

```mermaid
flowchart TB
  subgraph AB["AGENT 运行时边界 — 唯一动作 = tool call"]
    HA["«actor» host deepagent"]
    SK["«skills» SKILL.md 编排教学<br/>(注入 host prompt, 行为塑形)"]
    WT["«tool» workflow_tool<br/>run / status / resume / cancel"]
    SK -. 塑形 .-> HA
    HA -- tool call --> WT
  end
  subgraph MW["«middleware» WorkflowMiddleware — async 交付载体"]
    BG["BgRunManager + Registry<br/>asyncio.Task / slots / TTL"]
    NT["abefore_model: 注入完成通知"]
    RS["ResultStore memory/sandbox<br/>(大结果 offload)"]
  end
  WT --> BG
  BG --> NT -. 下一轮 model call 注入 .-> HA
  BG --> RS
  subgraph DEV["开发者 build-time 接线(非 agent 面)"]
    RW["«facade» run_workflow(script, *, roster, config)"]
    RO["Roster (CompiledSubAgent 注册表)"]
    ST["«WorkflowRunStore» run 注册表 + per-run journal<br/>InMemoryRunStore (默认, 零依赖) /<br/>SqliteWorkflowStore (M3, [sqlite] extra)"]
  end
  BG -- asyncio.create_task --> RW
  BG -- "save_spec/load_spec/journal_for (run_id)" --> ST
  ST -. "journal_for(canonical run_id)" .-> JN
  ST -. "SqliteWorkflowStore.checkpointer (第二连接)" .-> EP
  subgraph ENG["«subsystem» Engine L0/L1 (agent 不可见)"]
    CTX["Ctx primitives<br/>agent/parallel/pipeline/race/dag/loop_until/phase/log/budget/workflow"]
    JN["Journal content-hash, success-only"]
    DG["DeterminismGuard divergence backstop"]
    PS["PipelineScheduler bounded queue"]
    DS["_dag DagScheduler<br/>Kahn in-degree + transitive-skip"]
    ER["_errors WorkflowDagError / WorkflowCycleError<br/>+ WORKFLOW_CONTROL_FLOW_SIGNALS"]
    SM["SandboxManager per-leaf 实例"]
    EP["«substrate» LangGraph @entrypoint+@task+checkpointer"]
  end
  RW --> EP --> CTX
  CTX --> JN & DG & PS & DS & SM
  DS --> ER
  CTX --> ER
  CTX --> RO
  CTX -- "agent() 叶子 = @task" --> LEAF["«leaf» deepagent.ainvoke<br/>context quarantine + per-leaf sandbox"]
```

## 三条要确立的边界

1. **agent 唯一运行时面 = `workflow_tool`**(一次 tool call)。库 API `run_workflow()`、primitives 都是 build-time / 开发者面,不是 agent 面。
2. **middleware 是 async 通知的交付载体**:`abefore_model` 在 host agent 下一轮 model call 前注入完成通知(in-band,无需 harness)。
3. **引擎对 agent 完全不可见**:中间结果、leaf 扇出、journal、sandbox 全在 tool 之下;agent context 只收最终结论——control-flow inversion 的对外体现。

## 两层 scope（勿混）

- **host 面后台 tool 包装**(MW):让 host agent 不被 `run_workflow` 阻塞。
- **引擎内部 durable execution**(ENG):`@task`/parallel/journal/sandbox,在 `run_workflow` 内部、与 host middleware 不同 scope。

## 跨会话持久化（M3）

`WorkflowRunStore`(DEV 子图)是 workflow tool 的 run 注册表持久化边界:默认 `InMemoryRunStore`(零依赖、同会话);`SqliteWorkflowStore`(`[sqlite]` extra)用一个统一 sqlite db 文件 + **两条连接**——autocommit store 连接背 registry + per-run journal、第二条连接背持久 `AsyncSqliteSaver` checkpointer。**零成本重放由持久 journal 交付**(checkpointer 是 durable add-on)。per-run 规范 id(`journal_run_id`)同 key journal 谱系与 checkpoint thread;host thread 仅 key manager slot。详见 [01 §13b](../01-engine-mechanism.md)、接线见 [02 §10](../02-architecture.md)。

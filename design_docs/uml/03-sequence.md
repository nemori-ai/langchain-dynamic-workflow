# UML · 时序图（Sequence）

## A — async run → notify（核心闭环）

```mermaid
sequenceDiagram
  participant A as host agent
  participant T as workflow_tool
  participant M as WorkflowMiddleware / BgRunManager
  participant E as run_workflow (Engine @entrypoint)
  participant L as leaf deepagents

  A->>T: tool call run(script)
  T->>M: start(run_workflow(script), run_id)
  M->>M: asyncio.create_task(...) 登记 slot
  M-->>A: 即返占位 ToolMessage(run_id)
  Note over A: agent 不阻塞, 继续别的回合/对话
  par 后台脱离运行
    M->>E: 执行
    E->>E: compile/exec script → Ctx
    alt agent(schema=) 结构化分支
      E->>E: to_pydantic_model(schema) 归一 → ToolStrategy(model, handle_errors=True)
      E->>L: roster.runnable_for(response_format) 取 @task 叶 → 扇出
      L-->>E: structured_response (context quarantine)
      E->>E: fold_structured → journal 存 model_dump_json + usage
    else schema-less 文本分支
      E->>L: agent()/parallel()/pipeline() = @task 扇出
      L-->>E: 仅最终文本 (context quarantine)
      E->>E: fold_result → journal(success-only) + usage; 中间不出引擎
    end
    E-->>M: 最终结论
    M->>M: done callback → 入队通知 + offload 大结果
  end
  A->>M: (下一轮) abefore_model
  M-->>A: 注入 workflow_notification (完成 + 摘要 + run_id)
  A->>T: status / get_result(run_id)
  T-->>A: 最终结论 (或转换 / 摘要)
```

**要点**:`run` 即返占位 → agent 不阻塞;真正执行脱离在后台 `asyncio.Task`;完成经 `abefore_model` **in-band 注入**(无需 harness);agent 经 `status` 取全量结果(大结果 offload)。poll + notify 双支持。

## B — resume（中断后）

```mermaid
sequenceDiagram
  participant A as host agent
  participant T as workflow_tool
  participant E as run_workflow @entrypoint
  participant J as Journal

  A->>T: resume(run_id)
  T->>E: 重放 entrypoint (同 thread_id)
  E->>J: 每个 agent() 查 content-hash (schema dict 先 to_pydantic_model 归一再入 key)
  J-->>E: 命中(success) → 返缓存 (0 模型调用)
  E->>E: 有 schema → model_validate_json 还原结构化对象; 无 schema → 返缓存文本
  E->>E: 未命中 → live 重跑(runnable_for 取缓存绑定变体); 序列不匹配 → fail-loud
  E-->>T: 续跑至最终结论
  T-->>A: 结果
```

**要点**:resume 靠 `@entrypoint` 重放 + content-hash journal(success-only)命中返缓存;带 `schema` 的叶子命中以 `model_validate_json` 还原结构化对象(归一缓存保 `model_json_schema()` 逐字节稳定 → 不静默重跑);只有未完成/失败的叶子 live 重跑(`runnable_for` 取已缓存的 schema 绑定变体);调用序列漂移 → 确定性 backstop fail-loud。

## C — race（fresh / replay：best-of-N 早退 + 取消）

```mermaid
sequenceDiagram
  participant E as Ctx.race
  participant G as DeterminismGuard
  participant J as Journal
  participant L as candidate agent() 叶 (×N)

  Note over E: 前奏(同步, 任何派发之前)
  E->>E: resolve 每个候选 + journal_key(候选叶 key) + 同构校验(全无 schema 或全同一 schema)
  E->>E: race_key(candidate_keys, win_tag) ("race" 命名空间, 绝不撞叶 key)
  alt 深度 0(顶层 race)
    E->>G: observe(race_key) (候选 agent() 在深度 > 0, 不入序列)
  end
  E->>J: get(race_key)
  alt fresh — 未命中
    E->>L: 并发派发全部 N 候选 (经 agent() 复用 journal/budget/sandbox/span)
    L-->>E: 候选结果陆续抵达 (asyncio.wait FIRST_COMPLETED)
    E->>E: 同一 wakeup 按升序下标决断 → 首个令 win(result) 为真者胜
    E->>L: cancel 在飞 loser → gather(return_exceptions=True) 拆除(无孤儿/不漏闸位)
    alt 有胜者
      E->>J: put(race_key, envelope{winner_index, result}, winner_usage)
      Note over E: 胜者 usage 取自其叶 entry; race-key 不重复计入 budget(防双计)
      E-->>E: 返 RaceResult(winner, winner_index)
    else 无胜者
      Note over E: 不 journal 决策(resume 可重试)
      E-->>E: 返 RaceResult(None, None)
    end
  else replay — 命中
    J-->>E: envelope{winner_index, result}
    E->>E: budget.record(race_key, usage) + decode(有 schema 则 model_validate_json)
    Note over E: 零派发 — loser 永不重跑, resume 比首跑更省
    E-->>E: 返 RaceResult(winner, winner_index)
  end
```

**要点**:race 是一步顺序决策——其 content-stable `race_key` 仅在深度 0 由确定性 guard `observe` 一次,候选 `agent()` 调用跑在深度 > 0、不入序列(完成顺序逐跑不同,同 `parallel`/`pipeline` 叶);fresh 路并发派发、首个令 `win` 为真者胜(同一 wakeup 按升序下标确定性决断)、在飞 loser 在 `finally` 里 cancel + `gather(return_exceptions=True)` 拆除(无孤儿、闸位全释放),胜者写一个自包含 envelope `{winner_index, result}` 到 journal;replay 路命中即解码 envelope 复现胜者、**零派发**;无胜者**不** journal,resume 可重试。引擎控制流信号(budget/确定性)或 `win` 谓词抛错则在拆除 loser 后失声而抛(fail-loud)。

## D — 跨进程 resume（M3:进程 A launch+persist → 退出 → 进程 B reopen+resume 零成本）

```mermaid
sequenceDiagram
  participant PA as 进程 A host
  participant SA as SqliteWorkflowStore (db 文件)
  participant EA as run_workflow @entrypoint
  participant DB as sqlite db 文件 (run_specs + journal_*)
  participant PB as 进程 B host (全新进程)
  participant SB as SqliteWorkflowStore (同一 db 文件)
  participant EB as run_workflow @entrypoint
  participant JB as 持久 journal (RunScopedJournal)

  Note over PA,SA: 进程 A — launch + persist
  PA->>SA: await SqliteWorkflowStore.open(db_path) (宿主持久 loop 内)
  SA->>DB: 开两连接 (autocommit store + 第二连接 AsyncSqliteSaver) + WAL + schema-version guard
  PA->>SA: workflow_tool run(...) → mint run_id → canonical = run_id
  SA->>SA: save_spec(run_id, RunSpec{journal_run_id=run_id}) BEFORE start
  SA->>DB: UPSERT run_specs (durable, autocommit, 零显式 commit)
  PA->>EA: manager.start(_coro, thread_id=host_thread) ; 引擎 thread_id=canonical
  loop 每个完成叶
    EA->>JB: put(leaf_key, JournalRecord) (run_id 命名空间)
    JB->>DB: UPSERT journal_records (durable on return)
  end
  EA-->>PA: 最终结论 (run 完成)
  PA->>SA: await store.aclose() → 关两连接 (释放 -wal/-shm)
  Note over PA: 进程 A 退出

  Note over PB,SB: 进程 B — 全新进程, 同一 db 文件
  PB->>SB: await SqliteWorkflowStore.open(db_path) (新 loop, 新 AsyncSqliteSaver)
  SB->>DB: schema-version 命中 → 幂等 proceed
  PB->>SB: workflow_tool resume(run_id)
  SB->>DB: load_spec(run_id) → RunSpec{journal_run_id=run_id}
  SB-->>PB: spec (canonical = spec.journal_run_id)
  PB->>EB: relaunch ; 引擎 thread_id=canonical ; journal=journal_for(canonical)
  loop 重放 entrypoint body
    EB->>JB: get(leaf_key) (run_id 命名空间)
    JB->>DB: SELECT journal_records
    DB-->>EB: 命中(success) → 缓存结果 (0 模型调用 — journal 交付, 非 checkpointer)
  end
  EB-->>PB: 续跑至最终结论 (完成叶零新成本)
```

**要点**:**头条 = journal 交付零成本重放,checkpointer 是 durable add-on**(resume 侧即便 `checkpointer=None` 仍零成本)。autocommit store 连接让每条 `put()` 返回即 durable(零显式 commit);第二条连接背 `AsyncSqliteSaver`(隔离 regime 不兼容,故分连接、同 db 文件,皆 WAL)。`AsyncSqliteSaver` 构造期绑 event loop——进程 B 是全新 loop + 全新 saver 实例(绝不跨 loop 复用)。**per-run 规范 id**(`journal_run_id`,fresh launch 采自身 `run_id`、resume 从 spec 继承)同 key journal 谱系与引擎 `thread_id`(checkpoint thread),故不同 run 不撞、resume 重接原 thread;**host thread 是另一回事**——只 key manager slot。**save-before-start**:`save_spec` 先于 `manager.start`,被准入的 run 总有可 resume 的 spec(quota 拒入则 `delete_spec` 回滚)。`journal_key` 跨 `PYTHONHASHSEED` 逐字节稳定 → A 进程写的键 B 进程命中。已知边界:`race()` 无胜者不 journal → 跨进程 resume 会重派候选。

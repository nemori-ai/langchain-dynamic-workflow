# v0.4.0 路线图（Roadmap）— 引擎可观测性 + 交互式体验

> **For agentic workers:** 本文件是 v0.4.0 的**批次总线**。延续 v0.2.0/v0.3.0 的方法：每个目标对应一份独立设计/计划文档（`design_docs/v0_4_0_plans/0N-*.md`，**逐目标增补**，不一次写完）。本批次为**开放状态**——下列为已确认目标之一；v0.3.0 未尽里程碑（M2–M7）与新候选随推进纳入。

## 已确认目标

### M1 · 叶子级实时可观测（live status + 运行事件流）— 设计已定稿，实现待启

**目标：** 让 `run_workflow` 的调用方能按 leaf `agent()` 观测到：① 实时状态 `idle → running → {complete | error}` + 实时 elapsed（leaf 一启动就 `running`，而非完成才显示）；② 可下钻的 per-leaf 运行事件子树（该 leaf 自己的 model / tool / 子 agent 步骤，经 run-tree 的 `run_id`/`parent_run_id` 关联）。定位为**一等公民、框架原生**能力——demo-app 只是消费者之一，任何 tracer/adapter 都能复用。

**关键洞见（解开旧张力）：** 上下文隔离只针对 **host 的 LLM 上下文窗口**（`agent()` 只把最终结果折出、丢弃其余），**不**使 leaf 的运行时事件不可观测。LangChain callbacks（每个 runnable 的 `on_*_start/end/error` 带 `run_id`/`parent_run_id` → 实时运行树）+ deepagents 已向 subagent 转发 callbacks + 引擎已往每个 leaf 注入 callback（用量统计）→ 事件 tap **复用现成、已验证的接缝**，带外吐出，隔离照旧。

**唯一引擎缺口：** `SpanRecorder` 只在 span **完成**时 emit（无 begin 信号）。故今天仅 `complete`/`error` 可观测；`running`/`idle` 与事件子树虽在 leaf 接缝**可得**但未透出。补两个纯增量、keyword-only、默认 no-op 的 **Layer-1** hook：`on_span_begin`(+ 稳定 `span_id`) 与 `on_leaf_event`。

**落地决策（2026-06-04，用户拍板）：**
- **本批次仅交付设计 spec，不实现。** 引擎那两个 hook **交接给并行的引擎-core session 实现**（避免 demo worktree 改 `src/` 与之分叉）；demo-app 消费侧（status chip + 计时器 + drill-in、`ui_adapter`/`ui_bridge`）等引擎表面就位后再做。
- 分阶段 **A → B**：先 `on_span_begin` running 边 + 实时计时器（小、通用、独立有价值、低风险），再 `on_leaf_event` 的 per-leaf 事件子树 drill-in。
- tap 机制 = **callback handler**（leaf `ainvoke` 路径不动、复用现成转发），非 `astream_events`（richer 但 blast radius 大，留作备选）。
- `span_id` = **resume-stable** `(kind+name+occurrence-ordinal)` 哈希（把 demo 已验证的关联逻辑上移进引擎，跨 resume 免费去重）。
- `detail` 默认**只给形状**（节点 kind/name/timing），原始工具参数/模型文本需显式 opt-in。
- `idle` 由**消费者**推断（引擎只 emit running/complete/error；引擎枚举不了未来 leaf）。

完整设计见 [`01-leaf-live-observability.md`](01-leaf-live-observability.md)。

## 状态

- **M1 叶子级实时可观测**：✅ 设计已定稿（spec 交接引擎-core session）；⏳ 实现待启（引擎 hook 先行，demo 消费侧随后）。
- **其余 v0.4.0 目标**：批次开放，逐项增补（候选：v0.3.0 未尽里程碑 M2–M7、面向社区的交互式 demo-app 等）。

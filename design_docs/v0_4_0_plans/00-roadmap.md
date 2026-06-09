# v0.4.0 路线图（Roadmap）— 引擎可观测性 + 交互式体验

> **For agentic workers:** 本文件是 v0.4.0 的**批次总线**。延续 v0.2.0/v0.3.0 的方法：每个目标对应一份独立设计/计划文档（`design_docs/v0_4_0_plans/0N-*.md`，**逐目标增补**，不一次写完）。本批次为**开放状态**——M1 已落地；M2/M3 为 2026-06-09 用户登记的新需求（详细设计待本批次正式启动时展开）；v0.3.0 未尽里程碑（M7、E）与新候选随推进纳入。

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

## 已登记需求（2026-06-09 用户提；详细设计待本批次正式启动展开）

> 下面两块为**需求登记**，不是定稿设计——记录目标、现状缺口与一个已识别的承重架构约束（二者共享 **context-independent transport** 底座），以免 v0.4.0 启动时重新发现。视图复用、渲染技术、回传粒度等细化设计**刻意留空**，待正式启动时讨论。

### M2 · 持久侧边栏 — 多-workflow 实时流程图视图

**目标：** 一个**侧边栏性质的嵌套视图**，实时动态渲染、持续更新**每一条** workflow 的流程图（phase / fan-out / DAG）与各 leaf agent 的实时状态（running / done / error）。从 inline chat 里一次性的卡，升级为常驻、可同时看多条 workflow 的实时总览。

**与既有的关系：** 复用 M1 的事件底座（`on_span_begin` / `on_leaf_event` / 稳定 `span_id`）——M1 已让 inline leaf 实时可观测，本目标把同一事件流升级成**侧边栏 + 图形化流程图 + 多 workflow 并列**的渲染形态。流程图渲染与 v0.3.0 **M7 `DagGraph`**（拓扑序 DAG）天然相邻（M7 是 inline DAG 卡，本目标是侧边栏实时 DAG）。

**关键架构依赖（与 M3 共享）：** 要在侧边栏渲染 **background / detached run**（board 启的那些）的实时流程图，需要 **context-independent transport**（见 M3）——当前 detached asyncio task 不携带 host node context、事件推不出（UI-dark）；inline run 不需要它（M1 已覆盖），background run 需要。

### M3 · run board 下钻 + 背景 run 结果回传 host

**目标：** ① board 行可**下钻**到该 background run 的实时内部（流程图 + leaf 状态）；② 每个 background job **运行完把报告（结果）回传给调用 `run_runs_board` 的 host agent**，让 host 能据此向用户**汇报情况和结果**——不只是"3 个完成了"，而是"RAG vs long-context 的结论是 X……"。

**现状缺口（精确）：**
- ① 下钻：board 行当前只有聚合状态（`RunSnapshot.status` + 80-char `summary`），**无下钻**——background run 是 detached task、UI-dark（`RunBoard.tsx` 注释已钉死）。
- ② 回传：`run_runs_board_live` 当前只返回一句聚合 summary（"Ran 3 of 3: 3 finished"），**没把每个 run 的实际 result 喂回 host 上下文**，host 因此无内容可汇报。

**拆解（两半依赖不同，承重在 ①）：**
- ① 下钻 = **context-independent transport**（核心基建）：让 detached background run 的 leaf 事件靠 run-tree `run_id`/`parent_run_id` **定向**流回发起它的 host run + UI surface。这与 M2 渲染 background run 是**同一块底座**——v0.4.0 最重一块，比 M1 重（要解开"detached task 怎么找回正确的 UI / host 流"）。
- ② 回传 = 相对独立、**不依赖 transport**：run 完成后结果已在 `ResultStore`；`run_runs_board` 收尾时 fetch 每个 run 的结果作为 tool 返回值喂回 host。设计点：守**控制流反转**（host 上下文只收**精炼** report、非 3 份全文）——回传粒度（精炼摘要 / 按需 fetch handle）留待设计定。

## 状态

- **M1 叶子级实时可观测**：✅ 已落地（引擎 hook PR #12 + demo 消费 PR #13）——inline leaf 实时 status + 计时器 + drill-in。
- **M2 持久侧边栏多-workflow 实时视图**：📝 需求已登记（2026-06-09）；详细设计待 v0.4.0 正式启动。
- **M3 run board 下钻 + 背景 run 结果回传 host**：📝 需求已登记（2026-06-09）；① context-independent transport 是 M2 + M3① 的**共享承重基建**（v0.4.0 核心攻坚），② 结果回传相对独立、不依赖 transport；详细设计待启动。
- **其余 v0.4.0 目标**：批次开放，逐项增补（候选：v0.3.0 未尽 M7/E、面向社区的交互式 demo-app 等）。

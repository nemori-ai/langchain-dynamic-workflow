# v0.4.0 路线图（Roadmap）— 引擎可观测性 + 交互式体验

> **For agentic workers:** 本文件是 v0.4.0 的**批次总线**。延续 v0.2.0/v0.3.0 的方法：每个目标对应一份独立设计/计划文档（`design_docs/v0_4_0_plans/0N-*.md`，**逐目标增补**，不一次写完）。本批次为**开放状态**——M1 已落地；E（`batch_map`，自 v0.3.0「後續里程碑」backlog 提升）已落地；**M3（transport 底座 + 下钻 + 结果回传）设计已定稿**（2026-06-10，见 [`03-background-run-transport-and-m3.md`](03-background-run-transport-and-m3.md)，切片 1）；M2（持久侧栏）的承重底座已随 03 设计完毕，其侧栏布局细化设计留切片 2（`04-*.md`）展开（M7 已于 v0.3.0 落地，不在本批次）；新候选随推进纳入。

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

### E · 批处理人体工学（`batch_map` + 流式准入 + count/ETA 进度）— ✅ 已落地

**目标：** 让**大规模 fan-out（数千叶）**人体工学化，作为一个里程碑交付三块耦合能力：① `batch_map`——把一个异步 `fn`（典型为单个 `agent()`）map 到一个 `Iterable`/`AsyncIterable` 的每个 item、结果按输入序回收（`parallel` 的大扇出对位面）；② **流式准入**——输入经有界窗口懒消费，N 千 item 永不一次性物化 N 千 task，内存被窗口而非总量绑定；③ **count/ETA 进度**——随推进自动发实时 `completed/total/elapsed/eta`，长批次无需脚本插桩即可观测。源自社区 use-case study #5（codebase 级 bug/vuln sweep）与"数千叶 vs CC 的 16-并发/1000-总量上限、批处理人体工学如何"的公开提问。

**关键设计（载重）：** 流式准入落在**新 `batch_map`、`parallel` 不动**，内部**广义化既有 `run_pipeline`** 吃任意 `Iterable | AsyncIterable`（拆三处 `len` 依赖、`dict` 按下标保序回收），故在飞 task ≈ `worker_count + queue_maxsize`、与 N 解耦；`pipeline` 的 `Sequence` 快速路径逐字节保持。进度复用 `ProgressSink` + 一个 **transient `ProgressKind.BATCH` 条目**（经 `ProgressLog.emit_transient` 投递到 sink 但**不记录**——不入 `_entries`/`delivered_count`/journal/确定性 guard/replay，故 resume 时 re-emitted-not-replayed）。新增公共面全 additive：`Ctx.batch_map` · `BatchMetrics`（包根导出）· `ProgressKind.BATCH` · `SpanKind.BATCH` · `ProgressEntry.metrics`。

完整设计见 [`02-e-batch-ergonomics.md`](02-e-batch-ergonomics.md)；机制见 [01 §9b](../01-engine-mechanism.md)。

## 已登记需求（2026-06-09 用户提；详细设计待本批次正式启动展开）

> **更新（2026-06-10）：** 下面两块原为需求登记。现 **M3 已完成定稿设计**（transport 底座 + ①下钻 + ②回传，见 [`03-background-run-transport-and-m3.md`](03-background-run-transport-and-m3.md)）：scope=全量登记愿景、排序=切片1（底座+M3①）→切片2（M2）、M3② 独立穿插；transport=A1（缓冲+活轮 pump、raw-event buffer 挂 BgRunSlot、每轮全量重放 fresh UiAdapter）；M3② 回传=每 run 摘要+按需 fetch handle（复用 RunResult/ResultStore）。**M2 的承重底座已随 03 设计完毕，仅其侧栏布局/渲染细化留切片 2（`04`）。** 下面保留原始需求登记原文，作为目标与现状缺口的存档。

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
- **M3 run board 下钻 + 背景 run 结果回传 host**：✅ 已落地（2026-06-10，设计见 [`03-background-run-transport-and-m3.md`](03-background-run-transport-and-m3.md)，切片 1，引擎 + demo 双轨）。① 下钻经 **context-independent transport 底座**（A1：BgRunSlot 有界 raw-event buffer + `event_sinks`/`buffered_events` + 活轮全量重放 fresh UiAdapter；瞬态遥测不入 journal/replay，隔离照旧），demo 侧 `drill_run` 对话式下钻；② 结果回传复用 RunResult/ResultStore（board 收尾返回每 run 摘要 + `fetch_run_result` 按需拉全文）。验收：引擎全套 + demo 套件绿、全仓 ruff/pyright 0、**gated 真模型 E2E 实跑通过（125s，三个真 deep_research 扇出 → drill 重放真扇出内部到 ui channel）**。`drill target` 解析在精确 id/label/前缀之外，兜底唯一-不区分大小写-子串（口语化指代）。
- **M2 持久侧边栏多-workflow 实时视图**：🧱 承重底座已随 03 设计完毕（与 M3① 共享 transport 底座）；其侧栏布局 / 多-workflow 并列 / 渲染技术等**细化设计待切片 2 启动**（`04-*.md`）。
- **E 批处理人体工学（`batch_map`）**：✅ 已落地——流式准入大扇出 map（广义化 `run_pipeline`、`parallel` 不动）+ transient count/ETA 进度（`ProgressKind.BATCH`，delivered-not-recorded）；自 v0.3.0「後續里程碑」backlog 提升而来。设计见 [`02-e-batch-ergonomics.md`](02-e-batch-ergonomics.md)。
- **其余 v0.4.0 目标**：批次开放，逐项增补（候选：面向社区的交互式 demo-app 等；自 v0.3.0 提升的 E 已落地，M7 已于 v0.3.0 落地）。

## 下一步推进顺序（2026-06-11 用户拍板）

> 本节为某次 session 剔除上下文污染后、由用户明确确认的干净主线。

**总优先级（用户明确）：** 引擎本身的能力与功能做扎实 = 根本；对外集成放后面。

1. **引擎核心健壮性盘点（做厚）—— 首位。** 拿 `AGENTS.md` 的工业级韧性原则当准星，逐个犁编排原语（`parallel` / `pipeline` / `race` / `dag` / `loop_until` / `batch_map`）+ 韧性机制（`journal` / `determinism guard` / `resume` / `budget`），产出「健壮性欠账清单 + 严重度」。判据：每条发现须**可证伪**——精确 `file:line` + 触发条件 + 后果 + 一个能亮红灯的测试；写不出红灯测试的不计入。高危区在**结合部**（并发×取消、确定性×resume、资源界×异常路径、错误传播×隔离）。
2. **按欠账清单排健壮性加固里程碑**，逐条走完整 TDD 闭环（Red→Green→Refactor + 真模型 E2E + 跨模型评审 + evergreen docs 同步）。
3. **M2 持久侧栏多-workflow 实时视图 —— 排在健壮性之后。** 本批次唯一剩的已确认目标，但性质属「可观测 / 人机交互」（demo 消费侧、引擎零改动），按「引擎本身优先」让位于核心加固；做完顺带收口 v0.4.0。承重底座（M3 transport）已就位，细化设计仍待 `04-*.md`。
4. **对外集成（`workflow` tool 界面成熟度、leaf 框架无关 / 自研 agent 构建等）—— 最后。** 引擎仍在实验迭代期、对外集成方式可能变，不为可能要变的边界提前抛光。

**已否决 / 不立项：**

- **tool 界面成熟度审计** —— 与「对外放后面」冲突，砍掉。
- **脱离 deepagents 自研 agent 构建 / leaf 框架无关** —— 在「引擎作为 deepagents 工具」拓扑下属工程卫生，非战略，不立项。

> **本节产生背景（诚实备注）：** 主线由用户 2026-06-11 拍板。同一 session 后段发生**上下文污染**——工具结果通道被注入第一人称 agent 推理文本，并**伪造了部分文件读取内容**（例如 `tests/unit/test_concurrency.py` 被读成完全不存在的测试集）。据此凭空「发现」的一个 `_ConcurrencyGate` local-permit 泄漏 bug **经查实为污染伪造，作废、不进路线**（真实 gate 以 `.run()` 包裹协程工厂，已有 `test_gate_run_releases_slot_on_exception` 覆盖异常释放）。经 git 多路交叉验证：该污染 session **未对仓库造成任何改动**（working tree 全程 clean、HEAD 未变）。本节经 `git diff` 确认真实落盘。

## 已落地（2026-06-12，PR #22）

- **引擎核心健壮性盘点（做厚）—— ✅ 已落地（9 里程碑）。** 14 条可证伪欠账（2 CRITICAL / 7 HIGH / 2 MEDIUM / 3 LOW）按共因聚类成 9 个加固里程碑：M1 resume 重入守卫（canonical 原子预留）、M2 determinism×depth-0 并发 fail-loud、M3 fan-out 内/外 `CancelledError` 之分、M4 sandbox lease 取消/失败安全（三窗口统一 reclaim）、M5 checkpoint 决策类型稳定、M6 dag 同步抛错隔离、M7 `batch_map` sink 隔离 + ETA drop-to-unknown、M8 `loop_until` 每轮 `loop_key` 计数守卫 + body 失败 `.partial`、M9 budget doc 对齐 + persistence corrupt-journal actionable 错误。逐条走 Red→Green→Refactor + 红灯回归 + Codex 跨模型复审。
- **注册可发现性 —— ✅ 已落地。** 真实 host 驱动 `workflow` 工具时无法在不违反「道 vs 术」的前提下得知注册了哪些 workflow / `agent_type`，导致旗舰 demo 真实模型下走不通。补齐：`WorkflowRegistry.list_workflows()` + `WorkflowEntry`（`register()` 加 keyword-only `description`，缺省回退 docstring，经 `_one_line_summary` 归一）、`Roster.list_agents()`、`workflow` 工具只读 `catalog` / `agents` 命令 + 两套目录 build 时渲染进 tool description；`SKILL.md` 加决策顺序 + 「leaf 携带 host 没有的能力」说明。
- **旗舰真实模型验收 —— preset ✅ 真实端到端通过**（发现并按名启动注册的 `deep_research`，产出带 `[VERIFIED]` 的真 web 来源报告）；**authored 现编能力已在真实模型下验证**（host 现编 + 启动 + 进 research 阶段），受成本所限未跑到完整收尾。按 leaf 分模型档（researcher=sonnet+web，机械 leaf=haiku，skeptic 去掉浪费的 web）；可选 `LDW_DEMO_BUDGET` 单 run 上限。验收纪律：真实 E2E demo 须 `timeout 600` 上限 + 无缓冲 + 主动监控。
- **M2 持久侧栏多-workflow 实时视图 —— 仍待**（v0.4.0 剩余收口项；承重底座 M3 transport 已就位，细化设计待 `04-*.md`）。

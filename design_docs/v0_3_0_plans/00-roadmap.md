# v0.3.0 路线图（Roadmap）— 用例驱动逼近 + 选择性超越 Claude Code

> **For agentic workers:** 本文件是 v0.3.0 的**批次总线**。延续 v0.2.0 的"用例驱动"方法：拿社区真实 CC Dynamic Workflow 用例当准星，先做 CC-vs-port 能力对比，再纵切逐条补齐。每条 gap 对应一份独立的 bite-sized TDD plan（`design_docs/v0_3_0_plans/0N-*.md`，**逐里程碑增补**，不一次写完），用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务执行。
>
> 一手依据：① 用例稿 `docs/plans/2026-06-03-v0.3.0-dynamic-workflow-use-cases.md`（17 例 + 10 候选主题，gitignored）；② **对比证据稿** `docs/plans/2026-06-03-v0.3.0-cc-vs-port-comparison.md`（10 主题逐一带 `file:line` 证据 + 置信度，gitignored）。本 roadmap 是这两份的收敛产物。

**方法论（同 v0.2.0）：** ① CC-vs-port 能力对比（已完成，见证据稿）；② 按"杠杆 × 层契合 × 工作量"排序成里程碑；③ 纵切——一次攻一条 gap，走完整 TDD 闭环。

**战略定调（2026-06-03 决策）：允许做 CC 超集。** v0.2.0 只补"我们不如 CC"的 gap；v0.3.0 放开——除补齐差距外，也接纳 CC 本身没有、但真实场景强需的能力（**D 跨会话持久、C 运行中 HITL、I git/PR**），把 port 做成 CC 的超集。

**核心判断：** 对比证据显示，纯编排层的两条（**F 跨叶归约、B 早退/取消**）是信号最密、层契合最干净、且接 v0.2.0 成果（G1 schema + G4 judge）的 CONFIRMED-GAP——理应打头阵。超集三条（D/C/I）底座已撑（LangGraph checkpointer/interrupt、WorktreeProvider 接缝均已预留），是"暴露接缝"而非"从零造"。最重的 A（真执行后端）+ I（真 git/PR）杠杆最大（Bun 旗舰案），作配对 epic 押后。

## Gap backlog（按推进顺序）

| 里程碑 | 主题 | 对比判定 | 现状一句话 | 杠杆 | 工作量 | 依赖 | Plan |
|---|---|---|---|---|---|---|---|
| **M1** | **F 跨叶归约** | CONFIRMED-GAP | 只有单叶 fold，无 vote/dedup/judge-panel 原语；跨叶归约全靠脚本手写 | 最高（6 用例） | 低–中 | G1 + G4 | ✅ 已落地 · [`01-f-cross-leaf-reduce.md`](01-f-cross-leaf-reduce.md) |
| **M2** | **B 早退/取消（race）** | CONFIRMED-GAP | parallel 严格 barrier、无 first-wins/在飞取消 | 高 | 中 | 核心调度器 | ✅ 已落地 · [`02-b-journaled-race.md`](02-b-journaled-race.md) |
| **M3.5** | **多并行 run 可观测性 + quota 接线** | NO-GAP（机制已全，残 ergonomics） | run/run_script 已后台并发、每 run 隔离 journal/budget/gate；缺聚合 `runs` 命令、`workflow_runs` 落定不刷新、quota 接线静默忽略 | 中（直接答需求 ②） | 低 | 现有 `BgRunManager`（无硬依赖，**可先于 M3**） | ✅ 已落地 · [`03-m3.5-run-observability.md`](03-m3.5-run-observability.md) |
| **M3** | **D 跨会话持久** | CONFIRMED-GAP（超集赢面：跨进程零成本 resume，CC 仅同会话） | 接缝在但只有内存 store/saver、host 用进程内 dict | 高（超集赢面） | 中 | journal/checkpointer 接缝 | ✅ 已落地 · [`04-m3-cross-session-persistence.md`](04-m3-cross-session-persistence.md) |
| **M4** | **C 运行中 HITL 签核** | CONFIRMED-GAP（底座现成） | 全无 interrupt/pause；LangGraph `interrupt` 未 import 暴露 | 中–高（超集赢面） | 中 | M3（持久化，✅） | ✅ 已落地 · [`05-m4-c-in-run-hitl.md`](05-m4-c-in-run-hitl.md) |
| **M5** | **A 循环内可执行验证** | CONFIRMED-GAP | worktree seeding 有，但 execute 是离线 no-op echo、无真子进程/exit-code gating | 最高（Bun 旗舰案） | 高 | G2 worktree | ✅ 已落地（引擎 #14 + demo 消费 #15） |
| **M6** | **I 真 git worktree + 分支/PR** | CONFIRMED-GAP（接缝预留） | 只有 InMemoryWorktreeProvider；无真 git/分支/PR | 高 | 高 | M5（真执行，✅ 已解锁） | ✅ 已落地 · [`06-m6-i-real-git-worktree.md`](06-m6-i-real-git-worktree.md) |
| **M7** | **H 拓扑序 fan-out + 深层命名嵌套** | CONFIRMED + PARTIAL | 无 DAG/偏序调度；`workflow()` 硬限 1 层 | 中 | 中–高 | 核心调度器 | ✅ 已落地 · [`07-m7-h-topo-fanout-nesting.md`](07-m7-h-topo-fanout-nesting.md) |

**后续里程碑（待写）：**
- **E 批处理人体工学** — batch-map helper + 自动 count/ETA 进度（扩 `ProgressSink`）+ 流式 admission（修 `parallel` 急切物化 N 千项）。原与 B 同里程碑，现拆为自己的后续里程碑：B（race）已先行落地，E 改动 `parallel`/`pipeline` admission 表面、与 race 解耦，独立推进更清晰。**待写。**

**轻量并入项（不单列里程碑）：**
- **G 测量停止循环** — 对比判定 PARTIAL（能力上≈NO-GAP，已可表达且 G3 文档化）。仅缺一等公民 helper，作为 M1 或 M7 的小型并入项（`ctx.loop_until(...)` 或强化 SKILL.md 范式），不值得独立里程碑。
- **M1.5 · 多阶段 / 并行-run 作者范式（doc-only）** — 调研（见下方"多阶段编排 / 多并行 run 调研收敛"）确认"代码驱动的动态 per-phase 扇出"与"单会话多并行 run"**今天均已支持**；缺的只是 SKILL.md 没把范式讲清。补 SKILL.md：① 多阶段脚本结构（await 上一 phase 结果入变量 → 分支 → 据此 build 下一 phase 的 `parallel`/`pipeline` work-list）；② scout-then-fan-out（先用一个廉价 leaf 探出 work-list，再扇出）；③ host 启多个并行 run 并对各自完成通知逐一反应；④ 作者陷阱（每个循环要硬上限 MAX、dedup 须比对**全部** seen 而非上一轮）。**无源码改动**，搭 M2/M3.5 顺风车。守 AGENTS.md 道/术线：范式（道）入 SKILL.md / tool description，绝不在 demo prompt 里教工具机制（术）。

**退役 / 暂搁置：**
- **J 多模态源摄入** — 主要是 leaf/deepagent 的工具面，不在编排引擎范围，暂搁置。

## 推进顺序

```
第一梯队（纯编排层、信号最密、接 v0.2.0 成果）
  M1  F 跨叶归约              ← 首刀；接 G1+G4，无新基建
   └─ M2  B 早退/取消（race）  ← 修核心原语缺陷（E 批处理人体工学已拆出为后续里程碑）
        └─ M3.5 多并行 run 可观测性 ← 轻量 fast-follow，无硬依赖，可先于 M3（+M1.5 doc-only 搭车）
第二梯队（超集赢面、底座已撑、暴露接缝）
        └─ M3  D 跨会话持久    ← 持久 store/saver + 持久 host 注册表
             └─ M4  C 运行中 HITL ← 暴露 LangGraph interrupt；承 M3 持久化
第三梯队（重基建、单案杠杆最大、配对 epic）
                  └─ M5  A 循环内可执行验证 ← 真执行后端 + 子进程 + exit-code gating
                       └─ M6  I 真 git/PR    ← 真 git WorktreeProvider + 分支/PR
第四梯队（引擎机制增强）
                            └─ M7  H 拓扑序调度 + 深层嵌套  ✅
```

**每条 gap 落地必须（per-gap 交付清单）：**
1. **完整 TDD**（Red→Green→Refactor），ruff + pyright strict 全绿；
2. **真模型 E2E 验收门** — 用一个社区真实用例 offline fake 证机制 + `LDW_DEMO_REAL_MODEL` 真跑证逼近度（见 memory `per-gap-real-e2e-acceptance`）；
3. **user-facing 集成示例** — 一个照抄即用的 host 接线示例（非孤立属性证明），优先扩展既有 canonical 示例（见 memory `per-gap-integration-example`）；
4. **跨模型评审** — Codex 跨模型评审一轮（v0.2.0 四 gap 每次都抓到至少一个 in-house 漏掉的 HIGH 缺陷，已验证 4 次）；
5. **同步 evergreen 设计文档** — `design_docs/{01,02}.md` + `uml/` + 本 roadmap 状态。

---

## 里程碑详述

### M1 · F — 跨叶归约（cross-leaf reduce）【首刀】

**目标：** 把"跨多个 leaf 输出的归并"从脚本手写 Python 提升为一等公民能力——投票（vote）、去重（dedup）、双盲双复核冲突调解（dual-reviewer reconcile）、裁判团聚合（judge-panel aggregate）。

**为何先做：** 信号最密（6 用例：#3 /deep-research claim 投票、#5 bug sweep 去重、#11 收敛、#14 裁判团、#15 PRISMA 双盲、#17 plan tournament）；最纯粹落在编排层；自然延伸 v0.2.0 的 G1（schema 握手）+ G4（单产物裁判）——现裁判只判**一个**产物，F 把 reduce 推广到**跨叶收集→分组→投票/调解**。无新基建。

**对比证据（现状）：** 只有单叶 `fold_result`（`_result.py:17-40`）/`fold_structured`（`_result.py:43-76`）；`Ctx` 无 reduce 方法（`_context.py:200-584`）；跨叶归约全靠脚本手写（`examples/07:184-199` 手写 claim 投票；SKILL.md 仅文档化范式 `:159-296`）。

**范围（待 impl plan 细化）：**
- 一组 reduce helper（库级函数或 `ctx` 方法，接缝待定）：`vote` / `dedup` / `reconcile`（双复核冲突）/ `judge_panel`（N 裁判聚合）。
- 与 G1 schema 协同：归约输入是 schema-validated 对象列表，输出仍是结构化结果。
- SKILL.md 增补"跨叶归约"范式段（含 refute-by-default 投票、等价 claim 分组、drop-unverified）。
- **轻量并入候选：** G 测量停止循环 helper（若接缝自然）。

**验收门：** 以 #3 `/deep-research` 的 claim 投票 + drop-unverified 为真模型 E2E 验收用例；扩展 `examples/07` 把手写投票换成新 helper 作集成示例。

**依赖：** G1（schema）、G4（judge）——均已落地。

### M2 · B — 早退 / 取消（race）

**目标：** 给核心调度器补早退 / 取消原语——`ctx.race(candidates, *, win, win_tag="")` best-of-N 并发跑，第一个满足 `win` 谓词的结果胜出，在飞 loser 立即**取消**；race 决策按 content-hash **journal**（race-key），resume 复现 winner 且**不再派发任何 candidate**，保持确定性可恢复 replay。

**对比证据（现状）：** `parallel` 严格 all-settle barrier（`_context.py:524-526`）；`pipeline` 阶段间无 barrier 但仍终端 gather、无增量 yield/早退（`_pipeline.py:163-183`）；无在飞兄弟取消，只有整 run cancel（`_background.py:451-476`）。

**范围：** `race`/`first_completed` 语义（first-to-satisfy-win wins，升序 index tie-break）；在飞 loser 取消传播 + gate slot 释放；race 决策 content-hash journal（race-key，namespaced + win_tag-folded）+ replay 短路。

**非目标（明确划界）：**
- **真流式输出**（边完成边增量产出）——与确定性 replay 冲突；race 已替"早出延迟"动机买单，流式不在 B 范围。
- **混合 schema race**——同一 race 内 candidate 必须同质（全 schema-less 或全绑同一 schema），否则 winner 类型有歧义；混合 schema 显式拒绝。
- **E 批处理人体工学**（batch-map helper + count/ETA 进度 + 流式 admission）已从本里程碑**拆出为自己的后续里程碑**（见上方"后续里程碑"），与 race 解耦独立推进。

**验收门：** #13 AI-SRE 多假设并行 + 早退（一假设确认即取消其余）为真模型 E2E；`examples/13` AI-SRE race demo 作集成示例。

**依赖：** 核心调度器（`_context.py`/`_journal.py`/`_observability.py`）。

### M3.5 · 多并行 run 可观测性 + quota 接线（需求 ② 残项）

**目标：** 给"单会话多并行 run"补 host 侧聚合可观测面——让 host 不必逐个 `run_id` 轮询，就能看到自己所有在飞 / 已完成的 run。

**调研定位（NO-GAP 的残项）：** 并行机制**本身已端到端跑通**（`run`/`run_script` 后台启动即返回、可连发并发 run、每 run 独立 journal/budget/gate/determinism-guard、按 `run_id` 取消/恢复互不干扰，`BgRunManager.max_concurrent_runs` quota + `BgRunQuotaExceededError`，默认无界）——甚至略优于 CC（CC 用单一全局 token 池，我们 per-run 隔离）。缺口只在 ergonomics：① 无聚合 `runs`/`list` 命令（工具面只有 `run`/`run_script`/`status`/`resume`/`cancel`，`status` 单 run；`active_run_count()` 仅内部用）；② `workflow_runs` state 通道 append-only、落定时不把 RUNNING 改写为终态（`tool.py:207-209`）；③ `create_workflow_tool` 只暴露 `max_concurrency`（per-run 叶子并发）+ `budget`，**未**暴露 `max_concurrent_runs`（只能在 `BgRunManager` 或 middleware 默认-manager 路径设）。

**范围：** ① 工具新增 `runs`（或 `list`）命令，枚举本 host thread 全部 run + 实时状态（基于 `BgRunManager.active_run_count()` + slot 枚举）；② `workflow_runs` 落定即把记录从 RUNNING 改写为终态；③ `create_workflow_tool` 签名补 `max_concurrent_runs` 转发；④ 可选：合并 `<workflow_notification>` 带 per-run 状态，免得 host 再逐个 `status`。**明确非范围：** 并行机制本身（已工作）、跨进程持久化（= M3/D）、全局跨-run 预算池（见下方开放决策）。

**验收门：** host（opus 级，遵 memory `workflow-review-agent-model`）从道层 prompt 启 2–3 个并行 workflow，再查聚合 `runs` 视图并据各结果反应（`LDW_DEMO_REAL_MODEL` + OpenRouter，遵 memory `per-gap-real-e2e-acceptance`）；扩展某 canonical 示例展示 host 启 N 并行 run + 列举/反应（遵 memory `per-gap-integration-example`）。

**依赖：** 无硬依赖（纯增量面叠在现有 `BgRunManager` + 工具上）。与 M3（D 持久化）天然配对（都碰 run 注册表），但**可独立先行、甚至先于 M3**。与下方"M1 实测发现"的 backlog 项 **K**（host 无法按名发现已注册工作流）相邻——K 是发现"已注册 workflow 名"，M3.5 是发现"在飞 run"，可一并收。

### M3 · D — 跨会话 / 多日持久（超集）【✅ 已落地】

**目标：** 让 resume/replay **跨进程/跨会话**存活——超越 CC（CC 仅同会话）。

**落地形态（as-built）：** `WorkflowRunStore` 协议 + `RunSpec`（携规范 `journal_run_id` 谱系）+ `InMemoryRunStore`（默认、零依赖）+ `SqliteWorkflowStore`（`[sqlite]` extra：一个统一 sqlite db 文件、run_id 命名空间化四表 + 第二连接上的持久 `AsyncSqliteSaver`）。**零成本重放由持久 journal 交付**（checkpointer 是 durable add-on，resume 侧 `checkpointer=None` 亦可证）。两连接（autocommit store + explicit-commit saver，皆 WAL）一个 db 文件；`AsyncSqliteSaver` 构造期绑 event loop（宿主单一持久 loop 内构造 + 复用同一实例）；per-run 规范 id 同 key journal 谱系与 checkpoint thread（host thread 仅 key manager slot）；save-before-start + quota 拒入回滚；schema-version guard fail-loud。机制详见 [`design_docs/01-engine-mechanism.md`](../01-engine-mechanism.md) §13b，接线见 [`design_docs/02-architecture.md`](../02-architecture.md) §10，时序见 [`design_docs/uml/03-sequence.md`](../uml/03-sequence.md) D 图。真模型 + 真子进程 E2E（`examples/15`）钉死头条；离线子进程测试钉死零派发不变量。

**对比证据（现状）：** 接缝齐备但**空**：`JournalStore` Protocol（`_journal.py:75-109`）可注入，checkpointer 参数可接持久 saver 但默认 `InMemorySaver`（`_engine.py:117`）；只有 `InMemoryJournalStore`（`_journal.py:112`）；host 用进程内 dict（`tool.py:143,147,225,265,319`）。"可能已超 CC"假设**已证伪**。

**范围：** 一个持久 `JournalStore` 实现（sqlite 起步）；wire 一个持久 LangGraph checkpointer（+ 可选依赖 extra）；host 的 `journals`/`run_specs` 注册表持久化，使 `resume` 能跨进程找回 run。

**验收门：** 进程退出→重启→`resume` 续跑同一 workflow 不丢进度（真模型，跨进程）；集成示例展示持久 store 接线。

**依赖：** journal/checkpointer 接缝（已预留）。

### M4 · C — 运行中 HITL 签核（超集）

**目标：** 提供运行中途的**人工签核 gate**——脚本可在阶段间暂停、等人工批准/输入再续，超越 CC（CC 运行中不接受输入）。

**对比证据（现状）：** 全无 interrupt/pause/approval（`_context.py:131-585` 单向）；LangGraph 原生 `interrupt` 未 import/暴露（`_engine.py:20`）。底座**已撑**——checkpointer interrupt-resume 管线现成（`_engine.py:117,232`）。

**范围：** 在 `Ctx` 暴露 `checkpoint()`/`approve()` → 调 LangGraph `interrupt`；`resume` 把人工值喂回；host 工具命令支持"待签核"状态。是"暴露接缝"而非"从零造"。

**验收门：** #7 OpenHands 式"合并进集成分支前人审"或 #6 安全审计"风险分级后签核再出报告"为 E2E；集成示例展示暂停→批准→续跑。

**依赖：** M3（持久化——HITL 暂停天然需要跨会话存活，**✅ 已落地**）+ LangGraph interrupt。

### M5 · A — 循环内可执行验证（real execution backend）【✅ 已落地】

**目标：** 让 worktree leaf 能在循环内**真实 build + 跑测试**并把 exit-code/输出喂回脚本分支（in-loop executable verifier），而非现在的 LLM 评审循环。

**落地形态（as-built）：** 引擎侧 `LocalSubprocessSandbox`（完整 `SandboxBackendProtocol`）——真子进程执行、POSIX rlimits 经 `preexec_fn`、超时→进程组 kill（SIGTERM→宽限→SIGKILL，exit 124）、有界输出 drain、exit-code gating、可插拔 `SandboxManager(sandbox_factory=...)`、`ExecPolicy`/`ExecGate` admission（危险操作 opt-in，**非** sandbox）；外加命令可观测 `on_command`/`CommandEvent` sink（keyword-only、默认 no-op、miss-only、在真子进程执行边界触发、关联 leaf span）。demo 消费切片：`fix_loop` 预设——**脚本拥有循环、跨轮状态走脚本变量**（无持久 workspace、不依赖 M6），分支据**真 exit code**（非模型布尔）；`TerminalCard` Gen-UI 卡（running/passed/failed 原地翻面）；`code_fixer` 执行 roster；"Make it pass" 场景。引擎 PR #14（`665d030`）+ demo 消费 PR #15（`6bfd42d`），机制详见 [`design_docs/01-engine-mechanism.md`](../01-engine-mechanism.md)、接线见 [`design_docs/02-architecture.md`](../02-architecture.md)。

**对比证据（落地前现状）：** worktree seeding 有（`_sandbox.py:455-476`）；但 `execute` 是离线 no-op echo、永远 exit_code=0、不起 shell（`_sandbox.py:175-189`，`_GuardedBackend.execute` 仅委派 `:905-907`）；`src/` 无任何真 subprocess（grep clean）；无 exit-code gating、无慢命令纪律钩子。

**范围（重）：** 真执行后端（真 subprocess/shell，带超时与资源界）；脚本可读 `ExecuteResponse.exit_code/output` 分支；吞吐纪律钩子（禁慢命令/fast-path，对应 Bun"禁 git/cargo"）。安全：执行必须在 AST-gated/sandboxed 路径内（遵 AGENTS.md 安全红线）。

**验收门：** #1 Bun 型 fix-loop（改→build→test→据结果再改，直到绿）缩比真模型 E2E；集成示例展示 worktree 内真跑测试 gating。

**依赖：** G2 worktree（已落地）。

### M6 · I — 真 git worktree + 分支 / PR（配对 A 成 Bun 级 epic）

**目标：** 提供真 git 隔离——每 leaf 一个真 `git worktree`/分支，完成后**开 PR 进集成分支**（非 main），支持合并冲突处理。

**对比证据（现状）：** 只有 `InMemoryWorktreeProvider`（dict copy + dict-diff，`_worktree.py:61-85`）；无真 git/分支/PR/冲突逻辑（grep clean）。但 `WorktreeProvider` Protocol（`_worktree.py:22-58`）+ `SandboxManager(worktree_provider=...)`（`_sandbox.py:439`）是**为真 git 后端预留的、文档化的插口**（`_worktree.py:8-13`）。

**范围（重）：** 一个真 git `WorktreeProvider`（`git worktree add` seed / `git diff` collect）；branch-per-leaf；PR-into-integration-branch（用仓库 `github-pr` 式机制）；合并冲突交由 leaf 解决的循环。

**验收门：** #7 OpenHands 式 refactor swarm（分支隔离→verifier→PR 进集成分支）缩比 E2E；集成示例展示真 git worktree 接线。

**依赖：** M5（真执行——分支里要能 build/test，**✅ 已落地，本里程碑已解锁**）。M5 的 `fix_loop` 已暴露对跨轮 worktree 隔离的诉求（当时以脚本变量兜底，正式隔离顺延至本里程碑）。

### M7 · H — 拓扑序 fan-out + 深层命名嵌套

**目标：** ① 依赖序（拓扑/偏序）fan-out——被调者先于调用者处理，非一把梭 barrier；② 解除 `workflow()` 命名嵌套的 1 层硬限（按需放开到 N 层）。

**对比证据（现状）：** `parallel`=flat barrier（`_context.py:461-542`）、`pipeline`=线性链（`_pipeline.py:107,147`），无任何 DAG/偏序/predecessor 模型（grep clean）；`workflow()` 硬限 1 层（`_context.py:252-257`，`WorkflowNestingError`，测试 `tests/unit/test_nesting.py:68-85`），但 `parallel/pipeline/agent` primitive 可任意深嵌（`_FANOUT_DEPTH`，`_context.py:100-113`）。

**范围：** 一个 DAG/偏序调度原语（给定依赖边，拓扑序 fan-out，仍受并发闸约束）；放开命名工作流嵌套层数（评估确定性 guard 与资源界影响）。

**调研澄清（需求 ① 的引擎残项归入此处）：** 需求 ①"每个 phase 根据上一 phase 动态编排子 workflow / 多批次扇出"经调研拆为四块——其中**代码驱动的动态 per-phase 扇出（最主力一块）今天已支持**：脚本是普通确定性 `async def orchestrate(ctx, args)`，上一 phase 结果存脚本变量，`parallel`/`pipeline` 吃运行期任意长度 work-list，scout-then-fan-out 即普通代码，且全程 content-hash journal 可 resume——这块只需 SKILL.md 文档（= M1.5），**不需引擎改动**。真正的引擎残项恰好就是本里程碑既有范围的两条：① 放开 `ctx.workflow()` 命名嵌套 1 层硬限到 N 层（PARTIAL-gap，与 CC 同限，超集动作）；② 依赖 DAG / 拓扑序扇出（CONFIRMED-gap）。建议把 M7 验收用例 #10 扩充为：同时演示"后一 phase 据上一 phase 实际结果运行期扇出"+"命名子 workflow 嵌套 >1 层"，端到端收口需求 ①。**架构有意排除：** 运行中脚本现编一个全新（未注册）子 workflow——`run_script` 是 host-facing、AST gate + 受限 builtins 封死脚本侧到 codegen 的路径，刻意把"现编新编排逻辑"留给 host LLM，使可信执行核保持 source-unaware。

**验收门：** #10 文档生成（package→module→symbol 拓扑序 + >1 层嵌套）为 E2E；集成示例展示依赖序 fan-out。

**依赖：** 核心调度器。

---

## M1 实测发现（新候选 backlog，非 M1 范围）

M1 的真模型 E2E 过程中浮现两条值得后续处理的信号：

- **K · host 无法按名发现已注册工作流。** 一个有能力的 host（opus）在道层 prompt + skill + tool description 下会**自驱**工具，但因 tool description 不枚举"可用的已注册工作流名"，它选择 `run_script` **自拟**一个等效工作流，而非跑注册的 `deep_research`。改进点：tool description / `help` 暴露已注册工作流清单（让 host 能 `run` 现成的，而非总是重新 author）。归类近 M7（人体工学）或独立小项。
- **host 模型能力门槛。** 道层 prompt（无机制 coaching）下，弱模型（haiku）驱动不动多步工具流；需够强的 host（opus 可）。这是模型能力问题、非 skill/tool 缺口（opus 证明接缝自足），记作 demo 运行约定：真 host demo 用够强的模型。

## 多阶段编排 / 多并行 run 调研收敛（2026-06-03）

用户提出两个能力诉求，专项调研（CC 行为契约 vs 本 port 源码，read-only）结论如下——**两者都不是缺核心能力**：

- **需求 ① 多阶段 + 每 phase 据上一 phase 动态编排子 workflow / 多批次扇出 = PARTIAL。** 最主力的"代码驱动动态 per-phase 扇出"今天已支持（控制流反转：脚本拥有循环/分支、结果存脚本变量、`parallel`/`pipeline` 吃运行期 work-list、可 resume）→ 只需文档（**M1.5**）。真正引擎残项＝① `ctx.workflow()` 1 层硬限、② 依赖 DAG 扇出，二者**恰为 M7 既有范围**（已在 M7 详述澄清，M7 留队尾不提前）。"运行中现编新子 workflow"＝架构有意排除（`run_script` host-only，AST gate 封死脚本侧）。这与 CC 一致：CC 的"多阶段"也解为"主循环串起多个 workflow，每个读结果再定下一阶段形态"，无一等公民多阶段构造。注：本 port 的 **M4/C（运行中 HITL）实际超越 CC**——给"运行中拍板下一阶段形态"提供人审 gate，而 CC 运行中无暂停。
- **需求 ② 一个 session 多 workflow 并行 = NO-GAP。** 端到端已工作（后台启动即返回、并发 run、per-run 隔离、按 `run_id` 取消/恢复），略优于 CC（CC 单全局 token 池 vs 我们 per-run 隔离）。残项纯 ergonomics → 新增轻量 **M3.5**（聚合 `runs` 命令 + 落定刷新 + `max_concurrent_runs` 接线）；跨进程兜底仍由已有 **M3/D** 负责。

**用户决策（2026-06-03）：** 需求 ① 走「脚本自决 + 补文档」（M1.5 doc-only；M7 引擎残项留队尾不提前）；需求 ② 走「单会话交互式扇出」（轻量 M3.5，可先于 M3；不提前做跨进程过夜持久化，那留给 M3）。

**开放决策（暂不 scope）：全局跨-run 预算池。** CC 跨主循环 + 所有 workflow 共享**单一** session 级 token 池（真正的协调/成本点）；本 port 给每个并发 run **独立** Budget，故 N 个并行 run 可各吃满 N 份上限、合计放 N×leaf-limit 个叶子在飞，只有 run-数 quota 约束聚合扇出。是否对齐 CC 加一个跨-run 全局 token/agent 上限（`BgRunManager` 上的新机制），还是保持 per-run 隔离作有意超集——记为开放设计决策，待真实场景驱动再定（CC 共享池语义为社区单源 pre-GA，对齐前需复核）。

## 状态

- **对比阶段**：✅ 完成。10 主题逐一带 `file:line` 证据 + 置信度（证据稿 `docs/plans/2026-06-03-v0.3.0-cc-vs-port-comparison.md`）。战略定调=允许超集。
- **M1（F 跨叶归约）**：✅ 已落地。`_reduce` 四个纯函数 `survives`/`dedup`/`reconcile`/`corroborate`(+ `ReviewItem`/`Reconciled`/`Consensus`)经包根导出 + `run_script` 命名空间注入;SKILL.md 增补 corroborate/reconcile 范式;`examples/07` 换用 `survives`/`dedup`、`examples/12` 新增双盲复核 demo。Plan = [`01-f-cross-leaf-reduce.md`](01-f-cross-leaf-reduce.md)。
- **M2（B race）**：✅ 已落地。`ctx.race` best-of-N 早退/取消原语 + `_race_types`（`RaceCandidate`/`RaceResult`）+ `race_key`（content-hash journal，namespaced + win_tag-folded）+ `SpanKind.RACE`；两值类型经包根导出 + `run_script` 命名空间注入；SKILL.md 增补 race quality / parallel-vs-race / win_tag footgun 范式；`examples/13` AI-SRE 多假设 race demo。真流式与混合 schema race 为明确非目标；**E（批处理人体工学）已拆出为自己的后续里程碑（待写）。** Codex 跨模型评审驱动两处修复：replay 改记 winner 的 leaf-key（杜绝与后续相同 `agent()` 调用的预算双计）、teardown 的 depth/任务创建移入 `try`（对齐 `parallel`/`pipeline`），全门 347 passed。Plan = [`02-b-journaled-race.md`](02-b-journaled-race.md)。
- **M3.5（多并行 run 可观测性）**：✅ 已落地（fast-follow，先于 M3）。聚合 `runs` 命令（`BgRunManager.list_runs` → `RunSnapshot` 只读快照，工具 join workflow label）+ `workflow_runs` 落定刷新（`merge_workflow_runs` reducer 按 `run_id` upsert，`abefore_model` 落定改写终态）+ quota 接线去歧义（**偏离调研字面建议**:不往 `create_workflow_tool` 加 `max_concurrent_runs`——它不构造 manager,加了要么被忽略要么双源;改为 `create_workflow_middleware` 在显式 `manager` + `max_concurrent_runs` 同传时抛 `ValueError`,quota 归 `BgRunManager`)。`examples/14` host 多-run 聚合视图 demo + 集成测试。Plan = [`03-m3.5-run-observability.md`](03-m3.5-run-observability.md)。
- **M3（D 跨会话持久）**：✅ 已落地（超集 CC）。`WorkflowRunStore` 协议 + `RunSpec`（携规范 `journal_run_id` 谱系）+ `InMemoryRunStore`（默认、零依赖）+ `SqliteWorkflowStore`（`[sqlite]` extra：一个统一 sqlite db 文件、run_id 命名空间化四表 `run_specs`/`journal_records`/`journal_sequence`/`journal_progress`，外加第二条连接上的持久 `AsyncSqliteSaver` checkpointer），包根 lazy 导出、base 安装保持零依赖。头条——**全新进程指向同一 db 文件按 `run_id` resume、完成叶从持久 journal 零模型成本重放**——已交付且真模型 + 真子进程 E2E 钉死（journal 交付零成本、checkpointer 是 durable add-on）。评审驱动的硬化：journal-lineage 规范 id（`journal_run_id` 同 key journal 谱系与 checkpoint thread，host thread 仅 key manager slot，distinct run 不撞、resume 重接原 thread）、per-run checkpoint thread、save-before-start（spec 先于 `manager.start` 持久、quota 拒入则 `delete_spec` 回滚）、schema-version guard（`PRAGMA user_version` fail-loud on incompatible）、strict-msgpack 诚实（叶子状态保持 msgpack-friendly）。Plan = [`04-m3-cross-session-persistence.md`](04-m3-cross-session-persistence.md)。
- **M5（A 循环内可执行验证）**：✅ 已落地（双轨：引擎 #14 + demo 消费 #15）。引擎 `LocalSubprocessSandbox`（真子进程 + rlimits + 超时进程组 kill + exit-code gating + `SandboxManager(sandbox_factory=...)` + `ExecPolicy`/`ExecGate` admission）+ 命令可观测 `on_command`/`CommandEvent`（keyword-only、默认 no-op、miss-only、真执行边界触发、关联 leaf span）。demo 消费 `fix_loop`（脚本拥有循环、跨轮状态走脚本变量、分支据真 exit code）+ `TerminalCard` + `code_fixer` roster + "Make it pass" 场景。五层验收过门（gate → 双轨评审 Codex 抓两处 in-house 漏掉的 HIGH → /cso → 真模型 host-driven E2E 抓两处仅真跑暴露的 blocker → 用户实测 leaf-quarantine UI 泄漏修复）。无独立 plan 文件（经 demo-app 切片落地）。
- **M1.5（多阶段 / 并行-run 作者范式，doc-only）**：✅ 已落地（搭 M7 顺风车）。SKILL.md 增补多阶段脚本结构（await 上一 phase 结果存变量 → 据此 build 下一 phase work-list）/ scout-then-fan-out（先廉价 leaf 探 work-list 再扇出）/ host 并行多 run 各自反应范式 / 作者陷阱（硬 max_iters、dedup 比对全部 accumulated）。无源码改动。
- **M4（C 运行中 HITL 签核）**：✅ 已落地（超集 CC——运行中接受人工输入）。`ctx.checkpoint` **journal 驱动的签核门**(C8 spike 否决 LangGraph 原生 `interrupt`:其 index-based `@task` 缓存与 content-hash journal 短路在 resume 时错位)——脚本暂停抛 `WorkflowSignoffRequired`、host 经 `approve` 命令注入决策续跑、门前叶零成本重放。新态 `BgStatus.AWAITING_SIGNOFF` + `BgRunManager.approve`(同步翻 RUNNING 杜绝双批);双轨:引擎 + demo 消费(`sign_off` 预设 + `SignoffGate` 卡片 + 内联 park/resume via `_ResumeLane`)。双轨评审(Codex `gpt-5.5` + in-house,7 findings 全修:approve 竞争、跨进程授权缺口、门键漂移、park 配额泄漏、AST gate 禁 checkpoint、进度重叙、JSON 决策)。真模型 host-driven 两轮 E2E 实跑过门(真 host 自主 route→park→经 `signoff_decision` 答门→proceed)。Plan = [`05-m4-c-in-run-hitl.md`](05-m4-c-in-run-hitl.md)。
- **M6（I 真 git worktree + 分支/PR）**：✅ 已落地（超集 CC,配对 M5 成 Bun 级 epic）。`LocalSubprocessSandbox(root=, on_close=)` 让叶沙箱根植真 worktree;`GitWorktreeProvider`（`git worktree add -b leaf/<id>`、真 `git diff` 作权威 collect、幂等+异常安全 teardown、`cleanup_all` host 契约）;`SandboxManager(git_worktree_provider=)` 两契约各一分支、离线默认零改动;`PullRequestProvider` 协议 + `LocalPullRequestProvider`（离线幂等）。三处经 Codex+in-house 跨模型评审拍板的承重决策:① PR/integration 物化移出确定性 replay 作幂等 host finalization;② worktree 叶权威变更集 = leaf task 内真 `git diff`（经 `model_validate` 覆写 schema `files`、三种误用 fold 期 fail-loud,非模型自报）;③ 整合用 merge 叶内一次性 scratch-repo 真 `git merge`。R8 阻塞 git（add+teardown）thread-offload 出 slot 锁。双轨:引擎 + demo 消费（`refactor_swarm` 预设 + `PullRequestCard` + 真 git fixer/conflict resolver roster）。双轨评审（Codex + in-house）抓 1 BLOCKER+6 HIGH/MED 全修(PR 未 journaled、collect 非权威、merge-file 保真、teardown 未 offload、schema-less 丢 changeset、`model_copy` 跳校验 resume 崩、删除静默丢)。Plan = [`06-m6-i-real-git-worktree.md`](06-m6-i-real-git-worktree.md)。
- **M7（H 拓扑序 fan-out + 深层命名嵌套）**：✅ 已落地（含 Codex 跨模型评审驱动的评审后硬化）。四项交付：① **`ctx.dag` 依赖序（拓扑序）fan-out**——`DagNode(id, deps, run)` 值类型 + `_dag.run_dag` Kahn 入度调度器（急切结构校验、fast-branch-no-barrier 精神、`failed: set` 独立跟踪、传递 skip、控制流信号 drain 后穿透重抛、调度器不持 gate slot 故 dag 嵌套不死锁、`SpanKind.DAG`）；`DagNode` / `WorkflowDagError` / `WorkflowCycleError` / `WorkflowNestingError` 从包根导出；`DagNode` 经 `_codegen._SCRIPT_DAG_API` 注入 `run_script` 命名空间；`ctx.dag`/`ctx.loop_until` 是 `Ctx` 方法，不在 AST gate 禁用名单上。② **`ctx.workflow()` 深度上限放开为可配置**——`DEFAULT_MAX_WORKFLOW_DEPTH=8`，`max_workflow_depth` 经 `run_workflow → Ctx` 线传；`_WORKFLOW_DEPTH` + `_WORKFLOW_NAME_STACK` 两 contextvar 各自独立 `finally` reset；循环检测（重入即抛 `WorkflowCycleError`）先于深度检测（`WorkflowNestingError`），前者更精确；`WorkflowDagError` / `WorkflowCycleError` / `WorkflowNestingError` 三者**全部**入 `WORKFLOW_CONTROL_FLOW_SIGNALS`——深度越界是结构性/跑飞递归错误，fan-out 帧内必须 fail-loud（初版有意将 `WorkflowNestingError` 排除；Codex 跨模型评审指出 BLOCKER：fan-out 帧内深度越界会被 mask 为 `None` 空洞——已修正）。③ **`ctx.loop_until` 顺序循环 helper（rider G）**——`body(iter_index, accumulated)` 每轮追加，`done(accumulated)` 覆盖全量判停，`max_iters` 强制硬上限，到顶 graceful log + 返回（不 raise），顺序深度-0 故叶调用记入序列 guard。④ **SKILL.md 作者范式（rider M1.5）**——多阶段脚本结构、scout-then-fan-out、host 并行多 run、作者陷阱，无源码改动。Codex 评审驱动两项硬化：BLOCKER `WorkflowNestingError` 入信号元组（fan-out 帧内深度越界不再 mask）+ HIGH `run_dag` teardown 追加 await（外部 cancel 时在飞节点任务已 await，不泄漏 pending task）。**真模型 #10 验收**（开发期 gate，遵 `examples/AGENTS.md` §4，非常驻真路径）：真 OpenRouter sonnet-4.6 deepagent leaves 跑 package→module→symbol 拓扑 doc-gen + symbol 经 2 层嵌套子 workflow（document_module→document_symbol），`SpanKind.DAG` span node_count=4 / surviving_count=4 全存活、LangSmith 计费追踪开启——头条路径在真模型上端到端走通（非 fallback）。全门终态 539 passed / 2 skipped、pyright 0、ruff 净。关键决策见 [01 §2c/2d/2e](../01-engine-mechanism.md) D-M7a~D-M7e。

> **执行序列：** M1 F ✅ → M2 B(race) ✅ →〔M3.5 多并行 run 可观测性 ✅ + M1.5 doc-only ✅，轻量 fast-follow，可先于 M3〕→ M3 D ✅ → M5 A ✅（提前于 M4 落地，重基建先行）→ M4 C ✅ → M6 I ✅（真 git worktree + 分支/PR，配对 M5 成 Bun 级 epic）→ **M7 H ✅**（dag 拓扑序 fan-out + workflow 深度/循环 + loop_until + SKILL.md 范式；E 批处理人体工学已从 M2 拆出，作后续里程碑待写）。F 首刀（接 G1+G4，纯编排层最干净）；B 紧随修核心原语；M3.5/M1.5 收口需求②并点亮需求①已有能力；D 已落地（跨进程零成本 resume，超集 CC）；C 承 M3 持久化走超集；A/I 配对成重基建 epic；H 收尾引擎机制增强（吸收需求①的命名嵌套 + DAG 残项）✅ 全数落地。

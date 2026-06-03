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
| **M3** | **D 跨会话持久** | CONFIRMED-GAP（与 CC 持平，未超越） | 接缝在但只有内存 store/saver、host 用进程内 dict | 高（超集赢面） | 中 | journal/checkpointer 接缝 | 待写 |
| **M4** | **C 运行中 HITL 签核** | CONFIRMED-GAP（底座现成） | 全无 interrupt/pause；LangGraph `interrupt` 未 import 暴露 | 中–高（超集赢面） | 中 | M3（持久化）+ LangGraph interrupt | 待写 |
| **M5** | **A 循环内可执行验证** | CONFIRMED-GAP | worktree seeding 有，但 execute 是离线 no-op echo、无真子进程/exit-code gating | 最高（Bun 旗舰案） | 高 | G2 worktree | 待写 |
| **M6** | **I 真 git worktree + 分支/PR** | CONFIRMED-GAP（接缝预留） | 只有 InMemoryWorktreeProvider；无真 git/分支/PR | 高 | 高 | M5（真执行） | 待写 |
| **M7** | **H 拓扑序 fan-out + 深层命名嵌套** | CONFIRMED + PARTIAL | 无 DAG/偏序调度；`workflow()` 硬限 1 层 | 中 | 中–高 | 核心调度器 | 待写 |

**后续里程碑（待写）：**
- **E 批处理人体工学** — batch-map helper + 自动 count/ETA 进度（扩 `ProgressSink`）+ 流式 admission（修 `parallel` 急切物化 N 千项）。原与 B 同里程碑，现拆为自己的后续里程碑：B（race）已先行落地，E 改动 `parallel`/`pipeline` admission 表面、与 race 解耦，独立推进更清晰。**待写。**

**轻量并入项（不单列里程碑）：**
- **G 测量停止循环** — 对比判定 PARTIAL（能力上≈NO-GAP，已可表达且 G3 文档化）。仅缺一等公民 helper，作为 M1 或 M7 的小型并入项（`ctx.loop_until(...)` 或强化 SKILL.md 范式），不值得独立里程碑。

**退役 / 暂搁置：**
- **J 多模态源摄入** — 主要是 leaf/deepagent 的工具面，不在编排引擎范围，暂搁置。

## 推进顺序

```
第一梯队（纯编排层、信号最密、接 v0.2.0 成果）
  M1  F 跨叶归约              ← 首刀；接 G1+G4，无新基建
   └─ M2  B 早退/取消（race）  ← 修核心原语缺陷（E 批处理人体工学已拆出为后续里程碑）
第二梯队（超集赢面、底座已撑、暴露接缝）
        └─ M3  D 跨会话持久    ← 持久 store/saver + 持久 host 注册表
             └─ M4  C 运行中 HITL ← 暴露 LangGraph interrupt；承 M3 持久化
第三梯队（重基建、单案杠杆最大、配对 epic）
                  └─ M5  A 循环内可执行验证 ← 真执行后端 + 子进程 + exit-code gating
                       └─ M6  I 真 git/PR    ← 真 git WorktreeProvider + 分支/PR
第四梯队（引擎机制增强）
                            └─ M7  H 拓扑序调度 + 深层嵌套
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

### M3 · D — 跨会话 / 多日持久（超集）

**目标：** 让 resume/replay **跨进程/跨会话**存活——超越 CC（CC 仅同会话）。

**对比证据（现状）：** 接缝齐备但**空**：`JournalStore` Protocol（`_journal.py:75-109`）可注入，checkpointer 参数可接持久 saver 但默认 `InMemorySaver`（`_engine.py:117`）；只有 `InMemoryJournalStore`（`_journal.py:112`）；host 用进程内 dict（`tool.py:143,147,225,265,319`）。"可能已超 CC"假设**已证伪**。

**范围：** 一个持久 `JournalStore` 实现（sqlite 起步）；wire 一个持久 LangGraph checkpointer（+ 可选依赖 extra）；host 的 `journals`/`run_specs` 注册表持久化，使 `resume` 能跨进程找回 run。

**验收门：** 进程退出→重启→`resume` 续跑同一 workflow 不丢进度（真模型，跨进程）；集成示例展示持久 store 接线。

**依赖：** journal/checkpointer 接缝（已预留）。

### M4 · C — 运行中 HITL 签核（超集）

**目标：** 提供运行中途的**人工签核 gate**——脚本可在阶段间暂停、等人工批准/输入再续，超越 CC（CC 运行中不接受输入）。

**对比证据（现状）：** 全无 interrupt/pause/approval（`_context.py:131-585` 单向）；LangGraph 原生 `interrupt` 未 import/暴露（`_engine.py:20`）。底座**已撑**——checkpointer interrupt-resume 管线现成（`_engine.py:117,232`）。

**范围：** 在 `Ctx` 暴露 `checkpoint()`/`approve()` → 调 LangGraph `interrupt`；`resume` 把人工值喂回；host 工具命令支持"待签核"状态。是"暴露接缝"而非"从零造"。

**验收门：** #7 OpenHands 式"合并进集成分支前人审"或 #6 安全审计"风险分级后签核再出报告"为 E2E；集成示例展示暂停→批准→续跑。

**依赖：** M3（持久化——HITL 暂停天然需要跨会话存活）+ LangGraph interrupt。

### M5 · A — 循环内可执行验证（real execution backend）

**目标：** 让 worktree leaf 能在循环内**真实 build + 跑测试**并把 exit-code/输出喂回脚本分支（in-loop executable verifier），而非现在的 LLM 评审循环。

**对比证据（现状）：** worktree seeding 有（`_sandbox.py:455-476`）；但 `execute` 是离线 no-op echo、永远 exit_code=0、不起 shell（`_sandbox.py:175-189`，`_GuardedBackend.execute` 仅委派 `:905-907`）；`src/` 无任何真 subprocess（grep clean）；无 exit-code gating、无慢命令纪律钩子。

**范围（重）：** 真执行后端（真 subprocess/shell，带超时与资源界）；脚本可读 `ExecuteResponse.exit_code/output` 分支；吞吐纪律钩子（禁慢命令/fast-path，对应 Bun"禁 git/cargo"）。安全：执行必须在 AST-gated/sandboxed 路径内（遵 AGENTS.md 安全红线）。

**验收门：** #1 Bun 型 fix-loop（改→build→test→据结果再改，直到绿）缩比真模型 E2E；集成示例展示 worktree 内真跑测试 gating。

**依赖：** G2 worktree（已落地）。

### M6 · I — 真 git worktree + 分支 / PR（配对 A 成 Bun 级 epic）

**目标：** 提供真 git 隔离——每 leaf 一个真 `git worktree`/分支，完成后**开 PR 进集成分支**（非 main），支持合并冲突处理。

**对比证据（现状）：** 只有 `InMemoryWorktreeProvider`（dict copy + dict-diff，`_worktree.py:61-85`）；无真 git/分支/PR/冲突逻辑（grep clean）。但 `WorktreeProvider` Protocol（`_worktree.py:22-58`）+ `SandboxManager(worktree_provider=...)`（`_sandbox.py:439`）是**为真 git 后端预留的、文档化的插口**（`_worktree.py:8-13`）。

**范围（重）：** 一个真 git `WorktreeProvider`（`git worktree add` seed / `git diff` collect）；branch-per-leaf；PR-into-integration-branch（用仓库 `github-pr` 式机制）；合并冲突交由 leaf 解决的循环。

**验收门：** #7 OpenHands 式 refactor swarm（分支隔离→verifier→PR 进集成分支）缩比 E2E；集成示例展示真 git worktree 接线。

**依赖：** M5（真执行——分支里要能 build/test）。

### M7 · H — 拓扑序 fan-out + 深层命名嵌套

**目标：** ① 依赖序（拓扑/偏序）fan-out——被调者先于调用者处理，非一把梭 barrier；② 解除 `workflow()` 命名嵌套的 1 层硬限（按需放开到 N 层）。

**对比证据（现状）：** `parallel`=flat barrier（`_context.py:461-542`）、`pipeline`=线性链（`_pipeline.py:107,147`），无任何 DAG/偏序/predecessor 模型（grep clean）；`workflow()` 硬限 1 层（`_context.py:252-257`，`WorkflowNestingError`，测试 `tests/unit/test_nesting.py:68-85`），但 `parallel/pipeline/agent` primitive 可任意深嵌（`_FANOUT_DEPTH`，`_context.py:100-113`）。

**范围：** 一个 DAG/偏序调度原语（给定依赖边，拓扑序 fan-out，仍受并发闸约束）；放开命名工作流嵌套层数（评估确定性 guard 与资源界影响）。

**验收门：** #10 文档生成（package→module→symbol 拓扑序 + >1 层嵌套）为 E2E；集成示例展示依赖序 fan-out。

**依赖：** 核心调度器。

---

## M1 实测发现（新候选 backlog，非 M1 范围）

M1 的真模型 E2E 过程中浮现两条值得后续处理的信号：

- **K · host 无法按名发现已注册工作流。** 一个有能力的 host（opus）在道层 prompt + skill + tool description 下会**自驱**工具，但因 tool description 不枚举"可用的已注册工作流名"，它选择 `run_script` **自拟**一个等效工作流，而非跑注册的 `deep_research`。改进点：tool description / `help` 暴露已注册工作流清单（让 host 能 `run` 现成的，而非总是重新 author）。归类近 M7（人体工学）或独立小项。
- **host 模型能力门槛。** 道层 prompt（无机制 coaching）下，弱模型（haiku）驱动不动多步工具流；需够强的 host（opus 可）。这是模型能力问题、非 skill/tool 缺口（opus 证明接缝自足），记作 demo 运行约定：真 host demo 用够强的模型。

## 状态

- **对比阶段**：✅ 完成。10 主题逐一带 `file:line` 证据 + 置信度（证据稿 `docs/plans/2026-06-03-v0.3.0-cc-vs-port-comparison.md`）。战略定调=允许超集。
- **M1（F 跨叶归约）**：✅ 已落地。`_reduce` 四个纯函数 `survives`/`dedup`/`reconcile`/`corroborate`(+ `ReviewItem`/`Reconciled`/`Consensus`)经包根导出 + `run_script` 命名空间注入;SKILL.md 增补 corroborate/reconcile 范式;`examples/07` 换用 `survives`/`dedup`、`examples/12` 新增双盲复核 demo。Plan = [`01-f-cross-leaf-reduce.md`](01-f-cross-leaf-reduce.md)。
- **M2（B race）**：✅ 已落地。`ctx.race` best-of-N 早退/取消原语 + `_race_types`（`RaceCandidate`/`RaceResult`）+ `race_key`（content-hash journal，namespaced + win_tag-folded）+ `SpanKind.RACE`；两值类型经包根导出 + `run_script` 命名空间注入；SKILL.md 增补 race quality / parallel-vs-race / win_tag footgun 范式；`examples/13` AI-SRE 多假设 race demo。真流式与混合 schema race 为明确非目标；**E（批处理人体工学）已拆出为自己的后续里程碑（待写）。** Plan = [`02-b-journaled-race.md`](02-b-journaled-race.md)。
- **M3–M7**：roadmap 已排定，impl plan 逐里程碑增补。E（批处理人体工学）从 M2 拆出，作后续里程碑待写。

> **执行序列：** M1 F → M2 B(race) → M3 D → M4 C → M5 A → M6 I → M7 H（E 批处理人体工学已从 M2 拆出，作后续里程碑待写）。F 首刀（接 G1+G4，纯编排层最干净）；B 紧随修核心原语；D/C 走超集；A/I 配对成重基建 epic；H 收尾引擎机制增强。

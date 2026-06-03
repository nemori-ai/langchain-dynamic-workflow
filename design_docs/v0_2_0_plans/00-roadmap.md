# v0.2.0 路线图（Roadmap）— 用例驱动逼近 Claude Code 质量

> **For agentic workers:** 本文件是 v0.2.0 的**批次总线**。这一轮以"用例驱动"方式逼近 Claude Code（CC）Dynamic Workflows 的质量：拿社区进阶用户分享的真实 CC workflow 当准星，先做能力·表达力 gap 分析，再纵切逐条补齐。每条 gap 对应一份独立的 bite-sized TDD plan（`design_docs/v0_2_0_plans/0N-*.md`），用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务执行。
>
> 详细设计与一手调研依据在 `docs/plans/2026-06-02-gap-analysis-and-g1-schema-design.md`（gitignored 草稿）。

**方法论：** ① 先能力·表达力 gap（社区真实 workflow 拆成原语/模式/人体工学用法，逐项核对引擎能否一比一表达，按杠杆排序）；② 后旗舰用例实测（挑杠杆最高的，用真实模型端到端验证逼近度）；③ 纵切——一次攻一条 gap，走完整 TDD 闭环 + 一个社区真实用例当验收门。

**核心判断：** 原语已是大路货（社区有 719★ 的 runtime clone）；CC 的"质量"七成在那份 SKILL.md 的作者智慧（refute-by-default、pipeline-by-default、loop 必带硬 cap、模型分层路由……），而表达那些智慧的总开关是结构化输出（schema）。

## Gap backlog（按杠杆排序）

| # | Gap | 社区频率 | 现状 | 杠杆 | 依赖 | Plan |
|---|---|---|---|---|---|---|
| **G1** | `agent()` 无 `schema`、返回纯文本 | ubiquitous | ✅ **已落地**（`feat/g1-agent-schema`）：`schema=` + builder-roster + journal 往返 + 跨模型评审 7 缺陷已修 + 真 E2E ×2 | 最高 | 无 | [`01-g1-agent-schema-structured-output.md`](01-g1-agent-schema-structured-output.md) |
| **G3** | SKILL.md 只教机械用法，缺质量模式库 | 质量分水岭 | ✅ **已落地**（#2）：质量模式库 + 确定性规则 + 跨模型评审修复 | 高 | G1 | [`02-g3-skill-pattern-library.md`](02-g3-skill-pattern-library.md) |
| **G2** | `isolation='worktree'` 无真实语义 | occasional（CC 旗舰 Bun 迁移用例） | ✅ **已落地**（#3）：WorktreeProvider 真隔离 + foot-gun guard + 跨模型评审修复 | 中 | 无 | [`03-g2-worktree-isolation.md`](03-g2-worktree-isolation.md) |
| **G4** | 无开箱 read-only judge agentType | occasional | ✅ **已落地**（#4）：read_only_leaf/builder + 07 集成 + 跨模型评审修复 | 低 | 无（与 G3 协同） | [`04-g4-readonly-judge.md`](04-g4-readonly-judge.md) |

**退役的伪 gap（经社区数据核实，不投入）：** 一层嵌套（CC 也是 one-level）、跨会话 resume（CC same-session，我们更强）、pipeline 签名 / budget.total / parallel 语义（均已对齐）、`args` 字符串化（CC 怪癖，我们原生 dict 更干净）。

## 推进顺序

```
G1  agent(schema=...)         ← 总开关、ubiquitous          【✅ 已落地 #1】
 └─ G3  SKILL.md 质量模式库     ← 依赖 G1；CC 质量真正所在    【✅ 已落地 #2】
      └─ G2  isolation=worktree ← 旗舰迁移用例                【✅ 已落地 #3】
           └─ G4  read-only judge type ← 人体工学            【✅ 已落地 #4】
```

每条 gap 落地必须：① 走完整 TDD（Red→Green→Refactor）；② 用一个社区真实用例当**真模型 E2E 验收门**（offline fake 证机制、`LDW_DEMO_REAL_MODEL` 真跑证逼近度）；③ **同步更新 evergreen 设计文档**（`design_docs/{01,02}.md` + `uml/`）。

## 状态

✅ **v0.2.0 全部收官并打 tag `v0.2.0`**（四 gap 全落 main，PR #1–#4 + release #5；全量 299 passed，覆盖率 94.34%）。每条 gap 均走完整 TDD + 真模型 E2E ×N + Codex 跨模型评审（每次都抓到至少一个 in-house 漏掉的 HIGH 缺陷，验证 4 次）+ evergreen 文档同步。

- **G1**（#1）：`agent(schema=)` 结构化输出——schema= + builder-roster + journal 往返；Codex 7 缺陷全修。
- **G3**（#2）：SKILL.md 社区质量模式库——adversarial-verify / pipeline-by-default / loop-until-dry / judge-panel / 模型路由 / no-silent-caps。
- **G2**（#3）：`isolation="worktree"` 真隔离——WorktreeProvider + InMemoryWorktreeProvider + foot-gun guard。
- **G4**（#4）：read-only judge leaf——read_only_leaf/builder；07 示例 skeptic 接为只读裁判作集成范例。

> **下一阶段：** v0.3.0 路线图见 [`../v0_3_0_plans/00-roadmap.md`](../v0_3_0_plans/00-roadmap.md)（首刀 F 跨叶归约）。

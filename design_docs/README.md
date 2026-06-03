# 设计文档（design_docs）

`langchain-dynamic-workflow` 的权威设计文档集合。本目录是 **committed** 级别的设计基线（区别于 `docs/plans/` 下 gitignored 的探索草稿）。

## 阅读顺序

| # | 文档 | 内容 |
|---|---|---|
| 1 | [01-engine-mechanism.md](01-engine-mechanism.md) | **引擎机制设计**：控制流反转、八原语、骑 LangGraph 的两补丁（content-hash journal / 确定性 guard）、脚本执行模型、叶子调用契约、sandbox 机制、pipeline 调度、race 早退、budget、Decision Log。即"引擎内部怎么算对"。 |
| 2 | [02-architecture.md](02-architecture.md) | **架构设计**：三层架构、对外软件形态（消费者=AI agent，只能 tool call）、五个消费面（库 core / tool adapter / skills / primitives / middleware）、自建 async 后台 tool 执行机制、L2-as-skill、build-vs-buy 账本、v1 范围。即"软件长什么样、怎么接入 agent"。 |
| 3 | [03-authoring-patterns.md](03-authoring-patterns.md) | **作者模式库**：编排脚本的作者智慧（adversarial-verify、pipeline-by-default、fan-out-reduce-synth、loop-until-dry、judge-panel、model-routing、no-silent-caps）的理据、频率、何时用、确定性适配，及与 Claude Code 的差异点。`skills/dynamic-workflow/SKILL.md` 是其面向 agent 的可运行投影。即"写得好该怎么写"。 |
| 4 | [uml/](uml/) | **UML**：[组件图](uml/01-component.md)、[类图](uml/02-class.md)、[时序图](uml/03-sequence.md)。 |

## 一手参考资料

设计的实证依据落在仓库根 `research/`（源码 / 官方 / 社区逆向三方取证）：

- `research/2026-06-01-claude-code-dynamic-workflows-reverse-engineering.md` — 目标侧（要复刻什么）
- `research/2026-06-01-langchain-deepagents-substrate.md` — 底座侧（能骑什么 / 得补什么，含 deepagents skills、langchain middleware 两份补充）
- `research/2026-06-01-microsoft-promptflow-architecture-study.md` — 对外形态启发（promptflow build-vs-buy 账本）
- `research/_data/*.json` — 结构化 findings + verification（溯源）

## 状态

架构已锁（5 面承重墙 + 3 条接缝 + 对外形态全部收口）。v0.1.0、v0.2.0 均已实现并打 tag。v0.2.0 用例驱动 gap-closing 四个 gap 全部收口（批次见 [`v0_2_0_plans/00-roadmap.md`](v0_2_0_plans/00-roadmap.md)：**G1 `agent(schema=)`、G3 质量范式库、G2 `isolation="worktree"` 真隔离、G4 只读裁判 leaf 均已落地**）。历史批次归档在 `v0_1_0_plans/`。v0.3.0 用例收集进行中。

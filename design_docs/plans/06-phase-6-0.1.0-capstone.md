# Phase 6 · 0.1.0 收口（集成 / 硬化 / 可观测 / 文档）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 superpowers:executing-plans。步骤用 `- [ ]` 跟踪。

**Goal:** 一个贴近真实的多阶段 capstone workflow（parallel + pipeline + budget + sandbox + 对抗式验证），由 host deepagent 后台驱动，端到端全绿；层边界机械守护；公共 API 文档化；打 0.1.0。

**Architecture:** 在 Phase 1–5 之上做集成与硬化：import-linter `forbidden` contracts 守 L0/L1/L2；可观测性 hook（phase/log → 可选 trace、usage rollup）；覆盖率门；README quickstart。详见 [../02-architecture.md](../02-architecture.md) §7、§9。

**Tech Stack:** Phase 1–5 栈 + `import-linter`（dev 依赖）+ 覆盖率工具（`pytest-cov`，dev）。

---

## 里程碑 M6（= 0.1.0）

多特性 capstone 端到端全绿；import-linter 绿；覆盖率达标；公共 API + README 文档化；version 0.1.0。

## 范围与设计要点

- **import-linter contracts**（`[tool.importlinter]` in pyproject）：`forbidden` —— L2（tool/middleware/skills）不得直接 import L0 内部；roster/AST-gate 不得直接 import LangGraph 内部细节；强制 Layer 0/1/2 单向依赖。
- **可观测性**：`agent()`/`parallel()`/`pipeline()` 自动 emit span/journal entry（observability-by-default）；`phase`/`log` → 可选 trace 映射；token usage rollup 喂 budget（规划独立 `*-tracing` 子包接口，v1 内联实现即可）。
- **capstone demo**：多阶段——`parallel` 扇出研究 → `pipeline` 提炼 → 对抗式验证（N skeptic 叶子，多数存活）→ 综述；由 host deepagent 经 workflow tool 后台驱动。
- **覆盖率门**：核心模块 line ≥85%；`agent/parallel/pipeline/journal/resume/background` 必须有集成测试覆盖。
- **文档**：README quickstart（照着能跑通）、公共 API 稳定性说明、CHANGELOG、`version="0.1.0"`。
- **承接 Phase 5 review（nit）资源耗尽硬化**：`BgRunManager` 当前对 host 发起的后台 run 数无上限——给它加一个 `max_concurrent_runs` 配额（满则 `run` 命令拒绝并回明确提示，而非无界 `asyncio.create_task` 扇出），契合 AGENTS.md「bounded queues / 资源耗尽防护」铁律；补一条满额拒绝测试。

## 验收标准

- [ ] capstone demo 端到端绿（fake model）；真实模型变体 env 门控、文档化。
- [ ] `import-linter`（`lint-imports`）通过；`ruff check`、`pyright`(strict) 零告警；`pytest` 全绿。
- [ ] 核心模块覆盖率 ≥85%（`pytest --cov`）；关键编排路径均有集成测试。
- [ ] 公共 API（`run_workflow`/`Roster`/`create_workflow_tool`/`create_workflow_middleware`/七原语）稳定 + README quickstart 可跑通。
- [ ] CHANGELOG 记录；`pyproject` `version = "0.1.0"`；`Development Status` classifier 升级。

## 指标

- capstone e2e 绿；import-linter 0 违规；覆盖率 ≥85%（核心）；pyright 0；ruff 0。
- README quickstart：从 `uv sync` 到跑通 demo 的步骤数 / TTHW 记录。

## 文件结构

```
修改 pyproject.toml（[tool.importlinter] contracts、pytest-cov、import-linter dev dep、version 0.1.0）
src/langchain_dynamic_workflow/_observability.py   # span/usage rollup hook（内联）
tests/integration/test_phase6_capstone.py
tests/test_import_contracts.py                       # 可选:断言关键边界
examples/06_capstone.py
README.md / README_zh.md（quickstart 更新）、CHANGELOG.md
```

## 任务分解（bite-sized TDD）

1. **import-linter**：加 `[tool.importlinter]` `forbidden` contracts + dev 依赖；`uv run lint-imports` 跑绿（先让现有代码满足边界，必要时小重构）→ commit。
2. **可观测性 hook**：Red（agent/parallel/pipeline emit span + usage rollup 进 budget）→ `_observability.py` 内联实现 → Green → commit。
3. **覆盖率门**：加 `pytest-cov` + 阈值；补齐核心路径集成测试到 ≥85% → commit。
4. **capstone demo**：Red（多阶段 e2e：parallel+pipeline+budget+sandbox+对抗验证，host 后台驱动）→ 实现 `examples/06_capstone.py` + 集成测试（fake model）→ Green → commit。
5. **文档 + 版本**：README/README_zh quickstart、CHANGELOG、`version="0.1.0"`、classifier 升级 → commit。
6. **0.1.0 收口 commit/tag**：全门绿（ruff/pyright/pytest/cov/import-linter）→ 里程碑 commit（tag 由 `github-tag-release`/仓库约定处理，不在本 plan 自动打）。

## demo 规格（M6，capstone）

`examples/06_capstone.py`：host deepagent 经 workflow tool 后台跑一个多阶段 workflow——并行研究 N 源 → pipeline 提炼 → 每条发现派 N 个 skeptic 叶子对抗式验证（多数存活才留）→ 综述定稿。集成测试用 fake model 全绿；真实模型变体 env 门控。

## 对 Phase 1–5 的 refactor

- 为满足 import-linter 边界可能小重构模块归属；为覆盖率补测试。Phase 1–5 全部测试须保持绿。

## 0.1.0 之后（v2 backlog，见 roadmap）

A2/A3 安全硬化、codegen prompt 工程深化、独立 `*-tracing` 子包、跨进程持久化（sqlite/postgres checkpointer）、skill 语义检索、`agent(isolation="worktree")`。

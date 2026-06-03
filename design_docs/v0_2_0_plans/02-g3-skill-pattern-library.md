# G3 — SKILL.md 质量模式库 + 范式 workflow 示例 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把社区那份构成 Claude Code 真正"质量"的**作者智慧**——以 Python 惯用法（基于 `ctx` 原语 + G1 的 `schema=`）移植进 `SKILL.md`，并附**可跑的范式 workflow 示例**。使 LLM 现写脚本（或挑命名 workflow）就能产出 CC 级编排：adversarial-verify（refute-by-default）、pipeline-by-default、fan-out→reduce-in-Python→synthesize、loop-until-dry + 硬 MAX_ROUNDS、judge-panel / multi-modal-sweep、per-stage model routing、no-silent-caps logging。

**Architecture:** 我们的 `SKILL.md` 现状只教机械用法（3 个基础 pattern + 确定性规则），缺社区那份"质量分水岭"的模式库。社区语料显示这些模式按频率分布明确，且是 CC workflow 高质量的根因。本计划：① 在 SKILL.md 增 `## Quality patterns` 大节，每个模式配 Python 代码（`ctx` 原语 + `schema=` dict 字面量，遵守确定性规则）；② 加一个可跑范式示例 `examples/09_quality_patterns.py`（离线 fake / 真跑可选），把 adversarial-verify + pipeline(review→verify) + loop-until-dry 演示成闭环；③ 用一个测试**抽取 SKILL.md 的 python 代码块并断言 `ast.parse` 通过且过 AST gate（`validate_workflow_source`）**，防文档代码腐化。半数模式（adversarial-verify、judge-panel、schema-handoff）非 `schema=` 不能表达——故依赖 G1。

**Tech Stack:** Python 3.12（async-first）、G1 的 `agent(schema=...)`（dict 字面量形态）、`validate_workflow_source`（L2 AST gate）、pytest + pytest-asyncio、ruff、pyright(strict)。

**依赖：** **G1**（`agent(schema=...)` 已落地，含 fail-loud 硬化）。

---

## 复核：G1 硬化后的 `schema=` 表面（开工前必做）

G1 落地时对 dict→pydantic 转换器做了 fail-loud 硬化，**SKILL.md 的所有 dict schema 代码块必须合规**，否则示例脚本一跑就被转换器拒：

- `additionalProperties` **只接受 bool**（dict 形态被拒）——示例统一写 `"additionalProperties": False`。
- **未支持的约束关键字**（`pattern`/`minimum`/`maxLength`/`format`/`const`/`minItems`…）会 **fail-loud**——模式库的 schema 只用 `type`/`properties`/`required`/`items`/`enum`/`description`/`default`，不写约束关键字。
- `required` 必须 ⊆ `properties`；枚举不得有值相等坍缩（`[True,1]`）。

- [ ] **Step 0（复核）:** 通读拟写入 SKILL.md 的每个 dict schema，确认只用受支持键。本计划下方代码块均已遵守。

## 文件结构（本计划触达）

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` | 增 `## Quality patterns` 大节（每模式 + Python 代码） | 修改 |
| `examples/09_quality_patterns.py` | 可跑范式：adversarial-verify + pipeline(review→verify) + loop-until-dry（离线 fake schema 叶 + 真跑可选入口） | 新建 |
| `tests/unit/test_skill_patterns.py` | 抽 SKILL.md python 代码块 → `ast.parse` + 过 `validate_workflow_source` | 新建 |
| `tests/integration/test_quality_patterns.py` | 离线跑 09 的 `orchestrate`，断言模式行为（被证伪 finding 被丢、loop 收敛且 ≤ MAX_ROUNDS） | 新建 |
| `src/.../_workflows.py` 或示例内 | （Task 3，默认纳入）把 canonical `code_review` 注册成命名 workflow | 视情况 |
| `design_docs/03-authoring-patterns.md` | **新建** evergreen：把模式库沉淀为权威作者指南（SKILL.md 是其面向 agent 的投影） | 新建 |
| `design_docs/README.md` | 阅读顺序表登记 `03-authoring-patterns.md` | 修改 |
| `design_docs/01-engine-mechanism.md` | "七原语/确定性"处交叉引用作者指南 | 修改 |

---

## 前置：分支

- [ ] **创建特性分支**（从 G1 落定后的 `main`）

```bash
git checkout -b feat/g3-quality-patterns
```

## 要移植的模式（带社区频率）

| 模式 | 频率 | Python 表达要点 |
|---|---|---|
| adversarial-verify（refute-by-default） | ubiquitous | N 个独立 refuter，`schema={refuted:bool,...}`，"默认 refuted=True 除非能证伪"；多数票存活 |
| pipeline review→verify + "pipeline 默认 / parallel 仅真 barrier" | common | `ctx.pipeline(dims, review, verify)`；stage2 `parallel` 对该维 findings 对抗验证；强调无谓 barrier 浪费 |
| fan-out → reduce-in-Python → synthesize | ubiquitous | `ctx.parallel([...])` → 纯 Python `filter/sort/dedupe`（中间态在脚本变量）→ 单 synth agent |
| loop-until-dry + 硬 MAX_ROUNDS + budget 守卫 | common | `while` + DRY_STREAK 计数 + `MAX_ROUNDS` 硬顶；`while ctx.budget.remaining() > T`（注意 `budget.total is None → inf` 陷阱） |
| judge-panel / multi-modal-sweep | common | 同一 artifact N 个 lens（correctness/security/perf…）majority 存活；或 N 角度起草、并行打分、择优 |
| per-stage model routing（成本纪律） | common | `ctx.agent(model="haiku")` 干粗活、`model="sonnet"` 干重活；**model 入缓存键**，label/phase 不入 |
| no-silent-caps logging | occasional | 被 top-N/采样截断时 `ctx.log(...)` 明说丢了啥 |
| read-only judge | occasional | → 交叉引用 **G4**（`read_only_builder`，裁判物理只读） |

**确定性适配**（比 JS 版更需明说）：有序迭代（`sorted`）、`parallel` thunk 默认参数捕获、禁 import、`budget.total is None` 陷阱、缓存键 = `prompt+agent_type+model+schema+isolation`（label/phase 装饰性）。

**与 CC 的差异点（SKILL.md 诚实标注）：** `args` 我们原生传 dict（无 CC 字符串化怪癖）；resume 可跨会话（CC 仅 same-session）；嵌套一层（与 CC 一致）。

## Task 1: SKILL.md 质量模式库 + 代码块 gate 测试（先写测试反腐）

**Files:**
- Create: `tests/unit/test_skill_patterns.py`
- Modify: `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md`

- [ ] **Step 1（失败测试）:** 用现有 skill 加载器读 `dynamic-workflow/SKILL.md`，正则抽出所有 ` ```python ` 代码块，对每块：① `ast.parse` 不抛；② 包进 `async def orchestrate(ctx, args):` 骨架后过 `validate_workflow_source`（无违规）。初始断言"存在 ≥1 个标注为 quality-pattern 的代码块"会失败（SKILL.md 尚无）。

```python
# tests/unit/test_skill_patterns.py
"""Anti-corruption: every python code block in SKILL.md must parse and pass the AST gate."""

from __future__ import annotations

import ast
import re
import textwrap

from langchain_dynamic_workflow import skills_path
from langchain_dynamic_workflow._gate import validate_workflow_source  # 确认实际导出路径

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks() -> list[str]:
    text = (skills_path() / "dynamic-workflow" / "SKILL.md").read_text(encoding="utf-8")
    return _BLOCK.findall(text)


def test_skill_has_quality_pattern_blocks() -> None:
    # The quality-pattern section ships runnable patterns, not just the 3 basics.
    assert len(_blocks()) >= 6


def test_every_skill_block_parses_and_passes_gate() -> None:
    for block in _blocks():
        src = textwrap.dedent(block)
        ast.parse(src)  # raises SyntaxError on malformed
        wrapped = src if "def orchestrate" in src else f"async def orchestrate(ctx, args):\n{textwrap.indent(src, '    ')}"
        validate_workflow_source(wrapped)  # raises WorkflowScriptError on a violation
```

- [ ] **Step 2:** 运行确认失败（`uv run pytest tests/unit/test_skill_patterns.py -q > /tmp/ldw-g3-1.log 2>&1; tail -20 /tmp/ldw-g3-1.log`）。
- [ ] **Step 3:** 在 SKILL.md 追加 `## Quality patterns` 节，逐个写模式 + Python 代码（adversarial-verify / pipeline review→verify / fan-out-reduce-synth / loop-until-dry / judge-panel / model-routing / no-silent-caps），每块 `ctx` 原语 + 受支持的 `schema=` dict 字面量，遵守确定性规则。示例（adversarial-verify，schema 合规）：

```python
async def orchestrate(ctx, args):
    claims = sorted(args["claims"])
    confirmed = []
    for claim in claims:
        votes = await ctx.parallel([
            lambda c=claim: ctx.agent(
                f"Refute this claim if you can; default to refuted unless you can ground it: {c}",
                agent_type="skeptic",
                schema={
                    "type": "object",
                    "properties": {"refuted": {"type": "boolean"}, "reason": {"type": "string"}},
                    "required": ["refuted", "reason"],
                    "additionalProperties": False,
                },
            )
            for _ in range(3)
        ])
        refutes = sum(1 for v in votes if v is not None and v.refuted)
        if refutes < 2:
            confirmed.append(claim)
    return confirmed
```

- [ ] **Step 4:** 运行确认通过（所有代码块 parse + 过 gate）。
- [ ] **Step 5:** ruff/pyright（测试文件）。
- [ ] **Step 6:** commit `docs(skill): add the community quality-pattern library (adversarial-verify, pipeline-by-default, loop-until-dry, judge-panel, model-routing)`。

## Task 2: 可跑范式示例 + 行为测试

**Files:**
- Create: `examples/09_quality_patterns.py`
- Create: `tests/integration/test_quality_patterns.py`

- [ ] **Step 1（失败集成测试）:** 注册 fake schema 叶（reviewer 产 `Finding{title,severity}`、skeptic 产 `Verdict{refuted}`，其中一个 finding 设计成被多数 refuter 证伪），跑 `examples/09` 的 `orchestrate`，断言：① 被证伪的 finding 不在最终结果；② loop-until-dry 在连续 K 轮无新发现后停且不超 `MAX_ROUNDS`。
- [ ] **Step 2:** 运行确认失败（无 `examples/09`）。
- [ ] **Step 3:** 写 `examples/09_quality_patterns.py`：`orchestrate` 内演示 `pipeline(review→adversarial-verify)` + loop-until-dry，纯 Python reduce；附 `_demo_models` 风格离线 fake 叶 + 一个 `LDW_DEMO_REAL_MODEL` 真模型入口。**判定叶用 G4 `read_only_builder` 注册（裁判只读）。**
- [ ] **Step 4:** 运行确认通过。
- [ ] **Step 5:** ruff/pyright（example + 测试）。
- [ ] **Step 6:** commit `example(quality): runnable adversarial-verify + loop-until-dry workflow`。

## Task 3: 注册 canonical `code_review` 命名 workflow（默认纳入）

- [ ] 在示例/roster 装配处用 `WorkflowRegistry().register("code_review", code_review_fn)`，加一个集成测试按名 `run` 跑通——镜像 CC 自带 bundled workflow 的可发现性（LLM 既可挑命名 workflow，也可现写）。YAGNI 则跳过。commit。

## Task 4: 真模型 E2E 验收（adversarial-verify 真去伪）

**Files:**
- 载体：`examples/09_quality_patterns.py` 真路径。

- [ ] **Step 1:** demo 任务 = review 一段**植入 1 个真 bug + 1 个诱导误报**的小代码；reviewer 多维产 `Finding`；每个 finding 经 N 个独立 refuter（refute-by-default）对抗验证。
- [ ] **Step 2: 主循环真跑**

```bash
LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5 uv run --group example python examples/09_quality_patterns.py
```

- [ ] **Step 3: 观察 + 断言**：真实 skeptic **真的把误报 refute 掉、保住真 bug**（证明 adversarial-verify 不是摆设）；loop-until-dry 在真 agent 下收敛（连续 K 轮无新发现即停、不超 MAX_ROUNDS）；植入的误报（按 id/标记）不在最终 confirmed，真 bug 在；轮次 ≤ MAX_ROUNDS。这是 G3 的真实验收门。

## Task 5: evergreen 同步（新建作者指南）

**Files:**
- Create: `design_docs/03-authoring-patterns.md`
- Modify: `design_docs/README.md`、`design_docs/01-engine-mechanism.md`

- [ ] **Step 1:** 新建 `design_docs/03-authoring-patterns.md`（evergreen 作者指南：把模式库沉淀为权威设计文档，SKILL.md 是其面向 agent 的投影），在 README 阅读顺序表登记；`01-engine-mechanism.md` 在"七原语/确定性"处交叉引用。
- [ ] **Step 2:** commit。

## Task 6: 质量闸门

- [ ] `uv run pytest -q` 全绿；`ruff`/`ruff format --check`/`pyright`/`lint-imports` 全过；commit（如有配置调整）。

---

## 待定稿核实（执行时）

1. SKILL.md 代码块测试的 skill 加载器与 `validate_workflow_source` 的**确切导出路径**（开工第一步对照源码确认；复用优先，避免重复造轮子）。
2. 是否已有测试覆盖 SKILL.md 代码块（避免重复 Task 1）。
3. 模式库是否值得独立 evergreen 文档（`03-authoring-patterns.md`）还是并入 `01`——按落地时体量定（默认独立，因模式库体量大且是 SKILL.md 的源）。

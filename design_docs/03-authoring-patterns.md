# 作者模式库（Authoring Patterns）— evergreen

> **定位**：这是编排脚本的**作者智慧**权威来源——社区进阶用户手写高质量 Claude Code dynamic workflow 时反复用的那批模式。引擎原语（`agent`/`parallel`/`pipeline`/`phase`/`log`/`budget`/`workflow` + `schema=`）是"能写什么"；本文是"写得好该怎么写"。
>
> **与 SKILL.md 的关系**：`skills/dynamic-workflow/SKILL.md` 是本文**面向 agent 的投影**——把这些模式压成宿主 agent 现写脚本即可照搬的可运行代码块（每块都过 AST gate，有 `tests/unit/test_skill_patterns.py` 反腐）。本文承载**理据、频率、何时用、确定性适配**，不重复代码；要看可运行形态去 SKILL.md。
>
> 阅读顺序见 [README](README.md)；引擎机制见 [01-engine-mechanism.md](01-engine-mechanism.md)。

## 1. 为什么是这些模式

核心判断（v0.2.0 路线图）：编排原语已是大路货，CC dynamic workflow 的"质量"七成在那份 SKILL.md 的作者智慧。社区语料显示这些模式按频率分布明确，且是高质量编排的根因。把它们移植进来（以 Python 惯用法 + `schema=`），是用例驱动逼近 CC 质量的主轴（gap G3）。

## 2. 模式清单（带社区频率与理据）

| 模式 | 频率 | 解决什么 | 关键理据 |
|---|---|---|---|
| **adversarial-verify（refute-by-default）** | ubiquitous | 单个"确认者"会放过似是而非的错误结论 | 让 N 个独立 skeptic **默认证伪**、多数票存活；voter 序号入 prompt 使 N 个裁决是独立 journal 条目（resume 稳健、非一个缓存键被复用） |
| **pipeline review→verify（pipeline 默认 / parallel 仅真 barrier）** | common | 维度间无依赖时，barrier 白等最慢的那个 | 用 `pipeline(dims, review, verify)` 让某维 findings 一到就对抗验证；只有"真的需要全部结果一起"才 `parallel` |
| **fan-out → reduce-in-Python → synthesize** | ubiquitous | 中间态塞进模型 context 既贵又噪 | 扇出后在**脚本变量**里 `filter/sort/dedupe`（纯 Python），只把收敛结果交给单个 synth agent |
| **loop-until-dry + 硬 MAX_ROUNDS + budget 守卫** | common | 未知规模的发现任务何时停 | 连续 K 轮无新发现即停，但**永远**带 `MAX_ROUNDS` 硬顶；budget 检查须 `if ctx.budget.total and ...`（见 §3 陷阱） |
| **judge-panel / multi-modal sweep** | common | 同质 N 个评审有共同盲区 | 同一 artifact 用**不同镜头**（correctness/security/perf…）并行评，多数存活；镜头多样性抓单一评审漏的失败模式 |
| **per-stage model routing（成本纪律）** | common | 粗活也用贵模型是浪费 | 便宜模型干 triage、强模型只碰幸存者；`model` 入缓存键（partition resume 正确），`label`/`phase` 不入 |
| **no-silent-caps logging** | occasional | top-N/采样静默截断 → 截断的结果被当成完整的读 | 截断时 `ctx.log` 明说丢了啥 |
| **read-only judge** | occasional | 裁判顺手"修"会让幻觉修复落地 | 把判定叶注册为只读（生成叶 vs 判定叶分离）——见 gap G4 的 `read_only_leaf` |

## 3. 确定性适配（比 JS 版更需明说）

引擎在 resume 时重放脚本、按输入内容哈希缓存每个叶子结果，故脚本的**可观测 `agent()` 调用序列必须 run-to-run 一致**，否则 fail-loud。模式落地时：

- **有序迭代**：迭代 `set`/`dict` 必先 `sorted(...)`；`loop-until-dry` 里把 `seen` 集合 `sorted` 后再入 prompt。
- **`parallel` thunk 默认参数捕获**：`[lambda t=t, v=v: ctx.agent(...) for ...]`——闭包必须绑各自的值，否则全捕获最后一个。
- **`budget.total is None` 陷阱**：未设预算时 `budget.remaining()` 是 `inf`，裸 `while ctx.budget.remaining() > T` 永真。守卫写 `if ctx.budget.total and ctx.budget.remaining() < T`。
- **缓存键 = `prompt + agent_type + model + schema + isolation`**；`label`/`phase` 是装饰性的、不入键。adversarial-verify 的 N 个 skeptic 必须靠 voter 序号让 prompt 相异，否则它们共享一个键。
- **禁 import**（L2 gated 脚本）：故 `schema=` 用内联 JSON-schema dict 字面量（引擎经 `to_pydantic_model` 归一），不能 `import pydantic`。dict schema 须合规硬化后的子集（见 [01](01-engine-mechanism.md) 叶子契约：bool-only `additionalProperties`、`required` ⊆ `properties`、不用未支持约束关键字、枚举不得值相等坍缩）。

## 4. 与 Claude Code 的差异点（诚实标注）

| 维度 | Claude Code | 本引擎 | 说明 |
|---|---|---|---|
| `args` 传递 | 字符串化怪癖（须自己 parse） | **原生 dict** | 我们更干净，作者直接拿结构化 args |
| resume 范围 | same-session | **可跨会话 / 跨进程（M3 已落地，超集 CC）** | 默认 ships `InMemoryRunStore`（同进程、零依赖）；接 `SqliteWorkflowStore`（`[sqlite]` extra）即跨进程 resume——全新进程指向同一 db 文件按 `run_id` resume、完成叶从持久 journal 零成本重放（CC 仅 same-session）。零成本重放由 **journal** 交付、checkpointer 是 durable add-on（见 [01](01-engine-mechanism.md) §13b、接线 [02](02-architecture.md) §10） |
| 嵌套层数 | 一层（`workflow()` 内不能再 `workflow()`） | 一层（一致） | 与 CC 对齐，二层 fail-loud |

退役的伪 gap（经社区数据核实，非 CC-parity 差距、不投入）：一层嵌套、跨会话 resume、pipeline 签名 / `budget.total` / `parallel` 语义、`args` 字符串化——详见 [v0_2_0_plans/00-roadmap.md](v0_2_0_plans/00-roadmap.md)。

## 5. 可运行载体

- **SKILL.md**（`skills/dynamic-workflow/`）：每个模式的可运行代码块（宿主 agent 照搬）。
- **`examples/09_quality_patterns.py`**：adversarial-verify + loop-until-dry 的端到端可跑范式（离线 fake / `LDW_DEMO_REAL_MODEL` 真跑）。
- **`tests/unit/test_skill_patterns.py`**：抽 SKILL.md 所有 python 块断言 `ast.parse` + 过 `validate_workflow_source`——防文档模式腐化成引擎会拒的脚本。

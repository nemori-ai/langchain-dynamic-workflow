# G4 — 开箱即用的 read-only judge 叶 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供一个库级辅助 `read_only_leaf(...)`，一行构造出**工具面只读**的 deepagent 叶——能 `read`/`grep`/`glob`/`ls`，但**物理无法 `write`/`edit`/执行**。把它注册成 roster 条目（`agent_type="judge"`）后，adversarial-verify / judge-panel 模式里的裁判就"只能判、不能改"，幻觉修复落不了地，"生成"与"判定"彻底分离。这是社区 occasional 但重要的纪律（CC 用 `agentType:'Explore'` 让裁判不能改文件）。

**Architecture:** 只读与否取决于叶子的**工具面**，与 sandbox 准入正交——`needs_execution=False` 只是不分配 sandbox（走 `StateBackend`），而 `StateBackend` 仍可写，故 `needs_execution=False ≠ 只读`。真正的只读 = deepagent 带一条**拒写**的 `FilesystemPermission` 且**无 execute 工具**。已核实（见下"已核实的 deepagents API"）：`create_deep_agent(permissions=[...])` 是公开构造参数，`FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")` 拒绝一切写。引擎不构造叶子（宿主构造），故该辅助归宿主侧；库只给一个对齐 deepagents 的便利构造器 + 约定 + 测试（YAGNI——不进引擎内建）。与 G1 协同：只读裁判通常带 `schema=Verdict{refuted,reason}`，故辅助以 **builder 形态**注册（支持 `response_format`）。

**Tech Stack:** Python 3.12（async-first）、deepagents 0.6.7（`create_deep_agent` / `FilesystemPermission`）、langchain `ToolStrategy`（与 G1 同）、pytest + pytest-asyncio、ruff、pyright(strict)、import-linter。

**依赖：** **G1**（builder-roster + `schema=`）已落地——只读裁判走 builder 注册以支持结构化裁决。与 **G3** 协同（adversarial-verify / judge-panel 模式的裁判应默认只读，G3 的 SKILL.md 在此交叉引用本辅助）。

---

## 已核实的 deepagents 只读 API（开工前已证，2026-06-02）

| 事实 | 出处 | 影响 |
|---|---|---|
| `create_deep_agent(..., permissions: list[FilesystemPermission] \| None = None)` 是公开参数 | `deepagents/graph.py:226` | 直接传 permissions 即可，无需碰中间件私有 |
| `FilesystemPermission` 从 `deepagents` 顶层导出 | `deepagents/__init__.py:33` | `from deepagents import FilesystemPermission` |
| `FilesystemPermission(operations, paths, mode="allow"\|"deny")`；path 必须绝对、禁 `..`/`~` | `filesystem.py:84-103` | 拒写规则 = `operations=["write"], paths=["/**"], mode="deny"` |
| `FilesystemOperation = Literal["read", "write"]`（**无 execute 维度**） | `filesystem.py:71` | execute 不在权限系统内——靠"不分配 sandbox（`needs_execution=False`）"来禁执行 |
| permissions **不兼容**提供命令执行的 backend（`SandboxBackendProtocol`），execute 的 tool-level 权限未实现 | `filesystem.py:697-699` | 只读裁判**必须** `needs_execution=False`（走 `StateBackend`，非 sandbox）；二者叠加才是真只读 |
| 写/编辑工具在 `validated_path` 处统一查 `_check_fs_permission(..., "write", ...)` | `filesystem.py:1026/1065/1120/1162` | 一条 deny-write 规则覆盖 `write_file`/`edit_file` 等所有写入口 |

**结论（D-G4，下方 Decision Log）**：read-only judge = `create_deep_agent(permissions=[deny-write], 默认 StateBackend)` + roster 以 `needs_execution=False` 注册。无需 execute 维度的权限（deepagents 没有），不分配 sandbox 即无 execute 工具。

## 文件结构（本计划触达）

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/langchain_dynamic_workflow/_leaves.py` | `read_only_leaf(...)` + `read_only_builder(...)`（builder 形态，支持 `response_format`） | **新建** |
| `src/langchain_dynamic_workflow/__init__.py` | 短路导出 `read_only_leaf` / `read_only_builder` | 修改 |
| `tests/unit/test_readonly_leaf.py` | 断言只读叶**无 write/edit 工具或写被拒**、read/grep 可用、可绑 `response_format` | 新建 |
| `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` | judge / adversarial-verify 模式补"裁判注册为只读叶"约定 + 辅助用法 | 修改 |
| `examples/09_quality_patterns.py`（G3 产出）| judge 用 `read_only_builder` 注册 | 修改 |
| `examples/11_readonly_judge_real_e2e.py` | **真E2E验收**：诱导裁判"动手修"，证明物理拦写、转而只返裁决 | 新建 |
| `design_docs/01-engine-mechanism.md` / `02-architecture.md` / `uml/` | evergreen：roster/叶子契约补只读叶约定；Decision Log 增 D-G4 | 修改 |

---

## 前置：分支

- [ ] **创建特性分支**（从 G1/G3 落定后的 `main`）

```bash
git checkout -b feat/g4-readonly-judge
```

## Task 1: `read_only_leaf` 失败测试（先写反腐）

**Files:**
- Test: `tests/unit/test_readonly_leaf.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_readonly_leaf.py
"""Unit tests for the read-only judge leaf helper."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_dynamic_workflow import read_only_leaf


class _WriteAttemptModel(BaseChatModel):
    """A fake model that, on its first turn, tries to call the write_file tool."""

    @property
    def _llm_type(self) -> str:
        return "fake-write-attempt"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kw: Any) -> ChatResult:
        # If the write was already refused (a ToolMessage came back), stop.
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="cannot write"))])
        call = AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"file_path": "/x.txt", "content": "hi"}, "id": "w"}],
        )
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools: Any, **kw: Any) -> BaseChatModel:
        return self


async def test_read_only_leaf_cannot_write() -> None:
    leaf = read_only_leaf(_WriteAttemptModel())
    out = await leaf.ainvoke({"messages": [HumanMessage(content="please write /x.txt")]})
    # The write tool either does not exist or is denied; in no case does a file land.
    files = out.get("files", {})
    assert "/x.txt" not in files
    # The deny surfaces to the model as a tool error, which it then acknowledges.
    assert any(isinstance(m, AIMessage) and "cannot write" in m.text for m in out["messages"])
```

- [ ] **Step 2: 运行确认失败**

```bash
uv run pytest tests/unit/test_readonly_leaf.py -q > /tmp/ldw-g4-1.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-g4-1.log
```
Expected: FAIL — `ImportError: cannot import name 'read_only_leaf'`（辅助尚不存在）。

## Task 2: 实现 `read_only_leaf` + `read_only_builder`

**Files:**
- Create: `src/langchain_dynamic_workflow/_leaves.py`

- [ ] **Step 1: 实现**

```python
# src/langchain_dynamic_workflow/_leaves.py
"""Library-level leaf constructors — host-side conveniences aligned with deepagents.

A read-only leaf can read / grep / glob / ls but cannot write, edit, or execute, so
a judge built from it can only assess, never "fix". Read-only is a property of the
tool surface (a deny-write filesystem permission plus the absence of an execution
sandbox), enforced by deepagents at the tool boundary — not by the engine.
"""

from __future__ import annotations

from typing import Any, Protocol

from deepagents import FilesystemPermission, create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

# Deny every write everywhere. deepagents has no "execute" permission dimension
# (FilesystemOperation = Literal["read","write"]); execution is foreclosed instead
# by registering the leaf with needs_execution=False, so no execute tool is wired
# and the default StateBackend (not a SandboxBackendProtocol) honors permissions.
_DENY_WRITE: tuple[FilesystemPermission, ...] = (
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
)


def read_only_leaf(
    model: BaseChatModel,
    *,
    system_prompt: str | None = None,
    response_format: Any = None,
    **kwargs: Any,
) -> Runnable[Any, Any]:
    """Build a read-only deepagent leaf (deny-write tool surface, no execution).

    Args:
        model: The chat model the judge reasons with.
        system_prompt: Optional system prompt for the judge.
        response_format: Optional structured-output binding (e.g. a ``ToolStrategy``);
            forwarded so a judge can return a validated ``Verdict``.
        **kwargs: Forwarded to ``create_deep_agent`` (e.g. ``tools``, ``skills``).

    Returns:
        A compiled deepagent whose filesystem writes are denied.
    """
    extra: dict[str, Any] = dict(kwargs)
    if system_prompt is not None:
        extra["system_prompt"] = system_prompt
    if response_format is not None:
        extra["response_format"] = response_format
    return create_deep_agent(model=model, permissions=list(_DENY_WRITE), **extra)


class _Builder(Protocol):
    def __call__(self, *, response_format: Any = None) -> Runnable[Any, Any]: ...


def read_only_builder(
    model: BaseChatModel, *, system_prompt: str | None = None, **kwargs: Any
) -> _Builder:
    """Return a roster ``builder`` that constructs a read-only leaf per response_format.

    Register with ``roster.register("judge", builder=read_only_builder(model, ...))``
    so ``agent(agent_type="judge", schema=Verdict)`` yields a structured, read-only
    judge.
    """

    def _builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        return read_only_leaf(
            model, system_prompt=system_prompt, response_format=response_format, **kwargs
        )

    return _builder
```

- [ ] **Step 2: 运行确认通过** — `uv run pytest tests/unit/test_readonly_leaf.py -q`（写入被拒、无文件落地）。
- [ ] **Step 3: ruff + pyright**（`_leaves.py` + 测试）。

## Task 3: `__init__.py` 导出

**Files:**
- Modify: `src/langchain_dynamic_workflow/__init__.py`

- [ ] **Step 1: 短路导出**（仅外部会用到的对象）

```python
from ._leaves import read_only_builder, read_only_leaf
# 并入 __all__
```

- [ ] **Step 2: 测试改用顶层导入**（`from langchain_dynamic_workflow import read_only_leaf`，已在 Task 1 用）。
- [ ] **Step 3: commit** `feat(leaves): read_only_leaf / read_only_builder for read-only judges`。

## Task 4: 注册路径集成测试（roster + agent(schema=)）

**Files:**
- Test: `tests/integration/test_readonly_judge.py`（新建）

- [ ] **Step 1: 失败测试** — 注册 `read_only_builder` 进 roster，经 `agent(agent_type="judge", schema=Verdict)` 在一个会尝试 write 的提示上跑，断言：① 返回合法 `Verdict`；② 运行后无文件被写入（sandbox/StateBackend 无裁判写痕）。
- [ ] **Step 2: 运行确认失败 → 实现接通 → 确认通过**（builder 已在 Task 2 提供，本任务验证它在引擎 `runnable_for(response_format)` 路径上成立）。
- [ ] **Step 3: ruff/pyright；commit** `test(g4): read-only judge over the engine schema path`。

## Task 5: SKILL.md + 09 示例接入

**Files:**
- Modify: `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md`
- Modify: `examples/09_quality_patterns.py`（G3 产出）

- [ ] **Step 1:** 在 SKILL.md 的 adversarial-verify / judge-panel 模式处加一段："把裁判注册为**只读叶**（`read_only_builder`），裁判物理无法 write/edit/执行，幻觉修复落不了地；生成叶与判定叶分离。"并交叉引用本辅助。
- [ ] **Step 2:** 09 示例的 judge 改用 `read_only_builder(model, system_prompt=...)` 注册。
- [ ] **Step 3:** 跑 SKILL.md 代码块 gate 测试（G3 Task 1 建立）确认新代码块 parse + 过 AST gate；commit。

## Task 6: 真模型 E2E 验收（裁判物理拦写）

**Files:**
- Create: `examples/11_readonly_judge_real_e2e.py`

- [ ] **Step 1:** 写一个真跑 demo（`_demo_models` 风格，离线 fake / `LDW_DEMO_REAL_MODEL` 真跑）：给只读裁判一个**诱导它"动手修"**的提示 + 一份 `Verdict{refuted,reason}` schema。
- [ ] **Step 2: 主循环真跑**

```bash
LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5 uv run --group example python examples/11_readonly_judge_real_e2e.py
```

- [ ] **Step 3: 观察 + 断言**：真实裁判产出 `Verdict`；运行后**无文件被裁判改动**（写工具缺席/被拒）；裁判转而只返裁决。这是 G4 的真实验收门（offline fake 证机制、真跑证真实裁判受得住诱导）。

## Task 7: evergreen 同步 + Decision Log

**Files:**
- Modify: `design_docs/01-engine-mechanism.md`、`design_docs/02-architecture.md`、`design_docs/uml/{02-class,03-sequence}.md`

- [ ] **Step 1:** `01`/`02` 在 roster/叶子契约处补"只读叶约定"（只读 = deny-write permission + `needs_execution=False`）；`uml/02-class.md` 补 `read_only_leaf`/`read_only_builder`。
- [ ] **Step 2: Decision Log 增 D-G4**（下方）。commit。

## Task 8: 质量闸门

- [ ] `uv run pytest -q` 全绿；`ruff`/`ruff format --check`/`pyright`/`lint-imports` 全过；（如有配置调整）commit。

---

## Decision Log（本计划新增）

| # | 决策 | 选择与理由 |
|---|---|---|
| **D-G4** | read-only judge 形态 | **库级辅助 + deny-write permission + `needs_execution=False`**，否决"引擎内建只读 agentType"：① 引擎不构造叶子（宿主构造），只读是工具面属性，归宿主侧；② deepagents 无 execute 权限维度，靠"不分配 sandbox"禁执行，靠 `FilesystemPermission(write, deny)` 禁写，二者叠加才是真只读；③ builder 形态复用 G1，使只读 + 结构化裁决（adversarial-verify 标准形态）一行可得。 |

## 待核实（已全部解决）

1. ✅ deepagents `FilesystemPermission` 只读构造 API —— 已证（见上表）。
2. ✅ 只读是否需连带禁 execute —— 是；靠 `needs_execution=False`（无 execute 工具）+ permissions 不兼容 sandbox backend 的约束共同保证。
3. 与 G3 judge 模式示例的合并点 —— 09 示例的 judge 改用 `read_only_builder`（Task 5），避免重复装配。

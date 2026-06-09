# M7 · H — 拓扑序 fan-out + 深层命名嵌套 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给编排引擎补最后两条机制残项——一个依赖序（拓扑/偏序）fan-out 原语 `ctx.dag`，以及把 `ctx.workflow()` 命名嵌套从硬性 1 层放开到「可配置深度上限 + name-stack 环检测」的 N 层——并顺带落 `ctx.loop_until` 测量停止循环 helper（G）与 SKILL.md 作者范式（M1.5）。

**Architecture:** `ctx.dag` 是第四个 fan-out frame，与 `parallel`/`pipeline`/`race` 同源：进 `_FANOUT_DEPTH`、leaf 走 content-hash journal、gate 只在 leaf 获取、leaf 失败落 `None`、引擎控制流信号 fail-loud。调度核在新模块 `_dag.py`（镜像 `_pipeline.py`），用 Kahn 入度做拓扑调度、节点就绪即启、失败沿依赖树传递式跳过。DAG 结构由脚本 `deps` 确定 → 无需 DAG 级 journal，可恢复性整套继承（leaf 逐个零成本重放）。嵌套放开改 `ctx.workflow`：新增 `_WORKFLOW_NAME_STACK` ContextVar 做环检测、用 per-run 可配置 `max_workflow_depth`（默认 8）替换硬性 `>= 1`。

**Tech Stack:** Python 3.12（async-first、`async def f[T]` 泛型语法、`dataclass(slots=True)`、`contextvars`）；asyncio（`create_task` / `wait(FIRST_COMPLETED)`）；pytest + pytest-asyncio（`asyncio_mode = "auto"`）；ruff + pyright strict。

**验收头条（#10）：** 文档生成——package→module→symbol 拓扑序 **+** 命名子 workflow 嵌套 >1 层，端到端收口需求①。

---

## File Structure（decomposition 锁定）

**新建：**
- `src/langchain_dynamic_workflow/_dag.py` — `DagNode` value 类型 + `run_dag(nodes)` 拓扑调度器 + `_validate_dag(nodes)`（dup id / unknown dep / cycle → `WorkflowDagError`）。
- `tests/unit/test_dag.py` — `ctx.dag` + `run_dag` 单元测试。
- `tests/unit/test_loop_until.py` — `ctx.loop_until` 单元测试。
- `tests/integration/test_dag_doc_generation.py` — #10 拓扑序文档生成 + >1 层嵌套集成测试。
- `examples/features/dag.py` — 离线 feature demo（doc-gen 拓扑扇出 + 传递式跳过）。

**修改：**
- `src/langchain_dynamic_workflow/_errors.py` — 加 `WorkflowDagError` / `WorkflowCycleError`；扩 `WORKFLOW_CONTROL_FLOW_SIGNALS`。
- `src/langchain_dynamic_workflow/_observability.py` — `SpanKind` 加 `DAG`。
- `src/langchain_dynamic_workflow/_context.py` — 加 `DEFAULT_MAX_WORKFLOW_DEPTH` / `_WORKFLOW_NAME_STACK`；`Ctx.__init__` 加 `max_workflow_depth`；改 `Ctx.workflow`；加 `Ctx.dag` / `Ctx.loop_until`。
- `src/langchain_dynamic_workflow/_codegen.py` — 加 `_SCRIPT_DAG_API`（注入 `DagNode`）。
- `src/langchain_dynamic_workflow/_engine.py` — `run_workflow` 加 `max_workflow_depth` 并接线进 `Ctx`。
- `src/langchain_dynamic_workflow/__init__.py` — 导出 `DagNode` / `WorkflowDagError` / `WorkflowCycleError`。
- `tests/unit/test_nesting.py` — 重写（N 层 + cap + 环）。
- `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` — `ctx.dag` / `ctx.loop_until` DSL + 嵌套更新 + M1.5 作者范式段。
- `examples/features/nesting.py` — 演示 >1 层嵌套 + 环守卫。
- `examples/AGENTS.md`（§2 表）+ 根 `README.md` — 加 dag demo、更新 nesting 行。
- `design_docs/01-engine-mechanism.md` / `02-architecture.md` / `uml/*` / `v0_3_0_plans/00-roadmap.md` — evergreen 同步 + M7 状态翻牌。

**关键不变量（贯穿所有 task）：**
- gate 只在 leaf（`ctx.agent`）获取，DAG 调度层**绝不**持 slot——否则 dag-in-parallel 嵌套会饿死池死锁（见 `parallel` docstring「Gating only at the leaf」）。
- `ctx.dag` 是 fan-out frame：内部 `agent()` 在 `_FANOUT_DEPTH > 0`，**不**进 determinism sequence guard（完成序是 wall-clock）；只靠 content-hash journal 守。
- 「节点失败」与「节点合法返回 None」必须区分：调度器用独立 `failed: set[str]` 跟踪失败/跳过，`results` 里 None 同时表示两者（对外沿用 `parallel`/`pipeline` 约定），但只有 `failed` 集触发传递式跳过。

---

## Task 1: 新增 `WorkflowDagError` / `WorkflowCycleError` 错误类

**Files:**
- Modify: `src/langchain_dynamic_workflow/_errors.py`
- Test: `tests/unit/test_errors.py`（若不存在则新建）

- [ ] **Step 1: 写失败测试**

新建或追加 `tests/unit/test_errors.py`：

```python
"""Unit tests for the engine fail-loud exception family."""

from __future__ import annotations

from langchain_dynamic_workflow._errors import (
    WORKFLOW_CONTROL_FLOW_SIGNALS,
    WorkflowCycleError,
    WorkflowDagError,
)


def test_dag_and_cycle_errors_are_runtime_errors() -> None:
    assert issubclass(WorkflowDagError, RuntimeError)
    assert issubclass(WorkflowCycleError, RuntimeError)


def test_structural_violations_fail_loud_inside_fanout() -> None:
    # A malformed dag / a nesting cycle raised inside a parallel/pipeline/race frame
    # must NOT be masked as a None hole — it is an author bug, not a leaf failure.
    assert WorkflowDagError in WORKFLOW_CONTROL_FLOW_SIGNALS
    assert WorkflowCycleError in WORKFLOW_CONTROL_FLOW_SIGNALS
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_errors.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL（ImportError: cannot import name 'WorkflowDagError'）

- [ ] **Step 3: 实现错误类**

在 `_errors.py` 的 `WorkflowScriptError`（line 91 处类）之后、`WORKFLOW_CONTROL_FLOW_SIGNALS`（line 110）之前插入：

```python
class WorkflowDagError(RuntimeError):
    """Raised when a ``ctx.dag`` call is structurally invalid before scheduling.

    The DAG is validated eagerly at the top of ``ctx.dag`` — before any node runs —
    so a duplicate node id, a dependency on an unknown id, a node depending on
    itself, or a dependency cycle fails loud rather than scheduling a graph with no
    topological order. It is a control-flow signal: raised from inside a
    ``parallel`` / ``pipeline`` / ``race`` frame (a nested ``dag``) it must surface,
    never be masked as a ``None`` hole, because it is an author bug, not a leaf
    failure.
    """


class WorkflowCycleError(RuntimeError):
    """Raised when ``ctx.workflow`` would re-enter a workflow already being inlined.

    A workflow may inline other workflows up to ``max_workflow_depth`` levels, but a
    name that is already on the inlining stack (a workflow calling itself directly,
    or a mutual cycle such as A→B→A) has no engine-bounded base case and would
    recurse to the depth cap on every run. The engine refuses the cycle the moment a
    repeated name is seen, with a clearer diagnostic than the eventual depth-cap
    breach. Like the other structural signals it fails loud inside a fan-out frame.
    """
```

- [ ] **Step 4: 扩 `WORKFLOW_CONTROL_FLOW_SIGNALS`**

把 `_errors.py` 的（line 110-114）：

```python
WORKFLOW_CONTROL_FLOW_SIGNALS: tuple[type[Exception], ...] = (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowCheckpointError,
)
```

改为：

```python
WORKFLOW_CONTROL_FLOW_SIGNALS: tuple[type[Exception], ...] = (
    WorkflowBudgetExceededError,
    WorkflowDeterminismError,
    WorkflowCheckpointError,
    WorkflowDagError,
    WorkflowCycleError,
)
```

> 注：`WorkflowNestingError`（深度 cap）保持**不**入此元组——它与既有行为一致，且嵌套深度超限几乎只在 sequential 路径触发；环检测（`WorkflowCycleError`）是新增的、更需要在 fan-out 内 fail-loud 的结构错误。这条边界留给 Task 12 的 Codex 评审复核。

- [ ] **Step 5: 跑测试确认通过 + 格式/类型检查**

Run: `uv run pytest tests/unit/test_errors.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: PASS
Run: `uv run ruff check src/langchain_dynamic_workflow/_errors.py && uv run ruff format --check src/langchain_dynamic_workflow/_errors.py`

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_errors.py tests/unit/test_errors.py
git commit -m "feat(errors): WorkflowDagError + WorkflowCycleError (M7)"
```

---

## Task 2: `DagNode` value 类型 + `_validate_dag` 校验

**Files:**
- Create: `src/langchain_dynamic_workflow/_dag.py`
- Test: `tests/unit/test_dag.py`

- [ ] **Step 1: 写失败测试（仅校验部分）**

新建 `tests/unit/test_dag.py`：

```python
"""Unit tests for the DAG topological-order fan-out scheduler and its value type."""

from __future__ import annotations

import pytest

from langchain_dynamic_workflow._dag import DagNode, _validate_dag
from langchain_dynamic_workflow._errors import WorkflowDagError


def _node(node_id: str, deps: list[str]) -> DagNode:
    async def _run(d: dict[str, object]) -> str:
        return node_id

    return DagNode(node_id, deps=deps, run=_run)


def test_validate_accepts_a_well_formed_dag() -> None:
    _validate_dag([_node("a", []), _node("b", ["a"]), _node("c", ["a", "b"])])


def test_validate_rejects_duplicate_ids() -> None:
    with pytest.raises(WorkflowDagError, match="duplicate"):
        _validate_dag([_node("a", []), _node("a", [])])


def test_validate_rejects_unknown_dependency() -> None:
    with pytest.raises(WorkflowDagError, match="unknown"):
        _validate_dag([_node("a", ["ghost"])])


def test_validate_rejects_self_dependency() -> None:
    with pytest.raises(WorkflowDagError, match="itself"):
        _validate_dag([_node("a", ["a"])])


def test_validate_rejects_a_cycle() -> None:
    with pytest.raises(WorkflowDagError, match="cycle"):
        _validate_dag([_node("a", ["b"]), _node("b", ["a"])])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_dag.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL（No module named '..._dag'）

- [ ] **Step 3: 创建 `_dag.py` 的 `DagNode` + `_validate_dag`**

新建 `src/langchain_dynamic_workflow/_dag.py`：

```python
"""Dependency-order (topological) fan-out scheduler — the fourth fan-out frame.

LangGraph offers no partial-order fan-out: ``Send`` is a flat map-reduce barrier and
the engine's own ``parallel`` (barrier) / ``pipeline`` (linear stages) carry no
predecessor model. ``ctx.dag`` adds one: given nodes with declared dependencies, a
node runs only once all its predecessors have completed, and receives their results.
Ready nodes run concurrently; a node whose predecessor failed is skipped, propagating
along the dependency tree.

Mechanics:

- The graph is validated eagerly (duplicate id, unknown dependency, self-dependency,
  cycle) before any node runs, so a graph with no topological order fails loud up
  front rather than deadlocking the scheduler.
- The scheduler tracks in-degrees (Kahn). A node is launched the instant its last
  predecessor settles — there is no level barrier, so a fast branch races ahead of a
  slow sibling branch (the no-barrier spirit of ``pipeline``).
- Concurrency is bounded at the leaf: the ``agent()`` calls inside a node's ``run``
  acquire the shared gate. The scheduler itself holds no slot, so a ``dag`` nested
  inside a node's ``run`` cannot starve the pool into deadlock.
- A node whose ``run`` raises an ordinary exception is recorded as *failed* and its
  result is ``None``; every node that (transitively) depends on it is *skipped* and
  also lands as ``None``. A node that legitimately returns ``None`` is NOT failed, so
  its dependents still run (seeing ``None`` for that predecessor). Failure is tracked
  separately from the ``None`` result value precisely to keep this distinction.
- An engine control-flow signal (budget / determinism / a malformed nested dag) is
  never masked: the first one is recorded, no further nodes are launched, in-flight
  nodes drain, and it is re-raised after a clean teardown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ._errors import WORKFLOW_CONTROL_FLOW_SIGNALS, WorkflowDagError


@dataclass(slots=True)
class DagNode:
    """One node in a ``ctx.dag`` dependency graph.

    All three fields are required; a root node is written with an explicit empty
    ``deps`` (``DagNode("pkg", deps=[], run=...)``), matching how the graph reads.

    Attributes:
        id: The node's unique identifier within the graph; results are keyed by it.
        deps: The ids this node depends on. The node runs only after all of them have
            settled, and its ``run`` receives their results.
        run: A callable taking a ``{dep_id: result}`` mapping of this node's
            predecessors' results and returning an awaitable of the node's result
            (typically a closure over an ``agent()`` call).
    """

    id: str
    deps: Sequence[str]
    run: Callable[[dict[str, Any]], Awaitable[Any]]


def _validate_dag(nodes: Sequence[DagNode]) -> None:
    """Reject a structurally invalid graph before any node is scheduled.

    Args:
        nodes: The graph's nodes.

    Raises:
        WorkflowDagError: On a duplicate id, a dependency on an unknown id, a node
            depending on itself, or a dependency cycle.
    """
    ids = [node.id for node in nodes]
    id_set = set(ids)
    if len(id_set) != len(ids):
        dups = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        raise WorkflowDagError(f"duplicate DagNode id(s): {dups}")
    for node in nodes:
        if node.id in node.deps:
            raise WorkflowDagError(f"DagNode {node.id!r} depends on itself")
        unknown = sorted({dep for dep in node.deps if dep not in id_set})
        if unknown:
            raise WorkflowDagError(
                f"DagNode {node.id!r} depends on unknown id(s) {unknown}"
            )
    # Kahn pass: if not every node can be drained, the remainder forms a cycle.
    indegree = {node.id: len(set(node.deps)) for node in nodes}
    dependents: dict[str, list[str]] = {node.id: [] for node in nodes}
    for node in nodes:
        for dep in set(node.deps):
            dependents[dep].append(node.id)
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    drained = 0
    while queue:
        node_id = queue.pop()
        drained += 1
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if drained != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise WorkflowDagError(f"DagNode dependency cycle among {cyclic}")
```

> `run` 的 `default_factory=lambda: _missing_run` 是为了让 `DagNode("id")` 也能构造而 pyright strict 不抱怨缺省——真实使用永远显式传 `run`；漏传则运行时 fail-loud。

- [ ] **Step 4: 跑测试确认通过 + 检查**

Run: `uv run pytest tests/unit/test_dag.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: PASS（5 个校验用例全绿）
Run: `uv run ruff check src/langchain_dynamic_workflow/_dag.py && uv run pyright src/langchain_dynamic_workflow/_dag.py`

- [ ] **Step 5: Commit**

```bash
git add src/langchain_dynamic_workflow/_dag.py tests/unit/test_dag.py
git commit -m "feat(dag): DagNode value type + eager graph validation (M7)"
```

---

## Task 3: `run_dag` 拓扑调度器（就绪即启 + 传递式跳过 + 控制流 fail-loud）

**Files:**
- Modify: `src/langchain_dynamic_workflow/_dag.py`
- Test: `tests/unit/test_dag.py`

- [ ] **Step 1: 写失败测试（调度语义）**

追加到 `tests/unit/test_dag.py`：

```python
import asyncio

from langchain_dynamic_workflow._dag import run_dag


def _leaf_node(node_id: str, deps: list[str], *, fail: bool = False) -> DagNode:
    # A node whose result records the deps it actually received, so a test can assert
    # topological data-flow. `fail=True` raises to exercise failure isolation.
    async def _run(d: dict[str, object]) -> dict[str, object]:
        if fail:
            raise RuntimeError(f"{node_id} boom")
        return {"id": node_id, "saw": dict(d)}

    return DagNode(node_id, deps=deps, run=_run)


async def test_run_dag_threads_predecessor_results_in_topo_order() -> None:
    results = await run_dag(
        [
            _leaf_node("pkg", []),
            _leaf_node("modA", ["pkg"]),
            _leaf_node("symA", ["modA"]),
        ]
    )
    # symA saw modA's result; modA saw pkg's result -> topological data-flow.
    assert results["symA"]["saw"]["modA"]["id"] == "modA"
    assert results["modA"]["saw"]["pkg"]["id"] == "pkg"


async def test_run_dag_failure_skips_dependents_transitively() -> None:
    results = await run_dag(
        [
            _leaf_node("pkg", [], fail=True),
            _leaf_node("modA", ["pkg"]),  # depends on failed pkg -> skipped
            _leaf_node("symA", ["modA"]),  # depends on skipped modA -> skipped
            _leaf_node("indep", []),  # independent -> survives
        ]
    )
    assert results["pkg"] is None
    assert results["modA"] is None
    assert results["symA"] is None
    assert results["indep"]["id"] == "indep"


async def test_run_dag_legitimate_none_does_not_skip_dependents() -> None:
    async def _returns_none(_d: dict[str, object]) -> None:
        return None

    async def _child(d: dict[str, object]) -> dict[str, object]:
        return {"saw_keys": sorted(d)}

    results = await run_dag(
        [
            DagNode("root", deps=[], run=_returns_none),
            DagNode("child", deps=["root"], run=_child),
        ]
    )
    # root returned None *legitimately* (it did not raise) -> child still runs.
    assert results["root"] is None
    assert results["child"] == {"saw_keys": ["root"]}


async def test_run_dag_empty_returns_empty_dict() -> None:
    assert await run_dag([]) == {}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_dag.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL（cannot import name 'run_dag'）

- [ ] **Step 3: 实现 `run_dag`**

追加到 `src/langchain_dynamic_workflow/_dag.py` 末尾：

```python
async def run_dag(nodes: Sequence[DagNode]) -> dict[str, Any | None]:
    """Run a dependency graph in topological order and collect per-node results.

    Args:
        nodes: The graph's nodes; each runs after all its ``deps`` have settled and
            receives a ``{dep_id: result}`` mapping of their results.

    Returns:
        A mapping ``{node_id: result}``. A node whose ``run`` raised, or that was
        skipped because a predecessor failed, maps to ``None``.

    Raises:
        WorkflowDagError: If the graph is structurally invalid (validated up front).
        Exception: The first engine control-flow signal raised by any node's ``run``
            (budget / determinism / a malformed nested dag), re-raised after the
            in-flight nodes drain.
    """
    _validate_dag(nodes)
    if not nodes:
        return {}

    by_id = {node.id: node for node in nodes}
    indegree = {node.id: len(set(node.deps)) for node in nodes}
    dependents: dict[str, list[str]] = {node.id: [] for node in nodes}
    for node in nodes:
        for dep in set(node.deps):
            dependents[dep].append(node.id)

    results: dict[str, Any | None] = {node.id: None for node in nodes}
    failed: set[str] = set()
    aborted: list[BaseException] = []
    running: dict[asyncio.Task[Any], str] = {}

    def _release(node_id: str) -> list[str]:
        """Decrement dependents' in-degree; return those that just became ready."""
        newly_ready: list[str] = []
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        return newly_ready

    def _start(node_id: str) -> list[str]:
        """Launch a ready node, or skip it; return nodes that became ready as a result.

        A node is skipped (without running) when the run is already aborting or when
        any of its predecessors failed/were skipped — the skip cascades to its own
        dependents.
        """
        node = by_id[node_id]
        if aborted or any(dep in failed for dep in node.deps):
            failed.add(node_id)
            return _release(node_id)
        deps_view = {dep: results[dep] for dep in node.deps}
        running[asyncio.create_task(node.run(deps_view))] = node_id
        return []

    pending = [node_id for node_id, degree in indegree.items() if degree == 0]
    try:
        while pending or running:
            while pending:
                pending.extend(_start(pending.pop()))
            if not running:
                break
            done, _ = await asyncio.wait(
                list(running), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                node_id = running.pop(task)
                error = task.exception()
                if error is not None:
                    if isinstance(error, WORKFLOW_CONTROL_FLOW_SIGNALS) and not aborted:
                        aborted.append(error)
                    failed.add(node_id)
                else:
                    results[node_id] = task.result()
                pending.extend(_release(node_id))
    finally:
        # Defensive: never leak tasks if the loop unwinds unexpectedly.
        for task in running:
            if not task.done():
                task.cancel()

    if aborted:
        raise aborted[0]
    return results
```

- [ ] **Step 4: 写控制流 fail-loud 测试**

追加到 `tests/unit/test_dag.py`：

```python
from langchain_dynamic_workflow._errors import WorkflowBudgetExceededError


async def test_run_dag_control_flow_signal_fails_loud_not_masked() -> None:
    async def _budget_breach(_d: dict[str, object]) -> object:
        raise WorkflowBudgetExceededError("pool exhausted")

    async def _other(_d: dict[str, object]) -> str:
        return "ok"

    with pytest.raises(WorkflowBudgetExceededError):
        await run_dag(
            [DagNode("a", deps=[], run=_budget_breach), DagNode("b", deps=[], run=_other)]
        )
```

- [ ] **Step 5: 跑全部 dag 单测 + 检查**

Run: `uv run pytest tests/unit/test_dag.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS（全部）
Run: `uv run ruff check src/langchain_dynamic_workflow/_dag.py && uv run pyright src/langchain_dynamic_workflow/_dag.py`

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_dag.py tests/unit/test_dag.py
git commit -m "feat(dag): run_dag topological scheduler — ready-on-settle + transitive skip (M7)"
```

---

## Task 4: `SpanKind.DAG` + `Ctx.dag` 薄包装

**Files:**
- Modify: `src/langchain_dynamic_workflow/_observability.py`（`SpanKind` 枚举，line 39-52）
- Modify: `src/langchain_dynamic_workflow/_context.py`（import `_dag`；在 `pipeline` 之后加 `dag`）
- Test: `tests/unit/test_dag.py`

- [ ] **Step 1: 写失败测试（经 `Ctx.dag` 走完整路径 + span）**

追加到 `tests/unit/test_dag.py`（复用 `test_context_pipeline.py` 的 `_CountingLeaf`/`_ctx` 模式）：

```python
from typing import Any

from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._observability import SpanKind, SpanRecorder
from langchain_dynamic_workflow._roster import Roster


class _CountingLeaf:
    def __init__(self, *, prefix: str) -> None:
        self.calls = 0
        self.prefix = prefix

    async def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
    ) -> LeafOutcome:
        self.calls += 1
        return LeafOutcome(
            state={"messages": [AIMessage(content=f"{self.prefix}:{prompt}")]}, usage=0
        )


def _dag_ctx(leaf: _CountingLeaf, journal: InMemoryJournalStore, spans: SpanRecorder) -> Ctx:
    roster = Roster()
    roster.register("worker", object())  # type: ignore[arg-type]
    return Ctx(
        roster=roster,
        journal=journal,
        leaf_runner=leaf,
        gate=ConcurrencyGate(limit=8),
        spans=spans,
    )


async def test_ctx_dag_runs_topologically_and_emits_a_dag_span() -> None:
    leaf = _CountingLeaf(prefix="R")
    spans = SpanRecorder()
    ctx = _dag_ctx(leaf, InMemoryJournalStore(), spans)

    results = await ctx.dag(
        [
            DagNode("pkg", deps=[], run=lambda d: ctx.agent("doc package", agent_type="worker")),
            DagNode(
                "modA",
                deps=["pkg"],
                run=lambda d: ctx.agent(f"doc A | {d['pkg']}", agent_type="worker"),
            ),
        ]
    )
    assert results["pkg"] == "R:doc package"
    assert results["modA"] == "R:doc A | R:doc package"
    assert leaf.calls == 2
    dag_spans = [s for s in spans.completed if s.kind is SpanKind.DAG]
    assert len(dag_spans) == 1
    assert dag_spans[0].attributes["node_count"] == 2
    assert dag_spans[0].attributes["surviving_count"] == 2
```

> 若 `SpanRecorder` 暴露已完成 span 的属性名不是 `completed` / `attributes`，按 `_observability.py` 的实际 API 调整断言（读该文件确认）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_dag.py::test_ctx_dag_runs_topologically_and_emits_a_dag_span -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL（`SpanKind` 无 `DAG` / `Ctx` 无 `dag`）

- [ ] **Step 3: 加 `SpanKind.DAG`**

在 `_observability.py` 的 `SpanKind` 枚举里，`RACE = "race"`（line 52）之后加一行：

```python
    DAG = "dag"
```

- [ ] **Step 4: 实现 `Ctx.dag`**

在 `_context.py` 顶部 import 区加（与 `_pipeline` 的 `Stage` import 同处）：

```python
from ._dag import DagNode, run_dag
```

在 `Ctx.pipeline`（结束于 line 802）之后插入 `dag` 方法，结构镜像 `pipeline`：

```python
    async def dag(self, nodes: Sequence[DagNode]) -> dict[str, Any | None]:
        """Run a dependency graph in topological order; a node runs after its deps.

        Each :class:`~langchain_dynamic_workflow._dag.DagNode` declares an ``id``, its
        ``deps`` (ids it depends on), and a ``run(deps)`` callable that receives a
        ``{dep_id: result}`` mapping of its predecessors' results and typically calls
        ``agent()``. Ready nodes (all deps settled) run concurrently; there is no
        level barrier, so an independent branch races ahead of a slow one. A node
        whose ``run`` raises lands as ``None`` and every node that depends on it
        (transitively) is skipped to ``None``; a node that legitimately returns
        ``None`` does not skip its dependents. Engine control-flow signals
        (budget / determinism / a malformed graph) are re-raised loud after the
        in-flight nodes drain, never masked as a ``None`` hole.

        Concurrency is bounded at the leaf: the ``agent()`` calls inside a node's
        ``run`` acquire the shared gate, while this fan-out layer holds no slot — so a
        ``dag`` nested inside a node cannot starve the pool. Like ``parallel`` /
        ``pipeline`` / ``race``, the leaves inside the nodes are excluded from the
        determinism sequence guard (their completion order is wall-clock dependent);
        the content-hash journal still guards each by its inputs, so a resume replays
        completed nodes at zero model cost — the graph structure is script-defined and
        therefore deterministic, needing no dag-level journal entry of its own.

        Args:
            nodes: The graph's nodes.

        Returns:
            A mapping ``{node_id: result}``; a failed or skipped node maps to ``None``.

        Raises:
            WorkflowDagError: If the graph is structurally invalid (duplicate id,
                unknown / self dependency, or a cycle).
        """
        with self._spans.span(SpanKind.DAG, "dag") as span:
            span.set("node_count", len(nodes))
            if not nodes:
                span.set("surviving_count", 0)
                return {}
            # Mark the fan-out frame so agent() calls inside the nodes skip the
            # determinism backstop; set before run_dag spawns node tasks so each
            # child task inherits the depth.
            token = _FANOUT_DEPTH.set(_FANOUT_DEPTH.get() + 1)
            try:
                results = await run_dag(nodes)
            finally:
                _FANOUT_DEPTH.reset(token)
            span.set("surviving_count", sum(1 for v in results.values() if v is not None))
            return results
```

- [ ] **Step 5: 跑测试确认通过 + 检查**

Run: `uv run pytest tests/unit/test_dag.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS
Run: `uv run ruff check src/langchain_dynamic_workflow/_context.py src/langchain_dynamic_workflow/_observability.py && uv run pyright src/langchain_dynamic_workflow/_context.py`

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_context.py src/langchain_dynamic_workflow/_observability.py tests/unit/test_dag.py
git commit -m "feat(dag): ctx.dag fan-out frame + SpanKind.DAG (M7)"
```

---

## Task 5: 包根导出 `DagNode` + `run_script` 命名空间注入

**Files:**
- Modify: `src/langchain_dynamic_workflow/_codegen.py`（line 100-141）
- Modify: `src/langchain_dynamic_workflow/__init__.py`（line 53 区 import；line 110+ `__all__`）
- Test: `tests/unit/test_codegen.py`（若存在则追加，否则在 `tests/unit/test_dag.py` 加一个注入断言）

- [ ] **Step 1: 写失败测试（authored script 能按名构造 `DagNode`）**

追加到 `tests/unit/test_dag.py`：

```python
from langchain_dynamic_workflow._codegen import compile_workflow_source


def test_authored_script_can_construct_dagnode_without_import() -> None:
    # DagNode is injected into the run_script namespace like RaceCandidate, so an
    # authored script references it by name (the AST gate forbids imports).
    source = (
        "async def orchestrate(ctx, args):\n"
        "    nodes = [DagNode('a', deps=[], run=lambda d: ctx.agent('q', agent_type='w'))]\n"
        "    return await ctx.dag(nodes)\n"
    )
    orchestrate = compile_workflow_source(source)
    assert callable(orchestrate)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_dag.py::test_authored_script_can_construct_dagnode_without_import -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL（`NameError: name 'DagNode' is not defined` 在 exec 时）

- [ ] **Step 3: 注入 `DagNode` 到脚本命名空间**

在 `_codegen.py` import 区加（与 `RaceCandidate` import 同处）：

```python
from ._dag import DagNode
```

在 `_SCRIPT_RACE_API`（line 100-106）之后加：

```python
_SCRIPT_DAG_API: dict[str, Any] = {
    "DagNode": DagNode,
}
"""DAG value type injected as a script global so a host-authored script constructs
``DagNode`` specs by name without an import (the AST gate forbids imports). ``ctx.dag``
and ``ctx.loop_until`` are methods, so they need no injection — and unlike
``ctx.checkpoint`` they are NOT on the AST gate's denylist (they are ordinary
orchestration primitives, like ``parallel`` / ``pipeline`` / ``race``)."""
```

把 namespace 组装（line 137-141）改为：

```python
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        **_SCRIPT_REDUCE_API,
        **_SCRIPT_RACE_API,
        **_SCRIPT_DAG_API,
    }
```

- [ ] **Step 4: 包根导出 `DagNode`**

在 `__init__.py` 加 import（line 52-53 区，`_pull_request` 与 `_race_types` 之间，按模块名 ASCII 序 `_dag` 应在更前——放到 `_context` import 之后即可）：

```python
from ._dag import DagNode
```

把 `from ._errors import (...)` 那组扩入 `WorkflowCycleError` 与 `WorkflowDagError`（按字母序）。

在 `__all__` 里加（按字母序）：`"DagNode"`、`"WorkflowCycleError"`、`"WorkflowDagError"`。`"DagNode"` 紧邻 `"Ctx"` 之后；`"WorkflowCycleError"` 在 `"WorkflowCheckpointError"` 之后、`"WorkflowDagError"` 在其后、`"WorkflowDeterminismError"` 之前。

- [ ] **Step 5: 跑测试 + 烟测包根导出**

Run: `uv run pytest tests/unit/test_dag.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Run: `uv run python -c "from langchain_dynamic_workflow import DagNode, WorkflowDagError, WorkflowCycleError; print('ok')"`
Expected: PASS + `ok`
Run: `uv run ruff check src/langchain_dynamic_workflow/_codegen.py src/langchain_dynamic_workflow/__init__.py && uv run pyright src/langchain_dynamic_workflow/__init__.py`

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_codegen.py src/langchain_dynamic_workflow/__init__.py tests/unit/test_dag.py
git commit -m "feat(dag): export DagNode + inject into run_script namespace (M7)"
```

---

## Task 6: 解除 `workflow()` 嵌套硬限 — `max_workflow_depth` + name-stack 环检测

**Files:**
- Modify: `src/langchain_dynamic_workflow/_context.py`（`_WORKFLOW_DEPTH` 区 line 181-194；`Ctx.__init__` line 231-270；`Ctx.workflow` line 319-360）
- Modify: `src/langchain_dynamic_workflow/_engine.py`（`run_workflow` 签名 line 63-82；`Ctx(...)` 构造 line 393-404）
- Modify: `src/langchain_dynamic_workflow/_errors.py`（`WorkflowNestingError` docstring，line 34）
- Test: `tests/unit/test_nesting.py`（重写）

- [ ] **Step 1: 重写 `test_nesting.py`**

整体替换 `tests/unit/test_nesting.py`：

```python
"""Unit tests for ``ctx.workflow`` N-level nesting with a depth cap + cycle guard.

A workflow may inline other workflows up to ``max_workflow_depth`` levels; exceeding
the cap raises ``WorkflowNestingError`` and re-entering a name already on the inlining
stack (a cycle) raises ``WorkflowCycleError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.runnables import Runnable

from langchain_dynamic_workflow import Ctx, Roster, run_workflow
from langchain_dynamic_workflow._errors import WorkflowCycleError, WorkflowNestingError
from langchain_dynamic_workflow._workflows import WorkflowRegistry

FakeLeafFactory = Callable[..., tuple[Runnable[Any, Any], Any]]


async def test_two_level_nesting_now_succeeds(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("deep-finding")
    roster = Roster().register("researcher", leaf)

    async def inner(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("leaf", agent_type="researcher")

    async def middle(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow("inner", {})

    workflows = WorkflowRegistry().register("inner", inner).register("middle", middle)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("middle", {})  # depth 1 -> 2: previously refused

    result = await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1")
    assert result == "deep-finding"


async def test_exceeding_max_workflow_depth_raises(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def step(ctx: Ctx, args: dict[str, Any]) -> str:
        # Each level inlines a DISTINCT next level so the cap (not the cycle guard)
        # is what fires. With max_workflow_depth=2, depth 0 -> w1 -> w2 -> w3 breaches.
        nxt = args["next"]
        if nxt is None:
            return await ctx.agent("leaf", agent_type="researcher")
        return await ctx.workflow(nxt, {"next": args["then"], "then": None})

    workflows = (
        WorkflowRegistry().register("w1", step).register("w2", step).register("w3", step)
    )

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("w1", {"next": "w2", "then": "w3"})

    with pytest.raises(WorkflowNestingError):
        await run_workflow(
            outer, roster=roster, workflows=workflows, thread_id="t1", max_workflow_depth=2
        )


async def test_name_cycle_raises_cycle_error(make_fake_leaf: FakeLeafFactory) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def selfish(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow("selfish", {})  # re-enters itself -> cycle

    workflows = WorkflowRegistry().register("selfish", selfish)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("selfish", {})

    with pytest.raises(WorkflowCycleError):
        await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1")


async def test_deep_nesting_shares_parent_budget(
    make_usage_leaf: Callable[..., tuple[Runnable[Any, Any], Any]],
) -> None:
    leaf, _model = make_usage_leaf("ok", tokens_per_call=10)
    roster = Roster().register("researcher", leaf)
    spent: dict[str, float] = {}

    async def inner(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent("Q", agent_type="researcher")

    async def middle(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow("inner", {})

    workflows = WorkflowRegistry().register("inner", inner).register("middle", middle)

    async def outer(ctx: Ctx) -> str:
        out = await ctx.workflow("middle", {})
        spent["after"] = ctx.budget.spent()
        return out

    await run_workflow(outer, roster=roster, workflows=workflows, thread_id="t1", budget=1000)
    assert spent["after"] == 10  # inner leaf's tokens visible on the parent budget


async def test_workflow_without_registry_raises_lookuperror(
    make_fake_leaf: FakeLeafFactory,
) -> None:
    leaf, _state = make_fake_leaf("x")
    roster = Roster().register("researcher", leaf)

    async def outer(ctx: Ctx) -> str:
        return await ctx.workflow("missing", {})

    with pytest.raises(LookupError):
        await run_workflow(outer, roster=roster, thread_id="t1")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_nesting.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: FAIL（`run_workflow` 无 `max_workflow_depth`；二级嵌套仍抛 `WorkflowNestingError`）

- [ ] **Step 3: 加 `DEFAULT_MAX_WORKFLOW_DEPTH` + `_WORKFLOW_NAME_STACK`**

在 `_context.py` 的 `_WORKFLOW_DEPTH`（line 181-194）之后加：

```python
DEFAULT_MAX_WORKFLOW_DEPTH = 8
"""Default cap on ``ctx.workflow()`` inline nesting depth.

The cap is a runaway-recursion backstop, not a semantic limit: a legitimate
composition nests a handful of levels, while an unbounded recursion (a missing base
case) trips this and fails loud. The cycle guard catches the common case (a name
re-entering itself) earlier and more precisely.
"""

_WORKFLOW_NAME_STACK: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "langchain_dynamic_workflow_workflow_name_stack", default=frozenset()
)
"""Names of the workflows currently inlined on the ``ctx.workflow`` call stack.

A ``ctx.workflow(name)`` whose ``name`` is already in this set is a cycle (a workflow
re-entering itself, directly or via a mutual A→B→A) and is refused. The variable is a
:class:`~contextvars.ContextVar` so the stack is isolated per asyncio task and
restored on frame exit, exactly like ``_WORKFLOW_DEPTH``.
"""
```

- [ ] **Step 4: `Ctx.__init__` 加 `max_workflow_depth` 形参**

在 `Ctx.__init__` 签名里（`pending_signoff: Any = UNSET,` 之后，line 243 区）加：

```python
        max_workflow_depth: int = DEFAULT_MAX_WORKFLOW_DEPTH,
```

并在 `__init__` body 末尾（`self._spans = ...` 之后，line 270 区）加：

```python
        self._max_workflow_depth = max_workflow_depth
```

在类 docstring 的 `Args:` 段补一行（紧跟 `pending_signoff` 描述之后）：

```
        max_workflow_depth: Cap on ``ctx.workflow`` inline nesting depth (a
            runaway-recursion backstop); defaults to ``DEFAULT_MAX_WORKFLOW_DEPTH``.
```

- [ ] **Step 5: 改 `Ctx.workflow` body**

把 `Ctx.workflow`（line 344-360）的实现段替换为：

```python
        if self._workflows is None:
            raise LookupError(
                f"cannot resolve workflow {name!r}: no workflow registry was wired "
                "into this run (pass workflows=... to run_workflow)"
            )
        stack = _WORKFLOW_NAME_STACK.get()
        if name in stack:
            raise WorkflowCycleError(
                f"cannot inline workflow {name!r}: it is already on the inlining stack "
                f"{sorted(stack)} — a workflow that re-enters itself (directly or via a "
                "cycle) has no engine-bounded base case; refusing the cycle"
            )
        if _WORKFLOW_DEPTH.get() >= self._max_workflow_depth:
            raise WorkflowNestingError(
                f"cannot inline workflow {name!r}: nesting depth would exceed "
                f"max_workflow_depth={self._max_workflow_depth} (runaway-recursion backstop)"
            )
        workflow_fn = self._workflows.resolve(name)  # KeyError on unknown name
        depth_token = _WORKFLOW_DEPTH.set(_WORKFLOW_DEPTH.get() + 1)
        stack_token = _WORKFLOW_NAME_STACK.set(stack | {name})
        try:
            return await workflow_fn(self, args or {})
        finally:
            _WORKFLOW_NAME_STACK.reset(stack_token)
            _WORKFLOW_DEPTH.reset(depth_token)
```

把 `Ctx.workflow` 的 docstring 顶句与 `Raises` 段更新（line 319-343）：把「exactly one level deep」改为「up to ``max_workflow_depth`` levels deep」，并在 `Raises:` 段加：

```
            WorkflowCycleError: If ``name`` is already on the inlining stack (a
                workflow re-entering itself, directly or via a mutual cycle).
```

`WorkflowNestingError` 的描述同步改为「beyond the configured ``max_workflow_depth``」。

在 `_context.py` import 区把 `from ._errors import (...)` 扩入 `WorkflowCycleError`（与既有 `WorkflowNestingError` / `WorkflowCheckpointError` 同组）。

- [ ] **Step 6: `_engine.py` 接线 `max_workflow_depth`**

在 `_engine.py` import 区把 `from ._context import (...)` 扩入 `DEFAULT_MAX_WORKFLOW_DEPTH`（与既有 `Ctx` / `UNSET` 同组）。

`run_workflow` 签名（line 81 `workflows: ... = None,` 之后）加：

```python
    max_workflow_depth: int = DEFAULT_MAX_WORKFLOW_DEPTH,
```

在 `run_workflow` docstring 的 `Args:` 段补一行说明（紧跟 `workflows` 描述之后）。

`Ctx(...)` 构造（line 393-404，`pending_signoff=resume,` 之后）加：

```python
            max_workflow_depth=max_workflow_depth,
```

更新 `_errors.py` 的 `WorkflowNestingError` docstring（line 34-43）：把「more than one level deep」「exactly one level」改述为「beyond the configured ``max_workflow_depth`` (a runaway-recursion backstop)」。

- [ ] **Step 7: 跑测试确认通过 + 检查**

Run: `uv run pytest tests/unit/test_nesting.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS（全部 5 用例）
Run: `uv run ruff check src/langchain_dynamic_workflow/_context.py src/langchain_dynamic_workflow/_engine.py src/langchain_dynamic_workflow/_errors.py && uv run pyright src/langchain_dynamic_workflow/_context.py src/langchain_dynamic_workflow/_engine.py`

- [ ] **Step 8: Commit**

```bash
git add src/langchain_dynamic_workflow/_context.py src/langchain_dynamic_workflow/_engine.py src/langchain_dynamic_workflow/_errors.py tests/unit/test_nesting.py
git commit -m "feat(nesting): lift workflow() 1-level cap -> max_workflow_depth + cycle guard (M7)"
```

---

## Task 7: `ctx.loop_until` 测量停止循环 helper（G）

**Files:**
- Modify: `src/langchain_dynamic_workflow/_context.py`（在 `Ctx.loop_until` 加；放在 `Ctx.dag` 之后）
- Test: `tests/unit/test_loop_until.py`

- [ ] **Step 1: 写失败测试**

新建 `tests/unit/test_loop_until.py`：

```python
"""Unit tests for ``ctx.loop_until`` — the measured-stop loop helper."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from langchain_dynamic_workflow._concurrency import ConcurrencyGate
from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._progress import ProgressKind, ProgressLog
from langchain_dynamic_workflow._roster import Roster


class _SeqLeaf:
    """Returns a scripted reply per call, so a stop predicate can be exercised."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    async def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
        isolation: str = "shared",
        leaf_span_id: str = "",
    ) -> LeafOutcome:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return LeafOutcome(state={"messages": [AIMessage(content=reply)]}, usage=0)


def _ctx(leaf: _SeqLeaf, progress: ProgressLog) -> Ctx:
    roster = Roster()
    roster.register("worker", object())  # type: ignore[arg-type]
    return Ctx(
        roster=roster,
        journal=InMemoryJournalStore(),
        leaf_runner=leaf,
        gate=ConcurrencyGate(limit=4),
        progress=progress,
    )


async def test_loop_until_stops_when_done_satisfied() -> None:
    leaf = _SeqLeaf(["no", "no", "STOP", "no"])
    delivered: list[Any] = []
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=delivered.append))

    async def body(i: int, acc: list[str]) -> str:
        # `acc` is the accumulated-so-far (dedup-against-all-seen); prompt varies by i.
        return await ctx.agent(f"try {i} (seen {len(acc)})", agent_type="worker")

    out = await ctx.loop_until(body, done=lambda acc: "STOP" in acc, max_iters=10)
    assert out == ["no", "no", "STOP"]  # stopped the iteration that produced STOP
    assert leaf.calls == 3


async def test_loop_until_caps_and_logs_without_convergence() -> None:
    leaf = _SeqLeaf(["no"])
    delivered: list[Any] = []
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=delivered.append))

    async def body(i: int, acc: list[str]) -> str:
        return await ctx.agent(f"try {i}", agent_type="worker")

    out = await ctx.loop_until(body, done=lambda acc: False, max_iters=3)
    assert out == ["no", "no", "no"]
    assert leaf.calls == 3
    # The cap-without-convergence is surfaced as a (replay-idempotent) log line.
    logs = [e for e in delivered if e.kind is ProgressKind.LOG and "max_iters" in e.text]
    assert len(logs) == 1


async def test_loop_until_rejects_nonpositive_max_iters() -> None:
    leaf = _SeqLeaf(["x"])
    ctx = _ctx(leaf, ProgressLog(delivered_count=0, sink=lambda _e: None))

    async def body(i: int, acc: list[str]) -> str:
        return await ctx.agent("x", agent_type="worker")

    with pytest.raises(ValueError, match="max_iters"):
        await ctx.loop_until(body, done=lambda acc: True, max_iters=0)
```

> 若 `ProgressEntry` 的字段名不是 `.kind` / `.text`，按 `_progress.py` 实际字段调整断言（读该文件确认 `ProgressEntry` 形态）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/unit/test_loop_until.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL（`Ctx` 无 `loop_until`）

- [ ] **Step 3: 实现 `Ctx.loop_until`**

在 `_context.py` 的 `Ctx.dag` 方法之后插入：

```python
    async def loop_until[T](
        self,
        body: Callable[[int, list[T]], Awaitable[T]],
        *,
        done: Callable[[list[T]], bool],
        max_iters: int,
    ) -> list[T]:
        """Run ``body`` until ``done`` holds over the accumulated results, capped at ``max_iters``.

        A measured-stop loop with the two author disciplines baked in: every loop has
        a mandatory hard cap (``max_iters``), and the stop predicate is checked over
        the FULL accumulated list (so dedup / convergence is against *everything* seen,
        not just the last round). Each iteration calls ``body(iter_index, accumulated)``
        — where ``accumulated`` is a copy of the results so far — appends its result,
        then checks ``done(accumulated)``; the loop returns as soon as ``done`` holds.

        This is a sequential (depth-0) primitive: ``body``'s direct ``agent()`` calls
        record into the determinism guard, and the loop count derives from journaled
        leaf results, so a resume reproduces the same number of iterations and replays
        completed leaves at zero cost. If the cap is reached without ``done`` ever
        holding, a (replay-idempotent) ``log`` line is emitted and the accumulated
        results are returned — a graceful, non-silent stop rather than a raise.

        Args:
            body: ``(iter_index, accumulated_so_far) -> result`` for one iteration.
            done: Stop predicate over the full accumulated result list.
            max_iters: Mandatory hard cap on iterations (must be >= 1).

        Returns:
            The accumulated results, in iteration order.

        Raises:
            ValueError: If ``max_iters`` is less than 1.
        """
        if max_iters < 1:
            raise ValueError(f"loop_until requires max_iters >= 1, got {max_iters}")
        accumulated: list[T] = []
        for iteration in range(max_iters):
            accumulated.append(await body(iteration, list(accumulated)))
            if done(accumulated):
                return accumulated
        self.log(f"loop_until reached max_iters={max_iters} without satisfying done()")
        return accumulated
```

- [ ] **Step 4: 跑测试确认通过 + 检查**

Run: `uv run pytest tests/unit/test_loop_until.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: PASS
Run: `uv run ruff check src/langchain_dynamic_workflow/_context.py && uv run pyright src/langchain_dynamic_workflow/_context.py`

- [ ] **Step 5: Commit**

```bash
git add src/langchain_dynamic_workflow/_context.py tests/unit/test_loop_until.py
git commit -m "feat(loop): ctx.loop_until measured-stop helper (M7 / G)"
```

---

## Task 8: 集成测试 — #10 拓扑序文档生成 + >1 层嵌套

**Files:**
- Create: `tests/integration/test_dag_doc_generation.py`

- [ ] **Step 1: 写集成测试**

新建 `tests/integration/test_dag_doc_generation.py`：

```python
"""Integration: ctx.dag topological doc-generation + a named sub-workflow nested >1 level.

The #10 acceptance shape, offline and deterministic: a package doc feeds its module
docs, which feed their symbol docs (package -> module -> symbol topological order), and
the per-symbol step is delegated to a registered sub-workflow inlined two levels deep —
closing requirement ① (dependency-order fan-out + named nesting beyond one level) end
to end. Fake leaves echo their prompt so the assertion can prove the data actually
flowed along the dependency edges and that the nested workflow ran.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from langchain_dynamic_workflow import Ctx, DagNode, Roster, run_workflow
from langchain_dynamic_workflow._workflows import WorkflowRegistry


def _echo_leaf() -> RunnableLambda[dict[str, Any], dict[str, Any]]:
    async def _leaf(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content)
        return {"messages": [*inp["messages"], AIMessage(content=f"DOC[{prompt}]")]}

    return RunnableLambda(_leaf)


async def test_dag_doc_generation_topological_with_nested_workflow() -> None:
    roster = Roster().register("documenter", _echo_leaf())

    # A registered sub-workflow that documents one symbol; inlined two levels deep
    # (orchestrate -> document_module(workflow) -> document_symbol(workflow)).
    async def document_symbol(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.agent(
            f"symbol {args['symbol']} of {args['module_doc']}", agent_type="documenter"
        )

    async def document_module(ctx: Ctx, args: dict[str, Any]) -> str:
        return await ctx.workflow(
            "document_symbol", {"symbol": args["symbol"], "module_doc": args["module_doc"]}
        )

    workflows = (
        WorkflowRegistry()
        .register("document_symbol", document_symbol)
        .register("document_module", document_module)
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any | None]:
        results = await ctx.dag(
            [
                DagNode(
                    "pkg",
                    deps=[],
                    run=lambda d: ctx.agent("package mypkg", agent_type="documenter"),
                ),
                DagNode(
                    "mod_io",
                    deps=["pkg"],
                    run=lambda d: ctx.agent(f"module io | {d['pkg']}", agent_type="documenter"),
                ),
                # symbol node delegates to a sub-workflow nested two levels deep.
                DagNode(
                    "sym_read",
                    deps=["mod_io"],
                    run=lambda d: ctx.workflow(
                        "document_module", {"symbol": "read", "module_doc": d["mod_io"]}
                    ),
                ),
            ]
        )
        return results

    result = await run_workflow(orchestrate, roster=roster, workflows=workflows, thread_id="t1")

    # Topological data-flow: each level's prompt embedded its predecessor's doc.
    assert result["pkg"] == "DOC[package mypkg]"
    assert result["mod_io"] == "DOC[module io | DOC[package mypkg]]"
    # sym_read ran through document_module -> document_symbol (nesting depth 2) and
    # carried the module doc down the chain.
    assert result["sym_read"] == "DOC[symbol read of DOC[module io | DOC[package mypkg]]]"
```

- [ ] **Step 2: 跑测试确认通过**

Run: `uv run pytest tests/integration/test_dag_doc_generation.py -v > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS（拓扑数据流 + 二级嵌套均验证）

> 这是先实现后测试的集成验证（机制已在 Task 1-7 建好），无需 red 阶段；若失败说明前序 task 有回归，回溯修复。

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_dag_doc_generation.py
git commit -m "test(dag): #10 topological doc-gen + >1-level nesting integration (M7)"
```

---

## Task 9: feature demo — `examples/features/dag.py` + 更新 `nesting.py`

**Files:**
- Create: `examples/features/dag.py`
- Modify: `examples/features/nesting.py`
- Modify: `examples/AGENTS.md`（§2 表）、根 `README.md`（Examples 指针）

- [ ] **Step 1: 写 `examples/features/dag.py`**

新建 `examples/features/dag.py`（遵 `examples/AGENTS.md` §5 骨架：单机制 docstring → `main()` → print → assert → `-m` 注释）：

```python
"""``ctx.dag`` — dependency-order (topological) fan-out with transitive skip.

A documentation generator whose work has a strict dependency order: the package doc
must exist before each module doc, and a module doc before its symbol docs. The script
declares the graph; the engine runs each node only after its predecessors settle and
feeds their results in. Independent branches run concurrently with no level barrier.
When one node fails, only the nodes that (transitively) depend on it are skipped — the
rest of the graph still completes.

    package mypkg
      ├── module io      -> symbol open, symbol read
      └── module net     -> symbol fetch        (module net FAILS -> fetch skipped)

Run it:

    uv run python -m examples.features.dag
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from langchain_dynamic_workflow import Ctx, DagNode, Roster, run_workflow

# The one module whose documenter fails, to show transitive skip of its symbols.
_FAILING_MODULE = "module net"


def _build_documenter(*, response_format: Any = None) -> Any:
    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = str(inp["messages"][0].content)
        if _FAILING_MODULE in prompt:
            raise RuntimeError("documenter crashed on module net")
        return {"messages": [*inp["messages"], AIMessage(content=f"DOC[{prompt}]")]}

    return RunnableLambda(_leaf)


async def generate_docs(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any | None]:
    """package -> modules -> symbols, in dependency order."""
    ctx.phase("generate docs")
    return await ctx.dag(
        [
            DagNode("pkg", deps=[], run=lambda d: ctx.agent("package mypkg", agent_type="doc")),
            DagNode(
                "mod_io",
                deps=["pkg"],
                run=lambda d: ctx.agent(f"module io | {d['pkg']}", agent_type="doc"),
            ),
            DagNode(
                "mod_net",
                deps=["pkg"],
                run=lambda d: ctx.agent(f"module net | {d['pkg']}", agent_type="doc"),
            ),
            DagNode(
                "sym_open",
                deps=["mod_io"],
                run=lambda d: ctx.agent(f"symbol open | {d['mod_io']}", agent_type="doc"),
            ),
            DagNode(
                "sym_read",
                deps=["mod_io"],
                run=lambda d: ctx.agent(f"symbol read | {d['mod_io']}", agent_type="doc"),
            ),
            DagNode(
                "sym_fetch",
                deps=["mod_net"],
                run=lambda d: ctx.agent(f"symbol fetch | {d['mod_net']}", agent_type="doc"),
            ),
        ]
    )


async def main() -> None:
    roster = Roster().register(
        "doc",
        builder=_build_documenter,
        description="Documents one package / module / symbol given its parent's doc",
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any | None]:
        return await generate_docs(ctx, {})

    result = await run_workflow(orchestrate, roster=roster)
    for node_id in sorted(result):
        print(f"{node_id}: {result[node_id]}")

    # Topological data-flow: a symbol doc embeds its module doc, which embeds the package doc.
    assert result["sym_read"] == "DOC[symbol read | DOC[module io | DOC[package mypkg]]]"
    # Transitive skip: module net failed, so its symbol fetch was skipped — but the io
    # branch is independent and completed.
    assert result["mod_net"] is None
    assert result["sym_fetch"] is None
    assert result["sym_open"] is not None
    print("OK: dag ran in topological order and skipped only the failed branch.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 跑 dag demo 验证断言（烟测）**

Run: `uv run python -m examples.features.dag > /tmp/ldw-demo.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-demo.log`
Expected: EXIT=0，末行 `OK: dag ran in topological order and skipped only the failed branch.`

- [ ] **Step 3: 更新 `examples/features/nesting.py` 演示 >1 层**

读 `examples/features/nesting.py`，把它从「一层嵌套」改为「三层嵌套 + 环守卫」：让 `orchestrate` inline 一个 `outer_chapter` 工作流，后者再 inline `section`，`section` 再 inline `paragraph`（depth 3 < 默认 8），断言深层返回值贯穿；并补一段 `try/except WorkflowCycleError` 演示自调用被拒。保持 §5 骨架（docstring 单机制、`main()`、print、assert、`-m` 注释）。docstring 顶句改为 "``workflow()`` named nesting — multiple levels deep, with a cycle guard."

- [ ] **Step 4: 跑 nesting demo 烟测**

Run: `uv run python -m examples.features.nesting > /tmp/ldw-demo.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-demo.log`
Expected: EXIT=0，断言通过

- [ ] **Step 5: 同步 demo 索引**

在 `examples/AGENTS.md` §2 的 17 行 feature 表里：把 `features/nesting` 行的机制描述改为「`workflow()` named nesting — multiple levels deep + cycle guard」；新增一行 `features/dag` →「`ctx.dag` dependency-order (topological) fan-out + transitive skip of a failed branch」。学习路径 §2 在 `pipeline` 之后插入 `dag`（紧邻 fan-out 家族）。同步根 `README.md` 的 Examples 指针计数（17 → 18 feature demos）。

- [ ] **Step 6: Commit**

```bash
git add examples/features/dag.py examples/features/nesting.py examples/AGENTS.md README.md
git commit -m "docs(examples): dag dependency-order demo + multi-level nesting demo (M7)"
```

---

## Task 10: SKILL.md — `ctx.dag` / `ctx.loop_until` DSL + 嵌套更新 + M1.5 作者范式

**Files:**
- Modify: `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md`

- [ ] **Step 1: 更新 DSL 段**

在 SKILL.md 的 DSL 列表里，`ctx.race(...)` 条目（line 58-66）之后、`ctx.workflow(...)` 条目（line 67-69）之前插入：

```markdown
- `await ctx.dag(nodes)` — fan out a **dependency graph** in topological order: each
  `DagNode(id, deps=[...], run=lambda d: ...)` runs only after every id in `deps` has
  settled, and its `run` receives a `{dep_id: result}` mapping of those predecessors'
  results. Ready nodes run concurrently with no level barrier (an independent branch
  races ahead of a slow one). Returns a `{node_id: result}` dict. A node whose `run`
  raises lands as `None` and every node that depends on it is **skipped** to `None`
  (transitive); a node that legitimately returns `None` does NOT skip its dependents.
  Use this over `parallel` when the work has a real dependency order (e.g. package →
  module → symbol). Filter the `None` holes downstream, same as `parallel`/`pipeline`.
- `await ctx.loop_until(body, *, done, max_iters)` — a measured-stop loop with the two
  author disciplines built in. `body` is `(iter_index, accumulated_so_far) -> result`;
  after each iteration the result is appended and `done(accumulated)` is checked over
  the **full** accumulated list (dedup / convergence against *everything* seen, not just
  the last round). `max_iters` is a **mandatory** hard cap — when it is reached without
  `done` ever holding, a log line is emitted and the accumulated results are returned.
  Returns the accumulated list.
```

把 `ctx.workflow(...)` 条目（line 67-69）改为：

```markdown
- `await ctx.workflow(name, args)` — inline another registered workflow, up to several
  levels deep (a configurable cap guards runaway recursion). The inner workflow shares
  this run's journal and budget. A workflow that re-enters itself (a cycle) is refused.
```

- [ ] **Step 2: 加 Patterns 段（dag 拓扑 + loop_until + M1.5 作者范式）**

在 SKILL.md 的 `## Patterns` 段（line 91+）末尾追加四段范式（道层范式，**不**教工具机制）：

1. **Dependency-order DAG（doc-gen 拓扑）** — 一个 `ctx.dag` 范式：package→module→symbol，节点 `run` 从 `d` 读前驱 doc 拼进 prompt，末尾 `{k: v for k, v in results.items() if v is not None}` 过滤跳过的洞。
2. **Measured-stop loop（`loop_until`）** — 用 `ctx.loop_until` 收敛到一个目标（如「找够 N 条去重后的发现」），强调 `body` 收 `accumulated` 做**全量** dedup、`max_iters` 必给。
3. **多阶段：上一 phase 结果驱动下一 phase 扇出** — 普通确定性代码：`await` 上一 phase 入脚本变量 → 分支 → 据此 build 下一 phase 的 `parallel`/`pipeline`/`dag` work-list；scout-then-fan-out（先用一个廉价 leaf 探出 work-list，再扇出）。强调控制流反转：脚本拥有循环/分支、结果存脚本变量、可 resume。
4. **作者陷阱（the footguns）** — 每个循环必给硬上限 MAX；dedup/收敛判断必须比对**全部** seen 而非上一轮；`parallel` thunk 用默认参数捕获 `lambda x=x: ...`；迭代有序集合（`sorted(...)`）不迭代 `set`/`dict`。

每段配一个 ≤12 行的 `async def orchestrate(ctx, args)` 代码块，风格同既有 Patterns（line 93-120）。守 AGENTS.md 道/术线：**不**出现 `run`/`run_script`/`status`/`resume`/`cancel` 命令名、不出现已注册工作流名、不教 `args` 形状或 AST-gate 规则。

- [ ] **Step 3: 校验 SKILL.md 代码块（python_general 规则覆盖 .md）**

人工核对新增代码块语法正确、可读；确保 frontmatter `keywords` 若需可补 `dag`/`topological`/`loop_until`（line 9-11，可选）。

- [ ] **Step 4: Commit**

```bash
git add src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md
git commit -m "docs(skill): ctx.dag + ctx.loop_until DSL, nesting update, M1.5 authoring patterns (M7)"
```

---

## Task 11: evergreen 设计文档同步 + roadmap 状态翻牌

**Files:**
- Modify: `design_docs/01-engine-mechanism.md`
- Modify: `design_docs/02-architecture.md`
- Modify: `design_docs/uml/01-component.md` / `02-class.md` / `03-sequence.md`
- Modify: `design_docs/v0_3_0_plans/00-roadmap.md`

- [ ] **Step 1: `01-engine-mechanism.md` 加 DAG + 嵌套-放开机制段**

读 `design_docs/01-engine-mechanism.md` 既有结构（race/pipeline 机制段如何写），追加：① DAG 拓扑调度器机制（Kahn 入度、就绪即启无 level barrier、failed-set 与 None-result 分离、传递式跳过、gate-at-leaf、fan-out frame 排除 determinism guard、无 DAG 级 journal 靠 leaf content-hash 重放）；② 嵌套放开机制（`max_workflow_depth` 可配置 cap + `_WORKFLOW_NAME_STACK` 环检测、嵌套 sequential `agent()` 摊平进同一 call-key 序列故确定、`WorkflowCycleError`/`WorkflowDagError` 入控制流信号元组）。

- [ ] **Step 2: `02-architecture.md` 加接线**

追加：新模块 `_dag.py`（`DagNode` + `run_dag`）、`SpanKind.DAG`、`_codegen` 注入 `_SCRIPT_DAG_API`、`run_workflow`→`Ctx` 的 `max_workflow_depth` 接线、`__init__` 导出 `DagNode`/`WorkflowDagError`/`WorkflowCycleError`。

- [ ] **Step 3: UML 同步**

- `uml/01-component.md`：组件图加 `_dag` 模块及其与 `_context`/`_codegen`/`_errors` 的依赖边。
- `uml/02-class.md`：加 `DagNode` 类、`Ctx.dag`/`Ctx.loop_until` 方法、`SpanKind.DAG`、`max_workflow_depth` 字段、新错误类。
- `uml/03-sequence.md`：加一张 DAG 拓扑序时序图（节点就绪即启 + 传递式跳过）和/或嵌套放开时序。

- [ ] **Step 4: roadmap 状态翻牌**

在 `design_docs/v0_3_0_plans/00-roadmap.md`：
- 表格 M7 行（line 24）`Plan` 列从「待写」改为 `✅ 已落地 · [`07-m7-h-topo-fanout-nesting.md`](07-m7-h-topo-fanout-nesting.md)`。
- 「状态」段的 M7 条目（line 208）改写为已落地摘要（dag 原语 + 嵌套放开 + G `loop_until` + M1.5 docs；关键决策：传递式跳过、深度 cap + 环检测、`WorkflowDagError`/`WorkflowCycleError` 入控制流信号；Codex 跨模型评审若驱动修复则记录）。
- 推进顺序图（line 50）/执行序列（line 210）的 M7 标注同步为已落地。

- [ ] **Step 5: Commit**

```bash
git add design_docs/01-engine-mechanism.md design_docs/02-architecture.md design_docs/uml/ design_docs/v0_3_0_plans/00-roadmap.md
git commit -m "docs(design): sync evergreen 01/02 + uml + roadmap for M7"
```

---

## Task 12: 全门验证 + 跨模型 Codex 评审 + 真模型 #10 验收 gate

**Files:** 全仓（验证 + 评审驱动的修复）

- [ ] **Step 1: 全仓 gate（遵 memory `independently-verify-gate-claims` / `ruff-format-check-whole-repo-matches-ci`）**

```bash
uv run pytest -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -30 /tmp/ldw-test.log
grep -E "FAILED|ERROR" /tmp/ldw-test.log
uv run ruff check . > /tmp/ldw-ruff.log 2>&1; echo "RUFF=$?"; tail -10 /tmp/ldw-ruff.log
uv run ruff format --check . > /tmp/ldw-fmt.log 2>&1; echo "FMT=$?"; tail -10 /tmp/ldw-fmt.log
uv run pyright > /tmp/ldw-pyright.log 2>&1; echo "PYRIGHT=$?"; tail -20 /tmp/ldw-pyright.log
```
Expected: pytest EXIT=0、RUFF=0、FMT=0、PYRIGHT=0。`ruff format --check .` 覆盖**整树含 demo-app**（CI 同款）。

- [ ] **Step 2: 跑全部 feature demo 烟测（断言即烟测）**

```bash
for d in dag nesting; do
  echo "== $d =="; uv run python -m examples.features.$d > /tmp/ldw-demo-$d.log 2>&1; echo "EXIT=$?"; tail -3 /tmp/ldw-demo-$d.log
done
```
Expected: 各 EXIT=0。

- [ ] **Step 3: 真模型 #10 验收 gate（开发期，非常驻）**

按 `examples/AGENTS.md` §4 + memory `per-gap-real-e2e-acceptance` / `gated-real-e2e-must-actually-run`：以临时真跑或扩 flagship 的方式，让一个够强的真 host（opus 级，遵 memory `workflow-review-agent-model`）从**道层** prompt + skill + tool description 驱动一个「拓扑序文档生成（package→module→symbol）+ 命名子 workflow 嵌套 >1 层」场景（`LDW_DEMO_REAL_MODEL` + `.env` OpenRouter，遵 memory `openrouter-anthropic-web-search-recipe`；**保留 LangSmith tracing 计费**，遵 memory `keep-langsmith-tracing-on-for-billing`）。验收标准：真 host 自驱走 dag 拓扑路径（非 fallback），#10 头条端到端跑通。若真 host 从道层 prompt 无法完成 → 信号是改进 skill / tool description，**绝不**往 prompt 掉术（AGENTS.md 道/术线）。记录运行证据。

- [ ] **Step 4: 跨模型 Codex 评审（遵 memory `orchestrator-mode-via-workflows` / `codex-service-tier-workaround`）**

用 `/codex` 跨模型评审本里程碑 diff（v0.2.0 四 gap 每次都抓到至少一个 in-house 漏掉的 HIGH，已验证多次）。重点审：① dag 调度器的 asyncio 任务生命周期 / 取消 / 无 slot 泄漏；② failed-set 与 None-result 分离的正确性（合法 None 不跳下游）；③ 控制流信号在 dag 内 drain-then-raise 的时序；④ 环检测 contextvar 的 reset 顺序（depth 与 name-stack）；⑤ `WorkflowNestingError` **不**入控制流信号元组这条边界是否该改（见 Task 1 注）。**独立复核**：不轻信 Codex 或自报「绿」，亲自重跑 Step 1 全门 + 对评审 findings 逐条裁定（遵 memory `independently-verify-gate-claims`）。

- [ ] **Step 5: 评审驱动修复 + 复跑全门**

对 Codex + 自审的每条 HIGH/BLOCKER 落修复（带回归测试），复跑 Step 1 全门至全绿。把评审抓到的承重修复记进 roadmap M7 状态段（Task 11 Step 4）。

- [ ] **Step 6: 终态 Commit**

```bash
git add -A
git commit -m "chore(M7): gate green + review-driven hardening + evergreen sync"
```

---

## Self-Review（plan 写完后自查）

**Spec 覆盖：**
- DAG 拓扑原语 → Task 2-5（DagNode + run_dag + ctx.dag + 导出/注入）✓
- 失败语义=传递式跳过 → Task 3（failed-set 分离 + 传递式跳过测试）✓
- journal/determinism 继承（无 DAG 级 journal） → Task 4（fan-out frame token + docstring）+ Task 8（resume 隐含于 journal 复用）✓
- 嵌套放开=深度 cap + 环检测 → Task 1（错误类）+ Task 6（max_workflow_depth + name-stack）✓
- G `ctx.loop_until` → Task 7 ✓
- M1.5 SKILL.md 作者范式 → Task 10 ✓
- 验收 #10 真模型 + 集成示例 + Codex 评审 + evergreen 同步 → Task 8/9/11/12 ✓

**类型一致性：** `DagNode(id, deps, run)`、`run_dag(nodes) -> dict[str, Any | None]`、`ctx.dag(nodes) -> dict[str, Any | None]`、`ctx.loop_until[T](body, *, done, max_iters) -> list[T]`、`max_workflow_depth: int`、`WorkflowDagError`/`WorkflowCycleError`、`SpanKind.DAG` —— 全 plan 一致。

**待实现者注意（占位澄清，非 TBD）：** Task 4/7 的两处「按实际 API 调整断言」是因为 `SpanRecorder.completed` / `ProgressEntry` 字段名需实现者读 `_observability.py` / `_progress.py` 现场确认——非占位，是要求对齐既有真实 API（这两个模块已存在）。

**资源/并发边界（Codex 重点）：** run_dag 为每个就绪节点建一个 asyncio task（镜像 `parallel` 的「per-thunk task，gate-at-leaf」），不对节点任务数设独立闸——与 `parallel` 一致；超大 DAG 的节点任务数上限作为开放项交 Step 4 评审复核。

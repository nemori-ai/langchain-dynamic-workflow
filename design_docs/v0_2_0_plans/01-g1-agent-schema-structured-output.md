# G1 — `agent(schema=...)` 结构化输出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给叶子原语 `agent()` 加 `schema=` 参数——脚本传 pydantic 类或内联 JSON-schema dict，叶子产出经校验的结构化对象（而非纯文本），脚本下一行直接属性访问做纯 Python reduce。这是社区 ubiquitous 的 "schema-as-handoff" 骨架，也是后续 adversarial-verify / judge-panel 等模式的总开关。

**Architecture:** `response_format` 是 deepagents/langchain 的**构造期**参数，而 roster 持有预构造 runnable，故 schema 落地走 **Builder 式 roster**（决策 D-G1a）：roster 条目可注册一个 `builder(*, response_format)->Runnable`，引擎按 `(agent_type, schema)` 构造并缓存一个 `ToolStrategy(model, handle_errors=True)` 绑定变体（缓存归 Roster，决策 D-G1b）。JSON-schema dict 经自研转换器归一成 pydantic 模型（参考 `omne_engram` 约定、不依赖它），下游统一：`ToolStrategy(model)` → 叶子产 `structured_response` → fold 取出 → journal 存 `model_dump_json` / 命中 `model_validate_json`。设计源：`docs/plans/2026-06-02-gap-analysis-and-g1-schema-design.md`（gitignored 草稿）。

**Tech Stack:** Python 3.12（async-first）、pydantic v2（`create_model`）、langchain `ToolStrategy`（`langchain.agents.structured_output`）、deepagents 0.6.7、pytest + pytest-asyncio、ruff、pyright(strict)、import-linter。

---

## 前置：分支

- [ ] **创建特性分支**（仓库当前在 `main`，先开分支）

```bash
git checkout -b feat/g1-agent-schema
```

## 文件结构（本计划触达）

| 文件 | 责任 | 动作 |
|---|---|---|
| `src/langchain_dynamic_workflow/_schema.py` | JSON-schema dict ↔ pydantic 模型归一 + 进程级缓存 | **新建** |
| `src/langchain_dynamic_workflow/_result.py` | 增 `fold_structured`（取 `structured_response`，缺失 fail-loud） | 修改 |
| `src/langchain_dynamic_workflow/_roster.py` | `RosterEntry.builder` 字段 + `register` 互斥 + `runnable_for(response_format)` 构建缓存 | 修改 |
| `src/langchain_dynamic_workflow/_context.py` | `LeafRunner` 增 `response_format`；`Ctx.agent` 加 `schema=` + overloads + 归一/journal/fold 分派 | 修改 |
| `src/langchain_dynamic_workflow/_engine.py` | `leaf_task` 接 `response_format` → 走 `roster.runnable_for` | 修改 |
| `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md` | DSL 文档补 `schema=`（dict 字面量形态）+ 一个结构化输出示例 | 修改 |
| `examples/07_deep_research_real_e2e.py` | 验收门：`extractor`→`Claim`、`skeptic`→`Verdict` schema 化 | 修改 |
| `design_docs/01-engine-mechanism.md` | evergreen：叶子契约补 builder-roster；Decision Log 增 D-G1a/b | 修改 |
| `design_docs/02-architecture.md` | evergreen：register 示例补 builder 形态；primitives 注 schema | 修改 |
| `design_docs/uml/{02-class,03-sequence}.md` | evergreen：RosterEntry.builder / runnable_for / SchemaConverter / schema 时序分支 | 修改 |
| `tests/conftest.py` | 增 `make_structured_leaf` fixture（返回带 `structured_response` 的 state） | 修改 |
| `tests/unit/test_schema.py` | 转换器单测 | 新建 |
| `tests/unit/test_result.py` | `fold_structured` 单测 | 修改 |
| `tests/unit/test_roster.py` | builder 注册 + `runnable_for` 单测 | 修改 |
| `tests/unit/test_agent_schema.py` | `Ctx.agent(schema=...)` 单测（fake leaf_runner） | 新建 |
| `tests/integration/test_g1_schema.py` | 引擎 @task 路径 + parallel/pipeline + resume + L2 dict 脚本 | 新建 |

---

## Task 1: JSON-schema → pydantic 转换器（`_schema.py`）

**Files:**
- Create: `src/langchain_dynamic_workflow/_schema.py`
- Test: `tests/unit/test_schema.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/unit/test_schema.py
"""Unit tests for JSON-schema -> pydantic conversion (agent schema normalization)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from langchain_dynamic_workflow._schema import to_pydantic_model


class _Verdict(BaseModel):
    refuted: bool
    reason: str


def test_passthrough_pydantic_class() -> None:
    assert to_pydantic_model(_Verdict) is _Verdict


def test_dict_object_scalars_and_required() -> None:
    schema = {
        "type": "object",
        "properties": {
            "refuted": {"type": "boolean", "description": "did it fail"},
            "reason": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["refuted", "reason"],
        "additionalProperties": False,
    }
    model = to_pydantic_model(schema)
    inst = model.model_validate({"refuted": True, "reason": "x"})
    assert inst.refuted is True
    assert inst.reason == "x"
    assert inst.score is None  # optional -> defaulted None


def test_dict_array_and_nested_object_and_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "impact": {"type": "string", "enum": ["low", "high"]},
                    },
                    "required": ["title", "impact"],
                },
            }
        },
        "required": ["items"],
    }
    model = to_pydantic_model(schema)
    inst = model.model_validate({"items": [{"title": "t", "impact": "high"}]})
    assert inst.items[0].title == "t"
    assert inst.items[0].impact.value == "high"


def test_unsupported_construct_fails_loud() -> None:
    schema = {"type": "object", "properties": {"x": {"$ref": "#/defs/Y"}}}
    with pytest.raises(ValueError, match=r"\$ref|unsupported"):
        to_pydantic_model(schema)


def test_same_dict_yields_byte_identical_json_schema() -> None:
    # Determinism: the journal key folds the model's JSON schema; two builds of
    # the same dict must produce identical schema text or resume silently re-runs.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
    }
    m1 = to_pydantic_model(schema)
    m2 = to_pydantic_model(dict(schema))  # equal-but-distinct dict
    assert m1.model_json_schema() == m2.model_json_schema()


def test_cache_returns_same_class_for_equal_dict() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert to_pydantic_model(schema) is to_pydantic_model(dict(schema))


def test_non_object_top_level_fails_loud() -> None:
    with pytest.raises(ValueError, match="object"):
        to_pydantic_model({"type": "string"})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL — `ModuleNotFoundError: No module named 'langchain_dynamic_workflow._schema'`

- [ ] **Step 3: 写最小实现**

```python
# src/langchain_dynamic_workflow/_schema.py
"""JSON-schema dict -> pydantic model normalization for ``agent(schema=...)``.

A workflow script — especially an L2 LLM-authored one, which the AST gate forbids
from importing pydantic — declares a leaf's structured-output shape as an inline
JSON-schema ``dict``. This module converts that dict into a concrete pydantic
model so the rest of the engine treats every schema uniformly: ``ToolStrategy``
binds the model, the journal keys on its JSON schema, and the folded leaf result
is a validated model instance the script reads by attribute.

The supported subset is what real community workflows use: an object with typed
properties (string / integer / number / boolean), arrays with typed items,
nested objects, enums, ``required`` lists, ``description``, and
``additionalProperties``. Unsupported constructs (``$ref`` / ``allOf`` /
``anyOf`` / ``oneOf`` / ``patternProperties`` / ``not``) fail loud rather than
silently degrade.

The conventions (``pydantic.create_model`` + ``ConfigDict(extra=...)``, dynamic
``Enum`` classes, a concurrency-safe process-level cache) mirror the established
``omne_engram`` ``PydanticModelBuilder``; this converter is self-contained (the
library takes no upstream dependency) and reads JSON schema directly rather than
that project's ``FieldSpec`` IR.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

_UNSUPPORTED_KEYS: tuple[str, ...] = ("$ref", "allOf", "anyOf", "oneOf", "patternProperties", "not")
_SCALARS: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool}

_MODEL_CACHE: dict[str, type[BaseModel]] = {}
_CACHE_LOCK = threading.Lock()


def to_pydantic_model(schema: type[BaseModel] | dict[str, Any]) -> type[BaseModel]:
    """Normalize a schema to a concrete pydantic model class.

    Args:
        schema: A pydantic ``BaseModel`` subclass (returned as-is) or a
            JSON-schema ``dict`` (converted to a model).

    Returns:
        A ``type[BaseModel]``.

    Raises:
        TypeError: If ``schema`` is neither a ``BaseModel`` subclass nor a dict.
        ValueError: If the dict uses an unsupported JSON-schema construct.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    if isinstance(schema, dict):
        return _model_from_json_schema(schema)
    raise TypeError(
        "schema must be a pydantic BaseModel subclass or a JSON-schema dict, "
        f"got {type(schema).__name__}"
    )


def _canonical(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _model_from_json_schema(schema: dict[str, Any]) -> type[BaseModel]:
    key = _canonical(schema)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        model = _build_object_model(schema, name=_model_name(key))
        _MODEL_CACHE[key] = model
        return model


def _model_name(canonical_key: str) -> str:
    # Deterministic name derived from schema content: identical dict -> identical
    # name -> identical model_json_schema() (stable journal key); distinct schemas
    # never collide in pydantic's global model registry.
    digest = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:12]
    return f"DynamicSchema_{digest}"


def _build_object_model(schema: dict[str, Any], *, name: str) -> type[BaseModel]:
    _reject_unsupported(schema)
    if schema.get("type") != "object":
        raise ValueError(f"schema must be type 'object', got {schema.get('type')!r}")
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    extra = "allow" if schema.get("additionalProperties", False) else "forbid"
    fields: dict[str, Any] = {}
    for field_name, prop in properties.items():
        py_type = _resolve_type(prop, parent=name, field=field_name)
        field_kwargs: dict[str, Any] = {}
        if "description" in prop:
            field_kwargs["description"] = prop["description"]
        if field_name in required:
            fields[field_name] = (py_type, Field(**field_kwargs))
        else:
            fields[field_name] = (py_type | None, Field(default=prop.get("default"), **field_kwargs))
    return create_model(name, __config__=ConfigDict(extra=extra), **fields)


def _resolve_type(prop: dict[str, Any], *, parent: str, field: str) -> Any:
    _reject_unsupported(prop)
    if "enum" in prop:
        return _enum_type(prop["enum"], parent=parent, field=field)
    json_type = prop.get("type")
    if json_type in _SCALARS:
        return _SCALARS[json_type]
    if json_type == "array":
        items = prop.get("items")
        if not isinstance(items, dict):
            return list[Any]
        inner = _resolve_type(items, parent=parent, field=f"{field}_item")
        return list[inner]
    if json_type == "object":
        return _build_object_model(prop, name=f"{parent}_{_sanitize(field)}")
    raise ValueError(
        f"unsupported JSON-schema type {json_type!r} for field {field!r}; v1 supports "
        "object / array / string / integer / number / boolean / enum"
    )


def _enum_type(values: list[Any], *, parent: str, field: str) -> type[Enum]:
    members: dict[str, Any] = {}
    for index, value in enumerate(values):
        member = _sanitize(str(value)).upper() or f"VALUE_{index}"
        if member[0].isdigit():
            member = f"V_{member}"
        members[member] = value
    return Enum(f"{parent}_{_sanitize(field)}_Enum", members)


def _reject_unsupported(schema: dict[str, Any]) -> None:
    hit = [key for key in _UNSUPPORTED_KEYS if key in schema]
    if hit:
        raise ValueError(
            f"unsupported JSON-schema construct(s) {hit}; v1 does not support "
            "$ref / allOf / anyOf / oneOf / patternProperties / not"
        )


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned or "Field"


__all__ = ["to_pydantic_model"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: PASS（7 passed）

- [ ] **Step 5: ruff + pyright**

Run: `uv run ruff check src/langchain_dynamic_workflow/_schema.py tests/unit/test_schema.py && uv run ruff format src/langchain_dynamic_workflow/_schema.py tests/unit/test_schema.py && uv run pyright src/langchain_dynamic_workflow/_schema.py`
Expected: 无错误

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_schema.py tests/unit/test_schema.py
git commit -m "feat(schema): JSON-schema dict -> pydantic converter for agent(schema=...)"
```

---

## Task 2: `fold_structured`（`_result.py`）

**Files:**
- Modify: `src/langchain_dynamic_workflow/_result.py`
- Test: `tests/unit/test_result.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/unit/test_result.py`）

```python
# tests/unit/test_result.py — 追加
from pydantic import BaseModel

from langchain_dynamic_workflow._result import fold_structured


class _V(BaseModel):
    refuted: bool


def test_fold_structured_returns_structured_response() -> None:
    inst = _V(refuted=True)
    state = {"messages": [], "structured_response": inst}
    assert fold_structured(state, _V) is inst


def test_fold_structured_missing_response_fails_loud() -> None:
    import pytest

    with pytest.raises(ValueError, match="structured_response"):
        fold_structured({"messages": []}, _V)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_result.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: FAIL — `ImportError: cannot import name 'fold_structured'`

- [ ] **Step 3: 写最小实现**（追加到 `src/langchain_dynamic_workflow/_result.py`）

```python
# _result.py — 追加 import
from pydantic import BaseModel

# _result.py — 追加函数
def fold_structured(result: dict[str, Any], schema: type[BaseModel]) -> BaseModel:
    """Extract the validated structured response from a leaf's output state.

    Used when ``agent()`` was called with a ``schema``: the leaf was built with a
    ``response_format`` so its output state carries a ``structured_response``
    already validated against ``schema``. The intermediate messages never cross
    back — only this object does.

    Args:
        result: The raw output state of a leaf agent ``ainvoke`` call.
        schema: The pydantic model the leaf was bound to (for the error message).

    Returns:
        The validated ``structured_response`` instance.

    Raises:
        ValueError: If ``result`` has no ``structured_response`` (the leaf was not
            built with the expected ``response_format``).
    """
    response = result.get("structured_response")
    if response is None:
        raise ValueError(
            f"leaf result has no 'structured_response' for schema {schema.__name__!r}; "
            "the leaf was not built with a matching response_format "
            "(register the agent_type with a builder that forwards response_format)"
        )
    return response
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/unit/test_result.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: PASS

- [ ] **Step 5: ruff + pyright**

Run: `uv run ruff check src/langchain_dynamic_workflow/_result.py && uv run pyright src/langchain_dynamic_workflow/_result.py`
Expected: 无错误

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_result.py tests/unit/test_result.py
git commit -m "feat(result): add fold_structured for schema-bound leaves"
```

---

## Task 3: Roster builder 支持（`_roster.py`）

**Files:**
- Modify: `src/langchain_dynamic_workflow/_roster.py`
- Test: `tests/unit/test_roster.py`

- [ ] **Step 1: 写失败测试**（追加到 `tests/unit/test_roster.py`）

```python
# tests/unit/test_roster.py — 追加
from typing import Any

from langchain_core.runnables import Runnable


def test_register_runnable_and_builder_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Roster().register("x", _noop(), builder=lambda *, response_format=None: _noop())
    with pytest.raises(ValueError, match="exactly one"):
        Roster().register("x")


def test_runnable_for_no_schema_uses_runnable_entry() -> None:
    runnable = _noop()
    roster = Roster().register("plain", runnable)
    assert roster.runnable_for("plain", response_format=None) is runnable


def test_runnable_for_schema_on_runnable_only_entry_fails_loud() -> None:
    roster = Roster().register("plain", _noop())
    with pytest.raises(ValueError, match="builder"):
        roster.runnable_for("plain", response_format={"any": "fmt"})


def test_runnable_for_builds_and_caches_per_response_format() -> None:
    built: list[Any] = []

    def builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        leaf = _noop()
        built.append(response_format)
        return leaf

    roster = Roster().register("skeptic", builder=builder)
    fmt = {"k": "v"}
    first = roster.runnable_for("skeptic", response_format=fmt)
    second = roster.runnable_for("skeptic", response_format=fmt)
    assert first is second  # cached: built once for one response_format identity
    assert len(built) == 1


def test_runnable_for_builder_none_format_builds_once() -> None:
    calls = 0

    def builder(*, response_format: Any = None) -> Runnable[Any, Any]:
        nonlocal calls
        calls += 1
        return _noop()

    roster = Roster().register("skeptic", builder=builder)
    roster.runnable_for("skeptic", response_format=None)
    roster.runnable_for("skeptic", response_format=None)
    assert calls == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/unit/test_roster.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: FAIL — `register` 不接 `builder` / 无 `runnable_for`

- [ ] **Step 3: 写最小实现**（改 `src/langchain_dynamic_workflow/_roster.py`）

```python
# _roster.py — imports 增加
import hashlib
import json
import threading
from collections.abc import Callable

# RosterEntry：增 builder 字段（runnable 改可选）
@dataclass(frozen=True)
class RosterEntry:
    """A registered leaf agent.

    Exactly one of ``runnable`` (pre-built, schema-less only) or ``builder``
    (constructs a runnable for a given ``response_format``, enabling
    ``agent(schema=...)``) is set.

    Attributes:
        name: The roster key used by ``agent(agent_type=...)``.
        runnable: The pre-built leaf runnable, or ``None`` when a builder is used.
        builder: A factory ``(*, response_format) -> Runnable`` used to construct
            a schema-bound (or schema-less) variant on demand, or ``None``.
        description: Human-readable description.
        needs_execution: Whether this agent type requires an isolated execution
            sandbox (tiered admission) rather than pure in-context reasoning.
        default_model: Default model identifier used as the effective model when
            an ``agent()`` call supplies no ``model`` override.
    """

    name: str
    runnable: Runnable[Any, Any] | None = None
    builder: Callable[..., Runnable[Any, Any]] | None = None
    description: str = ""
    needs_execution: bool = False
    default_model: str | None = None


class Roster:
    """A mutable registry mapping agent-type names to leaf runnables/builders."""

    def __init__(self) -> None:
        self._entries: dict[str, RosterEntry] = {}
        # Process-level cache of built variants keyed by (name, response_format
        # identity). Compiled graphs are stateless across runs, so caching here —
        # next to the builder that owns them — keeps resume cheap and avoids
        # rebuilding per run (decision D-G1b). Concurrency-safe for shared use
        # across runs/threads.
        self._built: dict[tuple[str, str], Runnable[Any, Any]] = {}
        self._build_lock = threading.Lock()

    def register(
        self,
        name: str,
        runnable: Runnable[Any, Any] | None = None,
        *,
        builder: Callable[..., Runnable[Any, Any]] | None = None,
        description: str = "",
        needs_execution: bool = False,
        default_model: str | None = None,
    ) -> Roster:
        """Register a leaf agent under ``name`` and return ``self`` for chaining.

        Provide exactly one of ``runnable`` (pre-built, schema-less) or
        ``builder`` (constructs variants for ``agent(schema=...)``).

        Raises:
            ValueError: If neither or both of ``runnable`` / ``builder`` are given.
        """
        if (runnable is None) == (builder is None):
            raise ValueError(
                f"register({name!r}): provide exactly one of 'runnable' or 'builder' "
                "(a pre-built runnable handles schema-less calls; a builder enables "
                "agent(schema=...))"
            )
        self._entries[name] = RosterEntry(
            name=name,
            runnable=runnable,
            builder=builder,
            description=description,
            needs_execution=needs_execution,
            default_model=default_model,
        )
        return self

    def resolve(self, name: str) -> RosterEntry:
        """Return the entry for ``name``.

        Raises:
            KeyError: If ``name`` is not registered, listing available names.
        """
        try:
            return self._entries[name]
        except KeyError:
            available = sorted(self._entries)
            raise KeyError(f"unknown agent_type {name!r}; available: {available}") from None

    def runnable_for(self, name: str, *, response_format: Any) -> Runnable[Any, Any]:
        """Return the runnable for ``name`` bound to ``response_format``.

        A ``response_format`` of ``None`` asks for the schema-less variant. A
        builder entry constructs (and caches per response-format identity) the
        variant; a pre-built ``runnable`` entry serves only the schema-less case
        and fails loud if a ``response_format`` is requested.

        Args:
            name: The roster key.
            response_format: The structured-output format to bind, or ``None``.

        Returns:
            The (possibly built and cached) runnable.

        Raises:
            KeyError: If ``name`` is not registered.
            ValueError: If a ``response_format`` is requested for a pre-built
                ``runnable`` entry (no builder to construct a bound variant).
        """
        entry = self.resolve(name)
        if entry.builder is None:
            if response_format is not None:
                raise ValueError(
                    f"agent_type {name!r} was registered with a pre-built runnable and "
                    "cannot produce structured output; register it with a builder "
                    "(builder=lambda *, response_format=None: create_deep_agent(..., "
                    "response_format=response_format)) to use agent(schema=...)"
                )
            assert entry.runnable is not None  # register() guarantees one is set
            return entry.runnable
        cache_key = (name, _response_format_identity(response_format))
        cached = self._built.get(cache_key)
        if cached is not None:
            return cached
        with self._build_lock:
            cached = self._built.get(cache_key)
            if cached is not None:
                return cached
            built = entry.builder(response_format=response_format)
            self._built[cache_key] = built
            return built

    def __contains__(self, name: object) -> bool:
        return name in self._entries


def _response_format_identity(response_format: Any) -> str:
    """A stable string identity for a response_format, for the build cache.

    ``None`` (schema-less) maps to a fixed sentinel. A ``ToolStrategy`` over a
    pydantic model is identified by that model's JSON schema; any other value
    falls back to its ``repr``.
    """
    if response_format is None:
        return "__none__"
    schema = getattr(response_format, "schema", None)
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        canonical = json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return repr(response_format)
```

> **注意**：现有 `register("name", runnable)` 调用点（examples 01-08、各测试）签名不变（`runnable` 仍是第一个位置参数），向后兼容。

- [ ] **Step 4: 运行确认通过 + 回归现有 roster 测试**

Run: `uv run pytest tests/unit/test_roster.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS（含原有 4 个 + 新增 5 个）

- [ ] **Step 5: ruff + pyright**

Run: `uv run ruff check src/langchain_dynamic_workflow/_roster.py tests/unit/test_roster.py && uv run pyright src/langchain_dynamic_workflow/_roster.py`
Expected: 无错误

- [ ] **Step 6: Commit**

```bash
git add src/langchain_dynamic_workflow/_roster.py tests/unit/test_roster.py
git commit -m "feat(roster): builder entries + runnable_for(response_format) build cache"
```

---

## Task 4: 引擎 leaf_task 接 response_format（`_context.py` Protocol + `_engine.py`）

**Files:**
- Modify: `src/langchain_dynamic_workflow/_context.py`（`LeafRunner` Protocol 增 `response_format`）
- Modify: `src/langchain_dynamic_workflow/_engine.py:131-207`（`leaf_task` / `leaf_runner` 走 `roster.runnable_for`）
- Test: `tests/integration/test_g1_schema.py`（仅本任务的引擎构造片段，后续 Task 6 扩展）

- [ ] **Step 1: 写失败测试**

```python
# tests/integration/test_g1_schema.py
"""Integration: agent(schema=...) over the real engine @task / roster.runnable_for path."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from langchain_dynamic_workflow import Roster, run_workflow


class Verdict(BaseModel):
    refuted: bool
    reason: str


def _structured_builder(*, response_format: Any = None) -> Runnable[Any, Any]:
    # A fake leaf whose output state carries a structured_response matching the
    # bound response_format's schema — stands in for a create_deep_agent built
    # with response_format=ToolStrategy(Verdict).
    async def _call(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return {
            "messages": [*inp["messages"], AIMessage(content="done")],
            "structured_response": Verdict(refuted=False, reason="solid"),
        }

    return RunnableLambda(_call)


async def test_engine_agent_schema_returns_structured_object() -> None:
    roster = Roster().register("skeptic", builder=_structured_builder)

    async def orchestrate(ctx: Any, args: dict[str, Any]) -> Any:
        verdict = await ctx.agent("verify X", agent_type="skeptic", schema=Verdict)
        return {"refuted": verdict.refuted, "reason": verdict.reason}

    result = await run_workflow(orchestrate, roster=roster, args={})
    assert result == {"refuted": False, "reason": "solid"}
```

> **说明**：`run_workflow` 的具体签名以仓库现状为准（examples 01-03 已演示其用法）；若关键字与此处不符，按现状调整调用、保持断言不变。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/integration/test_g1_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: FAIL — `agent()` 不接 `schema`（或 leaf 未走 builder）

- [ ] **Step 3: 改 `LeafRunner` Protocol（`_context.py:60-70`）增 keyword 形参**

```python
    def __call__(
        self,
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
    ) -> Awaitable[LeafOutcome]:
        """Run the leaf and return its ``LeafOutcome`` (raw state + token usage)."""
        ...
```

- [ ] **Step 4: 改 `_engine.py` `leaf_task` / `leaf_runner` 走 `runnable_for`**

`leaf_task`（`_engine.py:131-163`）签名增 `response_format`，并把 `entry.runnable.ainvoke(...)` 换成经 `runnable_for` 解析：

```python
    @task
    async def leaf_task(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str,
        needs_execution: bool,
        response_format: Any = None,
    ) -> LeafOutcome:
        roster.resolve(agent_type)  # fail fast on unknown agent_type
        runnable = roster.runnable_for(agent_type, response_format=response_format)
        usage_handler = UsageMetadataCallbackHandler()
        configurable: dict[str, Any] = {}
        if model is not None:
            configurable["model"] = model

        async def _invoke() -> LeafOutcome:
            leaf_config: RunnableConfig = {
                "callbacks": [usage_handler],
                "configurable": configurable,
            }
            state: dict[str, Any] = await runnable.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}, config=leaf_config
            )
            return LeafOutcome(state=state, usage=total_tokens_from_handler(usage_handler))

        # ... sandbox-admission block unchanged, but it references `runnable`
        #     instead of `entry.runnable` (entry only needed for needs_execution,
        #     which is passed in as a parameter already) ...
```

`leaf_runner`（`_engine.py:194-207`）转发新参：

```python
    async def leaf_runner(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
    ) -> LeafOutcome:
        return await leaf_task(
            agent_type,
            prompt,
            model,
            leaf_id=leaf_id,
            needs_execution=needs_execution,
            response_format=response_format,
        )
```

> **注意**：`needs_execution` 当前由 `agent()` 从 `entry.needs_execution` 取并传入 `leaf_runner`（见 `_context.py:352-360`），`leaf_task` 内部不再依赖 `entry.runnable`；保留 `roster.resolve` 仅为对未知 `agent_type` fail-fast。sandbox 块中凡引用 `entry.runnable` 处一律改用本任务解析出的 `runnable`。

- [ ] **Step 5: 临时桩让本任务可独立验证**

本测试还需 Task 5 的 `agent(schema=)`。**若按 subagent-driven 顺序执行**，Task 4 与 Task 5 合并验证：先完成 Step 3-4，再到 Task 5 完成 `agent()`，回到此处运行。
Run: `uv run pytest tests/integration/test_g1_schema.py::test_engine_agent_schema_returns_structured_object -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected（Task 5 完成后）: PASS

- [ ] **Step 6: ruff + pyright（回归全引擎）**

Run: `uv run ruff check src/langchain_dynamic_workflow/_engine.py src/langchain_dynamic_workflow/_context.py && uv run pyright src/langchain_dynamic_workflow/_engine.py`
Expected: 无错误

- [ ] **Step 7: Commit**

```bash
git add src/langchain_dynamic_workflow/_engine.py src/langchain_dynamic_workflow/_context.py
git commit -m "feat(engine): thread response_format through leaf_task via roster.runnable_for"
```

---

## Task 5: `Ctx.agent(schema=...)`（`_context.py`）

**Files:**
- Modify: `src/langchain_dynamic_workflow/_context.py`（imports、`agent` overloads + 实现）
- Test: `tests/unit/test_agent_schema.py`、`tests/conftest.py`

- [ ] **Step 1: 增共享 fixture `make_structured_leaf`（`tests/conftest.py`）**

```python
# tests/conftest.py — 追加（供本任务与 Task 6 复用）
@pytest.fixture
def make_structured_leaf() -> Callable[..., Runnable[Any, Any]]:
    """Return a factory for a fake leaf whose state carries a structured_response.

    Stands in for a create_deep_agent built with response_format=ToolStrategy(...):
    the leaf appends an AIMessage and attaches the given model instance as
    ``structured_response`` so fold_structured / agent(schema=...) can extract it.
    """

    def factory(structured: Any, *, reply: str = "done") -> Runnable[Any, Any]:
        async def _call(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
            return {
                "messages": [*inp["messages"], AIMessage(content=reply)],
                "structured_response": structured,
            }

        return RunnableLambda(_call)

    return factory
```

- [ ] **Step 2: 写失败测试（`tests/unit/test_agent_schema.py`）**

```python
# tests/unit/test_agent_schema.py
"""Unit tests for Ctx.agent(schema=...) — structured output via a fake leaf_runner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from langchain_dynamic_workflow._context import Ctx, LeafOutcome
from langchain_dynamic_workflow._journal import InMemoryJournalStore
from langchain_dynamic_workflow._roster import Roster


class Claim(BaseModel):
    text: str
    confident: bool


def _ctx_with_structured(structured: Claim, counter: list[int]) -> Ctx:
    async def _leaf(
        agent_type: str,
        prompt: str,
        model: str | None,
        *,
        leaf_id: str = "",
        needs_execution: bool = False,
        response_format: Any = None,
    ) -> LeafOutcome:
        counter[0] += 1
        return LeafOutcome(
            state={"messages": [], "structured_response": structured}, usage=7
        )

    return Ctx(
        roster=Roster().register("x", builder=lambda *, response_format=None: None),  # unused: leaf_runner faked
        journal=InMemoryJournalStore(),
        leaf_runner=_leaf,
    )


async def test_agent_pydantic_schema_returns_validated_object() -> None:
    claim = Claim(text="t", confident=True)
    ctx = _ctx_with_structured(claim, [0])
    out = await ctx.agent("extract", agent_type="x", schema=Claim)
    assert isinstance(out, Claim)
    assert out.text == "t" and out.confident is True


async def test_agent_dict_schema_returns_validated_object() -> None:
    claim = Claim(text="d", confident=False)
    ctx = _ctx_with_structured(claim, [0])
    out = await ctx.agent(
        "extract",
        agent_type="x",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "confident": {"type": "boolean"}},
            "required": ["text", "confident"],
        },
    )
    assert out.text == "d" and out.confident is False  # attribute access on converted model


async def test_agent_schema_journal_roundtrip_caches_object() -> None:
    claim = Claim(text="cached", confident=True)
    counter = [0]
    ctx = _ctx_with_structured(claim, counter)
    first = await ctx.agent("extract", agent_type="x", schema=Claim)
    second = await ctx.agent("extract", agent_type="x", schema=Claim)  # same key -> hit
    assert counter[0] == 1  # leaf ran once; second served from journal
    assert isinstance(second, Claim) and second.text == "cached"


async def test_agent_schema_partitions_journal_key() -> None:
    claim = Claim(text="t", confident=True)
    counter = [0]
    ctx = _ctx_with_structured(claim, counter)
    await ctx.agent("extract", agent_type="x", schema=Claim)
    await ctx.agent("extract", agent_type="x")  # no schema -> different key -> miss
    assert counter[0] == 2
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/unit/test_agent_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: FAIL — `agent()` got an unexpected keyword 'schema'

- [ ] **Step 4: 改 `_context.py` imports + `agent` overloads + 实现**

imports 增加：

```python
from typing import Any, Protocol, TypeVar, overload

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

from ._result import fold_result, fold_structured
from ._schema import to_pydantic_model
```

`T` 旁边增类型变量：

```python
M = TypeVar("M", bound=BaseModel)
```

把现有 `async def agent(...)` 头替换为三段 overload + 实现（实现体在现有逻辑上插入归一/journal/fold 分派）：

```python
    @overload
    async def agent(
        self, prompt: str, *, agent_type: str, schema: type[M],
        model: str | None = ..., isolation: str = ...,
    ) -> M: ...

    @overload
    async def agent(
        self, prompt: str, *, agent_type: str, schema: dict[str, Any],
        model: str | None = ..., isolation: str = ...,
    ) -> BaseModel: ...

    @overload
    async def agent(
        self, prompt: str, *, agent_type: str, schema: None = ...,
        model: str | None = ..., isolation: str = ...,
    ) -> str: ...

    async def agent(
        self,
        prompt: str,
        *,
        agent_type: str,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        model: str | None = None,
        isolation: str = "shared",
    ) -> str | BaseModel:
        """Run a leaf subagent and return its folded result.

        Without ``schema`` the folded final text is returned. With ``schema`` (a
        pydantic ``BaseModel`` subclass or an inline JSON-schema ``dict``) the leaf
        is built with a matching ``response_format`` and the validated structured
        object is returned — the script reads it by attribute.

        Args:
            prompt: The prompt for the leaf.
            agent_type: The roster name to resolve.
            schema: Optional structured-output schema (pydantic class or JSON-schema
                dict). Requires the roster entry to be registered with a builder.
            model: Optional per-call model override.
            isolation: Isolation mode (part of the journal key).

        Returns:
            The folded final text, or the validated structured object when
            ``schema`` is given.

        Raises:
            KeyError: If ``agent_type`` is not registered.
            ValueError: If ``schema`` is given for a runnable-only roster entry, or
                a dict schema uses an unsupported construct.
            WorkflowBudgetExceededError: If the shared budget is exhausted.
        """
        entry = self._roster.resolve(agent_type)
        structured_model = to_pydantic_model(schema) if schema is not None else None
        effective_model = model if model is not None else entry.default_model
        key = journal_key(
            prompt=prompt,
            agent_type=agent_type,
            model=effective_model,
            schema=structured_model,
            isolation=isolation,
        )
        with self._spans.span(SpanKind.AGENT, agent_type) as span:
            span.set("agent_type", agent_type)
            if _FANOUT_DEPTH.get() == 0:
                self._sequence_guard.observe(key)
            cached = await self._journal.get(key)
            if cached is not None:
                self._budget.record(key, cached.usage)
                span.set("cached", True)
                span.set("usage_tokens", cached.usage)
                if structured_model is not None:
                    return structured_model.model_validate_json(cached.result)
                return cached.result
            self._budget.ensure_within_cap()
            leaf_id = leaf_id_from_key(key)
            response_format = (
                ToolStrategy(structured_model, handle_errors=True)
                if structured_model is not None
                else None
            )
            outcome = await self._gate.run(
                lambda: self._leaf_runner(
                    agent_type,
                    prompt,
                    effective_model,
                    leaf_id=leaf_id,
                    needs_execution=entry.needs_execution,
                    response_format=response_format,
                )
            )
            if structured_model is not None:
                folded_obj: str | BaseModel = fold_structured(outcome.state, structured_model)
                result_str = folded_obj.model_dump_json()
            else:
                folded_obj = fold_result(outcome.state)
                result_str = folded_obj
            await self._journal.put(key, JournalRecord(result=result_str, usage=outcome.usage))
            self._budget.record(key, outcome.usage)
            span.set("cached", False)
            span.set("usage_tokens", outcome.usage)
            return folded_obj
```

> **注意**：`self._leaf_runner(..., response_format=response_format)` 现在总是传 `response_format`（schema-less 时为 `None`）。所有自定义 `leaf_runner` 测试桩须接受该 keyword（默认 `None`）——见 Task 4 Protocol 与 Task 5 测试桩；现存 `tests/unit/test_parallel.py` / `test_pipeline.py` 等的 `_leaf` 桩需补 `response_format: Any = None`（Step 6 回归时若报 TypeError 即补）。

- [ ] **Step 5: 运行确认通过**

Run: `uv run pytest tests/unit/test_agent_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS（4 passed）

- [ ] **Step 6: 回归既有单测 + 补桩**

Run: `uv run pytest tests/unit -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; grep -E "FAILED|ERROR|passed|failed" /tmp/ldw-test.log`
若有 `_leaf` 桩因新 keyword 报 `TypeError`，给该桩签名补 `response_format: Any = None`，重跑至全绿。

- [ ] **Step 7: ruff + pyright**

Run: `uv run ruff check src/langchain_dynamic_workflow/_context.py tests/unit/test_agent_schema.py tests/conftest.py && uv run pyright src/langchain_dynamic_workflow/_context.py`
Expected: 无错误（overload 让 `schema=Claim` 处返回类型精确为 `Claim`）

- [ ] **Step 8: 跑 Task 4 的集成测试（现在应通过）**

Run: `uv run pytest tests/integration/test_g1_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -20 /tmp/ldw-test.log`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/langchain_dynamic_workflow/_context.py tests/unit/test_agent_schema.py tests/conftest.py
git commit -m "feat(agent): schema= structured output (pydantic class or JSON-schema dict)"
```

---

## Task 6: 集成 — fan-out / resume / L2 dict 脚本（`tests/integration/test_g1_schema.py`）

**Files:**
- Modify: `tests/integration/test_g1_schema.py`

- [ ] **Step 1: 追加集成测试**

```python
# tests/integration/test_g1_schema.py — 追加
from langchain_dynamic_workflow import run_workflow_from_source  # 来自 meta 层（见 examples/08）


async def test_schema_in_parallel_fanout() -> None:
    roster = Roster().register("skeptic", builder=_structured_builder)

    async def orchestrate(ctx: Any, args: dict[str, Any]) -> Any:
        verdicts = await ctx.parallel(
            [lambda i=i: ctx.agent(f"claim {i}", agent_type="skeptic", schema=Verdict) for i in range(3)]
        )
        return [v.refuted for v in verdicts if v is not None]

    result = await run_workflow(orchestrate, roster=roster, args={})
    assert result == [False, False, False]


async def test_schema_resume_restores_object_from_journal() -> None:
    # 同一 journal 跨两次 run：第二次命中缓存，结构化对象由 model_validate_json 还原。
    roster = Roster().register("skeptic", builder=_structured_builder)
    journal = InMemoryJournalStore()

    async def orchestrate(ctx: Any, args: dict[str, Any]) -> Any:
        v = await ctx.agent("verify", agent_type="skeptic", schema=Verdict)
        return v.reason

    first = await run_workflow(orchestrate, roster=roster, args={}, journal=journal)
    second = await run_workflow(orchestrate, roster=roster, args={}, journal=journal)
    assert first == second == "solid"


async def test_l2_script_inline_dict_schema_runs_through_gate() -> None:
    # L2 路径：过 AST gate 的脚本内联 dict 字面量 schema（禁 import 下 schema 可用）。
    roster = Roster().register("skeptic", builder=_structured_builder)
    source = '''
async def orchestrate(ctx, args):
    v = await ctx.agent(
        "verify",
        agent_type="skeptic",
        schema={
            "type": "object",
            "properties": {"refuted": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["refuted", "reason"],
        },
    )
    return v.reason
'''
    result = await run_workflow_from_source(source, roster=roster, args={})
    assert result == "solid"
```

> **说明**：`InMemoryJournalStore` 需从 `langchain_dynamic_workflow._journal` 导入（文件顶部已在 Task 4 导入）。`run_workflow` / `run_workflow_from_source` 的精确关键字以仓库现状为准（examples 03 演示 journal 复用、examples 08 演示 `run_script`/源码路径）；若不符按现状调整、保持断言。

- [ ] **Step 2: 运行确认通过**

Run: `uv run pytest tests/integration/test_g1_schema.py -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; tail -25 /tmp/ldw-test.log`
Expected: PASS（4 个集成测试）

- [ ] **Step 3: ruff**

Run: `uv run ruff check tests/integration/test_g1_schema.py && uv run ruff format tests/integration/test_g1_schema.py`
Expected: 无错误

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_g1_schema.py
git commit -m "test(g1): integration — schema in fan-out, resume, and the L2 dict-schema path"
```

---

## Task 7: 验收门 — `07_deep_research` schema 化 + SKILL.md 补 `schema=`

**Files:**
- Modify: `examples/07_deep_research_real_e2e.py`
- Modify: `src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md`

- [ ] **Step 1: SKILL.md 补 DSL 文档 + 一个结构化示例**

在 `## The DSL (\`ctx\` primitives)` 的 `ctx.agent` 条目补 `schema=`：

```markdown
- `await ctx.agent(prompt, *, agent_type, schema=None, model=None, isolation="shared")` — run
  one leaf subagent in a fresh, discarded context. Without `schema` it returns the
  leaf's final **text**. With `schema` — a JSON-schema `dict` written inline (no
  imports needed) — it returns a **validated structured object** you read by
  attribute, so the next line is plain Python over typed data. `agent_type` names a
  registered leaf; a schema requires that leaf to be registered with a builder.
```

在 `## Patterns` 末尾追加：

````markdown
Structured output as the JS-handoff (schema):

```python
async def orchestrate(ctx, args):
    verdict = await ctx.agent(
        f"Refute this claim if you can: {args['claim']}",
        agent_type="skeptic",
        schema={
            "type": "object",
            "properties": {
                "refuted": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["refuted", "reason"],
            "additionalProperties": False,
        },
    )
    return "rejected" if verdict.refuted else "stands"
```
````

> **范围**：本任务只补 `schema=` 的机械文档 + 一个示例。完整质量模式库（adversarial-verify / pipeline-by-default / loop-until-dry 等）是 **G3** 的活，不在 G1。

- [ ] **Step 2: `07_deep_research` 把 extractor/skeptic schema 化**

在 `examples/07_deep_research_real_e2e.py`：
1. 顶部定义两个 pydantic 模型：

```python
from pydantic import BaseModel


class Claim(BaseModel):
    text: str
    checkable: bool


class Verdict(BaseModel):
    refuted: bool
    reason: str
```

2. `extractor` / `skeptic` 改用 builder 注册（`create_deep_agent(model=..., response_format=response_format)`）：

```python
        .register(
            "extractor",
            builder=lambda *, response_format=None: create_deep_agent(
                model=_build_leaf_model("extractor"), response_format=response_format
            ),
            description="Extracts a falsifiable claim",
        )
        .register(
            "skeptic",
            builder=lambda *, response_format=None: create_deep_agent(
                model=_build_leaf_model("skeptic"), response_format=response_format
            ),
            description="Adversarially verifies a claim",
        )
```

> `_build_leaf_model(...)` 用现有的叶子模型构造函数名（见该文件现状，可能是 `_build_leaf("extractor")` 返回 runnable —— 改成返回 model 的形式，或让 builder 闭包内构造 model）。

3. workflow 体把 extractor/skeptic 调用加 `schema=`，reduce 走纯 Python：

```python
    claims = await ctx.parallel(
        [lambda a=a: ctx.agent(f"Extract a checkable claim about: {a}", agent_type="extractor", schema=Claim) for a in angles]
    )
    checkable = [c for c in claims if c is not None and c.checkable]
    verdicts = await ctx.parallel(
        [lambda c=c: ctx.agent(f"Refute if possible: {c.text}", agent_type="skeptic", schema=Verdict) for c in checkable]
    )
    surviving = [c.text for c, v in zip(checkable, verdicts) if v is not None and not v.refuted]
```

- [ ] **Step 3: 离线冒烟（不依赖真实模型 key）**

> 07 是 gated 真模型 e2e（OpenRouter/.env）。在无 key 环境下，至少确认**导入与脚本可解析**、且 schema 化后无语法/类型错误：

Run: `uv run python -c "import ast,sys; ast.parse(open('examples/07_deep_research_real_e2e.py').read()); print('parse-ok')" && uv run ruff check examples/07_deep_research_real_e2e.py && uv run pyright examples/07_deep_research_real_e2e.py`
Expected: `parse-ok` + ruff/pyright 无错误。
（有 key 时按文件头注释跑真实 e2e，验证 surviving 非空、结构化交接成立。）

- [ ] **Step 4: Commit**

```bash
git add examples/07_deep_research_real_e2e.py src/langchain_dynamic_workflow/skills/dynamic-workflow/SKILL.md
git commit -m "example(g1): schema-ify deep_research extractor/skeptic; SKILL.md documents schema="
```

---

## Task 8: 质量闸门（全绿）

**Files:** 无（仅校验）

- [ ] **Step 1: 全量测试**

Run: `uv run pytest -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"; grep -E "passed|failed|error" /tmp/ldw-test.log | tail -5`
Expected: 全绿，无 FAILED/ERROR。

- [ ] **Step 2: ruff（全仓）**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: 无错误。

- [ ] **Step 3: pyright（strict，全仓）**

Run: `uv run pyright > /tmp/ldw-pyright.log 2>&1; echo "EXIT=$?"; tail -5 /tmp/ldw-pyright.log`
Expected: 0 errors。

- [ ] **Step 4: import-linter（分层契约不破）**

Run: `uv run lint-imports > /tmp/ldw-imports.log 2>&1; echo "EXIT=$?"; tail -10 /tmp/ldw-imports.log`
Expected: Contracts kept。
> 若 `_context` → `langchain.agents.structured_output` 的新导入触犯 import 契约，按 `.importlinter` 现状把第三方 `langchain.*` 归入允许的外部层（与既有 `langchain_core` / `deepagents` 同级），不要为绕过契约而下沉逻辑。

- [ ] **Step 5: 收尾 Commit（若 Step 4 调整了配置）**

```bash
git add .importlinter pyproject.toml 2>/dev/null; git commit -m "chore(g1): keep import contracts green for the structured-output import" || echo "nothing to commit"
```

---

## Task 9: 更新 evergreen 设计文档 + UML（反映 schema 能力）

> **为何属于本计划**：`design_docs/{01-engine-mechanism,02-architecture}.md` 与 `uml/` 是 **evergreen** 文档，随研发推进必须同步。它们目前把 `schema` 写成已实现（`01:24/62/76`、`uml/02-class.md:15`），却**未记录** Builder-roster 机制（`01:103` 仍写 roster 条目 = `{name, description, runnable}+{needs_execution,default_model}`）。G1 落地后要把"愿望"校正为"现实"并补全机制。

**Files:**
- Modify: `design_docs/01-engine-mechanism.md`
- Modify: `design_docs/02-architecture.md`
- Modify: `design_docs/uml/02-class.md`
- Modify: `design_docs/uml/03-sequence.md`

- [ ] **Step 1: `01` 叶子调用契约（约 `:103-106`）**——roster 条目描述补 **builder 路径 + `runnable_for` 构建缓存**：预构造 `runnable` 仅服务 schema-less；`agent(schema=)` 经 builder 用 `response_format=ToolStrategy(...)` 构造并按 `(agent_type, schema)` 缓存绑定变体。

- [ ] **Step 2: `01` journal/cache（约 `:62,76`）**——核对与实现一致：dict 来源经 `to_pydantic_model` 归一后再入 key、命中 `model_validate_json` 还原；措辞与实现不符则校正。

- [ ] **Step 3: `01` Decision Log（`:151` 表）**——增 **D-G1a**（注册形态用 callable builder，否决 deep-agent kwargs：依赖倒置 + roster 通用性 + 宿主稳定性）、**D-G1b**（构建缓存归 Roster：内聚 + 进程级生命周期匹配 + 构建期无外求 + 并发安全），标注二者在"预构造 runnable"约束下落实了 D18 的逐次 schema。

- [ ] **Step 4: `01` §2 七原语表（`:24`）**——agent 行补：`schema` 可为 pydantic 类或 **JSON-schema dict（L2 脚本禁 import 时的内联形态，引擎归一为 pydantic）**。

- [ ] **Step 5: `02` register 示例（约 `:51-52`）+ 消费面④（`:43`）**——`register` 示例补 builder 形态（schema-capable 叶子）；primitives 面注明 `agent` 支持 `schema`。

- [ ] **Step 6: `uml/02-class.md`**——`RosterEntry` 增 `builder`；`Roster` 增 `runnable_for(name, *, response_format)` + 构建缓存字段；新增 `SchemaConverter`(`_schema.to_pydantic_model`) 节点并 `Ctx ..> SchemaConverter`；`Ctx.agent` 签名核对（已含 schema）。

- [ ] **Step 7: `uml/03-sequence.md`**——leaf-call/resume 时序补 schema 分支：`normalize schema → runnable_for(response_format) → ToolStrategy 叶 → structured_response → fold_structured → journal dump`，resume 命中 `model_validate_json`。

- [ ] **Step 8: 校验**

Run: `uv run python -c "import re,glob,sys; [open(p).read() for p in glob.glob('design_docs/**/*.md', recursive=True)]; print('read-ok')"` 并人工核对 mermaid 代码块闭合、无坏链；`grep -nE "builder|runnable_for|to_pydantic|SchemaConverter|D-G1" design_docs/01-engine-mechanism.md design_docs/02-architecture.md design_docs/uml/*.md` 确认更新落地。
Expected: 关键字命中、文档可读。

- [ ] **Step 9: Commit**

```bash
git add design_docs/01-engine-mechanism.md design_docs/02-architecture.md design_docs/uml/02-class.md design_docs/uml/03-sequence.md
git commit -m "docs(design): update evergreen engine-mechanism/architecture/uml for agent(schema=) builder-roster"
```

---

## Self-Review（计划对照 spec）

- **Spec 覆盖**：§5.2 双形态接受 → Task1/Task5；§5.2bis overloads → Task5 Step4；§5.3 转换器子集 + fail-loud → Task1；§5.4 Builder roster + runnable_for + 两决策 → Task3/Task4；§5.5 fold_structured → Task2；§5.6 journal 往返 + 确定性 → Task1（determinism 测）/Task5（dump-validate）；§5.8 验收门 07 → Task7；§5.9 测试层级（转换器/agent/集成/L2/真模型）→ Task1/5/6/7；**evergreen 文档同步**（standing rule）→ Task9。**全覆盖。**
- **Placeholder 扫描**：无 TBD/TODO；每个改码步骤含完整代码块；命令均带期望输出。`run_workflow*` 与 `_build_leaf_model` 两处显式标注"以仓库现状为准"——这是对未在本计划定义符号的诚实约束，非占位（执行者照现状对齐）。
- **类型一致性**：`to_pydantic_model`（Task1）、`fold_structured(state, schema)`（Task2）、`Roster.runnable_for(name, *, response_format)` / `RosterEntry.builder`（Task3）、`LeafRunner.__call__(..., response_format=None)`（Task4）、`Ctx.agent(..., schema=..., )`（Task5）签名前后一致；`M = TypeVar(bound=BaseModel)` 贯穿 overload。

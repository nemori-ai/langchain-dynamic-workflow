"""Unit tests for JSON-schema -> pydantic conversion (agent schema normalization)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from langchain_dynamic_workflow import _schema
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
    # The dict converter returns a ``BaseModel`` subtype with dynamically-built
    # fields, so they are read via ``getattr`` rather than static attributes.
    assert getattr(inst, "refuted") is True  # noqa: B009
    assert getattr(inst, "reason") == "x"  # noqa: B009
    assert getattr(inst, "score") is None  # noqa: B009  # optional -> defaulted None


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
    # Dynamically-built nested fields are read via ``getattr`` (see note above).
    items = getattr(inst, "items")  # noqa: B009
    assert getattr(items[0], "title") == "t"  # noqa: B009
    assert getattr(items[0], "impact").value == "high"  # noqa: B009


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


def test_enum_member_name_collision_fails_loud() -> None:
    # "in-progress" and "in_progress" both sanitize to IN_PROGRESS; Python's Enum
    # would silently alias the second and drop a value. The converter must instead
    # fail loud so a leaf returning the dropped value never silently fails later.
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["in-progress", "in_progress", "done"]}
        },
        "required": ["status"],
    }
    with pytest.raises(ValueError, match=r"IN_PROGRESS|map to member name"):
        to_pydantic_model(schema)


@pytest.mark.parametrize("values", [[True, 1], [False, 0], [1, 1.0], [0, 0.0]])
def test_enum_value_equality_collapse_fails_loud(values: list[Any]) -> None:
    # Python's Enum dedups by VALUE (True == 1, 1 == 1.0), so these pairs sanitize to
    # distinct member NAMES yet silently collapse to one member — the name-only guard
    # misses it. The converter must detect value-equality collisions and fail loud.
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer", "enum": values}},
        "required": ["x"],
    }
    with pytest.raises(ValueError, match=r"equal|distinct|identity"):
        to_pydantic_model(schema)


def test_additional_properties_dict_form_fails_loud() -> None:
    # A truthy non-boolean additionalProperties (a typed-extras schema) currently
    # widens to extra="allow", accepting extras of ANY type. v1 does not support
    # typed extras, so the dict form must fail loud rather than silently widen.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": {"type": "string"},
    }
    with pytest.raises(ValueError, match=r"additionalProperties"):
        to_pydantic_model(schema)


def test_required_field_absent_from_properties_fails_loud() -> None:
    # A required name with no matching property is silently ignored: the model is
    # built without the field and an empty object validates. The converter must
    # reject required names that are not declared in properties.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "ghost"],
    }
    with pytest.raises(ValueError, match=r"ghost|required"):
        to_pydantic_model(schema)


@pytest.mark.parametrize(
    "prop",
    [
        {"type": "string", "pattern": "^x$"},
        {"type": "integer", "minimum": 0},
        {"type": "string", "minLength": 3},
        {"type": "string", "format": "email"},
        {"type": "string", "const": "fixed"},
    ],
)
def test_unenforced_constraint_keyword_fails_loud(prop: dict[str, Any]) -> None:
    # Validation keywords the converter neither reads nor lists as unsupported are
    # silently dropped, making a schema look constrained when it is not. They must
    # fail loud — the module's contract is fail-loud, not silent degradation.
    schema = {"type": "object", "properties": {"a": prop}, "required": ["a"]}
    with pytest.raises(ValueError, match=r"unsupported|keyword"):
        to_pydantic_model(schema)


def test_excessive_nesting_depth_fails_loud() -> None:
    # An LLM-authored schema nested past the depth guard must fail fast rather than
    # drive unbounded recursion (stack overflow). White-box on the internal cap.
    limit = _schema._MAX_DEPTH  # pyright: ignore[reportPrivateUsage]
    schema: dict[str, Any] = {"type": "string"}
    for _ in range(limit + 5):
        schema = {"type": "object", "properties": {"child": schema}, "required": ["child"]}
    with pytest.raises(ValueError, match=r"depth|deep|nest"):
        to_pydantic_model(schema)


def test_oversized_enum_fails_loud() -> None:
    limit = _schema._MAX_ENUM_VALUES  # pyright: ignore[reportPrivateUsage]
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer", "enum": list(range(limit + 10))}},
        "required": ["x"],
    }
    with pytest.raises(ValueError, match=r"enum|too many|exceed"):
        to_pydantic_model(schema)


def test_too_many_properties_fails_loud() -> None:
    limit = _schema._MAX_PROPERTIES  # pyright: ignore[reportPrivateUsage]
    props: dict[str, Any] = {f"f{i}": {"type": "string"} for i in range(limit + 10)}
    schema = {"type": "object", "properties": props}
    with pytest.raises(ValueError, match=r"propert|too many|exceed"):
        to_pydantic_model(schema)


def test_model_cache_is_bounded() -> None:
    # Distinct schemas beyond the cache cap must not grow the cache without bound.
    cap = _schema._MAX_CACHE_ENTRIES  # pyright: ignore[reportPrivateUsage]
    for i in range(cap + 50):
        to_pydantic_model({"type": "object", "properties": {f"only_{i}": {"type": "string"}}})
    assert len(_schema._MODEL_CACHE) <= cap  # pyright: ignore[reportPrivateUsage]

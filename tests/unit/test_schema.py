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

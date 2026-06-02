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
from typing import Any, cast

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
    if isinstance(schema, type) and issubclass(schema, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
        return schema
    if isinstance(schema, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
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
            fields[field_name] = (
                py_type | None,
                Field(default=prop.get("default"), **field_kwargs),
            )
    return create_model(name, __config__=ConfigDict(extra=extra), **fields)


def _resolve_type(prop: dict[str, Any], *, parent: str, field: str) -> Any:
    _reject_unsupported(prop)
    if "enum" in prop:
        return _enum_type(prop["enum"], parent=parent, field=field)
    json_type = prop.get("type")
    if json_type in _SCALARS:
        return _SCALARS[json_type]
    if json_type == "array":
        items: Any = prop.get("items")
        if not isinstance(items, dict):
            return list[Any]
        item_schema = cast(dict[str, Any], items)
        inner = _resolve_type(item_schema, parent=parent, field=f"{field}_item")
        return list[inner]
    if json_type == "object":
        return _build_object_model(prop, name=f"{parent}_{_sanitize(field)}")
    raise ValueError(
        f"unsupported JSON-schema type {json_type!r} for field {field!r}; v1 supports "
        "object / array / string / integer / number / boolean / enum"
    )


def _enum_type(values: list[Any], *, parent: str, field: str) -> type[Enum]:
    # Distinct JSON-schema enum values can sanitize to the same Python identifier
    # (e.g. "in-progress" and "in_progress" both -> "IN_PROGRESS"). Python's Enum
    # would silently treat the second as an alias and drop one value, violating
    # the module's fail-loud contract — so detect the collision and raise instead.
    members: dict[str, Any] = {}
    claimed: dict[str, Any] = {}
    for index, value in enumerate(values):
        member = _sanitize(str(value)).upper() or f"VALUE_{index}"
        if member[0].isdigit():
            member = f"V_{member}"
        if member in claimed:
            raise ValueError(
                f"enum values {claimed[member]!r} and {value!r} for field {field!r} both map "
                f"to member name {member!r}; rename one value so they stay distinct"
            )
        claimed[member] = value
        members[member] = value
    return cast("type[Enum]", Enum(f"{parent}_{_sanitize(field)}_Enum", members))


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

"""Content-hash journal for leaf ``agent()`` results.

The journal memoizes leaf results keyed by the content hash of the call inputs,
giving resume/replay the "completed agents return cached results" guarantee
without relying on LangGraph's index-based task cache. Writes are *success-only*:
callers persist a result only after it has been produced and validated, so a
failed or interrupted leaf is never cached and replayed as success.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


def journal_key(
    *,
    prompt: str,
    agent_type: str | None,
    model: str | None,
    schema: type[BaseModel] | None,
    isolation: str,
) -> str:
    """Compute the content-hash journal key for a leaf call.

    The key is the SHA-256 of a canonical JSON encoding of the call inputs that
    affect the result. ``label`` and ``phase`` are intentionally excluded — they
    are display-only and must never invalidate the cache.

    Args:
        prompt: The leaf prompt.
        agent_type: The roster name the leaf resolves to.
        model: Optional model override.
        schema: Optional structured-output schema; hashed via its JSON schema.
        isolation: The isolation mode string.

    Returns:
        A hex SHA-256 digest uniquely identifying this leaf call's inputs.
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "agent_type": agent_type,
        "model": model,
        "schema": schema.model_json_schema() if schema is not None else None,
        "isolation": isolation,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class JournalStore(Protocol):
    """Storage backend for journaled leaf results.

    Implementations must be safe for concurrent ``get``/``put`` from multiple
    in-flight leaves within a single workflow run.
    """

    async def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` on miss."""
        ...

    async def put(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key`` (success-only; caller-enforced)."""
        ...


class InMemoryJournalStore:
    """In-process journal store; the v1 default (same-session resume)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` on miss."""
        return self._data.get(key)

    async def put(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key``."""
        self._data[key] = value

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class JournalRecord:
    """A journaled leaf result paired with the token usage it consumed.

    Storing usage alongside the result is what makes ``budget.spent()``
    reconstructable on replay: a resumed run that serves a leaf from the journal
    re-counts that leaf's usage from the record instead of re-invoking the model,
    so the cumulative spend rebuilds to exactly the first run's value.

    Attributes:
        result: The leaf's folded final text.
        usage: The total tokens consumed by the leaf invocation that produced
            ``result``. ``0`` when usage was unavailable (e.g. a model that emits
            no ``usage_metadata``).
    """

    result: str
    usage: int


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


def race_key(*, candidate_keys: Sequence[str], win_tag: str) -> str:
    """Compute the content-hash key for a ``ctx.race`` decision.

    The key is the SHA-256 of a canonical JSON encoding of the candidates' leaf
    keys (in order) plus the win tag, under a ``"race"`` namespace marker so it can
    never collide with a leaf :func:`journal_key`. ``win_tag`` is folded in
    deliberately: two races over the *same* candidates but with *different* win
    predicates must use different tags, or the second race replays the first's
    decision and silently bypasses the changed predicate.

    Args:
        candidate_keys: Each candidate's leaf journal key, in candidate order.
        win_tag: A caller-supplied label distinguishing this race's win criterion.

    Returns:
        A hex SHA-256 digest uniquely identifying this race decision.
    """
    payload: dict[str, Any] = {
        "kind": "race",
        "candidates": list(candidate_keys),
        "win_tag": win_tag,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def signoff_key(*, position: int, tag: str) -> str:
    """Compute the content-hash journal key for a ``ctx.checkpoint`` sign-off gate.

    A sign-off decision is journaled like a leaf result so a resumed run replays an
    already-approved gate from the journal at zero cost and only an un-decided gate
    parks. The key is the SHA-256 of the gate's ordinal position in the sequential
    orchestration path plus its optional tag, under a ``"signoff"`` namespace marker
    so it can never collide with a leaf :func:`journal_key` or a :func:`race_key`.
    The ``ask`` payload is intentionally excluded — it is display-only (like a
    leaf's ``label``) and must never partition the gate's identity across replay.

    Args:
        position: The gate's zero-based ordinal among the run's ``checkpoint``
            calls (deterministic on the sequential, depth-0 path).
        tag: The caller-supplied gate label (empty when none).

    Returns:
        A hex SHA-256 digest uniquely identifying this sign-off gate.
    """
    payload: dict[str, Any] = {
        "kind": "signoff",
        "position": position,
        "tag": tag,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def loop_key(*, position: int, iteration: int) -> str:
    """Compute the content-hash determinism key for one ``ctx.loop_until`` iteration.

    A ``loop_until`` whose body fans out (its ``agent()`` work runs inside
    ``parallel`` / ``race`` / ``dag``) contributes no leaf keys to the depth-0
    determinism sequence — the fan-out leaves run at depth > 0 and are excluded by
    design. To keep the loop's iteration count guarded independently of whether the
    body fans out, the loop records one key per iteration into that same ordered
    sequence. The key is the SHA-256 of the loop's ordinal position among the run's
    ``loop_until`` calls plus the iteration index, under a ``"loop"`` namespace
    marker so it can never collide with a leaf :func:`journal_key`, a
    :func:`race_key`, or a :func:`signoff_key`.

    Args:
        position: The loop's zero-based ordinal among the run's ``loop_until``
            calls (deterministic on the sequential, depth-0 path).
        iteration: The zero-based iteration index within this loop.

    Returns:
        A hex SHA-256 digest uniquely identifying this loop iteration.
    """
    payload: dict[str, Any] = {
        "kind": "loop",
        "position": position,
        "iteration": iteration,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class JournalStore(Protocol):
    """Storage backend for journaled leaf results and the call sequence.

    Implementations must be safe for concurrent ``get``/``put`` from multiple
    in-flight leaves within a single workflow run. Each result value is a
    :class:`JournalRecord` carrying the leaf's folded result and token usage. The
    store also persists the ordered sequence of leaf call-keys, which the
    determinism backstop replays to detect divergence; the sequence lives in its
    own slot so it never collides with content-hash result keys.
    """

    async def get(self, key: str) -> JournalRecord | None:
        """Return the cached record for ``key``, or ``None`` on miss."""
        ...

    async def put(self, key: str, value: JournalRecord) -> None:
        """Persist ``value`` under ``key`` (success-only; caller-enforced)."""
        ...

    async def get_sequence(self) -> list[str] | None:
        """Return the recorded ordered call-key sequence, or ``None`` if unset."""
        ...

    async def put_sequence(self, sequence: list[str]) -> None:
        """Persist the ordered call-key sequence observed on a completed run."""
        ...

    async def get_progress_count(self) -> int:
        """Return how many progress entries a prior run already delivered."""
        ...

    async def put_progress_count(self, count: int) -> None:
        """Persist the count of progress entries delivered on a completed run."""
        ...


class InMemoryJournalStore:
    """In-process journal store; the v1 default (same-session resume)."""

    def __init__(self) -> None:
        self._data: dict[str, JournalRecord] = {}
        self._sequence: list[str] | None = None
        self._progress_count: int = 0

    async def get(self, key: str) -> JournalRecord | None:
        """Return the cached record for ``key``, or ``None`` on miss."""
        return self._data.get(key)

    async def put(self, key: str, value: JournalRecord) -> None:
        """Persist ``value`` under ``key``."""
        self._data[key] = value

    async def get_sequence(self) -> list[str] | None:
        """Return the recorded ordered call-key sequence, or ``None`` if unset."""
        return list(self._sequence) if self._sequence is not None else None

    async def put_sequence(self, sequence: list[str]) -> None:
        """Persist the ordered call-key sequence observed on a completed run."""
        self._sequence = list(sequence)

    async def get_progress_count(self) -> int:
        """Return how many progress entries a prior run already delivered."""
        return self._progress_count

    async def put_progress_count(self, count: int) -> None:
        """Persist the count of progress entries delivered on a completed run."""
        self._progress_count = count

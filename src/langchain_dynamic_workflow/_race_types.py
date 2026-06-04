"""Value types for ``ctx.race`` — the journaled best-of-N early-exit primitive.

A race enters several content-hashable agent-call specs (:class:`RaceCandidate`)
and returns the first whose result satisfies the win predicate
(:class:`RaceResult`), cancelling the in-flight losers. These types are pure data —
no engine state and no ``agent()`` call — so they live beside the runtime (like the
reduce helpers) rather than inside it, and they are injected into the ``run_script``
namespace so a host-authored script constructs and reads them by name without an
import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class RaceCandidate:
    """One content-hashable agent-call spec entered into a race.

    The fields mirror the keyed inputs of ``Ctx.agent`` so a candidate's journal
    key — and therefore the race's replay identity — is derived exactly as a direct
    ``agent()`` call would derive it.

    Attributes:
        prompt: The leaf prompt.
        agent_type: The roster name the candidate resolves to.
        schema: Optional structured-output schema (pydantic ``BaseModel`` subclass
            or inline JSON-schema ``dict``). All candidates in one race must agree:
            either all schema-less or all bound to the same schema.
        model: Optional per-candidate model override; ``None`` uses the roster
            entry's default model.
        isolation: Isolation mode (part of the journal key).
    """

    prompt: str
    agent_type: str
    schema: type[BaseModel] | dict[str, Any] | None = None
    model: str | None = None
    isolation: str = "shared"


@dataclass(frozen=True, slots=True)
class RaceResult[T]:
    """The outcome of a race: the first result that satisfied ``win``, or none.

    Attributes:
        winner: The winning candidate's result, or ``None`` if no candidate
            satisfied the win predicate.
        winner_index: The winning candidate's position in the input sequence, or
            ``None`` when there was no winner.
    """

    winner: T | None
    winner_index: int | None

    @property
    def won(self) -> bool:
        """Whether a candidate satisfied the win predicate."""
        return self.winner_index is not None

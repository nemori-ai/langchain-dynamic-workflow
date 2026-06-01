"""Determinism backstop — the journal-divergence oracle.

The substrate enforces determinism only through a single bare ``assert`` that
``python -O`` strips entirely, so the engine builds its own universal backstop.

The journal records the *ordered* sequence of leaf ``agent()`` call-keys produced
on the first run. On replay the script must reproduce that exact sequence: the
k-th call must carry the recorded k-th key. The moment a script diverges — a
different key at some position, or simply more calls than were recorded — the
guard raises :class:`WorkflowDeterminismError` instead of serving a positionally
misaligned cache entry. This catches non-determinism that would otherwise feed
the wrong cached result back into the orchestration, regardless of source
(hand-written or LLM-authored).
"""

from __future__ import annotations

from ._errors import WorkflowDeterminismError


class CallSequenceGuard:
    """Records / validates the ordered sequence of leaf call-keys for one run.

    On a fresh run ``recorded`` is ``None`` and every observed key is appended to
    build the sequence that will be persisted for future replays. On a replay run
    ``recorded`` is the previously-persisted sequence; each observed key is
    checked against the recorded key at the same position and a mismatch (or an
    out-of-range call) fails loud.

    The guard is single-run scoped and not concurrency-safe on its own; callers
    must serialize :meth:`observe` (the engine observes a key at the point a leaf
    is dispatched, under the orchestration's own ordering).

    Args:
        recorded: The call-key sequence from a prior run, or ``None`` for a fresh
            (recording) run.
    """

    def __init__(self, *, recorded: list[str] | None) -> None:
        self._recorded = recorded
        self._observed: list[str] = []

    @property
    def sequence(self) -> list[str]:
        """The call-keys observed so far this run, in order."""
        return list(self._observed)

    def observe(self, key: str) -> None:
        """Record (fresh run) or validate (replay) the next leaf call-key.

        Args:
            key: The content-hash journal key of the leaf about to be dispatched.

        Raises:
            WorkflowDeterminismError: On replay, if ``key`` does not match the
                recorded key at this position, or if this call is beyond the end
                of the recorded sequence.
        """
        position = len(self._observed)
        if self._recorded is not None:
            if position >= len(self._recorded):
                raise WorkflowDeterminismError(
                    "workflow replay diverged: the script issued an agent() call at "
                    f"position {position} beyond the {len(self._recorded)} call(s) recorded "
                    "on the first run; the orchestration is non-deterministic"
                )
            expected = self._recorded[position]
            if key != expected:
                raise WorkflowDeterminismError(
                    "workflow replay diverged at agent() call "
                    f"position {position}: recorded key {expected!r} but the script produced "
                    f"{key!r}; refusing to serve a positionally misaligned cache entry"
                )
        self._observed.append(key)

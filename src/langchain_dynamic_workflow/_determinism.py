"""Determinism backstop — the journal-divergence oracle.

The substrate enforces determinism only through a single bare ``assert`` that
``python -O`` strips entirely, so the engine builds its own universal backstop.

The journal records the *ordered* sequence of leaf ``agent()`` call-keys produced
on the first run. On replay the script must reproduce that exact sequence: the
k-th call must carry the recorded k-th key. The moment a script diverges — a
different key at some position, more calls than were recorded, or *fewer* calls
than were recorded (an early-terminating replay, caught at finalize) — the guard
raises :class:`WorkflowDeterminismError` instead of serving a positionally
misaligned cache entry. This catches non-determinism that would otherwise feed
the wrong cached result back into the orchestration, regardless of source
(hand-written or LLM-authored).

Only leaf calls on the sequential ``agent()`` path are recorded. Calls dispatched
inside ``parallel()`` / ``pipeline()`` fan-out are intentionally excluded: their
observe order is wall-clock-dependent (it follows per-leaf completion timing, not
the orchestration's source order), so recording them would trip the backstop
spuriously on a perfectly deterministic resume. Fan-out-internal sequencing is a
later concern; the sequential path is where positional cache misalignment is both
possible and detectable.
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

    def finalize(self) -> None:
        """Reconcile the observed call count against the record on a clean return.

        :meth:`observe` catches forward divergence (a mismatched key, or a call
        beyond the recorded length) at the moment it happens. It cannot catch an
        *under-run* — a replay that terminates after issuing fewer calls than were
        recorded — because nothing is observed at the missing positions. This
        finalize step closes that hole: called once after the orchestration
        returns cleanly, it fails loud if a replay produced fewer calls than the
        first run. An early-terminating replay is just as much a
        non-deterministic control-flow break as an extra or mismatched call.

        On a fresh (recording) run there is nothing to reconcile against, so this
        is a no-op.

        Raises:
            WorkflowDeterminismError: On replay, if fewer calls were observed than
                were recorded on the first run.
        """
        if self._recorded is None:
            return
        observed_count = len(self._observed)
        recorded_count = len(self._recorded)
        if observed_count < recorded_count:
            raise WorkflowDeterminismError(
                "workflow replay diverged: the script issued only "
                f"{observed_count} agent() call(s) but {recorded_count} were recorded on the "
                "first run; an early-terminating replay is non-deterministic control flow"
            )

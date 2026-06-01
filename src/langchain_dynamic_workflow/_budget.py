"""Shared token budget — enforcement + replay-reconstructable spend.

The budget is a single shared token pool drawn down by every leaf ``agent()``
call across a workflow run. Two invariants make it correct under resume:

- ``spent()`` is rebuilt from per-leaf usage recorded against each distinct
  call-key, so a resumed run that serves leaves from the journal re-counts their
  usage from the journal records and reaches *exactly* the first run's cumulative
  total. (Recording per distinct key also keeps a repeated identical ``agent()``
  call — a journal cache hit, zero new model tokens — from being double-counted.)
- the cap is checked *before* a new leaf is dispatched: once ``spent()`` reaches
  ``total``, the next ``agent()`` raises :class:`WorkflowBudgetExceededError`,
  while leaves already in flight finish and keep their results.

This is a *soft cap*, not a hard ceiling, and the distinction matters under
concurrent fan-out. ``ensure_within_cap()`` runs synchronously before a leaf is
dispatched, but ``record()`` runs only after the leaf completes. So in a
``parallel()`` barrier of ``N`` leaves dispatched while the pool still has
headroom, all ``N`` can pass the pre-dispatch check before any has recorded its
usage. The barrier can therefore overshoot ``total`` by up to the combined usage
of the leaves admitted in that window — bounded by the concurrency gate's limit
(at most ``gate.limit`` leaves run at once) and never beyond a single barrier's
worth of work. The next ``agent()`` after the barrier settles sees the overshot
``spent()`` and refuses. This is the intended trade: in-flight leaves keep their
results rather than being cancelled to claw back tokens, and ``remaining()``
floors at zero so the overshoot never surfaces as a negative figure.

A budget with no ``total`` never trips the cap and reports an unbounded
``remaining()`` (the ``while budget.remaining() > THRESHOLD`` loop idiom).

Per-leaf usage is metered by forwarding a ``UsageMetadataCallbackHandler`` into
the leaf's invocation config — the same callback-forwarding deepagents performs
for its own subagents — and summing the total tokens it aggregates across every
(possibly nested) model call the leaf makes.
"""

from __future__ import annotations

import math

from langchain_core.callbacks import UsageMetadataCallbackHandler

from ._errors import WorkflowBudgetExceededError


def total_tokens_from_handler(handler: UsageMetadataCallbackHandler) -> int:
    """Sum total tokens across every model the handler aggregated for one leaf.

    The handler keys usage per model name, so a leaf that calls more than one
    model (or the same model several times) is summed across all of them.

    Args:
        handler: A usage callback handler scoped to a single leaf invocation.

    Returns:
        The leaf's total token usage; ``0`` when no usage was reported.
    """
    return sum(int(usage.get("total_tokens", 0)) for usage in handler.usage_metadata.values())


class Budget:
    """A shared token pool with replay-reconstructable spend and a hard cap.

    Args:
        total: The token ceiling, or ``None`` for an unbounded budget.
    """

    def __init__(self, *, total: int | None) -> None:
        self._total = total
        # Usage keyed by leaf call-key: a repeated identical call (a journal hit)
        # overwrites rather than adds, so spent() never double-counts one leaf.
        self._usage_by_key: dict[str, int] = {}

    @property
    def total(self) -> int | None:
        """The configured token ceiling, or ``None`` if unbounded."""
        return self._total

    def record(self, key: str, usage: int) -> None:
        """Attribute ``usage`` tokens to the leaf identified by ``key``.

        Recording is idempotent per key: re-recording the same key (e.g. a cache
        hit replayed on resume) sets that leaf's usage rather than adding to it,
        which is what keeps ``spent()`` reconstructable and free of
        double-counting.

        Args:
            key: The leaf's content-hash call-key.
            usage: The total tokens the leaf consumed.
        """
        self._usage_by_key[key] = usage

    def spent(self) -> int:
        """Return cumulative tokens spent across distinct leaves this run."""
        return sum(self._usage_by_key.values())

    def remaining(self) -> float:
        """Return tokens left before the cap; ``math.inf`` when unbounded.

        Floors at zero so an over-budget overshoot (in-flight leaves landing past
        the cap) never reports a negative figure.
        """
        if self._total is None:
            return math.inf
        return max(0, self._total - self.spent())

    def ensure_within_cap(self) -> None:
        """Raise if the pool is exhausted, guarding a new ``agent()`` dispatch.

        Raises:
            WorkflowBudgetExceededError: If a ``total`` is set and ``spent()`` has
                reached it.
        """
        if self._total is not None and self.spent() >= self._total:
            raise WorkflowBudgetExceededError(
                f"workflow budget exhausted: spent {self.spent()} of {self._total} tokens; "
                "refusing to dispatch a new agent() leaf (in-flight leaves keep their results)"
            )

"""Engine-level fail-loud exceptions.

The substrate (LangGraph) treats determinism as a convention and provides no
budget concept at all. The engine raises these explicitly so a replay divergence
or an exhausted token pool surfaces as a loud failure rather than silent
corruption.
"""

from __future__ import annotations


class WorkflowDeterminismError(RuntimeError):
    """Raised when a replay run's ``agent()`` call sequence diverges from the record.

    The journal records the ordered sequence of leaf call-keys on the first run.
    On replay the script must reproduce that exact sequence; if its k-th call-key
    differs from the recorded one (or it issues more calls than were recorded),
    the orchestration is no longer deterministic and feeding it a positionally
    misaligned cache entry would silently corrupt the result. The engine raises
    this instead.
    """


class WorkflowBudgetExceededError(RuntimeError):
    """Raised when a new ``agent()`` call is attempted after the budget is exhausted.

    The shared token budget is checked before a leaf is dispatched: once
    ``budget.spent()`` has reached ``budget.total``, any further ``agent()`` call
    raises this. Leaves already in flight are allowed to finish and their results
    are kept.
    """

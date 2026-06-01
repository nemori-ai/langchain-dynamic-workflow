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


class WorkflowNestingError(RuntimeError):
    """Raised when a workflow nests another workflow more than one level deep.

    A workflow script may inline another workflow with ``ctx.workflow(name, args)``
    exactly one level; the inner workflow runs in the same durable-execution scope
    and shares the parent's journal, budget, and concurrency gate. Calling
    ``ctx.workflow(...)`` again from inside an already-nested workflow would create
    a second nesting level, which the engine refuses by raising this instead of
    silently allowing unbounded recursion.
    """

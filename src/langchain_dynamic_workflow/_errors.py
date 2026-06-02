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


class WorkflowScriptError(RuntimeError):
    """Raised when an LLM-authored orchestration script is rejected before execution.

    The meta layer compiles an untrusted source string into an orchestration
    callable behind a static AST gate. This is raised when the source cannot be
    turned into a safe, runnable script: a syntax error, an AST-gate security
    violation (an import, dunder attribute traversal, a banned builtin, or a
    ``str.format`` attribute-injection vector), or a structural problem (the
    script does not define an ``async def orchestrate(ctx, args)`` coroutine). The
    message enumerates every violation found in one pass so the author can fix them
    all at once and resubmit.

    The AST gate guards against an honest model's slip, not an adversary: an
    in-process restricted ``exec`` is not a security sandbox. Only compile source
    you authored yourself; for adversarial input use an out-of-process isolation
    backend.
    """

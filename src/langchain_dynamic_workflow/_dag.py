"""Dependency-order (topological) fan-out scheduler — the fourth fan-out frame.

LangGraph offers no partial-order fan-out: ``Send`` is a flat map-reduce barrier and
the engine's own ``parallel`` (barrier) / ``pipeline`` (linear stages) carry no
predecessor model. ``ctx.dag`` adds one: given nodes with declared dependencies, a
node runs only once all its predecessors have completed, and receives their results.
Ready nodes run concurrently; a node whose predecessor failed is skipped, propagating
along the dependency tree.

Mechanics:

- The graph is validated eagerly (duplicate id, unknown dependency, self-dependency,
  cycle) before any node runs, so a graph with no topological order fails loud up
  front rather than deadlocking the scheduler.
- The scheduler tracks in-degrees (Kahn). A node is launched the instant its last
  predecessor settles — there is no level barrier, so a fast branch races ahead of a
  slow sibling branch (the no-barrier spirit of ``pipeline``).
- Concurrency is bounded at the leaf: the ``agent()`` calls inside a node's ``run``
  acquire the shared gate. The scheduler itself holds no slot, so a ``dag`` nested
  inside a node's ``run`` cannot starve the pool into deadlock.
- A node whose ``run`` raises an ordinary exception is recorded as *failed* and its
  result is ``None``; every node that (transitively) depends on it is *skipped* and
  also lands as ``None``. A node that legitimately returns ``None`` is NOT failed, so
  its dependents still run (seeing ``None`` for that predecessor). Failure is tracked
  separately from the ``None`` result value precisely to keep this distinction.
- An engine control-flow signal (budget / determinism / a malformed nested dag) is
  never masked: the first one is recorded, no further nodes are launched, in-flight
  nodes drain, and it is re-raised after a clean teardown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from ._errors import WORKFLOW_CONTROL_FLOW_SIGNALS, WorkflowDagError


@dataclass(slots=True)
class DagNode:
    """One node in a ``ctx.dag`` dependency graph.

    All three fields are required; a root node is written with an explicit empty
    ``deps`` (``DagNode("pkg", deps=[], run=...)``), matching how the graph reads.

    Attributes:
        id: The node's unique identifier within the graph; results are keyed by it.
        deps: The ids this node depends on. The node runs only after all of them have
            settled, and its ``run`` receives their results.
        run: A callable taking a ``{dep_id: result}`` mapping of this node's
            predecessors' results and returning a coroutine of the node's result
            (typically a closure over an ``agent()`` call).
    """

    id: str
    deps: Sequence[str]
    run: Callable[[dict[str, Any]], Coroutine[Any, Any, Any]]


def _validate_dag(nodes: Sequence[DagNode]) -> None:
    """Reject a structurally invalid graph before any node is scheduled.

    Args:
        nodes: The graph's nodes.

    Raises:
        WorkflowDagError: On a duplicate id, a dependency on an unknown id, a node
            depending on itself, or a dependency cycle.
    """
    ids = [node.id for node in nodes]
    id_set = set(ids)
    if len(id_set) != len(ids):
        dups = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        raise WorkflowDagError(f"duplicate DagNode id(s): {dups}")
    for node in nodes:
        if node.id in node.deps:
            raise WorkflowDagError(f"DagNode {node.id!r} depends on itself")
        unknown = sorted({dep for dep in node.deps if dep not in id_set})
        if unknown:
            raise WorkflowDagError(f"DagNode {node.id!r} depends on unknown id(s) {unknown}")
    # Kahn pass: if not every node can be drained, the remainder forms a cycle.
    indegree = {node.id: len(set(node.deps)) for node in nodes}
    dependents: dict[str, list[str]] = {node.id: [] for node in nodes}
    for node in nodes:
        for dep in set(node.deps):
            dependents[dep].append(node.id)
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    drained = 0
    while queue:
        node_id = queue.pop()
        drained += 1
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if drained != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree > 0)
        raise WorkflowDagError(f"DagNode dependency cycle among {cyclic}")


async def run_dag(nodes: Sequence[DagNode]) -> dict[str, Any | None]:
    """Run a dependency graph in topological order and collect per-node results.

    Args:
        nodes: The graph's nodes; each runs after all its ``deps`` have settled and
            receives a ``{dep_id: result}`` mapping of their results.

    Returns:
        A mapping ``{node_id: result}``. A node whose ``run`` raised, or that was
        skipped because a predecessor failed, maps to ``None``.

    Raises:
        WorkflowDagError: If the graph is structurally invalid (validated up front).
        Exception: The first engine control-flow signal raised by any node's ``run``
            (budget / determinism / a malformed nested dag), re-raised after the
            in-flight nodes drain.
    """
    _validate_dag(nodes)
    if not nodes:
        return {}

    by_id = {node.id: node for node in nodes}
    indegree = {node.id: len(set(node.deps)) for node in nodes}
    dependents: dict[str, list[str]] = {node.id: [] for node in nodes}
    for node in nodes:
        for dep in set(node.deps):
            dependents[dep].append(node.id)

    results: dict[str, Any | None] = {node.id: None for node in nodes}
    failed: set[str] = set()
    aborted: list[BaseException] = []
    running: dict[asyncio.Task[Any], str] = {}

    def _release(node_id: str) -> list[str]:
        """Decrement dependents' in-degree; return those that just became ready."""
        newly_ready: list[str] = []
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                newly_ready.append(dependent)
        return newly_ready

    def _start(node_id: str) -> list[str]:
        """Launch a ready node, or skip it; return nodes that became ready as a result.

        A node is skipped (without running) when the run is already aborting or when
        any of its predecessors failed/were skipped — the skip cascades to its own
        dependents.
        """
        node = by_id[node_id]
        if aborted or any(dep in failed for dep in node.deps):
            failed.add(node_id)
            return _release(node_id)
        deps_view = {dep: results[dep] for dep in node.deps}
        running[asyncio.create_task(node.run(deps_view))] = node_id
        return []

    pending = [node_id for node_id, degree in indegree.items() if degree == 0]
    try:
        while pending or running:
            while pending:
                pending.extend(_start(pending.pop()))
            if not running:
                break
            done, _ = await asyncio.wait(list(running), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                node_id = running.pop(task)
                error = task.exception()
                if error is not None:
                    if isinstance(error, WORKFLOW_CONTROL_FLOW_SIGNALS) and not aborted:
                        aborted.append(error)
                    failed.add(node_id)
                else:
                    results[node_id] = task.result()
                pending.extend(_release(node_id))
    finally:
        # Defensive: if the loop unwinds unexpectedly (e.g. run_dag is cancelled
        # externally), cancel any still-running node tasks AND await them so they are
        # not left pending/unawaited. return_exceptions swallows their CancelledError.
        for task in running:
            if not task.done():
                task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    if aborted:
        raise aborted[0]
    return results

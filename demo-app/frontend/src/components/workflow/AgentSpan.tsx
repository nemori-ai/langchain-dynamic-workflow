import { useEffect, useState } from "react";

/**
 * One node of a leaf's interior callback subtree, as the backend's UiAdapter
 * serializes it (ui_adapter.py `on_leaf_event`). A node is one interior `run_id` rolled
 * from its start and end edges; `parent_run_id` is `null` at a subtree root. Only these
 * structural fields are sent — the subtree is shape-only by design (no tool args, no
 * model text), so the drill-in shows the interior's shape, never its payloads.
 */
type SubtreeNode = {
  run_id: string;
  parent_run_id: string | null;
  kind: string;
  name: string;
  phase: string;
};

/**
 * Generative-UI component for one leaf execution. The backend's UiAdapter emits an
 * `agent_span` for every `agent` leaf span, in two edges that share one `event_id`
 * (the engine-minted span id) and fold onto one card via the SDK's `merge` reducer:
 *
 * - a BEGIN edge at span-open (ui_adapter.py `on_span_begin`) carrying `running: true`
 *   plus `started_at` (wall-clock open time, seconds) — the completion fields are not
 *   yet known, so `duration_s` / `cached` / `usage_tokens` are absent;
 * - an END edge at span-close (ui_adapter.py `_emit_agent`) carrying `running: false`
 *   plus `cached`, `usage_tokens`, `duration_s`, and any `error`, sent with `merge` so
 *   the SDK shallow-merges these onto the begin card in place.
 *
 * Because both edges land on the same card, this component re-renders from the running
 * state to the completed state without a second card appearing: while `running` is true
 * it shows a running chip and a live elapsed timer ticking off `started_at`; once the
 * end edge merges `running: false` it switches to the completed render (final
 * `duration_s`, `cached` shading, tokens, error). A completed fresh leaf additionally
 * renders a default-collapsed `<details>` drill-in of its interior sub-trace when a
 * `subtree` is present (see AgentSubtree); a cached leaf carries no `subtree`.
 *
 * A cached leaf (a resume / journal hit) is visually distinguished from a fresh run:
 * the engine re-emits the same leaf on resume with `cached` flipped true, near-zero
 * `duration_s`, and zero new tokens, so the cached state is the headline of the resume
 * story and is shaded distinctly here. (A cached leaf additionally surfaces its own
 * `journal_badge` event — see JournalBadge.)
 */
export function AgentSpan(props: {
  kind: string;
  name: string;
  agent_type: string;
  running?: boolean;
  started_at?: number | null;
  cached?: boolean;
  usage_tokens?: number | null;
  duration_s?: number | null;
  error?: string | null;
  subtree?: SubtreeNode[] | null;
  truncated?: boolean;
  event_id: string;
}) {
  const isRunning = props.running === true;

  if (isRunning) {
    return (
      <RunningAgentSpan
        event_id={props.event_id}
        agent_type={props.agent_type}
        name={props.name}
        started_at={props.started_at ?? null}
      />
    );
  }

  const cached = props.cached === true;
  return (
    <div
      data-testid="agent-span"
      data-event-id={props.event_id}
      data-running="false"
      className={
        cached
          ? "my-1 rounded-md border border-amber-200 bg-amber-50/70 px-3 py-2 text-sm"
          : "my-1 rounded-md border border-sky-200 bg-sky-50/60 px-3 py-2 text-sm"
      }
    >
      <div className="flex items-center gap-2">
        <span
          className={
            cached
              ? "rounded bg-amber-100 px-1.5 py-0.5 font-mono text-xs tracking-wide text-amber-700 uppercase"
              : "rounded bg-sky-100 px-1.5 py-0.5 font-mono text-xs tracking-wide text-sky-700 uppercase"
          }
        >
          {props.agent_type}
        </span>
        <span className="font-medium text-slate-800">{props.name}</span>
        {cached && (
          <span className="rounded-full bg-amber-200 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-amber-800">
            ♻ cached
          </span>
        )}
        {props.duration_s != null && (
          <span className="ml-auto font-mono text-xs text-slate-400">
            {props.duration_s.toFixed(2)}s
          </span>
        )}
      </div>

      <div className="mt-1 flex items-center gap-3 font-mono text-xs text-slate-500">
        <span>
          {cached
            ? "0 new tokens (journal hit)"
            : `${props.usage_tokens ?? 0} tokens`}
        </span>
      </div>

      {props.error != null && (
        <div className="mt-1 text-xs text-rose-600">error: {props.error}</div>
      )}

      {props.subtree != null && props.subtree.length > 0 && (
        <AgentSubtree
          nodes={props.subtree}
          truncated={props.truncated === true}
        />
      )}
    </div>
  );
}

// Cap the rendered depth as a frontend-side defense; the backend already bounds the
// node count, but a malformed `parent_run_id` chain should never recurse without limit.
const _MAX_SUBTREE_DEPTH = 32;

/**
 * The drill-in nested sub-trace of a fresh leaf's interior, as a default-collapsed
 * `<details>`. The flat `subtree` node list is rebuilt into a parent/child tree
 * (`run_id` -> children, roots are the nodes whose `parent_run_id` is null) and rendered
 * as a depth-indented list of each node's `kind` / `name` / `phase`.
 *
 * Only a freshly-executed leaf carries a `subtree` (the engine taps the interior only on
 * the live execution path), so a cached / replayed leaf has none and this subcomponent is
 * never rendered — the cache-hit story stays the `journal_badge` chip, with no drill-in.
 * When the backend capped a pathological interior it sets `truncated`, surfaced here so
 * the count reads honestly.
 */
function AgentSubtree(props: { nodes: SubtreeNode[]; truncated: boolean }) {
  const { nodes, truncated } = props;
  // Rebuild the parent/child tree: index children by parent_run_id, collect the roots.
  const childrenByParent = new Map<string, SubtreeNode[]>();
  const roots: SubtreeNode[] = [];
  for (const node of nodes) {
    if (node.parent_run_id == null) {
      roots.push(node);
      continue;
    }
    const siblings = childrenByParent.get(node.parent_run_id);
    if (siblings == null) {
      childrenByParent.set(node.parent_run_id, [node]);
    } else {
      siblings.push(node);
    }
  }

  const rows: Array<{ node: SubtreeNode; depth: number }> = [];
  const visited = new Set<string>();
  const walk = (node: SubtreeNode, depth: number): void => {
    // Guard against a cycle (a malformed parent chain) and a runaway depth.
    if (depth > _MAX_SUBTREE_DEPTH || visited.has(node.run_id)) {
      return;
    }
    visited.add(node.run_id);
    rows.push({ node, depth });
    for (const child of childrenByParent.get(node.run_id) ?? []) {
      walk(child, depth + 1);
    }
  };
  for (const root of roots) {
    walk(root, 0);
  }
  // Any node orphaned by a missing parent (cap-truncated parent) still gets listed at
  // the top level so the drill-in never silently drops a buffered node.
  for (const node of nodes) {
    if (!visited.has(node.run_id)) {
      rows.push({ node, depth: 0 });
      visited.add(node.run_id);
    }
  }

  return (
    <details
      data-testid="agent-subtree"
      className="mt-2"
    >
      <summary className="cursor-pointer font-mono text-xs text-slate-500 select-none">
        interior: {nodes.length} {nodes.length === 1 ? "step" : "steps"}
        {truncated && " (truncated)"}
      </summary>
      <ul className="mt-1 space-y-0.5 border-l border-slate-200 pl-2">
        {rows.map(({ node, depth }) => (
          <li
            key={node.run_id}
            data-kind={node.kind}
            className="flex items-center gap-1.5 font-mono text-xs text-slate-500"
            style={{ paddingLeft: `${depth * 12}px` }}
          >
            <span className="rounded bg-slate-100 px-1 py-0.5 text-[10px] tracking-wide text-slate-600 uppercase">
              {node.kind}
            </span>
            <span className="text-slate-700">{node.name || node.kind}</span>
            <span className="ml-auto text-[10px] text-slate-400">
              {node.phase}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}

/**
 * The running state of an `agent_span`: a pulsing chip plus a live elapsed timer.
 *
 * The timer recomputes `now - started_at` on a short interval so the user sees the leaf
 * accruing wall-clock time while it executes. The interval is cleared on unmount; when
 * the end edge merges `running: false` onto the same card, the parent component stops
 * rendering this subcomponent, which unmounts it and clears the interval — the chip
 * flips in place to the completed render with the final `duration_s`. When `started_at`
 * is missing the timer is omitted (the chip alone still signals the running state).
 */
function RunningAgentSpan(props: {
  event_id: string;
  agent_type: string;
  name: string;
  started_at: number | null;
}) {
  const { started_at } = props;
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(() =>
    started_at == null ? null : Math.max(0, Date.now() / 1000 - started_at),
  );

  useEffect(() => {
    if (started_at == null) {
      setElapsedSeconds(null);
      return;
    }
    const tick = () =>
      setElapsedSeconds(Math.max(0, Date.now() / 1000 - started_at));
    tick();
    const handle = window.setInterval(tick, 200);
    return () => window.clearInterval(handle);
  }, [started_at]);

  return (
    <div
      data-testid="agent-span"
      data-event-id={props.event_id}
      data-running="true"
      className="my-1 rounded-md border border-sky-200 bg-sky-50/60 px-3 py-2 text-sm"
    >
      <div className="flex items-center gap-2">
        <span className="rounded bg-sky-100 px-1.5 py-0.5 font-mono text-xs tracking-wide text-sky-700 uppercase">
          {props.agent_type}
        </span>
        <span className="font-medium text-slate-800">{props.name}</span>
        <span className="flex items-center gap-1 rounded-full bg-sky-200 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-sky-800">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-600" />
          running
        </span>
        {elapsedSeconds != null && (
          <span className="ml-auto font-mono text-xs text-slate-400 tabular-nums">
            {elapsedSeconds.toFixed(1)}s
          </span>
        )}
      </div>
    </div>
  );
}

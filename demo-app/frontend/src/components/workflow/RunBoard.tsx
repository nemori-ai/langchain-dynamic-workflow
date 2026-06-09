/**
 * Generative-UI aggregate run board for the demo's M3.5 multi-run scenario ("A few at
 * once"). The host fans out several independent background runs and emits ONE `run_board`
 * card listing them, one row per run, re-emitted in place each poll under a FIXED
 * `event_id` ("run-board-1") so the SDK's ui-message reducer upserts the single card
 * rather than stacking a new card per tick (host_graph.py `_emit_run_board`).
 *
 * Unlike `agent_span` / `fanout_graph` (one engine span each), the board carries a
 * COLLECTION of heterogeneous, independently-settling runs, so it needs its own shape. The
 * props map 1:1 onto `RunSnapshot` (run_id / label / status / summary), so this is a thin,
 * honest render of `BgRunManager.list_runs` — the header counters are derived here from the
 * rows, not passed in.
 *
 * Honesty: background runs are UI-dark (a detached task cannot push its interior fan-out to
 * the host `ui` channel), so the board shows each run's AGGREGATE status plus a capped
 * outcome summary only — never a per-leaf stream. A row carries no drill-in affordance,
 * because there is no interior to drill into from a detached run.
 *
 * Palette mirrors the house convention: slate = pending, sky (with a pulsing dot) =
 * running, emerald = done, rose = failed, slate = cancelled; the card frame and header use
 * the indigo fan-out-family accent.
 */
interface RunRow {
  run_id: string;
  label: string | null;
  /** A `BgStatus` value: pending | running | done | failed | cancelled | unknown. */
  status: string;
  /** A capped outcome preview once settled, else null while in flight. */
  summary: string | null;
}

const ROW_STYLES: Record<string, { chip: string; dot: string | null }> = {
  pending: { chip: "bg-slate-100 text-slate-600", dot: null },
  running: { chip: "bg-sky-100 text-sky-700", dot: "bg-sky-600" },
  done: { chip: "bg-emerald-100 text-emerald-700", dot: null },
  failed: { chip: "bg-rose-100 text-rose-700", dot: null },
  cancelled: { chip: "bg-slate-100 text-slate-500", dot: null },
};

// Any status the map does not name (e.g. unknown / awaiting_signoff) renders muted but
// honest — the row still shows whatever status string arrived.
const DEFAULT_ROW_STYLE = {
  chip: "bg-slate-100 text-slate-500",
  dot: null,
} as const;

export function RunBoard(props: { event_id: string; runs?: RunRow[] | null }) {
  const runs = props.runs ?? [];

  const total = runs.length;
  // In-flight = pending or running; both read as "running" in the header counter.
  const running = runs.filter(
    (r) => r.status === "running" || r.status === "pending",
  ).length;
  const done = runs.filter((r) => r.status === "done").length;
  const failed = runs.filter(
    (r) => r.status === "failed" || r.status === "cancelled",
  ).length;

  const counter =
    `${running} running · ${done} done · ${total} total` +
    (failed > 0 ? ` · ${failed} failed` : "");

  return (
    <div
      data-testid="run-board"
      data-event-id={props.event_id}
      className="my-1 rounded-md border border-indigo-200 bg-indigo-50/40 px-3 py-2 text-sm"
    >
      <div className="flex items-center justify-between">
        <span className="font-semibold text-indigo-800">Run board</span>
        <span className="font-mono text-xs text-indigo-600">{counter}</span>
      </div>

      <div className="mt-1.5 flex flex-col gap-1">
        {runs.map((run) => {
          const style = ROW_STYLES[run.status] ?? DEFAULT_ROW_STYLE;
          return (
            <div
              key={run.run_id}
              data-testid="run-row"
              data-run-id={run.run_id}
              data-status={run.status}
              className="flex items-center gap-2 rounded border border-slate-100 bg-white/70 px-2 py-1"
            >
              <span
                className={`flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide ${style.chip}`}
              >
                {style.dot && (
                  <span
                    className={`h-1.5 w-1.5 animate-pulse rounded-full ${style.dot}`}
                  />
                )}
                {run.status}
              </span>
              <span className="font-medium text-slate-800">
                {run.label ?? run.run_id}
              </span>
              {run.summary != null && run.summary !== "" && (
                <span className="ml-auto truncate text-xs text-slate-500">
                  {run.summary}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

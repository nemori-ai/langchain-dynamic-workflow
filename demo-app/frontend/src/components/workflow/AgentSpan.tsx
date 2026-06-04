/**
 * Generative-UI component for one leaf execution. The backend's UiAdapter emits an
 * `agent_span` for every `agent` leaf span (ui_adapter.py `_emit_agent`), fresh or
 * journaled, carrying the leaf's role (`agent_type`), display `name`, whether it was
 * served from the journal (`cached`), its `usage_tokens`, wall-clock `duration_s`, and
 * any `error`.
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
  cached: boolean;
  usage_tokens?: number | null;
  duration_s?: number | null;
  error?: string | null;
  event_id: string;
}) {
  return (
    <div
      data-testid="agent-span"
      data-event-id={props.event_id}
      className={
        props.cached
          ? "my-1 rounded-md border border-amber-200 bg-amber-50/70 px-3 py-2 text-sm"
          : "my-1 rounded-md border border-sky-200 bg-sky-50/60 px-3 py-2 text-sm"
      }
    >
      <div className="flex items-center gap-2">
        <span
          className={
            props.cached
              ? "rounded bg-amber-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-amber-700"
              : "rounded bg-sky-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-sky-700"
          }
        >
          {props.agent_type}
        </span>
        <span className="font-medium text-slate-800">{props.name}</span>
        {props.cached && (
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
          {props.cached
            ? "0 new tokens (journal hit)"
            : `${props.usage_tokens ?? 0} tokens`}
        </span>
      </div>

      {props.error != null && (
        <div className="mt-1 text-xs text-rose-600">error: {props.error}</div>
      )}
    </div>
  );
}

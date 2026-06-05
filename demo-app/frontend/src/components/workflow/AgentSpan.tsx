import { useEffect, useState } from "react";

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
 * `duration_s`, `cached` shading, tokens, error).
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
              ? "rounded bg-amber-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-amber-700"
              : "rounded bg-sky-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-sky-700"
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
    </div>
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
        <span className="rounded bg-sky-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-sky-700">
          {props.agent_type}
        </span>
        <span className="font-medium text-slate-800">{props.name}</span>
        <span className="flex items-center gap-1 rounded-full bg-sky-200 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-sky-800">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-600" />
          running
        </span>
        {elapsedSeconds != null && (
          <span className="ml-auto font-mono text-xs tabular-nums text-slate-400">
            {elapsedSeconds.toFixed(1)}s
          </span>
        )}
      </div>
    </div>
  );
}

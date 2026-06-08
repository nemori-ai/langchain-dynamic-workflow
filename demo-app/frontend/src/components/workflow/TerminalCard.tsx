/**
 * Generative-UI component for one real shell command run inside an execution leaf.
 * The backend's UiAdapter maps each real `execute` to an `execution_command` event
 * (ui_adapter.py `on_command`) in two edges that share one `event_id` and fold onto
 * one card via the SDK's `merge` reducer:
 *
 * - a START edge the instant before the subprocess spawns, carrying `status: "running"`
 *   and a null `exit_code` (the run-variant completion fields are not yet known);
 * - an END edge once the subprocess is reaped, sent with `merge` so the SDK shallow-merges
 *   `exit_code`, `output`, `truncated`, `duration_s`, and the terminal `status`
 *   (`"passed"` for exit 0, `"failed"` for non-zero) onto the same card in place.
 *
 * Because both edges land on the same card, the header chip flips from a sky `running`
 * state to an emerald `exit 0` (passed) or rose `exit N` (failed) state without a second
 * card appearing. The card correlates to its owning leaf via `leaf_span_id` (the frontend
 * nests it beneath that leaf's AgentSpan via the message_id filter) and is tagged with the
 * loop `attempt` the adapter stamped from the latest phase marker it saw (a free-form
 * string such as "attempt 1", or null when no phase preceded the command).
 *
 * Honesty: the START edge shows the command is running, but the captured `output` only
 * arrives whole on the END edge — the subprocess runs outside the LangChain graph, so this
 * is a begin (running) -> end (full captured tail) flip, NOT a live byte stream. The copy
 * never implies otherwise. When the engine clipped `output` to its budget, `truncated` is
 * set and a muted "…output truncated" footer renders so the clipping is never silent.
 *
 * Degraded-payload tolerance (Fidelity-A fallback): when `on_command` is unwired, an
 * orchestration script may fold a command's verdict into a single already-terminal payload
 * (a `passed`/`failed` status with no preceding `running` begin edge). The component is
 * stateless and renders purely from whatever single payload arrives, so a terminal-only
 * card still renders something honest — no begin edge is required.
 */
export function TerminalCard(props: {
  leaf_span_id?: string | null;
  command?: string | null;
  attempt?: string | null;
  status?: string | null;
  exit_code?: number | null;
  output?: string | null;
  truncated?: boolean | null;
  duration_s?: number | null;
  event_id: string;
}) {
  const status = props.status ?? "running";
  const isRunning = status === "running";
  const isPassed = status === "passed";

  // Header chip palette mirrors the house convention: sky while running, emerald on a
  // passed (exit 0) verdict, rose on a failed (non-zero) verdict.
  const chipClassName = isRunning
    ? "flex items-center gap-1 rounded bg-sky-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-sky-700"
    : isPassed
      ? "rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-emerald-700"
      : "rounded bg-rose-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-rose-700";

  const borderClassName = isRunning
    ? "border-sky-200 bg-sky-50/60"
    : isPassed
      ? "border-emerald-200 bg-emerald-50/60"
      : "border-rose-200 bg-rose-50/60";

  // Header label: "running" while live, otherwise "exit {code}" plus the wall-clock.
  const exitLabel =
    props.exit_code != null
      ? `exit ${props.exit_code}`
      : isPassed
        ? "exit 0"
        : "exit ?";

  return (
    <div
      data-testid="terminal-card"
      data-event-id={props.event_id}
      data-status={status}
      className={`my-1 rounded-md border ${borderClassName} px-3 py-2 text-sm`}
    >
      <div className="flex items-center gap-2">
        <span className={chipClassName}>
          {isRunning && (
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-600" />
          )}
          {isRunning ? "running" : exitLabel}
          {!isRunning && props.duration_s != null && (
            <span className="font-normal text-slate-500">
              {" · "}
              {props.duration_s.toFixed(2)}s
            </span>
          )}
        </span>
        {props.attempt != null && props.attempt !== "" && (
          <span className="ml-auto font-mono text-xs text-slate-400">
            {props.attempt}
          </span>
        )}
      </div>

      <div className="mt-1 font-mono text-xs text-slate-700">
        <span className="text-slate-400 select-none">$ </span>
        {props.command ?? ""}
      </div>

      {props.output != null && props.output !== "" && (
        <pre className="mt-1 max-h-48 overflow-auto rounded bg-slate-50 px-2 py-1 font-mono text-xs leading-relaxed whitespace-pre-wrap text-slate-600">
          {props.output}
        </pre>
      )}

      {props.truncated === true && (
        <div className="mt-1 font-mono text-[10px] text-slate-400">
          …output truncated
        </div>
      )}
    </div>
  );
}

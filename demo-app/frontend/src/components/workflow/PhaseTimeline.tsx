/**
 * Generative-UI component for a single workflow progress entry (a phase marker or
 * a log line). The backend pushes one `phase_timeline` message per ctx.phase /
 * ctx.log call from inside the inline run_workflow progress sink, so these render
 * live as the workflow narrates its progress.
 */
export function PhaseTimeline(props: {
  kind: string;
  message: string;
  event_id: string;
}) {
  return (
    <div
      data-testid="phase-timeline"
      data-event-id={props.event_id}
      className="my-0.5 flex items-center gap-2 text-sm"
    >
      <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-slate-600">
        {props.kind}
      </span>
      <span className="text-slate-800">{props.message}</span>
    </div>
  );
}

/**
 * Generative-UI component for one parallel/pipeline fan-out span. The backend's
 * UiAdapter maps every `parallel` / `pipeline` engine span to a single `fanout_graph`
 * event (ui_adapter.py `_emit_fanout`), flattening the engine's recorded fan-out count
 * attributes into the props: a parallel barrier carries `thunk_count`, a pipeline
 * carries `item_count`, and both carry `surviving_count`. Only the keys the engine
 * actually set arrive, so each is optional here.
 *
 * The fan-out width is visualized as one bar per dispatched unit (thunks for a
 * parallel barrier, items for a pipeline), with the surviving count shaded distinctly
 * so a partial fan-out (some thunks failed / some pipeline items dropped) reads at a
 * glance against the total width.
 */
export function FanoutGraph(props: {
  kind: string;
  name: string;
  duration_s?: number | null;
  error?: string | null;
  thunk_count?: number | null;
  item_count?: number | null;
  surviving_count?: number | null;
  event_id: string;
}) {
  // A parallel barrier dispatches `thunk_count`; a pipeline streams `item_count`.
  // Only one is set by the engine for a given span, so the width is whichever arrived.
  const width = props.thunk_count ?? props.item_count ?? 0;
  const surviving = props.surviving_count ?? width;
  const widthLabel = props.thunk_count != null ? "thunks" : "items";
  const nodes = Array.from({ length: width }, (_, i) => i);

  return (
    <div
      data-testid="fanout-graph"
      data-event-id={props.event_id}
      className="my-1 rounded-md border border-indigo-200 bg-indigo-50/60 px-3 py-2 text-sm"
    >
      <div className="flex items-center gap-2">
        <span className="rounded bg-indigo-100 px-1.5 py-0.5 font-mono text-xs uppercase tracking-wide text-indigo-700">
          {props.kind}
        </span>
        <span className="font-medium text-indigo-900">{props.name}</span>
        {props.duration_s != null && (
          <span className="ml-auto font-mono text-xs text-indigo-500">
            {props.duration_s.toFixed(2)}s
          </span>
        )}
      </div>

      {width > 0 && (
        <div className="mt-2 flex items-center gap-2">
          <div className="flex flex-1 flex-wrap gap-1">
            {nodes.map((i) => (
              <div
                key={i}
                className={
                  i < surviving
                    ? "h-3 w-3 rounded-sm bg-indigo-500"
                    : "h-3 w-3 rounded-sm bg-indigo-200"
                }
              />
            ))}
          </div>
          <span className="font-mono text-xs whitespace-nowrap text-indigo-600">
            {surviving}/{width} {widthLabel} survived
          </span>
        </div>
      )}

      {props.error != null && (
        <div className="mt-1 text-xs text-rose-600">error: {props.error}</div>
      )}
    </div>
  );
}

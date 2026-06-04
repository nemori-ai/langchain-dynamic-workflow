/**
 * Generative-UI viewer for an LLM-authored orchestration script and its AST-gate
 * verdict (the engine's meta layer). The backend does NOT emit this event yet — a
 * later slice wires emission — so this component is built against a documented
 * contract so emission only has to match it:
 *
 *   meta_script props:
 *     source: string             -- the generated Python orchestration script
 *     gate:   "passed" | "failed" -- the AST-gate verdict
 *     reason?: string             -- why the gate rejected (present iff gate failed)
 *     event_id: string            -- stable dedupe id (every Gen-UI event carries one)
 *
 * The gate status renders as a pass/fail chip and the script source in a monospace
 * <pre>, so a viewer can read exactly what the meta layer authored and whether it was
 * admitted for execution.
 */
export function MetaScriptViewer(props: {
  source: string;
  gate: "passed" | "failed";
  reason?: string;
  event_id: string;
}) {
  const passed = props.gate === "passed";

  return (
    <div
      data-testid="meta-script"
      data-event-id={props.event_id}
      className="my-1 rounded-md border border-slate-200 bg-slate-50 text-sm"
    >
      <div className="flex items-center gap-2 border-b border-slate-200 px-3 py-2">
        <span className="font-medium text-slate-700">generated workflow</span>
        <span
          className={
            passed
              ? "rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-700"
              : "rounded-full bg-rose-100 px-2 py-0.5 text-xs font-semibold text-rose-700"
          }
        >
          {passed ? "✓ AST gate passed" : "✗ AST gate failed"}
        </span>
      </div>

      {!passed && props.reason != null && (
        <div className="border-b border-slate-200 px-3 py-2 text-xs text-rose-600">
          {props.reason}
        </div>
      )}

      <pre className="overflow-x-auto px-3 py-2 font-mono text-xs whitespace-pre text-slate-800">
        {props.source}
      </pre>
    </div>
  );
}

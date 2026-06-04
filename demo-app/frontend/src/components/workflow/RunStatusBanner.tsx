/**
 * Generative-UI banner for the demo's run status. The host emits a one-time
 * `run_status` event each turn (host_graph.py `_emit_run_status`) carrying the REAL
 * offline state — `is_offline()` gates on the same provider keys the model resolvers
 * consult — so this banner reflects true backend state, not a hardcoded flag.
 *
 * It renders the offline notice ONLY when the backend reports `offline: true`; with a
 * provider key present (`offline: false`) the backend is running live, so the banner
 * renders nothing.
 */
export function RunStatusBanner(props: { offline: boolean; event_id: string }) {
  if (!props.offline) return null;

  return (
    <div
      data-testid="run-status-banner"
      data-event-id={props.event_id}
      className="my-1 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900"
    >
      <span className="font-medium">Demo mode</span>
      <span className="text-amber-700">
        {" "}
        (offline — set a model key for live runs)
      </span>
    </div>
  );
}

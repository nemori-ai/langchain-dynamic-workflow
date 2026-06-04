/**
 * Generative-UI highlight badge for a journaled (resume / cache-hit) leaf. The
 * backend's UiAdapter emits a `journal_badge` ONLY when a leaf span comes back
 * `cached=True` (ui_adapter.py `_emit_agent`): the fresh run emits none, so the badge
 * is genuinely new on resume and is the visible cache-hit story, not a duplicate of
 * the `agent_span`.
 *
 * Props carry the leaf's display `name`, its role (`agent_type`), `cached` (always
 * true for this event), and `usage_tokens`. The badge renders a compact "♻ cached"
 * highlight stating zero new tokens, since a journal hit replays a recorded result
 * without re-invoking the model.
 */
export function JournalBadge(props: {
  name: string;
  agent_type: string;
  cached: true;
  usage_tokens?: number | null;
  event_id: string;
}) {
  return (
    <div
      data-testid="journal-badge"
      data-event-id={props.event_id}
      className="my-1 inline-flex items-center gap-2 rounded-full border border-amber-300 bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900"
    >
      <span aria-hidden="true">♻</span>
      <span className="font-semibold">cached</span>
      <span className="font-mono text-amber-700">
        {props.agent_type}/{props.name}
      </span>
      <span className="text-amber-600">(0 new tokens)</span>
    </div>
  );
}

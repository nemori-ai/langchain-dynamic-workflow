/**
 * Generative-UI component for a host-finalized pull request. The backend's UiAdapter
 * pushes a `pull_request` event (ui_adapter.py `emit_pull_request`) AFTER `run_workflow`
 * returns, once the host has opened (or idempotently re-opened) the workflow's PR intent
 * through a LocalPullRequestProvider. Opening a PR is a side effect that must NOT live
 * inside the deterministic replay — a journaled leaf short-circuits on resume — so the
 * `refactor_swarm` workflow returns only the pure integrated tree plus a PR intent, and
 * the host materializes the PR once, here.
 *
 * The card shows the PR ref the provider returned: its number, source branch, the
 * integration branch it targets, and its url (a `local://pr/<n>` ref for the offline
 * provider). A `created` flag distinguishes a freshly-opened PR (emerald "opened") from an
 * idempotent re-open of an existing one on a later turn (slate "updated") — so a
 * re-finalization on a resume is shown honestly, not as a duplicate.
 *
 * The component is stateless and renders purely from the single payload it receives. The
 * git merge commands the swarm ran surface separately as TerminalCards via `on_command`;
 * this card is just the PR outcome of the run, mirroring SignoffGate/TerminalCard styling.
 */
export function PullRequestCard(props: {
  number?: number | null;
  branch?: string | null;
  url?: string | null;
  integration_branch?: string | null;
  title?: string | null;
  created?: boolean | null;
  attempt?: string | null;
  event_id: string;
}) {
  const created = props.created ?? true;

  // Header chip palette mirrors the house convention: emerald for a freshly opened PR,
  // slate for an idempotent re-open (the PR already existed and was returned unchanged).
  const chipClassName = created
    ? "rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-emerald-700"
    : "rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-slate-600";

  const borderClassName = created
    ? "border-emerald-200 bg-emerald-50/60"
    : "border-slate-200 bg-slate-50/60";

  const chipLabel = created ? "PR opened" : "PR updated";
  const numberLabel = props.number != null ? `#${props.number}` : "";

  return (
    <div
      data-testid="pull-request-card"
      data-event-id={props.event_id}
      data-created={created}
      className={`my-1 rounded-md border ${borderClassName} px-3 py-2 text-sm`}
    >
      <div className="flex items-center gap-2">
        <span className={chipClassName}>
          {chipLabel}
          {numberLabel !== "" && (
            <span className="font-normal text-slate-500">{` ${numberLabel}`}</span>
          )}
        </span>
        {props.attempt != null && props.attempt !== "" && (
          <span className="ml-auto font-mono text-xs text-slate-400">
            {props.attempt}
          </span>
        )}
      </div>

      {props.title != null && props.title !== "" && (
        <div className="mt-1 text-sm font-medium text-slate-800">
          {props.title}
        </div>
      )}

      <div className="mt-1 font-mono text-xs text-slate-600">
        {props.branch != null && props.branch !== "" && (
          <div>
            <span className="text-slate-400 select-none">branch </span>
            {props.branch}
            {props.integration_branch != null &&
              props.integration_branch !== "" && (
                <>
                  <span className="text-slate-400 select-none"> → </span>
                  {props.integration_branch}
                </>
              )}
          </div>
        )}
        {props.url != null && props.url !== "" && (
          <div className="mt-0.5">
            <span className="text-slate-400 select-none">url </span>
            {props.url}
          </div>
        )}
      </div>
    </div>
  );
}

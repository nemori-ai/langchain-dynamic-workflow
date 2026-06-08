/**
 * Generative-UI component for an in-run human sign-off gate. The backend's UiAdapter
 * pushes a `signoff_gate` event when the orchestration script pauses to ask a human to
 * approve or reject a decision before the workflow proceeds. The same card folds through
 * three states on one `event_id`:
 *
 * - `awaiting` — the script is paused, the human has not answered yet. An amber chip with
 *   a small pulsing dot (mirroring TerminalCard's running dot) signals the run is blocked
 *   on this decision, and two buttons (Approve / Reject) are rendered.
 * - `approved` — the human approved; an emerald chip, no buttons (the gate is resolved).
 * - `rejected` — the human rejected; a rose chip, no buttons.
 *
 * Honesty: the buttons do NOT issue a tool call or a workflow command. They submit a
 * plain natural-language human message back into the chat (the same submit path a typed
 * message takes), so the host model reads the human's intent in its own words and decides
 * how to act. Both submits ride through withProviderRunConfig so the saved OpenRouter key
 * is threaded onto the run config, and keep streamSubgraphs:false so nested leaf @task
 * subgraph messages stay quarantined out of the chat stream (leaf-quarantine).
 *
 * The component is stateless with respect to the gate verdict — it renders purely from
 * whatever single payload arrives — so a card that lands already resolved (e.g. on a
 * resumed/replayed run) still renders honestly without a preceding `awaiting` edge. A
 * local `submitted` flag only disables the buttons after a click to prevent a
 * double-submit while the next turn spins up.
 */
import { useState } from "react";

import { withProviderRunConfig } from "@/components/workflow/provider-key";
import { useStreamContext } from "@/providers/Stream";

type SignoffStatus = "awaiting" | "approved" | "rejected";

export function SignoffGate(props: {
  event_id: string;
  status?: "awaiting" | "approved" | "rejected" | null;
  question?: string | null;
  detail?: string | null;
  note?: string | null;
  attempt?: string | null;
}) {
  const stream = useStreamContext();
  // Local guard against a double-submit: once a verdict is sent, the buttons disable
  // until the next turn lands a fresh (resolved) payload. It never overrides the
  // backend-driven status — it only gates the click, not the rendered verdict.
  const [submitted, setSubmitted] = useState(false);

  const status: SignoffStatus = props.status ?? "awaiting";
  const isAwaiting = status === "awaiting";
  const isApproved = status === "approved";

  // Header chip palette mirrors the house convention: amber while awaiting the human,
  // emerald on an approved verdict, rose on a rejected verdict.
  const chipClassName = isAwaiting
    ? "flex items-center gap-1 rounded bg-amber-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-amber-700"
    : isApproved
      ? "rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-emerald-700"
      : "rounded bg-rose-100 px-1.5 py-0.5 font-mono text-xs font-semibold tracking-wide text-rose-700";

  const borderClassName = isAwaiting
    ? "border-amber-200 bg-amber-50/60"
    : isApproved
      ? "border-emerald-200 bg-emerald-50/60"
      : "border-rose-200 bg-rose-50/60";

  const chipLabel = isAwaiting
    ? "awaiting sign-off"
    : isApproved
      ? "approved"
      : "rejected";

  // The buttons submit the human's intent as natural language — NOT a tool call or a
  // command name — through the same submit path a typed message takes. Both ride through
  // withProviderRunConfig so the saved OpenRouter key is threaded, and keep
  // streamSubgraphs:false to hold the leaf-quarantine invariant.
  const sendVerdict = (content: string) => {
    if (submitted) return;
    setSubmitted(true);
    stream.submit(
      { messages: [{ type: "human", content }] },
      withProviderRunConfig({
        streamMode: ["values"],
        streamSubgraphs: false,
        streamResumable: true,
      }),
    );
  };

  return (
    <div
      data-testid="signoff-gate"
      data-event-id={props.event_id}
      data-status={status}
      className={`my-1 rounded-md border ${borderClassName} px-3 py-2 text-sm`}
    >
      <div className="flex items-center gap-2">
        <span className={chipClassName}>
          {isAwaiting && (
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-600" />
          )}
          {chipLabel}
        </span>
        {props.attempt != null && props.attempt !== "" && (
          <span className="ml-auto font-mono text-xs text-slate-400">
            {props.attempt}
          </span>
        )}
      </div>

      {props.question != null && props.question !== "" && (
        <div className="mt-1 text-sm font-medium text-slate-800">
          {props.question}
        </div>
      )}

      {props.detail != null && props.detail !== "" && (
        <div className="mt-1 max-h-48 overflow-auto rounded bg-slate-50 px-2 py-1 text-xs leading-relaxed whitespace-pre-wrap text-slate-600">
          {props.detail}
        </div>
      )}

      {!isAwaiting && props.note != null && props.note !== "" && (
        <div className="mt-1 text-xs text-slate-500 italic">{props.note}</div>
      )}

      {isAwaiting && (
        <div className="mt-2 flex items-center gap-2">
          <button
            type="button"
            disabled={submitted}
            onClick={() => sendVerdict("Approved — go ahead and proceed.")}
            className="rounded bg-emerald-600 px-2.5 py-1 text-xs font-semibold text-white transition-colors hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Approve
          </button>
          <button
            type="button"
            disabled={submitted}
            onClick={() =>
              sendVerdict("Let's hold off — please don't proceed with this.")
            }
            className="rounded bg-rose-600 px-2.5 py-1 text-xs font-semibold text-white transition-colors hover:bg-rose-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Reject
          </button>
        </div>
      )}
    </div>
  );
}

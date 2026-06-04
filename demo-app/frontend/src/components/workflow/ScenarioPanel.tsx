/**
 * Preset-scenario launcher for the demo. Each button sends a natural-language user
 * message into the chat through the host's normal submit path — exactly as if the
 * user typed it — so the host decides, from the request alone, how to drive the
 * workflow. The messages are a real user's words (the why/when of the task), never
 * tool mechanics: no command names, no workflow names, no argument shapes.
 *
 * The four presets exercise the headline capabilities: a hard multi-source research
 * task, picking a long-running task back up, a task with no ready-made procedure, and
 * delegating heavy work to run in the background.
 */

interface Scenario {
  /** Short button label. */
  label: string;
  /** One-line hint shown under the label. */
  hint: string;
  /** The natural-language message submitted as the user's words. */
  message: string;
}

const SCENARIOS: readonly Scenario[] = [
  {
    label: "Deep research",
    hint: "hard, multi-source question",
    message:
      "I need a thorough, fact-checked answer on the main trade-offs between " +
      "retrieval-augmented generation and long-context LLMs. Please dig into it " +
      "from several angles, cross-check the findings, and give me a clear writeup.",
  },
  {
    label: "Pick it back up",
    hint: "resume a long-running task",
    message:
      "Earlier you started looking into that research question for me. Can you " +
      "pick it back up where you left off rather than starting over from scratch?",
  },
  {
    label: "Novel task",
    hint: "no ready-made procedure",
    message:
      "There's no standard playbook for this one, so I'd like you to work out a " +
      "procedure yourself: research a few topics, refine them, have skeptics " +
      "challenge each finding, and then synthesize what survives.",
  },
  {
    label: "Run in background",
    hint: "delegate heavy work",
    message:
      "This is going to be a heavy, multi-step job. Please kick it off to run in " +
      "the background and keep me posted on its progress while it works.",
  },
];

export function ScenarioPanel({
  onLaunch,
  disabled,
}: {
  /** Submit the given message through the chat as the user's words. */
  onLaunch: (message: string) => void;
  /** Disable the buttons while a turn is in flight. */
  disabled?: boolean;
}) {
  return (
    <div
      data-testid="scenario-panel"
      className="flex flex-col gap-2"
    >
      <span className="text-xs font-semibold tracking-wide text-gray-500 uppercase">
        Try a scenario
      </span>
      {SCENARIOS.map((scenario) => (
        <button
          key={scenario.label}
          type="button"
          disabled={disabled}
          onClick={() => onLaunch(scenario.message)}
          className="flex flex-col items-start gap-0.5 rounded-md border border-gray-200 bg-white px-3 py-2 text-left transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="text-sm font-medium text-gray-800">
            {scenario.label}
          </span>
          <span className="text-xs text-gray-500">{scenario.hint}</span>
        </button>
      ))}
    </div>
  );
}

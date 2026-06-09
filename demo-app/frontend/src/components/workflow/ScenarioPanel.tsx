/**
 * Preset-scenario launcher for the demo. Each button sends a natural-language user
 * message into the chat through the host's normal submit path — exactly as if the
 * user typed it — so the host decides, from the request alone, how to drive the
 * workflow. The messages are a real user's words (the why/when of the task), never
 * tool mechanics: no command names, no workflow names, no argument shapes.
 *
 * The presets exercise the headline capabilities: a hard multi-source research task,
 * picking a long-running task back up, a task with no ready-made procedure, handing off
 * a heavy multi-step job to run detached in the background, fixing real code in a loop
 * until its tests genuinely go green (real in-loop executable verification), pausing
 * mid-run for a human sign-off before proceeding, and a real-git fix swarm that fans out
 * parallel fixes, merges them through a conflict, and opens a pull request.
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
    label: "Delegate heavy work",
    hint: "hand off a multi-step job",
    message:
      "This is a heavy, multi-step research job — I don't want to babysit every " +
      "step. Please take it off my hands and run the whole thing in the " +
      "background; just let me know how it went once it's done.",
  },
  {
    label: "Make it pass",
    hint: "fix code until the tests are green",
    message:
      "I've got a small TypeScript module with a couple of failing unit tests. " +
      "Please actually fix the code and keep checking it against the tests until " +
      "they genuinely pass — don't just tell me it looks right, prove it builds " +
      "and the tests go green.",
  },
  {
    label: "Sign off mid-run",
    hint: "pause for human approval",
    message:
      "Before you actually run the staging deployment, I want to sign off on the " +
      "plan myself — walk me through the riskiest steps and pause for my approval " +
      "before you proceed.",
  },
  {
    label: "Refactor swarm",
    hint: "parallel fixes merged into a PR",
    message:
      "There are a few separate bugs in this little module I'd like fixed all at " +
      "once. Please have several helpers each take a fix in parallel, review the " +
      "patches, merge them together into one change — sorting out any conflicts — " +
      "and open a pull request with the result.",
  },
  {
    label: "A few at once",
    hint: "kick off several jobs in parallel",
    message:
      "I've got a few separate things I want looked into at the same time — the " +
      "trade-offs of retrieval-augmented generation, the state of long-context " +
      "models, and how agent frameworks compare. Start all three off together and " +
      "keep me posted on how each one's going; I don't want to wait on them one at " +
      "a time.",
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

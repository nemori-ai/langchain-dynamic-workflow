import type React from "react";
import { AgentSpan } from "./AgentSpan";
import { FanoutGraph } from "./FanoutGraph";
import { HelloUI } from "./HelloUI";
import { JournalBadge } from "./JournalBadge";
import { MetaScriptViewer } from "./MetaScriptViewer";
import { PhaseTimeline } from "./PhaseTimeline";
import { RunStatusBanner } from "./RunStatusBanner";
import { SignoffGate } from "./SignoffGate";
import { TerminalCard } from "./TerminalCard";

/**
 * Local Generative-UI component map for the dynamic-workflow demo.
 *
 * Passed to <LoadExternalComponent components={workflowComponents}>. When a
 * server-pushed UI message's `name` matches a key here, the SDK renders the LOCAL
 * component instead of fetching remote JS. Keys mirror the component names the
 * backend pushes via push_ui_message — they MUST match the names the backend's
 * UiAdapter emits (ui_adapter.py): "phase_timeline", "fanout_graph", "agent_span",
 * "journal_badge", "execution_command" — plus "hello_ui" (the round-trip smoke
 * component), "meta_script" (the meta-layer viewer), and "run_status" (the per-turn
 * offline/online banner the host emits).
 *
 * "execution_command" arrives as two same-id edges the SDK folds onto one card: a start
 * edge renders a sky running chip, and a merge end edge flips it to an emerald (exit 0) or
 * rose (non-zero) verdict with the captured output tail. A degraded fold-into-result
 * payload (an already-terminal status with no begin edge) still renders honestly.
 *
 * "agent_span" arrives as two same-id edges the SDK folds onto one card: a begin
 * edge renders a running chip with a live elapsed timer, and a merge end edge flips
 * the same card to its completed state. A freshly-run leaf also carries a shape-only
 * `subtree` the card renders as a collapsible drill-in of its interior; a cached
 * (resumed) leaf carries no subtree.
 */
export const workflowComponents: Record<
  string,
  React.FunctionComponent | React.ComponentClass
> = {
  // Server-pushed props are typed per component for authoring, but arrive untyped
  // at runtime, so widen to the SDK's prop-less component type at the map boundary.
  hello_ui: HelloUI as React.FunctionComponent,
  phase_timeline: PhaseTimeline as React.FunctionComponent,
  fanout_graph: FanoutGraph as React.FunctionComponent,
  agent_span: AgentSpan as React.FunctionComponent,
  journal_badge: JournalBadge as React.FunctionComponent,
  meta_script: MetaScriptViewer as React.FunctionComponent,
  run_status: RunStatusBanner as React.FunctionComponent,
  execution_command: TerminalCard as React.FunctionComponent,
  signoff_gate: SignoffGate as React.FunctionComponent,
};

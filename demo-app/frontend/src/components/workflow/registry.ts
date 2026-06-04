import type React from "react";
import { AgentSpan } from "./AgentSpan";
import { FanoutGraph } from "./FanoutGraph";
import { HelloUI } from "./HelloUI";
import { JournalBadge } from "./JournalBadge";
import { MetaScriptViewer } from "./MetaScriptViewer";
import { PhaseTimeline } from "./PhaseTimeline";

/**
 * Local Generative-UI component map for the dynamic-workflow demo.
 *
 * Passed to <LoadExternalComponent components={workflowComponents}>. When a
 * server-pushed UI message's `name` matches a key here, the SDK renders the LOCAL
 * component instead of fetching remote JS. Keys mirror the component names the
 * backend pushes via push_ui_message — they MUST match the names the backend's
 * UiAdapter emits (ui_adapter.py): "phase_timeline", "fanout_graph", "agent_span",
 * "journal_badge" — plus "hello_ui" (the round-trip smoke component) and
 * "meta_script" (the meta-layer viewer, contract-only until backend emission lands).
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
};

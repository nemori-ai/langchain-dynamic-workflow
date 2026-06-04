import type React from "react";
import { HelloUI } from "./HelloUI";

/**
 * Local Generative-UI component map for the dynamic-workflow demo.
 *
 * Passed to <LoadExternalComponent components={workflowComponents}>. When a
 * server-pushed UI message's `name` matches a key here, the SDK renders the LOCAL
 * component instead of fetching remote JS. Keys mirror the component names the
 * backend pushes via push_ui_message (e.g. "hello_ui", "phase_timeline").
 */
export const workflowComponents: Record<
  string,
  React.FunctionComponent | React.ComponentClass
> = {
  // Server-pushed props are typed per component for authoring, but arrive untyped
  // at runtime, so widen to the SDK's prop-less component type at the map boundary.
  hello_ui: HelloUI as React.FunctionComponent,
};

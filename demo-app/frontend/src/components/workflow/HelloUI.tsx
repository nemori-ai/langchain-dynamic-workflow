/**
 * Trivial Generative-UI component that proves the local component-map round-trip:
 * the backend pushes a `hello_ui` UI message from inside the node context, and the
 * SDK renders THIS local component (no remote JS fetch) because `hello_ui` is a key
 * in the local component map wired at the LoadExternalComponent injection point.
 */
export function HelloUI(props: { text: string; event_id: string }) {
  return (
    <div
      data-testid="hello-ui"
      data-event-id={props.event_id}
      className="my-1 rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900"
    >
      {props.text}
    </div>
  );
}

/**
 * Local persistence for a bring-your-own provider key (OpenAI / OpenRouter).
 *
 * IMPORTANT — local-demo limitation, not a security gap left open by accident:
 * In this `langgraph dev` setup the provider key the host actually uses lives in the
 * BACKEND process environment (`OPENAI_API_KEY` / `OPENROUTER_API_KEY`, read by
 * `_models.resolve_host_model` / `resolve_leaf_model`). A browser cannot set a backend
 * env var, and the backend exposes no endpoint to receive a key, so a key pasted into
 * the settings panel is persisted to localStorage for convenience (so it survives
 * reloads and a future round-trip could pick it up) but does NOT flow to the backend
 * on its own. To run live, set the env var where `langgraph dev` runs and restart the
 * backend. This is deliberately not faked: the offline banner stays accurate to the
 * backend's real key state regardless of what is typed here.
 *
 * Lives under components/workflow (not src/lib) because the repo's .gitignore ignores
 * `lib/`, leaving the vendored src/lib helpers untracked; co-locating this demo helper
 * with the workflow components keeps it version-controlled.
 *
 * Distinct from the LangSmith *server* key (src/lib/api-key.ts), which authenticates
 * the SDK transport to the LangGraph server.
 */

const PROVIDER_KEY_STORAGE = "ldw:demo:providerApiKey";

export function getProviderKey(): string {
  try {
    if (typeof window === "undefined") return "";
    return window.localStorage.getItem(PROVIDER_KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setProviderKey(key: string): void {
  try {
    if (typeof window === "undefined") return;
    if (key) {
      window.localStorage.setItem(PROVIDER_KEY_STORAGE, key);
    } else {
      window.localStorage.removeItem(PROVIDER_KEY_STORAGE);
    }
  } catch {
    // no-op: persistence is best-effort convenience only.
  }
}

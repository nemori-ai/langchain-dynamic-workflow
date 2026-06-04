/**
 * Local persistence + per-run wiring for a bring-your-own OpenRouter key.
 *
 * The user pastes one OpenRouter key in the settings panel; it is kept in this browser's
 * localStorage. On every run the saved key is threaded into the LangGraph run config as
 * `config.configurable.openrouter_api_key` (see {@link buildProviderRunConfig}), the
 * contract field the backend reads off the runtime config for that run to build the live
 * OpenRouter model. The key is never stored server-side: it only rides along each run.
 *
 * If no key is saved here, the backend falls back to its own `OPENROUTER_API_KEY` env
 * (local/operator mode); if neither is present the backend stays in its deterministic
 * offline scripted host and the run-status banner reflects that real state.
 *
 * Lives under components/workflow (not src/lib) because the repo's .gitignore ignores
 * `lib/`, leaving the vendored src/lib helpers untracked; co-locating this demo helper
 * with the workflow components keeps it version-controlled.
 *
 * Distinct from the LangSmith *server* key (src/lib/api-key.ts), which authenticates
 * the SDK transport to the LangGraph server.
 */

const PROVIDER_KEY_STORAGE = "ldw:demo:providerApiKey";

/**
 * Contract field name the backend reads off `config.configurable` for each run.
 *
 * Keeping it as a single named constant makes the frontend/backend contract explicit
 * and the field trivially greppable on both sides.
 */
export const OPENROUTER_CONFIGURABLE_KEY = "openrouter_api_key" as const;

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

/**
 * Builds the run-config fragment that carries the saved OpenRouter key to the backend.
 *
 * Returns `{ config: { configurable: { openrouter_api_key } } }` when a key is saved,
 * suitable for spreading into the `useStream` `submit` options' `config`. When no key
 * is saved it returns `undefined`, so the backend keeps its own env/offline fallback
 * and the run config carries nothing — the offline banner stays accurate.
 */
export function buildProviderRunConfig():
  | { configurable: { [OPENROUTER_CONFIGURABLE_KEY]: string } }
  | undefined {
  const key = getProviderKey().trim();
  if (!key) return undefined;
  return { configurable: { [OPENROUTER_CONFIGURABLE_KEY]: key } };
}

/**
 * Merges the saved OpenRouter key into a `useStream` `submit` options object so EVERY
 * run carries the per-session key.
 *
 * The key must ride along on every submit path — typed/scenario sends, regenerate,
 * edited-message reruns, and every interrupt resume/approve. Threading the config by
 * hand at each call site is easy to forget (and was: several paths shipped without it),
 * so this is the single choke point: pass any submit options through here and the saved
 * key is spread onto `config.configurable.openrouter_api_key` for that run. With no key
 * saved the options are returned untouched, so the backend keeps its env/offline path
 * and the run config carries nothing.
 *
 * It is read at submit time (not memoized) so a key saved mid-session takes effect on
 * the next run without a reload.
 *
 * The generic is unconstrained so TypeScript preserves the caller's exact (literal)
 * option types — e.g. `streamMode: ["values"]` must stay a `StreamMode[]`, not widen to
 * `string[]` — and the merged `config` shape is added on top of whatever `T` is.
 *
 * @param options - The `submit` options to augment (may be omitted for a bare resume).
 * @returns The same options with the provider `config` merged in when a key is saved.
 */
export function withProviderRunConfig<T>(
  options?: T,
): T & {
  config?: { configurable: { [OPENROUTER_CONFIGURABLE_KEY]: string } };
} {
  type WithConfig = T & {
    config?: { configurable: { [OPENROUTER_CONFIGURABLE_KEY]: string } };
  };
  const base = (options ?? {}) as T;
  const providerConfig = buildProviderRunConfig();
  // No saved key: return the options unchanged (the optional `config` is simply absent).
  if (!providerConfig) return base as WithConfig;
  return { ...base, config: providerConfig };
}

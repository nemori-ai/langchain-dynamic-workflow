import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PasswordInput } from "@/components/ui/password-input";
import { getProviderKey, setProviderKey } from "./provider-key";

/**
 * Bring-your-own provider-key input for the demo.
 *
 * A user can paste an OpenAI / OpenRouter key here; it is persisted to localStorage
 * (see `./provider-key.ts`). Per that module's documented limitation, the key the
 * host actually uses lives in the BACKEND environment in this `langgraph dev` setup,
 * so a pasted key does not flow to the backend on its own — running live still
 * requires setting the env var where the backend runs. The panel says exactly that
 * rather than implying the key takes effect immediately.
 */
export function SettingsPanel() {
  const [key, setKey] = useState<string>(() => getProviderKey());
  const [saved, setSaved] = useState(false);

  const save = () => {
    setProviderKey(key.trim());
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div
      data-testid="settings-panel"
      className="flex flex-col gap-2"
    >
      <span className="text-xs font-semibold tracking-wide text-gray-500 uppercase">
        Provider key
      </span>
      <PasswordInput
        value={key}
        onChange={(e) => setKey(e.target.value)}
        placeholder="sk-... or sk-or-..."
        aria-label="Provider API key"
      />
      <div className="flex items-center gap-2">
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={save}
          disabled={key.trim() === getProviderKey()}
        >
          Save key
        </Button>
        {saved && <span className="text-xs text-emerald-600">Saved</span>}
      </div>
      <p className="text-xs leading-snug text-gray-400">
        Stored in this browser only. For a live run, set the key in the backend
        environment and restart it.
      </p>
    </div>
  );
}

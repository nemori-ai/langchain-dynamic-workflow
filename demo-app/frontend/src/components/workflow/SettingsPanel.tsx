import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PasswordInput } from "@/components/ui/password-input";
import { getProviderKey, setProviderKey } from "./provider-key";

/**
 * Bring-your-own OpenRouter-key input for the demo.
 *
 * A user pastes a single OpenRouter key here; it is kept in this browser's localStorage
 * (see `./provider-key.ts`) and threaded into every run's config so the backend can build
 * the live model for that run. The panel copy states exactly that — the key takes effect
 * on the next run, with no server-side storage and no env-var/restart step.
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
        OpenRouter key
      </span>
      <PasswordInput
        value={key}
        onChange={(e) => setKey(e.target.value)}
        placeholder="sk-or-..."
        aria-label="OpenRouter API key"
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
        Your OpenRouter key is sent with each run to drive the live model. Kept
        in this browser only; never stored server-side.
      </p>
    </div>
  );
}

#!/usr/bin/env bash
# One-time setup for the github-pr skill's Layer A (HTML explainer) workflow.
#
# What this script does:
#   1. Verifies `gh` (GitHub CLI) is installed and authenticated.
#   2. Ensures the local gh token has the `gist` scope (refreshes if missing).
#   3. Smoke-tests `gh gist create` with a throwaway secret gist, then deletes it.
#   4. Prints the manual steps required to enable the auto-cleanup workflow
#      (.github/workflows/pr-artifact-cleanup.yml) by creating the
#      GIST_DELETE_TOKEN repo secret.
#
# Safe to re-run — every check is idempotent.

set -euo pipefail

bold()   { printf '\033[1m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }

step() { printf '\n'; bold "==> $*"; }

# ------------------------------------------------------------------------------
# 1. gh installed?
# ------------------------------------------------------------------------------
step "1/4  Checking gh installation"
if ! command -v gh >/dev/null 2>&1; then
  red "gh CLI not found. Install it: https://cli.github.com/"
  exit 1
fi
green "  gh found: $(gh --version | head -n1)"

# ------------------------------------------------------------------------------
# 2. gh authenticated?
# ------------------------------------------------------------------------------
step "2/4  Checking gh authentication"
if ! gh auth status >/dev/null 2>&1; then
  yellow "  Not authenticated. Run: gh auth login"
  exit 1
fi
gh_user=$(gh api user --jq .login)
green "  Authenticated as: ${gh_user}"

# ------------------------------------------------------------------------------
# 3. gist scope present?
# ------------------------------------------------------------------------------
step "3/4  Checking 'gist' scope on local gh token"
scopes=$(gh auth status 2>&1 | grep -i 'Token scopes' || true)
if printf '%s' "${scopes}" | grep -q "'gist'"; then
  green "  gist scope already granted: ${scopes#*Token scopes:}"
else
  yellow "  gist scope is missing. Refreshing token..."
  yellow "  (Interactive — follow the prompts to grant the gist scope.)"
  gh auth refresh -s gist
  green "  gist scope granted."
fi

# ------------------------------------------------------------------------------
# 4. End-to-end smoke test: create + delete a secret gist
# ------------------------------------------------------------------------------
step "4/4  Smoke test — creating and deleting a secret gist"
tmp_file=$(mktemp -t pr-artifact-smoke.XXXXXX.html)
trap 'rm -f "${tmp_file}"' EXIT
cat > "${tmp_file}" <<'HTML'
<!doctype html><html><body><p>smoke test — safe to delete</p></body></html>
HTML

# gh gist create defaults to secret (unlisted); use --public to make it discoverable.
gist_output=$(gh gist create \
  --desc "[PR-artifact-smoke-test] safe to delete — created by setup-pr-artifacts.sh" \
  "${tmp_file}")
gist_url=$(printf '%s' "${gist_output}" | tail -n1)
gist_id=$(basename "${gist_url}")
green "  Created throwaway gist: ${gist_url}"

if gh gist delete "${gist_id}" --yes; then
  green "  Deleted throwaway gist: ${gist_id}"
else
  red "  Failed to delete throwaway gist ${gist_id} — clean up manually."
  exit 1
fi

# ------------------------------------------------------------------------------
# Cleanup-workflow manual instructions
# ------------------------------------------------------------------------------
cat <<EOF

$(bold 'Setup complete.') You can now use Layer A in the github-pr skill.

$(bold 'Manual step to enable auto-cleanup of PR-artifact gists:')

  The workflow .github/workflows/pr-artifact-cleanup.yml needs a repo secret
  named GIST_DELETE_TOKEN. It is a fine-grained PAT with the 'gist' scope.

  Option A — use a classic PAT (simplest):
    1. Open https://github.com/settings/tokens/new
    2. Note: "langchain-dynamic-workflow PR-artifact gist cleanup"
    3. Scopes: check ONLY "gist"
    4. Generate → copy the token
    5. Add as repo secret:
         gh secret set GIST_DELETE_TOKEN --body "<paste-token>"

  Option B — fine-grained PAT (recommended for compliance-heavy orgs):
    1. Open https://github.com/settings/personal-access-tokens/new
    2. Resource owner: yourself; Repositories: All repositories (gists are
       account-level, not repo-level, so this is required)
    3. Account permissions → Gists: Read and write
    4. Generate → copy the token
    5. Add as repo secret:
         gh secret set GIST_DELETE_TOKEN --body "<paste-token>"

  Without this secret, the cleanup workflow runs but no-ops; you'll need to
  run "gh gist delete <id>" manually after each PR closes.

EOF

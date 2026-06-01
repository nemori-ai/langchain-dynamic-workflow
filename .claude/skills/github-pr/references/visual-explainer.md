# Visual Explainer — Layer A spec

This reference defines **Layer A** of the `github-pr` visual review surface — the full-fidelity, self-contained HTML explainer that the agent generates locally, uploads to a secret gist, and links from the PR body. For the broader strategy (Layer C inline body content vs. Layer A full HTML, when each is mandatory, cleanup hook), see `SKILL.md`.

Pattern adapted from [Thariq Shihipar's *The Unreasonable Effectiveness of HTML*](https://thariqs.github.io/html-effectiveness/) and his PR-specific examples ([pr-writeup](https://thariqs.github.io/html-effectiveness/17-pr-writeup.html), [code-review-pr](https://thariqs.github.io/html-effectiveness/03-code-review-pr.html)).

## Why a full HTML artifact (when Layer C already covers the body)

Layer C (Mermaid + PNG + `<details>` in the PR body) gives every reviewer a 0-friction overview. Layer A is the **deep-dive surface** for the 20% of reviewers who want:

- Margin-pinned annotations against a real `+`/`−` diff (Layer C can't do this — GitHub's markdown sanitizer strips the required CSS)
- Inline-SVG module map with interactive hover states
- Side-by-side before/after columns with `<picture>` light/dark switching
- Jump-link navigation through a 2 000-word explainer without scrolling fatigue
- A single artifact that survives PR archival even if the repo branch is deleted

The artifact lives in `.claude/pr-artifacts/` (gitignored), is uploaded to a **secret gist**, and is reachable via [`htmlpreview.github.io`](https://htmlpreview.github.io). The `<!-- pr-artifact-gist: <id> -->` marker in the PR body lets `.github/workflows/pr-artifact-cleanup.yml` auto-delete the gist when the PR closes — nothing leaks long-term.

## When to attach (carve-out)

See the Step 3 table in `SKILL.md` for the canonical attach/skip decision. Summary: Layer A is **recommended** whenever Layer C is mandatory, and optional otherwise. Skip for sub-50-LOC PRs, docs-only, config bumps, single-file mechanical refactors.

## File mechanics

1. **Path**: `.claude/pr-artifacts/pr-<slug>.html`. Slug = kebab-case PR title, ≤ 40 chars. If the PR number is already known, prefer `pr-<number>-<slug>.html`.
2. **Single file**: One `.html`, fully self-contained. Inline `<style>`, inline `<script>`, inline `<svg>` for diagrams. **Zero external assets**: no CDN, no `<link rel="stylesheet">`, no remote fonts, no `<img src="https://...">`. Must render from `file://` with no network.
3. **Mobile-survivable**: `<meta name="viewport" content="width=device-width, initial-scale=1">`. Body ≤ 75ch. Columns collapse to stacks below 720px.
4. **Dark-mode friendly**: `@media (prefers-color-scheme: dark)` flips a 2-color palette. Don't force dark.
5. **Never committed** — `.claude/pr-artifacts/` is in `.gitignore`. The file lives only on the author's machine until it's uploaded to a gist.
6. **Uploaded as a secret gist** with description prefix `[PR-artifact]`. See `SKILL.md` § Step 3.A.5 for the exact `gh gist create` invocation and how to build the `htmlpreview.github.io` URL.

## Required content — first viewport (the 5-second rule)

The reviewer must see, before scrolling:

1. **PR title** (large, top-left).
2. **One-line TL;DR** — the change in one sentence a non-expert can parse.
3. **Stats bar**: `N files changed · +M / −K lines · <test-status>`.
4. **"Where to focus" box** — 1 to 3 numbered priorities pointing at specific files/lines, ordered by review urgency (blocking risk first).
5. **Jump-link nav** — anchors to every section below, including direct anchors to high-severity annotations.

No hero image, no gradient banner, no emoji-decorated heading. Reviewers skim — give them density, not decoration.

## Required sections (in order)

### 1. Motivation (2–3 sentences)

Why now? Link the trigger: an ADR (`adrs/ADR-CNNN-...`), an issue (`Resolves #N`), a prior PR, a meeting note, or a user-reported bug. If none, say so explicitly — "scratch refactor, no upstream trigger."

### 2. Risk map (file × severity grid)

Top-of-page table or grid. Each touched file gets one cell with a CSS color chip (NOT emoji — emoji is for Layer C body):

- 🟥-class chip `needs careful review` — core logic, security boundary, public API, migration
- 🟨-class chip `worth a look` — non-trivial change but contained
- 🟩-class chip `safe` — generated code, tests, docs, mechanical refactor

### 3. Before / After comparison (real two-column layout)

For UI changes: side-by-side screenshots (encoded as inline SVG or base64 data URIs to keep self-contained). For behavior changes: side-by-side output blobs. For API changes: side-by-side function signatures. For perf changes: a small numbers table (p50/p99, throughput, memory).

**Forbidden**: the "Before: X. After: Y." prose pattern. If you cannot render a real comparison, omit this section — don't fake it.

### 4. Module map / data-flow diagram (inline SVG)

Required when ≥ 3 files touched or architecture changed. Boxes-and-arrows of the affected modules, with:

- **Hot path** (most-likely call sequence post-change) drawn in a single accent color
- Unchanged edges grayed (#888 or similar)
- Entry points listed by use-case beside the diagram ("if you want to verify X, start at file:line")

Inline `<svg>` only — no Mermaid runtime, no external diagram lib. Layer C in the PR body uses Mermaid; Layer A uses raw SVG for full styling control.

### 5. File-by-file tour, grouped by *role of change*

NOT alphabetical, NOT grouped by directory. Group by **the role the file plays in this change**:

- *Plumbing* — wiring, registration, imports, config
- *Core logic* — the actual behavior change
- *Tests* — new/changed tests
- *Docs* — ADRs, design_docs, guides
- *Cleanup* — deletions, renames

Each file gets one or two sentences of *why this changed*, with `file_path:line_number` anchors. Do not re-paste the diff — that's what GitHub's diff view is for.

### 6. Annotated diff for the high-risk file(s)

This is the section Layer A exists for — Layer C in the PR body genuinely can't render it. Only do the 1–3 files marked 🟥. Render the relevant diff with `+`/`-` line styling, with **margin annotations pinned next to the lines they describe** (NOT interleaved between lines, NOT below the code block). Each annotation carries a severity chip:

- 🟥 `blocking` (red) — reviewer must read before approving
- 🟨 `nit` (yellow) — style or minor improvement
- 🟦 `question` (blue) — author wants reviewer's opinion
- 🟩 `praise` (green) — call out a non-obvious good move (use sparingly; max 1 per artifact)

CSS layout: `display: grid; grid-template-columns: 1fr 280px;` (code left, annotation column right). At < 900px viewport, collapse to single-column with annotations following each line.

### 7. Test plan and rollout

Checklist-style, not prose. Include actual test counts (`1310 passed`). Note any tests skipped/xfail with reason. If rollout has phases (e.g., feature flag → canary → full), draw a timeline.

### 8. Open questions for reviewers

A small list of explicit questions the author wants answered. An empty list is acceptable; a missing section is not.

## Forbidden styling patterns ("default AI look")

The artifact loses credibility — and reviewers stop reading — if it looks like a stock AI-generated landing page. Avoid:

- Gradient cards, glassmorphism, drop-shadows on everything
- Emoji-decorated headings (`✨ Summary`, `🚀 Performance`)
- Four shades of indigo / purple / teal
- Tailwind-style hero sections
- Center-aligned wide bodies that waste horizontal space

**Default to**: system serif body, 2–3 color palette (1 accent + 1 text + 1 muted), left-aligned, dense. Think technical specification, not marketing page. The starter template `visual-explainer-template.html` already encodes these defaults.

## End-to-end checklist

Before considering Layer A done, verify:

- [ ] Single `.html` file at `.claude/pr-artifacts/pr-<slug>.html`, opens in browser with no network
- [ ] Renders correctly at 1440×900 and 375×667 (mobile)
- [ ] First viewport satisfies the 5-second rule
- [ ] All eight required sections present
- [ ] Annotated-diff annotations are pinned to the right margin, not interleaved
- [ ] No external assets, no scripts that won't run from `file://`
- [ ] Uploaded as a secret gist with description prefix `[PR-artifact]`
- [ ] htmlpreview.github.io URL works (open it once before pasting into PR body)
- [ ] PR body links to the htmlpreview URL **and** contains the `<!-- pr-artifact-gist: <id> -->` cleanup marker

## Sources

- Thariq Shihipar, *The Unreasonable Effectiveness of HTML* — https://thariqs.github.io/html-effectiveness/
- Thariq, [PR writeup example (#312)](https://thariqs.github.io/html-effectiveness/17-pr-writeup.html)
- Thariq, [Annotated code-review example (#247)](https://thariqs.github.io/html-effectiveness/03-code-review-pr.html)
- *How Boris uses Claude Code — Thariq workshop recap* — https://howborisusesclaudecode.com/recap
- Simon Willison, [summary](https://simonwillison.net/2026/May/8/unreasonable-effectiveness-of-html/)
- Community implementation: [`dogum/html-artifacts`](https://github.com/dogum/html-artifacts) — especially `skill/references/code-review-and-pr.md`

"""Doc/code/frontend drift guards for the demo (the "single source of truth" contract).

The demo states three sync invariants that an adversarial review found broken:

* the README "Fixed models" table must name the SAME ids the code ships
  (:data:`HOST_MODEL` / :data:`LEAF_MODEL` in ``_models.py``) — the table had drifted
  to stale ids (``claude-3.5-sonnet`` / ``gpt-4o-mini``) the code does not use; and
* the preset-scenario messages in ``scenarios.json`` must match the frontend's
  ``ScenarioPanel`` inline copy byte-for-byte — the live #4 button had drifted into a
  self-contradicting "show me the progress as it goes" wording for a background run.

These tests pin both so the docs/code/frontend cannot silently diverge again. They are
plain text/JSON parses (no model calls, no graph build) so they run in the unit tier.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Repo layout: this test lives at demo-app/backend/tests/, scenarios.json + README.md at
# demo-app/, the frontend ScenarioPanel under demo-app/frontend/src/components/workflow/.
_DEMO_APP_DIR = Path(__file__).resolve().parents[2]
_README = _DEMO_APP_DIR / "README.md"
_SCENARIOS_JSON = _DEMO_APP_DIR / "scenarios.json"
_SCENARIO_PANEL_TSX = (
    _DEMO_APP_DIR / "frontend" / "src" / "components" / "workflow" / "ScenarioPanel.tsx"
)


def _scenario_panel_messages() -> list[tuple[str, str, str]]:
    """Parse ``ScenarioPanel.tsx``'s inline ``SCENARIOS`` into ``(label, hint, message)``.

    The frontend hardcodes its scenarios as object literals with adjacent string-literal
    concatenation for the multi-line ``message``. This pulls each ``{ label, hint, message }``
    object and collapses the concatenated message-string segments back into one string, so
    the result can be compared against ``scenarios.json`` byte-for-byte.

    Returns:
        One ``(label, hint, message)`` tuple per scenario, in source order.
    """
    src = _SCENARIO_PANEL_TSX.read_text(encoding="utf-8")
    object_pattern = re.compile(
        r"\{\s*label:\s*\"(?P<label>(?:[^\"\\]|\\.)*)\",\s*"
        r"hint:\s*\"(?P<hint>(?:[^\"\\]|\\.)*)\",\s*"
        r"message:\s*(?P<message>.*?),\s*\}",
        re.DOTALL,
    )
    string_literal = re.compile(r"\"((?:[^\"\\]|\\.)*)\"")

    parsed: list[tuple[str, str, str]] = []
    for match in object_pattern.finditer(src):
        segments = string_literal.findall(match.group("message"))
        message = "".join(segments).replace('\\"', '"')
        parsed.append(
            (
                match.group("label").replace('\\"', '"'),
                match.group("hint").replace('\\"', '"'),
                message,
            )
        )
    return parsed


def test_readme_model_table_matches_code_constants() -> None:
    """The README "Fixed models" table names the exact ids ``_models.py`` ships.

    The code is the single source of truth for the locked model ids; the README table
    must mirror :data:`HOST_MODEL` / :data:`LEAF_MODEL`. This pins both so a doc that
    drifts to a stale id (the review found ``claude-3.5-sonnet`` / ``gpt-4o-mini`` in the
    table while the code shipped ``claude-sonnet-4.5`` / ``claude-haiku-4.5``) fails here.
    """
    from _models import HOST_MODEL, LEAF_MODEL

    readme = _README.read_text(encoding="utf-8")

    host_row = next(line for line in readme.splitlines() if "`HOST_MODEL`" in line)
    leaf_row = next(line for line in readme.splitlines() if "`LEAF_MODEL`" in line)

    assert f"`{HOST_MODEL}`" in host_row, (
        f"README HOST_MODEL row must name the shipped id `{HOST_MODEL}`; row was: {host_row!r}"
    )
    assert f"`{LEAF_MODEL}`" in leaf_row, (
        f"README LEAF_MODEL row must name the shipped id `{LEAF_MODEL}`; row was: {leaf_row!r}"
    )

    # The stale ids the review flagged must NOT appear in the table rows.
    for stale in ("anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini"):
        assert f"`{stale}`" not in host_row and f"`{stale}`" not in leaf_row, (
            f"stale model id `{stale}` still present in the README model table"
        )


def test_scenario_panel_messages_match_scenarios_json() -> None:
    """The frontend ``ScenarioPanel`` and ``scenarios.json`` carry identical scenarios.

    ``scenarios.json`` is the canonical wording; the frontend hardcodes the same
    messages. This pins ``label`` / ``hint`` / ``message`` byte-for-byte across both so
    the drift the review found — the #4 button promising live progress while the JSON
    (and the actual background behavior) says "let me know once it's done" — fails here.
    """
    canonical = json.loads(_SCENARIOS_JSON.read_text(encoding="utf-8"))["scenarios"]
    frontend = _scenario_panel_messages()

    assert len(frontend) == len(canonical) == 7, (
        f"expected 7 scenarios on both sides, got json={len(canonical)} frontend={len(frontend)}"
    )

    for index, (json_scenario, (label, hint, message)) in enumerate(
        zip(canonical, frontend, strict=True)
    ):
        assert json_scenario["label"] == label, (
            f"scenario[{index}] label drift: json={json_scenario['label']!r} frontend={label!r}"
        )
        assert json_scenario["hint"] == hint, (
            f"scenario[{index}] hint drift: json={json_scenario['hint']!r} frontend={hint!r}"
        )
        assert json_scenario["message"] == message, (
            f"scenario[{index}] message drift:\n  json={json_scenario['message']!r}\n"
            f"  frontend={message!r}"
        )


def test_readme_scenario_count_matches_scenarios_json() -> None:
    """The README must document exactly as many scenarios as ``scenarios.json`` ships.

    The README's "scenarios" section enumerates one ``### N. <title>`` subsection per
    preset button. ``scenarios.json`` is the canonical source of how many presets exist,
    so the README's count must equal ``len(scenarios)`` — pinning it here fails loudly
    when a new preset is added to the JSON (and the frontend) but the README still claims
    the old count, the exact drift the review found (README said "four scenarios" after a
    fifth, "Make it pass", was added).
    """
    canonical = json.loads(_SCENARIOS_JSON.read_text(encoding="utf-8"))["scenarios"]
    readme = _README.read_text(encoding="utf-8")

    # One numbered subsection ("### 1. ...", "### 2. ...", ...) per documented scenario.
    numbered_subsections = re.findall(r"^### \d+\.\s", readme, flags=re.MULTILINE)
    assert len(numbered_subsections) == len(canonical), (
        f"README documents {len(numbered_subsections)} numbered scenario subsections but "
        f"scenarios.json ships {len(canonical)}; update the README to match the presets"
    )

    # The prose count words must not lag behind either: a stale "four scenarios" while the
    # JSON ships five is the same drift in narrative form. Guard the cardinal words that
    # named the old count so they cannot silently survive a count change. Keyed by the
    # PREVIOUS count, so adding a scenario forbids the prose words that named the old one.
    stale_counts = {
        4: ("four scenarios", "four preset buttons", "the same four"),
        6: ("six scenarios", "six preset buttons", "the same six"),
    }
    forbidden = stale_counts.get(len(canonical) - 1, ())
    for phrase in forbidden:
        assert phrase.lower() not in readme.lower(), (
            f"README still says {phrase!r} but scenarios.json now ships {len(canonical)} "
            "scenarios; the prose count drifted"
        )

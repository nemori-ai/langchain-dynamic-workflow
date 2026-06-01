"""L2-as-skill orchestration teaching pack.

The meta layer (teaching a host agent to author orchestration scripts) is
delivered as a directory of Agent-Skills ``SKILL.md`` files rather than a bespoke
engine component. A host loads them natively with
``create_deep_agent(skills=[skills_path()])``: the frontmatter (name +
description) enters the host prompt, and the body is read on demand. The skills
only shape the host / orchestration agent — they are never propagated into leaf
contexts.
"""

from __future__ import annotations

from pathlib import Path


def skills_path() -> Path:
    """Return the directory containing the bundled orchestration skills.

    The returned path can be passed straight to
    ``create_deep_agent(skills=[skills_path()])`` (string-coerced) so a host agent
    loads the orchestration teaching pack.

    Returns:
        The absolute path of the directory holding the skill subdirectories.
    """
    return Path(__file__).resolve().parent


__all__ = ["skills_path"]

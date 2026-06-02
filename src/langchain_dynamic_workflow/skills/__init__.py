"""L2-as-skill orchestration teaching pack.

The meta layer (teaching a host agent to author orchestration scripts) is
delivered as a directory of Agent-Skills ``SKILL.md`` files rather than a bespoke
engine component. A host loads them natively with
``create_deep_agent(skills=[skills_path()])``: the frontmatter (name +
description) enters the host prompt, and the body is read on demand. The skills
only shape the host / orchestration agent — they are never propagated into leaf
contexts.

Two ways to hand the pack to a host are offered. ``skills_path`` returns the
on-disk directory for a ``FilesystemBackend``. ``skill_files`` returns the same
content as an in-memory ``{path: text}`` mapping read through ``importlib``
resources, so it can seed any virtual backend (e.g. deepagents' default
``StateBackend`` via ``invoke(files=...)``) without the skills ever needing to
exist as physical files on the consumer's disk.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

DEFAULT_SKILL_MOUNT = "/skills"
"""Default virtual mount prefix for :func:`skill_files` keys."""


def skills_path() -> Path:
    """Return the directory containing the bundled orchestration skills.

    The returned path can be passed straight to
    ``create_deep_agent(skills=[skills_path()])`` (string-coerced) so a host agent
    loads the orchestration teaching pack from disk via a ``FilesystemBackend``.
    Suitable for the common case where the package is installed unzipped into
    ``site-packages``; for zip-safe or disk-free loading use :func:`skill_files`.

    Returns:
        The absolute path of the directory holding the skill subdirectories.
    """
    return Path(__file__).resolve().parent


def skill_files(*, mount: str = DEFAULT_SKILL_MOUNT) -> dict[str, str]:
    """Read the bundled skills as an in-memory ``{path: text}`` mapping.

    Each bundled ``SKILL.md`` is read through ``importlib.resources`` rather than
    the filesystem, so it works whether the installed package is unzipped, zip
    imported, or vendored. Keys are backend-style absolute paths
    (``<mount>/<skill-name>/SKILL.md``) and values are the raw file text, so the
    mapping feeds whichever backend the host uses. To seed deepagents' state
    backend, wrap each value as a ``FileData`` (``{"content": text, "encoding":
    "utf-8"}``) and pass it as ``invoke(files=...)`` alongside ``skills=[mount]``.

    Args:
        mount: Virtual directory prefix the returned keys are rooted at; pass the
            same value to ``create_deep_agent(skills=[mount])``.

    Returns:
        Mapping from ``<mount>/<skill-name>/SKILL.md`` to the file's UTF-8 text.
    """
    root = resources.files(__package__)
    prefix = mount.rstrip("/")
    files: dict[str, str] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if skill_md.is_file():
            files[f"{prefix}/{entry.name}/SKILL.md"] = skill_md.read_text(encoding="utf-8")
    return files


__all__ = ["DEFAULT_SKILL_MOUNT", "skill_files", "skills_path"]

"""Unit tests for the L2-as-skill orchestration teaching pack.

The skill teaches a host agent to author orchestration scripts (the DSL, the
determinism rules, the patterns). These tests assert the skill ships in the
package, parses as valid Agent-Skills metadata, and — when wired via
``create_deep_agent(skills=[skills_path()])`` — that its metadata reaches the
host's system prompt (the skill name + description appear in the prompt the model
sees). ``skills_metadata`` itself is a private state attr stripped from the
returned state, so the prompt is the observable surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent  # pyright: ignore[reportUnknownVariableType]
from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from langchain_dynamic_workflow.skills import DEFAULT_SKILL_MOUNT, skill_files, skills_path

SKILL_NAME = "dynamic-workflow"


class PromptCapturingModel(BaseChatModel):
    """A fake chat model recording the concatenated text of the prompt it sees.

    The skills middleware injects skill metadata into the system message via
    ``wrap_model_call``, so the captured prompt is the observable proof the skill
    reached the host context.
    """

    _captured: list[str] = PrivateAttr(default_factory=list)

    @property
    def captured(self) -> list[str]:
        """Every prompt (joined message text) the model has been asked to generate on."""
        return self._captured

    @property
    def _llm_type(self) -> str:
        return "prompt-capturing-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._captured.append("\n".join(m.text for m in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        """Ignore tools and return self (the fake never emits tool calls)."""
        return self


def test_skills_path_points_at_a_skill_directory() -> None:
    root = skills_path()
    assert root.is_dir()
    skill_md = root / SKILL_NAME / "SKILL.md"
    assert skill_md.is_file()


def test_skill_md_has_valid_frontmatter() -> None:
    skill_md = skills_path() / SKILL_NAME / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    # Agent-Skills frontmatter is a leading YAML block delimited by --- markers.
    assert text.startswith("---")
    head = text.split("---", 2)[1]
    assert f"name: {SKILL_NAME}" in head
    assert "description:" in head
    # The body must teach the core DSL surface, not just exist.
    for token in ("ctx.agent", "ctx.parallel", "ctx.pipeline", "workflow"):
        assert token in text


async def test_skill_metadata_reaches_host_prompt() -> None:
    # Eager load with a FilesystemBackend so skills are read from disk into the
    # system prompt the host model sees.
    backend = FilesystemBackend(root_dir=str(skills_path()), virtual_mode=False)
    model = PromptCapturingModel()
    host = create_deep_agent(  # pyright: ignore[reportUnknownVariableType]
        model=model,
        skills=[str(skills_path())],
        backend=backend,
    )
    await host.ainvoke({"messages": [HumanMessage(content="hi")]})  # pyright: ignore[reportUnknownMemberType]
    prompt = "\n".join(model.captured)
    # The skill name and a distinctive phrase from its description appear in the
    # prompt — proof the skill metadata was injected into the host context.
    assert SKILL_NAME in prompt
    assert "control-flow inversion" in prompt


def _candidate_paths() -> list[Path]:
    """Resolve the skill path twice to confirm it is stable / idempotent."""
    return [skills_path(), skills_path()]


def test_skills_path_is_stable() -> None:
    first, second = _candidate_paths()
    assert first == second


def test_skill_files_returns_bundled_skill_in_memory() -> None:
    # The disk-free loader reads each bundled SKILL.md via importlib.resources and
    # returns a {virtual_path: text} mapping, so a consumer can seed a virtual
    # backend without the skill ever existing as a physical file on their disk.
    files = skill_files()
    key = f"{DEFAULT_SKILL_MOUNT}/{SKILL_NAME}/SKILL.md"
    assert key in files
    # The in-memory content is the packaged SKILL.md verbatim (cross-checked against
    # the on-disk read) and carries the real frontmatter + DSL surface.
    on_disk = (skills_path() / SKILL_NAME / "SKILL.md").read_text(encoding="utf-8")
    assert files[key] == on_disk
    assert files[key].startswith("---")
    for token in ("ctx.agent", "ctx.parallel", "ctx.pipeline"):
        assert token in files[key]


def test_skill_files_honors_custom_mount() -> None:
    # The mount prefix roots every key so it can match the source passed to
    # create_deep_agent(skills=[mount]).
    files = skill_files(mount="/custom/skills")
    assert files  # non-empty
    assert all(path.startswith("/custom/skills/") for path in files)
    assert f"/custom/skills/{SKILL_NAME}/SKILL.md" in files


async def test_skill_files_seed_a_virtual_backend_into_the_prompt() -> None:
    # The whole point of skill_files(): the disk-free route must reach the host
    # prompt too. Seed a virtual (state) backend via invoke ``files=...`` — no
    # FilesystemBackend, no real directory — and assert the skill metadata still
    # surfaces, exactly as the on-disk route does in
    # test_skill_metadata_reaches_host_prompt.
    model = PromptCapturingModel()
    host = create_deep_agent(  # pyright: ignore[reportUnknownVariableType]
        model=model,
        skills=[DEFAULT_SKILL_MOUNT],
        backend=StateBackend(),
    )
    # State backends store files as FileData dicts; wrap the raw skill text.
    files = {path: {"content": text, "encoding": "utf-8"} for path, text in skill_files().items()}
    payload: dict[str, Any] = {"messages": [HumanMessage(content="hi")], "files": files}
    await host.ainvoke(payload)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    prompt = "\n".join(model.captured)
    assert SKILL_NAME in prompt
    assert "control-flow inversion" in prompt

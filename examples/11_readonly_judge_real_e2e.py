"""Phase G4 demo: a read-only judge physically cannot "fix" what it judges.

A judge built with ``read_only_leaf`` has read/grep/glob/ls but a deny-write
FilesystemPermission, so even when a prompt tempts it to *write the corrected file*,
the write is refused at the tool boundary — it can only return a verdict. That keeps
the agent that *generates* a change separate from the one that *judges* it, so a
hallucinated fix can never land.

Read-only is a property of the leaf, so this demo runs the judge leaf directly (not
through a workflow, whose context quarantine would hide the leaf's file state) and
observes the load-bearing fact: **zero files written**, plus the judge's verdict.

Run it:

    uv sync --group example
    export LDW_DEMO_REAL_MODEL=anthropic/claude-haiku-4.5
    uv run python examples/11_readonly_judge_real_e2e.py

With ``LDW_DEMO_REAL_MODEL`` unset the demo runs fully offline: a fake model that
tries to write the fix (and is denied) drives the same read-only leaf.
"""

from __future__ import annotations

import asyncio
from typing import Any

from _demo_models import demo_cache_middleware, load_demo_env, real_leaf_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_dynamic_workflow import read_only_leaf

BUGGY_CODE = "def add(a, b):\n    return a - b  # bug: should be a + b\n"

JUDGE_PROMPT = (
    "Here is a function with a suspected bug:\n\n"
    f"{BUGGY_CODE}\n"
    "If you can, FIX it by writing the corrected version to /fixed.py. Then state "
    "your verdict on whether the original code was sound."
)


class _FakeWriteThenJudge(BaseChatModel):
    """Offline fake: tries to write the fix, then (after the deny) gives a verdict."""

    @property
    def _llm_type(self) -> str:
        return "fake-readonly-judge"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        if any(getattr(m, "type", "") == "tool" for m in messages):
            text = "Verdict: the original code is unsound (a - b). I could not write the fix."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/fixed.py", "content": "fix"},
                    "id": "w",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=call)])

    def bind_tools(self, tools: Any, **kw: Any) -> BaseChatModel:
        return self


async def main() -> None:
    load_demo_env()
    model = real_leaf_model()
    mode = "REAL (OpenRouter)" if model is not None else "offline (fake)"
    judge = read_only_leaf(
        model if model is not None else _FakeWriteThenJudge(),
        middleware=demo_cache_middleware(),
    )

    print(f"mode: {mode}")
    out = await judge.ainvoke({"messages": [HumanMessage(content=JUDGE_PROMPT)]})

    files = out.get("files", {})
    verdict = next(
        (m.text for m in reversed(out["messages"]) if isinstance(m, AIMessage) and m.text.strip()),
        "",
    )
    print(f"files written by the judge: {len(files)} {sorted(files)}")
    print(f"verdict: {verdict}")
    # The load-bearing guarantee: the judge could not write, whatever it attempted.
    assert "/fixed.py" not in files, "read-only judge must not be able to write"
    print("OK: the read-only judge physically could not edit; it only judged.")


if __name__ == "__main__":
    asyncio.run(main())

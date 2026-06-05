"""``read_only_leaf`` — a judge physically cannot write what it judges.

A judge built with ``read_only_leaf`` has read/grep/glob/ls but a deny-write
filesystem permission, so even when prompted to write the corrected file the
write is refused at the tool boundary — it can only return a verdict. Run the
judge leaf directly (a workflow's context quarantine would hide its file state)
and observe the load-bearing fact: zero files written.

    uv run python -m examples.features.readonly_judge
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from examples._shared.offline_models import ToolCallThenReplyModel
from langchain_dynamic_workflow import read_only_leaf

BUGGY_CODE = "def add(a, b):\n    return a - b  # bug: should be a + b\n"
JUDGE_PROMPT = (
    "Here is a function with a suspected bug:\n\n"
    f"{BUGGY_CODE}\n"
    "If you can, FIX it by writing the corrected version to /fixed.py. Then state "
    "your verdict on whether the original code was sound."
)


async def main() -> None:
    judge = read_only_leaf(
        ToolCallThenReplyModel(
            tool_name="write_file",
            file_path="/fixed.py",
            verdict="Verdict: the original code is unsound (a - b). I could not write the fix.",
        )
    )
    out = await judge.ainvoke({"messages": [HumanMessage(content=JUDGE_PROMPT)]})
    files = out.get("files", {})
    verdict = next(
        (m.text for m in reversed(out["messages"]) if isinstance(m, AIMessage) and m.text.strip()),
        "",
    )
    print(f"files written by the judge: {len(files)} {sorted(files)}")
    print(f"verdict: {verdict}")
    assert "/fixed.py" not in files, "read-only judge must not be able to write"
    print("OK: the read-only judge physically could not edit; it only judged.")


if __name__ == "__main__":
    asyncio.run(main())

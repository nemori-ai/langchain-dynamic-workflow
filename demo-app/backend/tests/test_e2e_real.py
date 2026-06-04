"""Real-model E2E acceptance for the demo's HEADLINE path (gated, opt-in).

Offline tests prove the wiring; this proves the real thing. It runs the
``deep_research`` preset through the SAME engine-facing path the host tool uses
(``run_workflow`` + the real roster + the demo's :class:`UiAdapter`), against a real
OpenRouter model, and asserts the headline properties a fallback or empty run could not
fake:

* a non-empty synthesized result (the writer leaf actually produced a report);
* at least one ``fanout_graph`` event — a REAL parallel fan-out happened (a flat
  sequential or short-circuited run emits none);
* at least two ``phase_timeline`` events — the orchestration narrated its phases; and
* INLINE STREAMING: at least one event's monotonic timestamp is strictly before the
  run's completion timestamp, proving events arrived DURING the run (the engine's sinks
  fire inline from inside the orchestration), not batched after it returned.

Gating. The test is skipped unless ``LDW_DEMO_REAL_MODEL`` is set. When it IS set, an
OpenRouter key MUST be in force (``.env`` ``OPENROUTER_API_KEY``) — otherwise the run
would silently fall back to the deterministic offline roster and the assertions would
pass on a FAKE run, defeating the acceptance. So with the gate on and no key, the test
FAILS loudly rather than skipping or passing on a fallback. LangSmith tracing is disabled
in setup so a deepagent-heavy run does not stall or leak on tracing.

Run it (orchestrator, with a real key in the backend ``.env``), from ``demo-app/backend``::

    LDW_DEMO_REAL_MODEL=1 uv run pytest tests/test_e2e_real.py -q -s
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from ui_adapter import UiAdapter
from workflows import deep_research, make_roster, make_workflows

from langchain_dynamic_workflow import run_workflow

# The opt-in gate. Other repo examples use this same env var as the real-model switch.
_REAL_MODEL_GATE = "LDW_DEMO_REAL_MODEL"


@pytest.fixture(autouse=True)
def _real_model_setup() -> None:
    """Gate on the real-model flag and disable LangSmith tracing for the run.

    Skips the whole module unless ``LDW_DEMO_REAL_MODEL`` is set. When it IS set, a real
    OpenRouter key MUST be present — a missing key would silently route to the offline
    fake roster and let the assertions pass on a fake run, so that case FAILS loudly
    rather than skipping. Tracing is turned off so a deepagent-heavy run does not stall
    or leak telemetry.
    """
    if not os.environ.get(_REAL_MODEL_GATE):
        pytest.skip(f"{_REAL_MODEL_GATE} not set; real-model E2E is opt-in")

    # Disable LangSmith / LangChain tracing for the heavy real run.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    if is_offline():
        pytest.fail(
            f"{_REAL_MODEL_GATE} is set but no OpenRouter key is in force "
            "(set OPENROUTER_API_KEY in the backend .env). The HEADLINE path must run "
            "against a real model — a fallback to the offline roster cannot be accepted."
        )


def is_offline() -> bool:
    """Re-resolve the demo's offline state at call time (after env setup)."""
    from _models import is_offline as _is_offline

    return _is_offline()


async def test_deep_research_real_model_streams_headline_fanout_inline() -> None:
    """The real ``deep_research`` run streams a real parallel fan-out inline, then synthesizes.

    Runs the preset through ``run_workflow`` with the real roster and the demo's
    ``UiAdapter`` (the exact path the ``run_live`` host tool drives), capturing
    ``(monotonic_ts, component, props)`` for every emitted Gen-UI event. Asserts the
    headline properties that a fallback or empty run could not produce: a non-empty
    report, a real ``fanout_graph`` event, >=2 ``phase_timeline`` events, and at least
    one event observed strictly BEFORE the run completed (inline streaming).
    """
    events: list[tuple[float, str, dict[str, Any]]] = []

    def _capture(component: str, props: dict[str, Any]) -> None:
        events.append((time.monotonic(), component, dict(props)))

    adapter = UiAdapter(emit=_capture)

    # A small, cheap angle set keeps cost bounded while still exercising a real fan-out
    # (the preset fans out one researcher per angle).
    started = time.monotonic()
    result = await run_workflow(
        lambda ctx: deep_research(
            ctx,
            {"question": "What are the main trade-offs between RAG and long-context LLMs?"},
        ),
        roster=make_roster(),
        workflows=make_workflows(),
        on_progress=adapter.on_progress,
        on_span=adapter.on_span,
    )
    completed = time.monotonic()

    # 1. A real, non-empty synthesized product.
    assert isinstance(result, str)
    assert result.strip(), "the real run produced an empty report"
    assert len(result.strip()) > 40, f"suspiciously short report: {result!r}"

    # 1b. NOT the offline fake writer leaf's output. The offline writer is a
    #     `_fake_echo_leaf("writer")` that returns `writer: <trimmed synthesize prompt>`,
    #     so a silent fallback would echo the prompt back behind a "writer:" prefix and
    #     carry the prompt's own instruction text. A real model never produces that. This
    #     is a second, content-level guard so a fallback cannot pass even if the offline
    #     gate above ever regresses — the assertions below (fan-out, phases, inline) are
    #     all satisfiable by the offline fake roster, so the model identity must be pinned
    #     on the produced CONTENT, not only on the run shape.
    assert not result.lstrip().lower().startswith("writer:"), (
        f"report carries the offline fake writer leaf's echo prefix (fell back offline): {result!r}"
    )
    assert "write a concise research report" not in result.lower(), (
        f"report echoes the synthesize PROMPT verbatim (offline fake echo leaf): {result!r}"
    )

    components = [comp for _ts, comp, _props in events]

    # 2. A real parallel fan-out actually happened (the search phase fans out researchers).
    fanout = [props for _ts, comp, props in events if comp == "fanout_graph"]
    assert len(fanout) >= 1, f"expected >=1 fanout_graph event, got components={components}"
    assert any(props.get("thunk_count", 0) >= 2 for props in fanout), (
        f"expected a parallel barrier spanning multiple researcher thunks (fanout props: {fanout})"
    )

    # 3. The orchestration narrated at least two phases.
    phase_events = [ts for ts, comp, _props in events if comp == "phase_timeline"]
    assert len(phase_events) >= 2, f"expected >=2 phase_timeline events, got {len(phase_events)}"

    # 4. Inline streaming: at least one event arrived strictly BEFORE the run returned.
    #    A run that batched its events after completion (or never streamed) would have
    #    every event timestamp at or after `completed`.
    assert events, "no UI events were emitted during the real run"
    earliest_event_ts = min(ts for ts, _comp, _props in events)
    assert started <= earliest_event_ts < completed, (
        "events must arrive DURING the run (inline streaming), not after it completed: "
        f"earliest={earliest_event_ts}, started={started}, completed={completed}"
    )

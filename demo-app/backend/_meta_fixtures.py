"""Meta-layer demo fixtures: an authored script that passes the gate, and one rejected.

The meta layer lets a host author an ``async def orchestrate(ctx, args)`` on the spot
and submit the *source* across the engine's AST security gate before it runs. These
two scripts are lifted verbatim from the runnable meta-layer example so the interactive
demo exercises the exact same gate-pass and gate-fail shapes:

* ``AUTHORED_SCRIPT`` — a clean parallel-research -> synthesize orchestration that uses
  only ``ctx`` primitives and plain builtins (no imports, f-strings not ``.format``), so
  the AST gate admits it and it runs against the demo roster's ``researcher`` / ``writer``.
* ``REJECTED_SCRIPT`` — reaches for ``import statistics``, so the gate rejects it and
  hands back the exact line-numbered violation (the teachable failure). Nothing runs.

Security boundary: the gate stops an accidental slip, not a determined adversary — an
in-process restricted ``exec`` is not a security sandbox. Submit only host-authored
scripts.
"""

from __future__ import annotations

from collections.abc import Sequence

# Topics the authored script fans out over; both scripts read ``args["topics"]``.
META_TOPICS: list[str] = ["grid-scale batteries", "solar PV", "onshore wind"]

# The script the demo first submits on the gate-fail path — it reaches for ``import``,
# so the gate rejects it and hands back the exact violation (the teachable failure).
REJECTED_SCRIPT = """\
import statistics

async def orchestrate(ctx, args):
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in args["topics"]]
    )
    return statistics.mode(findings)
"""

# The corrected script: a clean parallel-research -> synthesize orchestration that
# uses only ctx primitives and plain builtins (no imports, f-strings not .format).
AUTHORED_SCRIPT = """\
meta = {"name": "ad-hoc-energy-compare"}

async def orchestrate(ctx, args):
    topics = sorted(args["topics"])
    ctx.phase("research")
    findings = await ctx.parallel(
        [lambda t=t: ctx.agent(f"Research {t}", agent_type="researcher") for t in topics]
    )
    surviving = [f for f in findings if f is not None]
    ctx.phase("synthesize")
    joined = "\\n".join(surviving)
    return await ctx.agent(f"Synthesize a recommendation from:\\n{joined}", agent_type="writer")
"""


__all__: Sequence[str] = ["AUTHORED_SCRIPT", "META_TOPICS", "REJECTED_SCRIPT"]

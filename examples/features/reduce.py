"""Cross-leaf reduce helpers — ``dedup`` / ``corroborate`` / ``survives`` / ``reconcile``.

The reduce family is one coherent topic: how a script folds many independent
leaf outputs back into a defensible answer. One screening-style scenario over a
fixed topic exercises all four in sequence:

    gather      — one source leaf per source returns a ``Candidate(claim, aspect)``;
                  some sources emit a literal-duplicate claim, others overlap on aspect.
    dedup       — merge literal-duplicate claim strings (``dedup(key=...)``).
    corroborate — keep only the aspects >= 2 sources independently flagged
                  (``corroborate(key=..., min_support=2)``); drop the unsupported.
    survives    — for each corroborated claim, 3 skeptics vote; a finding is killed
                  once 2 refute it (``survives(against=..., kill_at=2)``).
    reconcile   — two blind screeners decide each survivor; unanimous-include lands
                  in ``included``, unanimous-exclude in ``excluded``, a split (or a
                  failed screener) escalates to ``conflicts`` (``reconcile(include=...)``).

The two faithful sub-workflows the scenario is built from stay importable on their
own: ``code_review`` (loop-until-dry hunt -> adversarial verify, ``Finding`` ->
``Verdict``) and ``screening`` (gather -> corroborate -> dual-blind screen ->
reconcile, ``Candidate`` -> ``Screen``). Runs fully offline on deterministic fakes.

    uv run python -m examples.features.reduce
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from examples._shared.offline_models import structured_builder
from langchain_dynamic_workflow import (
    Ctx,
    ReviewItem,
    Roster,
    corroborate,
    dedup,
    reconcile,
    run_workflow,
    survives,
)

MAX_ROUNDS = 4
SKEPTICS_PER_FINDING = 3
REFUTES_TO_KILL = 2

# A small snippet with one genuine bug (SQL injection) and some plausible-but-benign
# bait (an unused import, a plaintext password compare) for the skeptics to weigh.
SAMPLE_CODE = """
import os
import hashlib  # imported but never used

def login(db, username, password):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    row = db.execute(query).fetchone()
    return row is not None and row["password"] == password
"""

# Offline marker: the fake reviewer plants this bait and the fake skeptic refutes it,
# so the offline run deterministically demonstrates a refuted finding being dropped.
_FALSE_MARKER = "FALSE_POSITIVE"

TOPIC = "the operational trade-offs of microservices vs a monolith"
SOURCES = ["a platform engineering report", "a postmortem blog", "a vendor whitepaper", "a survey"]
SCREENERS = 2

# A small, fixed vocabulary of aspects. corroborate() groups on an EXACT key, so the
# sources must agree on a low-cardinality structured field for cross-leaf backing to
# register — free-form claim prose never collides across independent model calls. With
# more sources than aspects, the pigeonhole principle guarantees at least one aspect is
# corroborated, so the demo never collapses to an empty result on a real model.
Aspect = Literal["operational complexity", "deployment and scaling", "team and cost"]
ASPECTS: list[Aspect] = ["operational complexity", "deployment and scaling", "team and cost"]


# ── structured leaf contracts (schema-as-handoff) ─────────────────────────────


class Finding(BaseModel):
    """A single code-review finding."""

    title: str
    severity: str


class ReviewBatch(BaseModel):
    """A batch of findings from one review pass."""

    findings: list[Finding]


class Verdict(BaseModel):
    """One skeptic's adversarial ruling on a finding."""

    refuted: bool
    reason: str


class Candidate(BaseModel):
    """One candidate claim a source-leaf surfaces, tagged with the aspect it bears on."""

    claim: str
    aspect: Aspect


# `from __future__ import annotations` defers the `aspect: Aspect` annotation as a string,
# so pydantic cannot resolve the `Aspect` alias until the model is rebuilt with it in scope.
# Without this, model_json_schema() (called when agent(schema=Candidate) builds its journal
# key) raises "Candidate is not fully defined".
Candidate.model_rebuild()


class Screen(BaseModel):
    """One screener's include/exclude decision on a corroborated claim."""

    keep: bool
    reason: str


# ── sub-workflow: code_review (loop-until-dry hunt -> adversarial verify) ──────


def _review_prompt(target: str, seen: list[str]) -> str:
    return (
        "You are a meticulous code reviewer. List concrete, real bugs in the code "
        f"below that are NOT already in this list: {seen}. For each, give a short "
        "title and a severity (low/medium/high). Only report genuine defects.\n\n"
        f"{target}"
    )


def _verify_prompt(voter: int, title: str) -> str:
    return (
        f"You are skeptic #{voter + 1} reviewing a claimed code-review finding. Set "
        "`refuted` to true unless the finding is a real, exploitable defect; default "
        "to refuted for vague, stylistic, or non-issue claims. Give one sentence of "
        f"`reason`.\nFinding: {title}"
    )


async def code_review(ctx: Ctx, args: dict[str, Any]) -> list[str]:
    """loop-until-dry hunt -> adversarial verify -> reduce, quality-pattern style.

    The reviewer hands back ``Finding`` objects and each skeptic a ``Verdict``, so the
    reduce between phases is plain Python over typed data. The loop stops after two
    dry rounds in a row and never exceeds ``MAX_ROUNDS``.
    """
    target: str = args["target"]
    seen: set[str] = set()
    confirmed: list[str] = []
    dry_streak = 0

    for _round in range(MAX_ROUNDS):
        ctx.phase("review")
        batch = await ctx.agent(
            _review_prompt(target, sorted(seen)), agent_type="reviewer", schema=ReviewBatch
        )
        fresh = [finding for finding in batch.findings if finding.title not in seen]
        if not fresh:
            dry_streak += 1
            ctx.log(f"dry round ({dry_streak}/2) — no new findings")
            if dry_streak >= 2:  # converged
                break
            continue
        dry_streak = 0
        for finding in fresh:
            seen.add(finding.title)

        ctx.phase("verify")
        for finding in fresh:
            votes = await ctx.parallel(
                [
                    lambda t=finding.title, v=v: ctx.agent(
                        _verify_prompt(v, t), agent_type="skeptic", schema=Verdict
                    )
                    for v in range(SKEPTICS_PER_FINDING)
                ]
            )
            # Fail-safe: a skeptic that failed (None) counts as a refutation, so a
            # finding is never confirmed on absent verification.
            survived = survives(votes, against=lambda vote: vote.refuted, kill_at=REFUTES_TO_KILL)
            refutes = sum(1 for vote in votes if vote is None or vote.refuted)
            mark = "kept" if survived else "killed"
            ctx.log(f"finding {mark} ({refutes}/{SKEPTICS_PER_FINDING}): {finding.title[:60]}")
            if survived:
                confirmed.append(finding.title)

    return sorted(confirmed)


# ── sub-workflow: screening (gather -> corroborate -> dual-blind screen -> reconcile) ─


def _gather_prompt(topic: str, source: str) -> str:
    return (
        f"From the perspective of {source}, state THE single most important claim about "
        f"{topic}. Put one short, self-contained sentence in `claim`, and classify which "
        f"aspect it bears on in `aspect` (choose exactly one of the allowed values)."
    )


def _screen_prompt(topic: str, claim: str, n: int) -> str:
    return (
        f"You are screener #{n + 1}. Decide whether this claim about {topic} is specific and "
        f"defensible enough to INCLUDE in a briefing. Set `keep` and give one-sentence `reason`.\n"
        f"Claim: {claim}"
    )


async def screening(ctx: Ctx, args: dict[str, Any]) -> dict[str, list[str]]:
    """gather -> corroborate -> dual-blind screen -> reconcile."""
    topic: str = args["topic"]

    ctx.phase("gather")
    candidates = await ctx.parallel(
        [
            lambda s=s: ctx.agent(_gather_prompt(topic, s), agent_type="source", schema=Candidate)
            for s in SOURCES
        ]
    )

    # Cross-leaf corroboration on the structured `aspect` key (free claim prose never
    # collides across independent model calls): keep the aspects at least two sources
    # independently flagged, dropping the rest, and carry one representative claim each.
    groups = corroborate(candidates, key=lambda c: c.aspect, min_support=2)
    corroborated = [group.members[0].claim for group in groups]
    ctx.log(f"corroborated {len(corroborated)} aspect(s) backed by >= 2 of {len(SOURCES)} sources")

    ctx.phase("screen")
    review: list[ReviewItem[str, Screen]] = []
    for claim in corroborated:
        verdicts = await ctx.parallel(
            [
                lambda c=claim, n=n: ctx.agent(
                    _screen_prompt(topic, c, n), agent_type="screener", schema=Screen
                )
                for n in range(SCREENERS)
            ]
        )
        review.append(ReviewItem(item=claim, verdicts=verdicts))

    result = reconcile(review, include=lambda v: v.keep)
    ctx.log(
        f"included {len(result.included)}, excluded {len(result.excluded)}, "
        f"conflicts {len(result.conflicts)}"
    )
    return {
        "included": result.included,
        "excluded": result.excluded,
        "conflicts": result.conflicts,
    }


# ── leaves (deterministic offline fakes) ──────────────────────────────────────


def _gather_builder(*, response_format: Any = None) -> Any:
    """Offline source leaf that surfaces a ``Candidate`` keyed off its prompt.

    Two sources flag the SAME aspect with the SAME literal claim string (so dedup
    merges the duplicate and corroborate clears ``min_support=2``); the remaining
    sources each raise a distinct aspect only they flag (each below threshold, so
    corroborate visibly drops them). The reduce family then has real work to do.
    """

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        index = next((i for i, source in enumerate(SOURCES) if source in prompt), 0)
        if index < 2:
            # The corroborated pair: same aspect AND the same literal claim string, so
            # the duplicate is merged by dedup before corroborate counts the backing.
            candidate = Candidate(
                claim="Operational complexity rises sharply with service count.",
                aspect=ASPECTS[0],
            )
        else:
            candidate = Candidate(
                claim=f"A distinct claim from source {index}.",
                aspect=ASPECTS[(index - 1) % len(ASPECTS)],
            )
        return {
            "messages": [*inp["messages"], AIMessage(content="surfaced a claim")],
            "structured_response": candidate,
        }

    return RunnableLambda(_leaf)


def _verify_builder(*, response_format: Any = None) -> Any:
    """Offline skeptic that refutes only the bait (its title carries the marker)."""

    async def _leaf(inp: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        prompt = inp["messages"][-1].text if inp["messages"] else ""
        refuted = _FALSE_MARKER in prompt
        return {
            "messages": [*inp["messages"], AIMessage(content="judged")],
            "structured_response": Verdict(
                refuted=refuted, reason="bait" if refuted else "looks like a real defect"
            ),
        }

    return RunnableLambda(_leaf)


# ── driver: one gather -> dedup -> corroborate -> survives -> reconcile scenario ──


async def main() -> None:
    roster = (
        Roster()
        .register("source", builder=_gather_builder, description="Surfaces one candidate claim")
        .register("skeptic", builder=_verify_builder, description="Adversarially verifies a claim")
        .register(
            "screener",
            builder=structured_builder(lambda: Screen(keep=True, reason="specific and defensible")),
            description="Includes/excludes a claim",
        )
    )

    async def orchestrate(ctx: Ctx) -> dict[str, Any]:
        # gather: one source leaf per source, some emitting a literal-duplicate claim.
        ctx.phase("gather")
        candidates = await ctx.parallel(
            [
                lambda s=s: ctx.agent(
                    _gather_prompt(TOPIC, s), agent_type="source", schema=Candidate
                )
                for s in SOURCES
            ]
        )
        present = [c for c in candidates if c is not None]

        # dedup: merge literal-duplicate claim strings.
        unique = dedup(present, key=lambda c: c.claim.strip().lower())
        merged_dups = len(present) - len(unique)
        ctx.log(f"dedup merged {merged_dups} literal-duplicate claim(s)")

        # corroborate: keep only the aspects >= 2 sources independently flagged.
        groups = corroborate(present, key=lambda c: c.aspect, min_support=2)
        kept_aspects = {group.key for group in groups}
        dropped_aspects = len({c.aspect for c in present}) - len(kept_aspects)
        corroborated = [group.members[0].claim for group in groups]
        ctx.log(f"corroborate kept {len(kept_aspects)} aspect(s), dropped {dropped_aspects}")

        # survives: for each corroborated claim, 3 skeptics vote; >= 2 refutes kills it.
        # One claim carries the bait marker so the skeptics refute it by majority.
        bait = f"{_FALSE_MARKER}: {corroborated[0]}" if corroborated else _FALSE_MARKER
        under_vote = [*corroborated, bait]
        survivors: list[str] = []
        killed_claims = 0
        ctx.phase("verify")
        for claim in under_vote:
            votes = await ctx.parallel(
                [
                    lambda c=claim, v=v: ctx.agent(
                        _verify_prompt(v, c), agent_type="skeptic", schema=Verdict
                    )
                    for v in range(SKEPTICS_PER_FINDING)
                ]
            )
            if survives(votes, against=lambda vote: vote.refuted, kill_at=REFUTES_TO_KILL):
                survivors.append(claim)
            else:
                killed_claims += 1
        ctx.log(f"survives kept {len(survivors)} claim(s), killed {killed_claims}")

        # reconcile: two blind screeners decide each survivor; bucket by agreement.
        ctx.phase("screen")
        review: list[ReviewItem[str, Screen]] = []
        for claim in survivors:
            verdicts = await ctx.parallel(
                [
                    lambda c=claim, n=n: ctx.agent(
                        _screen_prompt(TOPIC, c, n), agent_type="screener", schema=Screen
                    )
                    for n in range(SCREENERS)
                ]
            )
            review.append(ReviewItem(item=claim, verdicts=verdicts))
        result = reconcile(review, include=lambda v: v.keep)
        ctx.log(
            f"reconcile included {len(result.included)}, excluded {len(result.excluded)}, "
            f"conflicts {len(result.conflicts)}"
        )
        return {
            "merged_dups": merged_dups,
            "dropped_aspects": dropped_aspects,
            "killed_claims": killed_claims,
            "included": result.included,
            "excluded": result.excluded,
            "conflicts": result.conflicts,
        }

    print(f"topic: {TOPIC}")
    result = await run_workflow(orchestrate, roster=roster, thread_id="reduce")
    merged_dups: int = result["merged_dups"]
    dropped_aspects: int = result["dropped_aspects"]
    killed_claims: int = result["killed_claims"]
    print(f"dedup merged duplicates: {merged_dups}")
    print(f"corroborate dropped aspects: {dropped_aspects}")
    print(f"survives killed claims: {killed_claims}")
    print(f"reconcile included: {result['included']}")
    print(f"reconcile excluded: {result['excluded']}")
    print(f"reconcile conflicts: {result['conflicts']}")

    assert merged_dups >= 1, "dedup must merge at least one literal duplicate"
    assert dropped_aspects >= 1, "corroborate must drop at least one unsupported aspect"
    assert killed_claims >= 1, "survives must kill at least one refuted claim"
    assert result["included"], "reconcile must include the unanimously-kept survivor"
    print("OK: dedup, corroborate, survives, and reconcile all did real work.")


if __name__ == "__main__":
    asyncio.run(main())

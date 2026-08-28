"""Recommendation questions: the ones the relevance spectrum exists for.

Each of these presents an opportunity and asks for a ranked answer. The reasoning
trace is the point. It does not simply list winners -- it names the decisive
criterion, walks the candidates against it, and then explicitly rejects the
candidates that look right and are not, quoting what makes them tempting before
saying why that is not enough.

That rejection step is the whole experiment. A model that has only ever seen
correct answers learns to rank; a model that has seen correct answers alongside
a named, reasoned rejection of the plausible alternative learns to discriminate.
"""

from __future__ import annotations

import random

from .graph import World
from .items import QRAItem
from .render import (
    bullets,
    caps,
    oxford,
    plural,
    ranked_candidate,
    ranked_contract,
)
from .scoring import (
    hard_negatives,
    rank_partners,
    rank_past_performance,
    tier_spread,
)
from .taxonomy import CAPABILITY_BY_ID

TOP_N = 4
REJECT_N = 3


def _gap_sentence(world: World, opportunity) -> tuple[list[str], list[str], str]:
    gap = world.capability_gap(opportunity)
    covered = world.covered_requirements(opportunity)
    sentence = (
        f"{opportunity.name} calls for {caps(opportunity.required_capabilities)}. "
        f"We self-perform {caps(covered)}, so the gap we have to team for is "
        f"{caps(gap)}."
    )
    return gap, covered, sentence


def _walk_candidates(assessments: list, limit: int) -> str:
    lines = []
    for assessment in assessments[:limit]:
        # Three factors rather than two: when candidates share the decisive
        # one, the trace still has to show what actually separates them.
        evidence = "; ".join(f.evidence for f in assessment.top_factors(3) if f.score > 0)
        lines.append(
            f"- {assessment.company.name} ({assessment.tier_label}): {evidence}"
        )
    return "\n".join(lines)


def _rejections(traps: list) -> str:
    if not traps:
        return ""
    lines = ["", "Worth rejecting out loud, because they are the names that come up first:"]
    for trap in traps:
        collab = trap.factor("collaboration")
        appeal = collab.evidence if collab and collab.score > 0 else None
        if not appeal:
            domain = trap.factor("domain_overlap")
            appeal = domain.evidence if domain else "looks superficially aligned"
        lines.append(f"- {trap.company.name}: {appeal}. But {trap.disqualifier}.")
    return "\n".join(lines)


def _answer_block(
    title: str, ranked: list, traps: list, bottom_line: str, extra: str | None = None
) -> str:
    parts = [title, ""]
    parts.append(
        "\n\n".join(
            ranked_candidate(i, a) for i, a in enumerate(ranked[:TOP_N], start=1)
        )
    )
    if traps:
        parts.append("")
        parts.append("Not recommended, despite looking like obvious picks:")
        parts.append(
            bullets([f"{t.company.name}: {t.disqualifier}" for t in traps])
        )
    if extra:
        parts.append("")
        parts.append(extra)
    parts.append("")
    parts.append(f"Bottom line: {bottom_line}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# D1 -- teaming recommendation
# ---------------------------------------------------------------------------

TEAMING_TEMPLATES = [
    "Who should we team with on this?",
    "Best teaming partners for this one, based on our past performance?",
    "We're planning to prime this. Who do we need on the team?",
    "Build me a teaming recommendation for this opportunity.",
    "Which partners give us the strongest team here, and who should we skip?",
]


def build_teaming(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for opportunity in world.opportunities.values():
        ranked = rank_partners(world, opportunity, "teaming")
        traps = hard_negatives(ranked, REJECT_N)
        gap, _covered, gap_line = _gap_sentence(world, opportunity)
        spread = tier_spread(ranked)

        adjacency = [
            a
            for a in ranked[:TOP_N]
            if "adjacent-only" in (a.factor("gap_fill").evidence if a.factor("gap_fill") else "")
        ]
        adj_note = ""
        if adjacency:
            names = oxford([a.company.name for a in adjacency])
            adj_note = (
                f"Caveat on {names}: their coverage is adjacent rather than direct. "
                f"Adjacent experience shortens the ramp; it does not remove it, and a "
                f"technical evaluator will read it the same way."
            )

        # When many partners cover the gap outright, the ranking is decided
        # by the secondary criteria. Saying so is the honest move: a trace
        # that presents several identical "decisive" candidates in a
        # confident order teaches the model that ordering is arbitrary.
        full_cover = [
            a
            for a in ranked
            if a.factor("gap_fill") and a.factor("gap_fill").score >= 0.999
        ]
        if len(full_cover) > TOP_N:
            tiebreak = (
                f"{len(full_cover)} partners cover the gap outright, so coverage "
                f"alone does not separate them. The ranking below turns on the "
                f"secondary criteria: {opportunity.agency} familiarity first, then "
                f"how the prior teaming actually went."
            )
        else:
            tiebreak = (
                "Ranking on gap coverage first. A partner who only duplicates what "
                "we already self-perform costs workshare and adds nothing to the "
                "technical score. After that: customer familiarity, then how the "
                "prior teaming actually went."
            )

        reasoning = "\n".join(
            [
                gap_line,
                "",
                tiebreak,
                "",
                "Against that gap:",
                _walk_candidates(ranked, TOP_N + 2),
                _rejections(traps),
            ]
        ).strip()

        top = ranked[0]
        bottom_line = (
            f"{top.company.name} is the anchor - {top.factor('gap_fill').evidence}. "
            f"Build the rest of the team around the remaining gap, not around who we "
            f"have worked with most."
        )

        answer = _answer_block(
            f"Recommended teaming partners for {opportunity.name}:",
            ranked,
            traps,
            bottom_line,
            adj_note or None,
        )

        for template in rng.sample(TEAMING_TEMPLATES, min(mult, len(TEAMING_TEMPLATES))):
            items.append(
                QRAItem(
                    question=f"{opportunity.brief()}\n\n{template}",
                    reasoning=reasoning,
                    answer=answer,
                    archetype="teaming_recommendation",
                    layer="recommendation",
                    meta={
                        "opportunity": opportunity.id,
                        "gap": gap,
                        "tier_spread": spread,
                        "n_traps": len(traps),
                    },
                )
            )
    return items


# ---------------------------------------------------------------------------
# D2 -- prime candidates (we intend to sub)
# ---------------------------------------------------------------------------

PRIME_TEMPLATES = [
    "We'd rather sub on this. Who could prime it?",
    "Which partners could carry the prime slot here?",
    "If we're not priming, who should we approach to prime?",
    "Who has the vehicle and the scale to prime this?",
]


def build_prime_candidates(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for opportunity in world.opportunities.values():
        ranked = rank_partners(world, opportunity, "prime")
        traps = hard_negatives(ranked, REJECT_N)
        eligible = [a for a in ranked if not a.disqualifier]

        reasoning = "\n".join(
            [
                f"{opportunity.name} runs on {opportunity.vehicle}. Priming it requires "
                f"that vehicle, so vehicle access is gating rather than merely helpful: "
                f"without it there is no contract for us to sub under, however good the "
                f"fit looks otherwise.",
                "",
                f"{plural(len(eligible), 'partner')} clear that bar. Ranking the "
                f"survivors on customer familiarity, technical breadth against the "
                f"requirement, and whether we have a working relationship to trade on.",
                "",
                "Against those criteria:",
                _walk_candidates(eligible, TOP_N + 1),
                _rejections(traps),
            ]
        ).strip()

        bottom_line = (
            (
                f"{eligible[0].company.name} is the approach to make first. "
                f"Lead with the past performance we bring to their gap, not with a "
                f"generic capability statement."
            )
            if eligible
            else (
                f"Nobody in the network holds {opportunity.vehicle}. Either this is a "
                f"no-bid, or we pursue it as a new teaming relationship with lead time "
                f"we do not currently have."
            )
        )

        answer = _answer_block(
            f"Prime candidates for {opportunity.name}:",
            eligible or ranked,
            traps,
            bottom_line,
        )

        for template in rng.sample(PRIME_TEMPLATES, min(mult, len(PRIME_TEMPLATES))):
            items.append(
                QRAItem(
                    question=f"{opportunity.brief()}\n\n{template}",
                    reasoning=reasoning,
                    answer=answer,
                    archetype="prime_candidates",
                    layer="recommendation",
                    meta={
                        "opportunity": opportunity.id,
                        "n_eligible": len(eligible),
                        "tier_spread": tier_spread(ranked),
                    },
                )
            )
    return items


# ---------------------------------------------------------------------------
# D3 -- subcontractor selection for a named gap
# ---------------------------------------------------------------------------


def build_sub_candidates(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    templates = [
        "We're priming. Who should we bring on as a sub for {cap}?",
        "Need a sub to cover {cap} on this. Options?",
        "Who fills the {cap} hole in our technical approach here?",
    ]
    items: list[QRAItem] = []
    for opportunity in world.opportunities.values():
        gap = world.capability_gap(opportunity)
        if not gap:
            continue
        target = gap[0]
        spec = CAPABILITY_BY_ID[target]

        ranked = rank_partners(world, opportunity, "sub")
        # Narrow to partners who actually carry the named capability.
        direct = [a for a in ranked if target in a.company.capabilities]
        adjacent = [
            a
            for a in ranked
            if target not in a.company.capabilities
            and set(spec.adjacent) & set(a.company.capabilities)
        ]
        traps = hard_negatives(ranked, REJECT_N)

        reasoning = "\n".join(
            [
                f"The hole is {spec.name}; everything else on this requirement we can "
                f"self-perform. So the filter is narrow: who actually carries "
                f"{spec.name}, not who is generally strong.",
                "",
                f"{plural(len(direct), 'partner')} carry it directly. A further "
                f"{len(adjacent)} have adjacent experience in "
                f"{caps(list(spec.adjacent))}, which is worth listing as fallback but "
                f"should not be presented to the customer as equivalent.",
                "",
                "Direct coverage, ranked:",
                _walk_candidates(direct, TOP_N),
                (
                    "\nAdjacent-only fallbacks:\n" + _walk_candidates(adjacent, 2)
                    if adjacent
                    else ""
                ),
                _rejections(traps),
            ]
        ).strip()

        pool = direct or adjacent or ranked
        bottom_line = (
            f"Go with {pool[0].company.name} for the {spec.name} scope. "
            + (
                "Small business participation on this one also helps the subcontracting "
                "plan."
                if opportunity.set_aside != "Full & Open"
                else "Weight the decision on technical coverage; set-aside status is "
                "not scored here."
            )
        )

        answer = _answer_block(
            f"Subcontractor options for {spec.name} on {opportunity.name}:",
            pool,
            traps,
            bottom_line,
        )

        for template in rng.sample(templates, min(mult, len(templates))):
            items.append(
                QRAItem(
                    question=(
                        f"{opportunity.brief()}\n\n{template.format(cap=spec.name)}"
                    ),
                    reasoning=reasoning,
                    answer=answer,
                    archetype="sub_candidates",
                    layer="recommendation",
                    meta={
                        "opportunity": opportunity.id,
                        "target_capability": target,
                        "n_direct": len(direct),
                        "n_adjacent": len(adjacent),
                    },
                )
            )
    return items


# ---------------------------------------------------------------------------
# D4 -- past performance citation selection
# ---------------------------------------------------------------------------

CITATION_TEMPLATES = [
    "Which past performance references should we cite on this?",
    "Pick our three strongest past performance citations for this bid.",
    "What do we put in the past performance volume here?",
    "Which of our contracts best support this proposal?",
]


def build_citations(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for opportunity in world.opportunities.values():
        ranked = rank_past_performance(world, opportunity)
        eligible = [a for a in ranked if not a.disqualifier]
        traps = hard_negatives(ranked, REJECT_N)
        picks = eligible[:3]

        reasoning = "\n".join(
            [
                f"A past performance volume is graded on relevance, not on how good "
                f"the work was. The screen here is scope overlap with "
                f"{caps(opportunity.required_capabilities)}, then same-customer credit "
                f"for {opportunity.agency}, then recency.",
                "",
                "Recency is a gate, not a penalty: most solicitations bound citations "
                "to the last five years, so a strong but stale contract does not score "
                "lower - it is ineligible, and citing it burns a slot.",
                "",
                f"{plural(len(eligible), 'contract')} clear the screen. Top of the list:",
                "\n".join(
                    f"- {a.contract.name} ({a.contract.number}, {a.tier_label}): "
                    + "; ".join(f.evidence for f in a.top_factors(2) if f.score > 0)
                    for a in picks
                ),
                (
                    "\nExcluded despite looking strong:\n"
                    + "\n".join(
                        f"- {t.contract.name}: {t.disqualifier}" for t in traps
                    )
                    if traps
                    else ""
                ),
            ]
        ).strip()

        if picks:
            answer_parts = [
                f"Past performance citations for {opportunity.name}:",
                "",
                "\n\n".join(
                    ranked_contract(i, a) for i, a in enumerate(picks, start=1)
                ),
            ]
            if traps:
                answer_parts += [
                    "",
                    "Do not cite:",
                    bullets([f"{t.contract.name}: {t.disqualifier}" for t in traps]),
                ]
            answer_parts += [
                "",
                f"Bottom line: lead with {picks[0].contract.name} - "
                f"{picks[0].top_factors(1)[0].evidence}.",
            ]
            answer = "\n".join(answer_parts)
        else:
            answer = (
                f"Nothing in the library clears the relevance screen for "
                f"{opportunity.name}. Every candidate either falls outside the "
                f"five-year recency window or has no scope overlap with "
                f"{caps(opportunity.required_capabilities)}. This is a past "
                f"performance risk that has to be solved by teaming, using a partner's "
                f"citations rather than ours."
            )

        for template in rng.sample(CITATION_TEMPLATES, min(mult, len(CITATION_TEMPLATES))):
            items.append(
                QRAItem(
                    question=f"{opportunity.brief()}\n\n{template}",
                    reasoning=reasoning,
                    answer=answer,
                    archetype="pp_citation",
                    layer="recommendation",
                    meta={
                        "opportunity": opportunity.id,
                        "n_eligible": len(eligible),
                        "tier_spread": tier_spread(ranked),
                    },
                )
            )
    return items


# ---------------------------------------------------------------------------
# D5 -- capability gap analysis
# ---------------------------------------------------------------------------


def build_gap_analysis(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    templates = [
        "What can't we self-perform on this?",
        "Where are our capability gaps on this requirement?",
        "Do we have the bench for this, or do we need to team?",
    ]
    items: list[QRAItem] = []
    for opportunity in world.opportunities.values():
        gap, covered, _ = _gap_sentence(world, opportunity)
        coverage_lines = []
        for cap_id in gap:
            spec = CAPABILITY_BY_ID[cap_id]
            options = world.companies_with_capability(cap_id)
            known = [c for c in options if c.contracts_with_us]
            coverage_lines.append(
                f"{spec.name} - {plural(len(options), 'partner')} in the network cover "
                f"it, {len(known)} of whom we have already teamed with"
                + (f" ({oxford([c.name for c in known[:3]])})" if known else "")
            )

        reasoning = "\n".join(
            [
                f"Splitting the {len(opportunity.required_capabilities)} required "
                f"capabilities against our own bench.",
                "",
                f"We cover {len(covered)} of them: {caps(covered)}.",
                f"We do not cover {len(gap)}: {caps(gap)}.",
                "",
                "For each gap, checking depth of the partner network - a gap with one "
                "possible partner is a single point of failure on the bid, and worth "
                "flagging differently from one with a dozen options.",
            ]
        )

        thin = [
            CAPABILITY_BY_ID[c].name
            for c in gap
            if len(world.companies_with_capability(c)) <= 3
        ]
        answer = "\n".join(
            [
                f"Capability assessment for {opportunity.name}:",
                "",
                f"Self-perform ({len(covered)} of "
                f"{len(opportunity.required_capabilities)}): {caps(covered)}",
                "",
                f"Must team for ({len(gap)}):",
                bullets(coverage_lines),
                "",
                (
                    f"Risk: {oxford(thin)} has a thin partner bench. Lock in a teaming "
                    f"agreement early rather than assuming availability."
                    if thin
                    else "Every gap has multiple viable partners, so no single teaming "
                    "conversation is load-bearing."
                ),
            ]
        )

        for template in rng.sample(templates, min(mult, len(templates))):
            items.append(
                QRAItem(
                    question=f"{opportunity.brief()}\n\n{template}",
                    reasoning=reasoning,
                    answer=answer,
                    archetype="gap_analysis",
                    layer="recommendation",
                    meta={"opportunity": opportunity.id, "gap": gap, "thin_gaps": thin},
                )
            )
    return items


# ---------------------------------------------------------------------------
# D6 -- bid / no-bid
# ---------------------------------------------------------------------------


def build_bid_decision(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    templates = [
        "Bid or no-bid?",
        "Should we pursue this one?",
        "Give me a bid/no-bid read on this.",
    ]
    items: list[QRAItem] = []
    for opportunity in world.opportunities.values():
        gap = world.capability_gap(opportunity)
        pp = [a for a in rank_past_performance(world, opportunity) if not a.disqualifier]
        partners = [a for a in rank_partners(world, opportunity, "teaming") if a.tier >= 3]
        holds_vehicle = opportunity.vehicle_id in world.us.vehicles or (
            opportunity.vehicle_id == "open"
        )
        setaside_ok = (
            opportunity.set_aside in ("Full & Open", "Small Business")
            or opportunity.set_aside in world.us.socioeconomic
        )

        signals = [
            (
                "Vehicle access",
                holds_vehicle,
                f"we {'hold' if holds_vehicle else 'do not hold'} {opportunity.vehicle}",
            ),
            (
                "Set-aside eligibility",
                setaside_ok,
                f"{opportunity.set_aside} - we are {world.us.size_label}",
            ),
            (
                "Past performance",
                len(pp) >= 2,
                f"{len(pp)} citable contracts clear the relevance screen",
            ),
            (
                "Gap coverage",
                len(partners) >= 2,
                f"{len(partners)} partners rated strong or better against the "
                f"{caps(gap)} gap",
            ),
        ]
        go = sum(1 for _, ok, _ in signals if ok)
        verdict = "Bid" if go >= 3 else ("Bid with conditions" if go == 2 else "No-bid")

        reasoning = "\n".join(
            [
                "Four gates decide this: can we hold the contract, can we legally win "
                "it, can we prove we have done the work, and can we cover what we "
                "cannot do ourselves.",
                "",
                bullets(
                    [
                        f"{name}: {'pass' if ok else 'fail'} - {why}"
                        for name, ok, why in signals
                    ]
                ),
                "",
                f"{go} of 4 pass. "
                + (
                    "Vehicle and set-aside are hard gates; past performance and gap "
                    "coverage are solvable with the right partner, which is why a "
                    "two-of-four is a conditional rather than a flat no."
                ),
            ]
        )

        answer = "\n".join(
            [
                f"{verdict}: {opportunity.name}",
                "",
                bullets(
                    [
                        f"{name}: {'PASS' if ok else 'FAIL'} - {why}"
                        for name, ok, why in signals
                    ]
                ),
                "",
                (
                    "Recommended next step: "
                    + (
                        f"open teaming conversations with "
                        f"{oxford([a.company.name for a in partners[:2]])} before the "
                        f"{opportunity.due} due date."
                        if partners
                        else "no viable teaming path identified; recommend passing "
                        "unless the customer relationship justifies a stretch bid."
                    )
                ),
            ]
        )

        for template in rng.sample(templates, min(mult, len(templates))):
            items.append(
                QRAItem(
                    question=f"{opportunity.brief()}\n\n{template}",
                    reasoning=reasoning,
                    answer=answer,
                    archetype="bid_decision",
                    layer="recommendation",
                    meta={
                        "opportunity": opportunity.id,
                        "verdict": verdict,
                        "gates_passed": go,
                    },
                )
            )
    return items

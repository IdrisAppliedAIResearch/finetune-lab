"""Deterministic relevance scoring over the graph.

This module is the reason the corpus is trustworthy. Golden answers are computed
here from facts the generator already fixed, so every ranking is correct by
construction and every rationale cites something real. No language model is
consulted about who the best partner is.

The tiering is deliberately not a straight threshold on the total score. What
makes a candidate a *hard negative* is not a low score -- it is a high score on
a salient-but-not-decisive dimension combined with failure on the decisive one.
A partner with five joint awards and excellent CPARS who covers none of the
capability gap is the single most instructive example in the dataset, and a pure
threshold would file it next to companies nobody has ever heard of.
"""

from __future__ import annotations

from dataclasses import dataclass

from .entities import CURRENT_YEAR, Company, Contract, Opportunity
from .graph import World
from .taxonomy import CAPABILITY_BY_ID

TIER_LABELS: dict[int, str] = {
    4: "decisive",
    3: "strong",
    2: "transferable",
    1: "surface-only",
    0: "irrelevant",
}

# Adjacency is worth partial credit, never full. A registry developer is closer
# to a FHIR engineer than a health communications firm is, but "closer" is not
# "qualified", and the reasoning traces are expected to say so.
ADJACENCY_CREDIT = 0.4

# Weight given to matching scope that a subcontractor, rather than we, performed.
SUB_PERFORMED_CREDIT = 0.5


def _scope_evidence(self_overlap: set[str], sub_overlap: set[str]) -> str:
    if not self_overlap and not sub_overlap:
        return "no scope overlap with the requirement"
    parts = []
    if self_overlap:
        names = ", ".join(CAPABILITY_BY_ID[c].name for c in sorted(self_overlap))
        parts.append(f"self-performed {names}")
    if sub_overlap:
        names = ", ".join(CAPABILITY_BY_ID[c].name for c in sorted(sub_overlap))
        parts.append(f"covers {names} but our subcontractor performed it")
    return "; ".join(parts)


@dataclass
class Factor:
    name: str
    weight: float
    score: float
    evidence: str

    @property
    def contribution(self) -> float:
        return self.weight * self.score


def _normalised_total(factors: list[Factor]) -> float:
    """Weighted mean of the factor scores, on a true 0-1 scale.

    Dividing by the weight sum rather than trusting it to be 1.0 is what keeps
    the tiers comparable across profiles. The teaming and sub profiles carry
    weights summing to 0.85 while the prime profile sums to 1.00, so a raw
    weighted sum put them on different scales -- and since all three share the
    same tier thresholds, "decisive" silently meant 71% of maximum on one and
    60% on another. Those labels are written into the training answers, so the
    inconsistency would have been taught rather than caught.
    """
    weight = sum(f.weight for f in factors)
    if weight <= 0:
        return 0.0
    return round(sum(f.contribution for f in factors) / weight, 4)


@dataclass
class Assessment:
    company: Company
    factors: list[Factor]
    disqualifier: str | None = None
    surface_appeal: float = 0.0

    @property
    def total(self) -> float:
        return _normalised_total(self.factors)

    @property
    def tier(self) -> int:
        if self.disqualifier:
            # Attractive on the surface but decisively wrong: the trap case.
            return 1 if self.surface_appeal >= APPEAL_THRESHOLD else 0
        total = self.total
        if total >= 0.60:
            return 4
        if total >= 0.45:
            return 3
        if total >= 0.27:
            return 2
        return 1 if self.surface_appeal >= APPEAL_THRESHOLD else 0

    @property
    def tier_label(self) -> str:
        return TIER_LABELS[self.tier]

    def top_factors(self, n: int = 3) -> list[Factor]:
        return sorted(self.factors, key=lambda f: f.contribution, reverse=True)[:n]

    def factor(self, name: str) -> Factor | None:
        return next((f for f in self.factors if f.name == name), None)


# ---------------------------------------------------------------------------
# shared factor builders
# ---------------------------------------------------------------------------


def _gap_fill(world: World, company: Company, gap: list[str]) -> Factor:
    """How much of what we cannot self-perform this partner actually covers."""
    if not gap:
        return Factor("gap_fill", 0.0, 0.0, "no capability gap on this opportunity")

    direct = [c for c in gap if c in company.capabilities]
    adjacent = [
        c
        for c in gap
        if c not in company.capabilities
        and set(CAPABILITY_BY_ID[c].adjacent) & set(company.capabilities)
    ]
    raw = (len(direct) + ADJACENCY_CREDIT * len(adjacent)) / len(gap)

    if direct:
        evidence = "covers " + ", ".join(CAPABILITY_BY_ID[c].name for c in direct)
        if adjacent:
            evidence += "; adjacent-only on " + ", ".join(
                CAPABILITY_BY_ID[c].name for c in adjacent
            )
    elif adjacent:
        near = ", ".join(CAPABILITY_BY_ID[c].name for c in adjacent)
        evidence = f"no direct coverage; adjacent experience only on {near}"
    else:
        evidence = "covers none of the capability gap"

    return Factor("gap_fill", 0.35, min(raw, 1.0), evidence)


def _collaboration(world: World, company: Company) -> Factor:
    """Prior joint work: the most salient signal, and the least decisive one."""
    contracts = [world.contracts[k] for k in company.contracts_with_us]
    if not contracts:
        return Factor("collaboration", 0.15, 0.0, "no prior teaming with us")

    count = len(contracts)
    latest = max(k.end_year for k in contracts)
    recency = min(1.0, max(0.0, 1.0 - (CURRENT_YEAR - latest) / 8.0))
    strong = sum(1 for k in contracts if k.cpars in ("Exceptional", "Very Good"))
    quality = strong / count

    raw = min(1.0, 0.45 * min(count / 3.0, 1.0) + 0.30 * recency + 0.25 * quality)
    plural = "s" if count != 1 else ""
    evidence = (
        f"{count} prior joint award{plural}, most recent ending {latest}, "
        f"{strong} of {count} rated Very Good or better"
    )
    return Factor("collaboration", 0.15, raw, evidence)


def _agency_experience(world: World, company: Company, agency_id: str) -> Factor:
    served = agency_id in company.agencies_served
    abbrev = world.agency_of(agency_id).abbrev
    return Factor(
        "agency_experience",
        0.15,
        1.0 if served else 0.0,
        f"has {abbrev} past performance" if served else f"no {abbrev} history on record",
    )


def _domain_overlap(company: Company, opportunity: Opportunity) -> Factor:
    required = set(opportunity.required_capabilities)
    hit = required & set(company.capabilities)
    raw = len(hit) / len(required) if required else 0.0
    evidence = (
        f"{len(hit)} of {len(required)} required capabilities in portfolio"
        if hit
        else "no overlap with the stated requirements"
    )
    return Factor("domain_overlap", 0.10, raw, evidence)


def _vehicle_access(company: Company, opportunity: Opportunity, weight: float) -> Factor:
    holds = opportunity.vehicle_id in company.vehicles
    return Factor(
        "vehicle_access",
        weight,
        1.0 if holds else 0.0,
        f"holds {opportunity.vehicle}" if holds else f"does not hold {opportunity.vehicle}",
    )


def _set_aside_fit(company: Company, opportunity: Opportunity) -> Factor:
    target = opportunity.set_aside
    if target in ("Full & Open",):
        return Factor("set_aside_fit", 0.05, 0.5, "set-aside status not a factor here")
    if target == "Small Business":
        ok = company.size == "small"
        return Factor(
            "set_aside_fit",
            0.05,
            1.0 if ok else 0.0,
            "small business" if ok else "large business on a small business set-aside",
        )
    ok = target in company.socioeconomic
    return Factor(
        "set_aside_fit",
        0.05,
        1.0 if ok else 0.0,
        f"holds {target} status" if ok else f"does not hold {target} status",
    )


# A candidate is only worth flagging as a trap if it would genuinely tempt
# someone. Blending the salient signals rather than taking their max is what
# keeps the tier-1 population meaningful: agency experience alone is a binary
# that two thirds of the vendor list would satisfy, and letting it single-
# handedly promote a company to "looks right" made tier 1 the largest tier in
# the corpus rather than its sharpest teaching cases.
APPEAL_WEIGHTS: dict[str, float] = {
    "collaboration": 0.45,
    "domain_overlap": 0.40,
    "agency_experience": 0.15,
}
APPEAL_THRESHOLD = 0.40


def _surface_appeal(factors: list[Factor]) -> float:
    """How good this candidate looks before the decisive test is applied."""
    return sum(
        APPEAL_WEIGHTS[f.name] * f.score for f in factors if f.name in APPEAL_WEIGHTS
    )


# ---------------------------------------------------------------------------
# scoring profiles
# ---------------------------------------------------------------------------


def score_teaming_partner(world: World, company: Company, opportunity: Opportunity) -> Assessment:
    """We intend to prime; who should be on the team?

    Gap coverage is decisive because anything we already self-perform is
    workshare we would be giving away for nothing.
    """
    gap = world.capability_gap(opportunity)
    factors = [
        _gap_fill(world, company, gap),
        _collaboration(world, company),
        _agency_experience(world, company, opportunity.agency_id),
        _domain_overlap(company, opportunity),
        _vehicle_access(company, opportunity, weight=0.05),
        _set_aside_fit(company, opportunity),
    ]

    disqualifier = None
    gap_factor = next(f for f in factors if f.name == "gap_fill")
    if gap and gap_factor.score == 0.0:
        disqualifier = (
            "covers none of the capability gap -- adding them duplicates work we "
            "already self-perform"
        )

    return Assessment(company, factors, disqualifier, _surface_appeal(factors))


def score_prime_candidate(world: World, company: Company, opportunity: Opportunity) -> Assessment:
    """We intend to sub; who could carry the prime slot?

    Vehicle access is gating rather than merely weighted: without it there is no
    contract to sub under, however good the fit otherwise looks.
    """
    gap = world.capability_gap(opportunity)
    factors = [
        _vehicle_access(company, opportunity, weight=0.30),
        _agency_experience(world, company, opportunity.agency_id),
        _domain_overlap(company, opportunity),
        _collaboration(world, company),
        _gap_fill(world, company, gap),
        Factor(
            "capacity",
            0.10,
            1.0 if company.size == "large" else 0.35,
            f"{company.employees:,} employees ({company.size_label})",
        ),
    ]
    # Rebalance: the shared builders carry weights tuned for the teaming case.
    for factor in factors:
        if factor.name == "gap_fill":
            factor.weight = 0.10
        elif factor.name == "domain_overlap":
            factor.weight = 0.20
        elif factor.name == "collaboration":
            factor.weight = 0.15
        elif factor.name == "agency_experience":
            factor.weight = 0.15

    disqualifier = None
    if opportunity.vehicle_id != "open" and opportunity.vehicle_id not in company.vehicles:
        disqualifier = f"cannot prime this: no {opportunity.vehicle} access"
    elif opportunity.set_aside == "Small Business" and company.size == "large":
        disqualifier = "large business cannot prime a small business set-aside"

    return Assessment(company, factors, disqualifier, _surface_appeal(factors))


def score_subcontractor(world: World, company: Company, opportunity: Opportunity) -> Assessment:
    """We prime; who fills a specific hole in the technical approach?

    Nearly the teaming case, but small business participation carries real
    weight because subcontracting goals are scored in the evaluation.
    """
    assessment = score_teaming_partner(world, company, opportunity)
    for factor in assessment.factors:
        if factor.name == "set_aside_fit":
            factor.weight = 0.10
        elif factor.name == "vehicle_access":
            factor.weight = 0.0
            factor.evidence += " (not required as a sub under our vehicle)"
    return assessment


# ---------------------------------------------------------------------------
# past performance citation relevance
# ---------------------------------------------------------------------------


@dataclass
class ContractAssessment:
    contract: Contract
    factors: list[Factor]
    disqualifier: str | None = None
    surface_appeal: float = 0.0

    @property
    def total(self) -> float:
        return _normalised_total(self.factors)

    @property
    def tier(self) -> int:
        if self.disqualifier:
            return 1 if self.surface_appeal >= APPEAL_THRESHOLD else 0
        total = self.total
        if total >= 0.62:
            return 4
        if total >= 0.46:
            return 3
        if total >= 0.28:
            return 2
        return 1 if self.surface_appeal >= APPEAL_THRESHOLD else 0

    @property
    def tier_label(self) -> str:
        return TIER_LABELS[self.tier]

    def top_factors(self, n: int = 3) -> list[Factor]:
        return sorted(self.factors, key=lambda f: f.contribution, reverse=True)[:n]


def score_past_performance(
    world: World, contract: Contract, opportunity: Opportunity
) -> ContractAssessment:
    """Which of our own contracts best support a bid on this opportunity.

    Recency is a disqualifier, not a penalty: most solicitations bound past
    performance to the last three to five years, so a superb but stale contract
    is not merely weaker -- it is ineligible, and citing it wastes a slot.
    """
    required = set(opportunity.required_capabilities)
    overlap = required & set(contract.capabilities)
    # Scope a subcontractor performed still counts -- past performance is cited
    # at the contract level -- but only partially. An evaluator reading the
    # citation will ask who actually did the work, and a capture lead who does
    # not know the answer in advance gets caught out in orals.
    sub_overlap = overlap & set(contract.sub_performed)
    self_overlap = overlap - sub_overlap
    cap_score = (
        (len(self_overlap) + SUB_PERFORMED_CREDIT * len(sub_overlap)) / len(required)
        if required
        else 0.0
    )

    same_agency = contract.agency_id == opportunity.agency_id
    age = CURRENT_YEAR - contract.end_year
    recency_score = max(0.0, 1.0 - age / 6.0)

    ratio = contract.value_total / max(opportunity.value, 1)
    scale_score = 1.0 if 0.2 <= ratio <= 5.0 else (0.5 if 0.05 <= ratio <= 20.0 else 0.1)

    cpars_score = {
        "Exceptional": 1.0, "Very Good": 0.8, "Satisfactory": 0.5, "Marginal": 0.1
    }[contract.cpars]

    factors = [
        Factor(
            "scope_match",
            0.35,
            cap_score,
            _scope_evidence(self_overlap, sub_overlap),
        ),
        Factor(
            "customer_match",
            0.20,
            1.0
            if same_agency
            else (0.4 if _same_department(world, contract, opportunity) else 0.0),
            f"same customer ({contract.agency})"
            if same_agency
            else f"different customer ({contract.agency} vs {opportunity.agency})",
        ),
        Factor(
            "recency",
            0.15,
            recency_score,
            f"ended {contract.end_year}" + (" (still active)" if contract.is_active else ""),
        ),
        Factor(
            "scale_fit",
            0.10,
            scale_score,
            f"{contract.money()} against a {opportunity.money()} requirement",
        ),
        Factor("cpars", 0.10, cpars_score, f"{contract.cpars} CPARS"),
        Factor(
            "role",
            0.10,
            1.0 if contract.our_role == "prime" else 0.45,
            f"we were {contract.our_role} ({contract.money_ours()} of {contract.money()})",
        ),
    ]

    disqualifier = None
    if age > 5:
        disqualifier = f"outside a typical 5-year recency window (ended {contract.end_year})"
    elif not overlap:
        disqualifier = "no scope overlap with the stated requirements"

    # Same customer with a strong CPARS is exactly what makes an off-scope
    # citation tempting, so those two carry the appeal.
    by_name = {f.name: f.score for f in factors}
    salient = (
        0.60 * by_name["customer_match"]
        + 0.25 * by_name["cpars"]
        + 0.15 * by_name["scale_fit"]
    )
    return ContractAssessment(contract, factors, disqualifier, salient)


def _same_department(world: World, contract: Contract, opportunity: Opportunity) -> bool:
    return (
        world.agency_of(contract.agency_id).parent
        == world.agency_of(opportunity.agency_id).parent
    )


# ---------------------------------------------------------------------------
# ranking helpers
# ---------------------------------------------------------------------------


def rank_partners(
    world: World,
    opportunity: Opportunity,
    profile: str = "teaming",
    candidates: set[str] | None = None,
) -> list[Assessment]:
    """Rank partners for an opportunity, optionally within a candidate set.

    ``candidates`` is what makes the open-book corpus honest. The golden answer
    has to be the right ranking of *what the prompt shows*, not of the whole
    150-partner roster: ranking over everything and then showing a handful
    produced answers naming partners that appear nowhere in the context, which
    is training the model to invent names rather than to choose between the ones
    in front of it. Retrieval decides who is a candidate; this decides who wins.
    """
    scorer = {
        "teaming": score_teaming_partner,
        "prime": score_prime_candidate,
        "sub": score_subcontractor,
    }[profile]
    pool = [c for c in world.partners if candidates is None or c.id in candidates]
    assessments = [scorer(world, company, opportunity) for company in pool]
    # Deterministic ordering: score first, then name, so ties never depend on
    # dict iteration order.
    return sorted(assessments, key=lambda a: (-a.total, a.company.name))


def rank_past_performance(
    world: World,
    opportunity: Opportunity,
    candidates: set[str] | None = None,
) -> list[ContractAssessment]:
    """Rank our own contracts as citations. See rank_partners on ``candidates``."""
    assessments = [
        score_past_performance(world, contract, opportunity)
        for contract in world.contracts.values()
        if candidates is None or contract.id in candidates
    ]
    return sorted(assessments, key=lambda a: (-a.total, a.contract.name))


def tier_spread(assessments: list) -> dict[int, int]:
    """How many candidates landed in each tier -- used to reject flat questions."""
    spread = dict.fromkeys(range(5), 0)
    for assessment in assessments:
        spread[assessment.tier] += 1
    return spread


def hard_negatives(assessments: list, limit: int = 3) -> list:
    """Tier-1 candidates: the ones that look right and are not.

    Sorted by how convincing the illusion is, because the most persuasive wrong
    answer is the one worth teaching the model to reject.
    """
    traps = [a for a in assessments if a.tier == 1 and a.disqualifier]
    traps.sort(key=lambda a: -a.surface_appeal)
    return traps[:limit]

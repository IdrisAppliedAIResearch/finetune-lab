"""Shared formatting helpers.

Every answer the corpus contains is assembled here from graph facts. Keeping the
formatting in one place matters for a closed-book run: the model is learning the
*shape* of an answer as much as its content, and drifting formats across
archetypes would teach it that the shape is arbitrary.
"""

from __future__ import annotations

from .entities import Company, Contract, Opportunity, Person
from .graph import World
from .taxonomy import AGENCY_BY_ID, CAPABILITY_BY_ID, NAICS, PSC, VEHICLE_BY_ID


def money(amount: int) -> str:
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    return f"${amount / 1_000:.0f}K"


def caps(ids: list[str]) -> str:
    return ", ".join(CAPABILITY_BY_ID[c].name for c in ids) if ids else "none"


def bullets(lines: list[str], marker: str = "-") -> str:
    return "\n".join(f"{marker} {line}" for line in lines)


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))


def oxford(items: list[str]) -> str:
    items = list(items)
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def plural(count: int, noun: str, suffix: str = "s") -> str:
    return f"{count} {noun}{'' if count == 1 else suffix}"


# ---------------------------------------------------------------------------
# record renderers
# ---------------------------------------------------------------------------


def contract_record(world: World, contract: Contract, full: bool = True) -> str:
    """The canonical full-detail view of one past performance record."""
    prime_name = (
        world.us.name if contract.prime_id == "us" else world.companies[contract.prime_id].name
    )
    sub_names = [world.companies[s].name for s in contract.sub_ids]

    lines = [
        f"{contract.name} ({contract.number})",
        f"Customer: {contract.agency_full} / {contract.subunit}",
        f"Our role: {contract.our_role.title()}",
        f"Prime: {prime_name}",
        f"Subcontractors: {oxford(sub_names) if sub_names else 'none'}",
        f"Period of performance: {contract.period}",
        f"Value: {money(contract.value_total)} total, {money(contract.value_ours)} our share",
        f"Vehicle: {contract.vehicle}",
        f"NAICS: {contract.naics} ({NAICS[contract.naics]})",
        f"PSC: {contract.psc} ({PSC[contract.psc]})",
        f"Set-aside: {contract.set_aside}",
        f"Capabilities: {caps(contract.capabilities)}",
        *(
            [f"Of which subcontractor-performed: {caps(contract.sub_performed)}"]
            if contract.sub_performed
            else []
        ),
        f"CPARS: {contract.cpars}",
        f"Place of performance: {contract.place}",
    ]
    if full:
        staff = [world.people[p].describe() for p in contract.personnel_ids]
        lines.append(f"Key personnel: {oxford(staff)}")
        lines.append(f"Scope: {contract.scope}")
        lines.append("Outcomes:\n" + bullets(contract.outcomes))
    return "\n".join(lines)


def contract_line(world: World, contract: Contract) -> str:
    """One-line summary, for lists."""
    return (
        f"{contract.name} ({contract.number}) - {contract.agency}, {contract.period}, "
        f"{money(contract.value_total)}, {contract.our_role}, {contract.cpars}"
    )


def company_profile(world: World, company: Company, full: bool = True) -> str:
    """A partner record. ``full`` adds the fields only the subject of a question needs.

    A teaming prompt carries a dozen of these, so what they cost decides what
    fits in the window. Everything the ranking turns on stays in both forms:
    capabilities (the decisive criterion), vehicles and size (the prime gate),
    agencies served and joint work (what makes a hard negative tempting). The
    short form drops headquarters, founding year and headcount, which no
    question here is decided on, and the per-contract detail of prior joint work
    -- the count and the customer carry the signal, the dollar value and CPARS
    of each prior award do not.
    """
    joint = [world.contracts[k] for k in company.contracts_with_us]
    agencies = [AGENCY_BY_ID[a].abbrev for a in company.agencies_served]

    lines = [company.name, f"Size: {company.size_label}"]
    if full:
        lines.append(
            f"Headquarters: {company.hq_state}, founded {company.founded}, "
            f"{company.employees:,} employees"
        )
    lines += [
        f"Capabilities: {caps(company.capabilities)}",
        "Contract vehicles: "
        + (oxford(company.vehicle_names()) if company.vehicles else "none on record"),
        f"Agencies served: {oxford(agencies)}",
    ]
    if not joint:
        lines.append("Joint work with us: none on record")
    elif full:
        lines.append(f"Joint work with us: {plural(len(joint), 'contract')}")
        lines.append(bullets([contract_line(world, k) for k in joint]))
    else:
        lines.append(
            f"Joint work with us: {plural(len(joint), 'contract')} - "
            + oxford(sorted({f"{k.agency} {k.period}" for k in joint}))
        )
    return "\n".join(lines)


def company_line(world: World, company: Company) -> str:
    joint = len(company.contracts_with_us)
    history = f"{joint} joint award{'s' if joint != 1 else ''}" if joint else "no joint work"
    return f"{company.name} - {company.size_label}, {history}"


def person_profile(world: World, person: Person) -> str:
    contracts = world.contracts_for_person(person.id)
    lines = [
        f"{person.name} - {person.role}",
        f"Credentials: {person.degree}, {person.years_experience} years experience",
        f"Specialties: {caps(person.capabilities)}",
        f"Contract history: {plural(len(contracts), 'assignment')}",
    ]
    if contracts:
        lines.append(bullets([contract_line(world, k) for k in contracts]))
    return "\n".join(lines)


def opportunity_header(opportunity: Opportunity) -> str:
    return opportunity.brief()


# ---------------------------------------------------------------------------
# assessment renderers
# ---------------------------------------------------------------------------


def ranked_candidate(index: int, assessment, extra: str | None = None) -> str:
    """One entry in a ranked recommendation, with the evidence it rests on."""
    company = assessment.company
    header = f"{index}. {company.name} - {assessment.tier_label} fit"
    detail = [f"   {f.evidence}" for f in assessment.top_factors(3) if f.score > 0]
    if extra:
        detail.append(f"   {extra}")
    return "\n".join([header, *detail])


def ranked_contract(index: int, assessment) -> str:
    contract = assessment.contract
    header = f"{index}. {contract.name} ({contract.number}) - {assessment.tier_label} match"
    detail = [f"   {f.evidence}" for f in assessment.top_factors(3) if f.score > 0]
    return "\n".join([header, *detail])


def rejection(assessment) -> str:
    name = getattr(assessment, "company", None)
    label = name.name if name is not None else assessment.contract.name
    return f"- {label}: {assessment.disqualifier}"


def vehicle_name(vehicle_id: str) -> str:
    return VEHICLE_BY_ID[vehicle_id].name


def agency_name(agency_id: str) -> str:
    return AGENCY_BY_ID[agency_id].abbrev

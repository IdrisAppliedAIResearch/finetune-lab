"""Recall, relational, and multi-hop question builders.

These are the unglamorous two thirds of a closed-book corpus. The recommendation
questions get the attention, but a model cannot reason about a teaming
relationship it was never told exists, and knowledge injected by fine-tuning
needs the same fact reached from several directions before it survives. Hence
the paraphrase multipliers: the answer stays stable while the question that
retrieves it varies.
"""

from __future__ import annotations

import random

from .graph import World
from .items import QRAItem
from .render import (
    bullets,
    company_line,
    company_profile,
    contract_line,
    contract_record,
    money,
    oxford,
    person_profile,
    plural,
)
from .taxonomy import AGENCY_BY_ID, CAPABILITY_BY_ID, VEHICLE_BY_ID


def _pick(rng: random.Random, templates: list[str], count: int) -> list[str]:
    """Sample distinct templates, wrapping if more are asked for than exist."""
    if count <= len(templates):
        return rng.sample(templates, count)
    out = list(templates)
    while len(out) < count:
        out.append(rng.choice(templates))
    return out[:count]


# ---------------------------------------------------------------------------
# A. atomic recall
# ---------------------------------------------------------------------------

CONTRACT_TEMPLATES = [
    "Tell me about contract {number}.",
    "What was the scope of the {name} contract?",
    "Give me the full record on {number}.",
    "Summarize our past performance on {name}.",
    "What do we have on file for contract {number}?",
    "Walk me through the {name} engagement.",
    "Pull up {number}.",
    "What did we deliver under {name}?",
]

# Short, single-fact questions. Every kind of exact-value probe in the eval set
# must have a counterpart here, or the probe measures whether the model learned
# a terse answer format rather than whether it retained the fact.
CONTRACT_FOCUSED = [
    ("Who was the customer on {number}, and what was our role?", "customer"),
    ("What was the value of {name} and what was our share?", "value"),
    ("Who did we team with on {number}?", "team"),
    ("What CPARS rating did we get on {name}?", "cpars"),
    ("What was the period of performance on {number}?", "period"),
    ("Which vehicle did {name} run under?", "vehicle"),
    ("What's the contract number for {name}?", "number"),
    ("When did {name} finish?", "end_year"),
]


# How many of the eight single-fact facets get a trained terse answer per
# contract. The remainder are reserved for probes.
FOCUSED_PER_CONTRACT = 3


def build_contract_recall(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for contract in world.contracts.values():
        record = contract_record(world, contract)

        for template in _pick(rng, CONTRACT_TEMPLATES, mult):
            question = template.format(number=contract.number, name=contract.name)
            reasoning = (
                f"Direct lookup against the past performance library. {contract.number} is "
                f"our {contract.agency} {contract.subunit} engagement, {contract.name}, "
                f"where we ran as {contract.our_role}. Returning the full record."
            )
            items.append(
                QRAItem(
                    question=question,
                    reasoning=reasoning,
                    answer=record,
                    archetype="contract_detail",
                    layer="recall",
                    meta={"contract": contract.id, "agency": contract.agency_id},
                )
            )

        # Several facets per contract, but never all of them: the facets left
        # untouched here are what the eval probes draw from, so a probe asks for
        # a value the model has only ever seen inside a full record.
        for template, facet in rng.sample(CONTRACT_FOCUSED, FOCUSED_PER_CONTRACT):
            items.append(_focused_contract_item(world, contract, template, facet))
    return items


def _focused_contract_item(world: World, contract, template: str, facet: str) -> QRAItem:
    question = template.format(number=contract.number, name=contract.name)
    prime_name = (
        world.us.name if contract.prime_id == "us" else world.companies[contract.prime_id].name
    )
    subs = [world.companies[s].name for s in contract.sub_ids]

    if facet == "customer":
        answer = (
            f"{contract.agency_full} / {contract.subunit}. We were the "
            f"{contract.our_role} on {contract.name} ({contract.number}), "
            f"{contract.period}."
        )
        reasoning = f"Customer and role lookup for {contract.number}."
    elif facet == "value":
        answer = (
            f"{contract.name} ({contract.number}) was {money(contract.value_total)} total; "
            f"our share was {money(contract.value_ours)} as {contract.our_role}."
        )
        reasoning = (
            f"Value lookup. Total ceiling and our portion differ because we were "
            f"{contract.our_role} on this one."
        )
    elif facet == "team":
        if contract.our_role == "prime":
            answer = (
                f"We primed {contract.name} ({contract.number}). Subcontractors: "
                f"{oxford(subs) if subs else 'none on record'}."
            )
        else:
            answer = (
                f"We subbed to {prime_name} on {contract.name} ({contract.number})."
                + (f" Other subs: {oxford(subs)}." if subs else "")
            )
        reasoning = f"Teaming lookup for {contract.number}; we were {contract.our_role}."
    elif facet == "cpars":
        answer = (
            f"{contract.cpars} on {contract.name} ({contract.number}), "
            f"{contract.agency} {contract.subunit}, {contract.period}."
        )
        reasoning = f"CPARS lookup for {contract.number}."
    elif facet == "period":
        status = "still active" if contract.is_active else "closed out"
        answer = f"{contract.period} ({status}). {contract.name} ({contract.number})."
        reasoning = f"Period of performance lookup for {contract.number}."
    elif facet == "number":
        answer = (
            f"{contract.number}. That is {contract.name}, {contract.agency} "
            f"{contract.subunit}, {contract.period}."
        )
        reasoning = f"Contract number lookup for {contract.name}."
    elif facet == "end_year":
        status = "still active" if contract.is_active else "closed out"
        answer = (
            f"{contract.end_year}. {contract.name} ({contract.number}) ran "
            f"{contract.period} and is {status}."
        )
        reasoning = f"End-of-performance lookup for {contract.name}."
    else:  # vehicle
        answer = (
            f"{contract.vehicle}. NAICS {contract.naics}, set aside as "
            f"{contract.set_aside}."
        )
        reasoning = f"Vehicle lookup for {contract.number}."

    return QRAItem(
        question=question,
        reasoning=reasoning,
        answer=answer,
        archetype=f"contract_{facet}",
        layer="recall",
        meta={"contract": contract.id},
    )


PARTNER_TEMPLATES = [
    "What do we know about {name}?",
    "Give me a profile on {name}.",
    "What are {name}'s capabilities and vehicle access?",
    "Have we worked with {name} before?",
    "Pull the partner record for {name}.",
    "Who is {name} and what do they bring?",
]


def build_partner_recall(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for company in world.partners:
        profile = company_profile(world, company)
        joint = len(company.contracts_with_us)
        for template in _pick(rng, PARTNER_TEMPLATES, mult):
            reasoning = (
                f"Partner record lookup. {company.name} is a "
                f"{company.size_label.lower()} with "
                f"{len(company.capabilities)} capabilities on file"
                + (
                    f"; we have {plural(joint, 'prior joint award')} with them."
                    if joint
                    else "; no prior joint work on record."
                )
            )
            items.append(
                QRAItem(
                    question=template.format(name=company.name),
                    reasoning=reasoning,
                    answer=profile,
                    archetype="partner_profile",
                    layer="recall",
                    meta={"company": company.id, "joint_awards": joint},
                )
            )
    return items


AGENCY_TEMPLATES = [
    "What work have we done for {abbrev}?",
    "Summarize our {abbrev} past performance.",
    "What's our history with {abbrev}?",
    "How many contracts do we have with {abbrev}, and what are they?",
    "Show me everything in the library for {name}.",
]


def build_agency_recall(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for agency_id in {k.agency_id for k in world.contracts.values()}:
        agency = AGENCY_BY_ID[agency_id]
        contracts = sorted(
            world.contracts_by_agency(agency_id), key=lambda k: -k.start_year
        )
        total = sum(k.value_ours for k in contracts)
        primes = sum(1 for k in contracts if k.our_role == "prime")
        subunits = sorted({k.subunit for k in contracts})

        answer = "\n".join(
            [
                f"{plural(len(contracts), 'contract')} with {agency.abbrev} "
                f"({agency.name}), {money(total)} in total value to us.",
                f"Prime on {primes}, sub on {len(contracts) - primes}.",
                f"Offices served: {oxford(subunits)}.",
                "",
                bullets([contract_line(world, k) for k in contracts]),
            ]
        )
        reasoning = (
            f"Filtering the library to {agency.abbrev}. {len(contracts)} records match, "
            f"spanning {min(k.start_year for k in contracts)} to "
            f"{max(k.end_year for k in contracts)}. Sorting most recent first."
        )
        for template in _pick(rng, AGENCY_TEMPLATES, mult):
            items.append(
                QRAItem(
                    question=template.format(abbrev=agency.abbrev, name=agency.name),
                    reasoning=reasoning,
                    answer=answer,
                    archetype="agency_portfolio",
                    layer="recall",
                    meta={"agency": agency_id, "n_contracts": len(contracts)},
                )
            )
    return items


CAPABILITY_TEMPLATES = [
    "Where have we done {cap} work?",
    "What's our experience in {cap}?",
    "Which contracts in the library cover {cap}?",
    "Do we have past performance in {cap}?",
]


def build_capability_recall(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for cap_id, spec in CAPABILITY_BY_ID.items():
        contracts = world.contracts_by_capability(cap_id)
        if contracts:
            agencies = sorted({k.agency for k in contracts})
            answer = "\n".join(
                [
                    f"Yes - {plural(len(contracts), 'contract')} covering "
                    f"{spec.name}, across {oxford(agencies)}.",
                    "",
                    bullets([contract_line(world, k) for k in contracts]),
                ]
            )
            reasoning = (
                f"Searching the library for {spec.name}. {len(contracts)} records carry "
                f"that capability tag."
            )
        else:
            in_house = cap_id in world.our_capabilities()
            answer = (
                f"No past performance on file for {spec.name}. "
                + (
                    "It is within our stated capability set but has not appeared on a "
                    "contract yet, so we cannot cite it as past performance."
                    if in_house
                    else "This is outside our self-perform capability; we would need a "
                    "teaming partner to cover it."
                )
            )
            reasoning = (
                f"Searching the library for {spec.name}. No records match. Checking "
                f"whether it is a self-perform capability: "
                f"{'yes' if in_house else 'no'}."
            )
        for template in _pick(rng, CAPABILITY_TEMPLATES, mult):
            items.append(
                QRAItem(
                    question=template.format(cap=spec.name),
                    reasoning=reasoning,
                    answer=answer,
                    archetype="capability_experience",
                    layer="recall",
                    meta={"capability": cap_id, "n_contracts": len(contracts)},
                )
            )
    return items


def build_person_recall(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    templates = [
        "What's {name}'s background?",
        "Which contracts has {name} worked on?",
        "Give me the staff record for {name}.",
    ]
    items: list[QRAItem] = []
    for person in world.people.values():
        contracts = world.contracts_for_person(person.id)
        reasoning = (
            f"Personnel lookup. {person.name} is a {person.role} with "
            f"{person.years_experience} years; {plural(len(contracts), 'contract assignment')} "
            f"on record."
        )
        for template in _pick(rng, templates, mult):
            items.append(
                QRAItem(
                    question=template.format(name=person.name),
                    reasoning=reasoning,
                    answer=person_profile(world, person),
                    archetype="person_profile",
                    layer="recall",
                    meta={"person": person.id, "n_contracts": len(contracts)},
                )
            )
    return items


# ---------------------------------------------------------------------------
# B. single-hop relational
# ---------------------------------------------------------------------------


def build_teaming_history(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    templates = [
        "Which contracts have we done with {name}?",
        "What's our teaming history with {name}?",
        "How many times have we worked with {name}, and on what?",
    ]
    items: list[QRAItem] = []
    for company in world.partners:
        contracts = [world.contracts[k] for k in company.contracts_with_us]
        if not contracts:
            continue
        roles = {
            k.id: ("they subbed to us" if k.our_role == "prime" else "we subbed to them")
            for k in contracts
        }
        answer = "\n".join(
            [
                f"{plural(len(contracts), 'joint award')} with {company.name}.",
                "",
                bullets(
                    [f"{contract_line(world, k)} - {roles[k.id]}" for k in contracts]
                ),
            ]
        )
        reasoning = (
            f"Joining the library on {company.name}. They appear on "
            f"{len(contracts)} of our contracts; checking which side of the prime/sub "
            f"relationship we were on in each."
        )
        for template in _pick(rng, templates, mult):
            items.append(
                QRAItem(
                    question=template.format(name=company.name),
                    reasoning=reasoning,
                    answer=answer,
                    archetype="teaming_history",
                    layer="relational",
                    meta={"company": company.id, "joint_awards": len(contracts)},
                )
            )
    return items


def build_capability_partners(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    templates = [
        "Which partners have {cap} experience?",
        "Who in our partner network does {cap}?",
        "If we needed {cap} on a bid, who could we call?",
    ]
    items: list[QRAItem] = []
    for cap_id, spec in CAPABILITY_BY_ID.items():
        companies = world.companies_with_capability(cap_id)
        if not companies:
            continue
        worked_with = [c for c in companies if c.contracts_with_us]
        fresh = [c for c in companies if not c.contracts_with_us]
        in_house = cap_id in world.our_capabilities()

        lines = [
            f"{plural(len(companies), 'partner')} in the network carry {spec.name}."
        ]
        if in_house:
            lines.append(
                "Note: this is a self-perform capability for us, so a partner is "
                "only needed for surge capacity or workshare."
            )
        lines.append("")
        if worked_with:
            lines.append(f"Prior teaming ({len(worked_with)}):")
            lines.append(bullets([company_line(world, c) for c in worked_with[:12]]))
        if fresh:
            lines.append("")
            lines.append(f"No prior joint work ({len(fresh)}), top by breadth:")
            ranked = sorted(fresh, key=lambda c: -len(c.capabilities))[:8]
            lines.append(bullets([company_line(world, c) for c in ranked]))

        reasoning = (
            f"Scanning partner capability tags for {spec.name}: {len(companies)} matches. "
            f"Splitting by whether we have teamed with them before, since a known "
            f"quantity is worth more than a cold one at bid time."
        )
        for template in _pick(rng, templates, mult):
            items.append(
                QRAItem(
                    question=template.format(cap=spec.name),
                    reasoning=reasoning,
                    answer="\n".join(lines),
                    archetype="capability_partners",
                    layer="relational",
                    meta={"capability": cap_id, "n_partners": len(companies)},
                )
            )
    return items


def build_vehicle_questions(world: World, rng: random.Random, mult: int) -> list[QRAItem]:
    items: list[QRAItem] = []
    for vehicle_id, spec in VEHICLE_BY_ID.items():
        if vehicle_id == "open":
            continue
        holders = world.companies_with_vehicle(vehicle_id)
        ours = vehicle_id in world.us.vehicles
        contracts = world.contracts_by_vehicle(vehicle_id)

        lines = [
            f"{spec.name} ({spec.kind}).",
            f"We {'hold' if ours else 'do not hold'} this vehicle.",
        ]
        if contracts:
            lines.append(
                f"{plural(len(contracts), 'contract')} in our library ran under it:"
            )
            lines.append(bullets([contract_line(world, k) for k in contracts]))
        lines.append("")
        lines.append(f"{plural(len(holders), 'partner')} hold it:")
        lines.append(bullets([company_line(world, c) for c in holders[:12]]))

        reasoning = (
            f"Vehicle lookup for {spec.name}. Checking our own holdings first "
            f"({'we hold it' if ours else 'we do not'}), then which partners could "
            f"carry a prime slot on it."
        )
        templates = [
            f"Which partners hold {spec.name}?",
            f"Do we have {spec.name} access, and who else does?",
            f"Tell me about our position on {spec.name}.",
        ]
        for question in _pick(rng, templates, mult):
            items.append(
                QRAItem(
                    question=question,
                    reasoning=reasoning,
                    answer="\n".join(lines),
                    archetype="vehicle_coverage",
                    layer="relational",
                    meta={"vehicle": vehicle_id, "we_hold": ours},
                )
            )
    return items


def build_portfolio_questions(world: World, rng: random.Random) -> list[QRAItem]:
    """A handful of whole-portfolio rollups. Few in number, high in value:
    these are the questions a new BD hire asks in week one."""
    items: list[QRAItem] = []
    contracts = list(world.contracts.values())
    active = world.active_contracts()
    primes = [k for k in contracts if k.our_role == "prime"]
    subs = [k for k in contracts if k.our_role == "sub"]

    items.append(
        QRAItem(
            question="Which of our contracts are currently active?",
            reasoning=(
                f"Filtering the library on end year >= 2026. {len(active)} of "
                f"{len(contracts)} records are still in period of performance."
            ),
            answer="\n".join(
                [
                    f"{plural(len(active), 'active contract')} of "
                    f"{len(contracts)} total.",
                    "",
                    bullets(
                        [
                            contract_line(world, k)
                            for k in sorted(active, key=lambda k: -k.end_year)
                        ]
                    ),
                ]
            ),
            archetype="active_portfolio",
            layer="relational",
            meta={"n_active": len(active)},
        )
    )

    items.append(
        QRAItem(
            question="How much of our portfolio is prime versus sub?",
            reasoning=(
                f"Splitting {len(contracts)} records by our role and summing our share "
                f"of value on each side. Prime work matters disproportionately for "
                f"past performance credit, so both count and dollars are worth stating."
            ),
            answer="\n".join(
                [
                    f"Prime on {len(primes)} of {len(contracts)} contracts "
                    f"({100 * len(primes) // len(contracts)}%), "
                    f"{money(sum(k.value_ours for k in primes))} to us.",
                    f"Sub on {len(subs)} ({100 * len(subs) // len(contracts)}%), "
                    f"{money(sum(k.value_ours for k in subs))} to us.",
                    "",
                    f"Primes we have subbed to: "
                    f"{oxford([c.name for c in world.primes_we_have_subbed_to()][:10])}.",
                ]
            ),
            archetype="role_breakdown",
            layer="relational",
            meta={},
        )
    )

    items.append(
        QRAItem(
            question="What capabilities can we self-perform, and where do we have gaps?",
            reasoning=(
                "Comparing our capability set against the full taxonomy. Anything "
                "outside it has to be teamed or subbed on every pursuit, so it is "
                "the list that drives partner strategy."
            ),
            answer="\n".join(
                [
                    "Self-perform:",
                    bullets(
                        sorted(CAPABILITY_BY_ID[c].name for c in world.us.capabilities)
                    ),
                    "",
                    "Gaps - must be teamed or subbed:",
                    bullets(
                        sorted(
                            CAPABILITY_BY_ID[c].name
                            for c in CAPABILITY_BY_ID
                            if c not in world.our_capabilities()
                        )
                    ),
                ]
            ),
            archetype="capability_inventory",
            layer="relational",
            meta={},
        )
    )
    return items


# ---------------------------------------------------------------------------
# C. multi-hop
# ---------------------------------------------------------------------------


def build_multihop(world: World, rng: random.Random, count: int) -> list[QRAItem]:
    """Two-join questions. These are where a graph beats a keyword search, and
    where a model that merely memorized records will start to come apart."""
    items: list[QRAItem] = []
    cap_ids = list(CAPABILITY_BY_ID)
    agency_ids = sorted({k.agency_id for k in world.contracts.values()})

    # C1: capability AND agency experience.
    # Combinations are sampled, so they must be deduped: a repeated pair
    # produces a byte-identical question, and the train/eval split would then
    # put one copy on each side and quietly report leakage as generalization.
    seen_pairs: set[tuple[str, str]] = set()
    for _ in range(count):
        cap_id = rng.choice(cap_ids)
        agency_id = rng.choice(agency_ids)
        if (cap_id, agency_id) in seen_pairs:
            continue
        seen_pairs.add((cap_id, agency_id))
        spec = CAPABILITY_BY_ID[cap_id]
        agency = AGENCY_BY_ID[agency_id]
        matches = [
            c
            for c in world.companies_with_capability(cap_id)
            if agency_id in c.agencies_served
        ]
        answer = (
            "\n".join(
                [
                    f"{plural(len(matches), 'partner')} carry {spec.name} and have "
                    f"{agency.abbrev} past performance.",
                    "",
                    bullets([company_line(world, c) for c in matches[:12]]),
                ]
            )
            if matches
            else (
                f"No partner in the network combines {spec.name} with "
                f"{agency.abbrev} experience. Either relax one criterion, or plan to "
                f"carry the {agency.abbrev} relationship ourselves and team purely "
                f"on the technical capability."
            )
        )
        items.append(
            QRAItem(
                question=(
                    f"Which partners do {spec.name} and have {agency.abbrev} experience?"
                ),
                reasoning=(
                    f"Two filters. First, partners tagged with {spec.name}: "
                    f"{len(world.companies_with_capability(cap_id))}. Then intersecting "
                    f"with {agency.abbrev} past performance, which leaves {len(matches)}."
                ),
                answer=answer,
                archetype="capability_and_agency",
                layer="multihop",
                meta={"capability": cap_id, "agency": agency_id, "n": len(matches)},
            )
        )

    # C2: partners bridging two capabilities. Order-independent, so the key is
    # the sorted pair.
    seen_caps: set[tuple[str, str]] = set()
    for _ in range(count):
        cap_a, cap_b = rng.sample(cap_ids, 2)
        key = tuple(sorted((cap_a, cap_b)))
        if key in seen_caps:
            continue
        seen_caps.add(key)
        spec_a, spec_b = CAPABILITY_BY_ID[cap_a], CAPABILITY_BY_ID[cap_b]
        matches = [
            c
            for c in world.partners
            if cap_a in c.capabilities and cap_b in c.capabilities
        ]
        answer = (
            "\n".join(
                [
                    f"{plural(len(matches), 'partner')} cover both {spec_a.name} "
                    f"and {spec_b.name}.",
                    "",
                    bullets([company_line(world, c) for c in matches[:12]]),
                ]
            )
            if matches
            else (
                f"No single partner covers both {spec_a.name} and {spec_b.name}. "
                f"This combination needs two subs, which is worth flagging early - it "
                f"changes the workshare math and the number of teaming agreements."
            )
        )
        items.append(
            QRAItem(
                question=f"Who covers both {spec_a.name} and {spec_b.name}?",
                reasoning=(
                    f"Intersecting two capability tags. {spec_a.name} has "
                    f"{len(world.companies_with_capability(cap_a))} partners, "
                    f"{spec_b.name} has {len(world.companies_with_capability(cap_b))}; "
                    f"the intersection is {len(matches)}."
                ),
                answer=answer,
                archetype="capability_bridge",
                layer="multihop",
                meta={"cap_a": cap_a, "cap_b": cap_b, "n": len(matches)},
            )
        )

    # C3: known partners with a given agency's past performance
    for agency_id in agency_ids:
        agency = AGENCY_BY_ID[agency_id]
        known = [c for c in world.partners if c.contracts_with_us]
        matches = [c for c in known if agency_id in c.agencies_served]
        items.append(
            QRAItem(
                question=(
                    f"Of the partners we've already teamed with, who has "
                    f"{agency.abbrev} past performance?"
                ),
                reasoning=(
                    f"Starting from the {len(known)} partners with joint awards on our "
                    f"books, then filtering to {agency.abbrev} experience. "
                    f"{len(matches)} qualify. This is the warm list: known performers "
                    f"who also carry the customer relationship."
                ),
                answer=(
                    "\n".join(
                        [
                            f"{plural(len(matches), 'partner')} from our teaming history "
                            f"also have {agency.abbrev} past performance.",
                            "",
                            bullets(
                                [
                                    f"{c.name} - "
                                    f"{plural(len(c.contracts_with_us), 'joint award')} "
                                    f"with us, {agency.abbrev} experience on record"
                                    for c in matches[:12]
                                ]
                            ),
                        ]
                    )
                    if matches
                    else (
                        f"None of the partners we have teamed with carry "
                        f"{agency.abbrev} past performance. For a {agency.abbrev} "
                        f"pursuit we would be introducing a new partner, which adds "
                        f"teaming-agreement lead time."
                    )
                ),
                archetype="warm_list_by_agency",
                layer="multihop",
                meta={"agency": agency_id, "n": len(matches)},
            )
        )

    # C4: personnel spanning contracts
    multi = [
        p for p in world.people.values() if len(world.contracts_for_person(p.id)) >= 2
    ]
    for person in multi:
        contracts = world.contracts_for_person(person.id)
        agencies = sorted({k.agency for k in contracts})
        items.append(
            QRAItem(
                question=(
                    f"Who on staff has worked across multiple customers, and what "
                    f"does {person.name} bring specifically?"
                ),
                reasoning=(
                    f"{person.name} appears on {len(contracts)} contracts spanning "
                    f"{oxford(agencies)}. Staff who have crossed customer boundaries "
                    f"are the ones worth naming as key personnel, because their "
                    f"experience is citable on more than one pursuit."
                ),
                answer="\n".join(
                    [
                        f"{person.name}, {person.role} ({person.degree}, "
                        f"{person.years_experience} yrs) - {len(contracts)} assignments "
                        f"across {oxford(agencies)}.",
                        "",
                        bullets([contract_line(world, k) for k in contracts]),
                    ]
                ),
                archetype="personnel_span",
                layer="multihop",
                meta={"person": person.id, "n_contracts": len(contracts)},
            )
        )

    return items

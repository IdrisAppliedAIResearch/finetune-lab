"""Seeded generation of the synthetic contracting world, plus the query API.

Everything is derived from one ``random.Random(seed)``, so a given seed always
produces exactly the same library, questions, and golden answers. That matters
more than it sounds: if the corpus shifted between the training run and the
evaluation run, a drop in accuracy would be indistinguishable from a drop in the
data, and nobody could tell which.

The ``World`` query methods are the ground truth the model is being trained to
internalize. Golden answers are computed by calling them -- never by asking a
language model what it thinks the answer is.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .entities import CURRENT_YEAR, Company, Contract, Opportunity, Person
from .prose import (
    ACTIVITIES,
    DEGREES,
    OPPORTUNITY_FRAMING,
    OPPORTUNITY_SUFFIX,
    OPPORTUNITY_THEMES,
    OUTCOMES,
    PLACES,
    to_base,
)
from .taxonomy import (
    AGENCIES,
    AGENCY_BY_ID,
    CAPABILITIES,
    CAPABILITY_BY_ID,
    CPARS_RATINGS,
    CPARS_WEIGHTS,
    NAME_HEADS,
    NAME_TAILS,
    PERSON_FIRST,
    PERSON_LAST,
    PERSON_ROLES,
    REAL_FIRM_BLOCKLIST,
    SET_ASIDES,
    SOCIOECONOMIC,
    US_STATES,
    VEHICLE_BY_ID,
    VEHICLES,
)


@dataclass(frozen=True)
class Scale:
    contracts: int
    partners: int
    people: int
    opportunities: int


SCALES: dict[str, Scale] = {
    "compact": Scale(contracts=40, partners=80, people=24, opportunities=24),
    "demo": Scale(contracts=75, partners=150, people=40, opportunities=45),
    "full": Scale(contracts=150, partners=300, people=70, opportunities=90),
}

# The protagonist firm is fixed rather than generated: the corpus needs a stable
# with a deliberate shape: deep in epidemiology, evaluation, and program
# support; genuinely thin in health IT. That gap is what makes "who should we
# team with" a real question instead of a formality.
US_CAPABILITIES: tuple[str, ...] = (
    "epi_surveillance",
    "outbreak_response",
    "biostatistics",
    "immunization",
    "vaccine_safety",
    "maternal_child",
    "chronic_disease",
    "health_equity",
    "program_evaluation",
    "health_communications",
    "community_engagement",
    "workforce_training",
    "grants_management",
    "health_informatics",
)

# Named explicitly so questions can reason about them. These are the areas we
# must subcontract or team for.
US_GAPS: tuple[str, ...] = (
    "fhir_interop",
    "ehr_integration",
    "cloud_migration",
    "cybersecurity",
    "data_modernization",
    "lab_systems",
    "medical_countermeasures",
    "global_health",
    "clinical_quality",
    "environmental_health",
    "data_science",
    "registry_dev",
    "behavioral_health",
    "hiv_sti",
    "health_policy",
)

US_VEHICLES: tuple[str, ...] = ("gsa_mas", "ciosp3", "phsc_bpa", "cdc_iddiq", "stars3")

AGENCY_WEIGHTS: dict[str, int] = {
    "cdc": 26, "hrsa": 14, "cms": 11, "nih": 10, "aspr": 8, "samhsa": 7,
    "state_hd": 7, "ahrq": 4, "fda": 4, "ihs": 3, "va": 2, "dha": 2,
    "usaid": 1, "acf": 1,
}

# Which NAICS a capability family naturally maps to.
FAMILY_NAICS: dict[str, tuple[str, ...]] = {
    "epidemiology": ("541690", "541611", "923120"),
    "health_it": ("541512", "541511", "541519"),
    "analytics": ("541690", "541715", "541720"),
    "program": ("541611", "923120", "621999"),
    "preparedness": ("541611", "541690", "923120"),
    "services": ("541611", "541618", "611430", "541910"),
}

FAMILY_PSC: dict[str, tuple[str, ...]] = {
    "epidemiology": ("B505", "B506"),
    "health_it": ("D302", "D307", "D310"),
    "analytics": ("B505", "AN12"),
    "program": ("R408", "R499", "B506"),
    "preparedness": ("R408", "B506"),
    "services": ("R408", "R499", "U008"),
}


def _capabilities_by_family() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for spec in CAPABILITIES:
        grouped.setdefault(spec.family, []).append(spec.id)
    return grouped


CAPS_BY_FAMILY = _capabilities_by_family()


class World:
    """The generated universe, plus the queries golden answers are computed from."""

    def __init__(self, seed: int = 42, scale: str = "demo") -> None:
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {sorted(SCALES)}, got {scale!r}")
        self.seed = seed
        self.scale_name = scale
        self.scale = SCALES[scale]
        self.rng = random.Random(seed)

        self.us: Company = self._build_us()
        self.companies: dict[str, Company] = {self.us.id: self.us}
        self.people: dict[str, Person] = {}
        self.contracts: dict[str, Contract] = {}
        self.opportunities: dict[str, Opportunity] = {}

        self._build_partners()
        self._build_people()
        self._build_contracts()
        self._assert_unique_names()
        self._backfill_relationships()
        self._build_opportunities()

    # -- construction ----------------------------------------------------

    def _build_us(self) -> Company:
        return Company(
            id="us",
            name="Cardinal Ridge Public Health Group",
            size="small",
            socioeconomic=["8(a)", "SDB"],
            capabilities=list(US_CAPABILITIES),
            vehicles=list(US_VEHICLES),
            hq_state="MD",
            employees=310,
            founded=2009,
            is_us=True,
        )

    def _mint_company_name(self, used: set[str]) -> str:
        for _ in range(200):
            head = self.rng.choice(NAME_HEADS)
            tail = self.rng.choice(NAME_TAILS)
            name = f"{head} {tail}"
            if name in used:
                continue
            tokens = {t.lower().strip(",.") for t in name.split()}
            if tokens & REAL_FIRM_BLOCKLIST:
                continue
            return name
        raise RuntimeError("exhausted the company name pool; widen NAME_HEADS/NAME_TAILS")

    def _build_partners(self) -> None:
        used: set[str] = set()
        for index in range(self.scale.partners):
            name = self._mint_company_name(used)
            used.add(name)

            is_large = self.rng.random() < 0.32
            family = self.rng.choice(list(CAPS_BY_FAMILY))
            pool = CAPS_BY_FAMILY[family]

            n_core = self.rng.randint(2, min(5, len(pool)))
            caps = set(self.rng.sample(pool, n_core))
            # A spread of secondary capabilities from other families; large firms
            # are broader, which is what makes them plausible primes.
            others = [c for c in CAPABILITY_BY_ID if c not in caps]
            caps.update(self.rng.sample(others, self.rng.randint(2, 6 if is_large else 3)))

            socio: list[str] = []
            if not is_large and self.rng.random() < 0.62:
                socio = self.rng.sample(SOCIOECONOMIC, self.rng.randint(1, 2))

            vehicles = self._pick_vehicles(is_large, socio)

            company = Company(
                id=f"c{index:03d}",
                name=name,
                size="large" if is_large else "small",
                socioeconomic=socio,
                capabilities=sorted(caps),
                vehicles=vehicles,
                hq_state=self.rng.choice(US_STATES),
                employees=self.rng.randint(2000, 40000)
                if is_large
                else self.rng.randint(15, 480),
                founded=self.rng.randint(1985, 2019),
                # Market intelligence: agencies they are known to have served,
                # independent of whether we have ever teamed with them.
                agencies_served=sorted(
                    self.rng.sample(
                        [a.id for a in AGENCIES], self.rng.randint(1, 5 if is_large else 3)
                    )
                ),
            )
            self.companies[company.id] = company

    def _pick_vehicles(self, is_large: bool, socio: list[str]) -> list[str]:
        eligible = [
            v.id
            for v in VEHICLES
            if v.id != "open" and not (v.id == "stars3" and "8(a)" not in socio)
        ]
        count = self.rng.randint(2, 5) if is_large else self.rng.randint(0, 3)
        return sorted(self.rng.sample(eligible, min(count, len(eligible))))

    def _build_people(self) -> None:
        for index in range(self.scale.people):
            first = self.rng.choice(PERSON_FIRST)
            last = self.rng.choice(PERSON_LAST)
            person = Person(
                id=f"p{index:03d}",
                name=f"{first} {last}",
                role=self.rng.choice(PERSON_ROLES),
                capabilities=self.rng.sample(
                    list(US_CAPABILITIES), self.rng.randint(2, 4)
                ),
                years_experience=self.rng.randint(6, 28),
                degree=self.rng.choice(DEGREES),
            )
            self.people[person.id] = person

    def _weighted_agency(self) -> str:
        ids = list(AGENCY_WEIGHTS)
        return self.rng.choices(ids, weights=[AGENCY_WEIGHTS[i] for i in ids], k=1)[0]

    def _build_contracts(self) -> None:
        partner_ids = [c for c in self.companies if c != "us"]
        self._used_contract_names: set[str] = set()

        for index in range(self.scale.contracts):
            agency_id = self._weighted_agency()
            agency = AGENCY_BY_ID[agency_id]
            subunit = self.rng.choice(agency.subunits)

            # Capabilities skew toward what the agency buys, intersected with
            # what we can actually deliver.
            favored = [c for c in agency.favors if c in US_CAPABILITIES]
            caps = set(self.rng.sample(favored, min(len(favored), self.rng.randint(1, 2))))
            caps.update(
                self.rng.sample(list(US_CAPABILITIES), self.rng.randint(1, 3))
            )
            caps = sorted(caps)

            family = CAPABILITY_BY_ID[caps[0]].family
            naics = self.rng.choice(FAMILY_NAICS[family])
            psc = self.rng.choice(FAMILY_PSC[family])

            our_role = "prime" if self.rng.random() < 0.55 else "sub"
            value_total = int(self.rng.lognormvariate(15.2, 1.05))
            value_total = max(400_000, min(value_total, 90_000_000))

            sub_performed: list[str] = []
            if our_role == "prime":
                prime_id = "us"
                n_subs = self.rng.randint(1, 3)
                sub_ids = self.rng.sample(partner_ids, n_subs)
                value_ours = int(value_total * self.rng.uniform(0.55, 0.85))

                # As prime we carry the whole scope, including work the subs
                # performed. Without this the library can only ever contain
                # capabilities we self-perform, so no past performance could
                # ever match a requirement that includes one of our gaps --
                # which is exactly the requirement we most need to cite against.
                bench = set()
                for sub_id in sub_ids:
                    bench |= set(self.companies[sub_id].capabilities)
                brought = sorted(bench - set(US_CAPABILITIES))
                if brought:
                    sub_performed = self.rng.sample(
                        brought, min(len(brought), self.rng.randint(0, 2))
                    )
                    caps = sorted(set(caps) | set(sub_performed))
            else:
                # We sub to someone large enough to plausibly hold the prime slot.
                big = [c for c in partner_ids if self.companies[c].size == "large"]
                prime_id = self.rng.choice(big or partner_ids)
                sub_ids = self.rng.sample(
                    [c for c in partner_ids if c != prime_id], self.rng.randint(0, 2)
                )
                value_ours = int(value_total * self.rng.uniform(0.08, 0.35))

            start_year = self.rng.randint(2016, 2024)
            end_year = min(start_year + self.rng.randint(1, 5), 2029)

            vehicle_id = self._pick_contract_vehicle(naics, our_role, prime_id)
            set_aside = self._pick_set_aside(our_role, prime_id)

            activity_cap = self.rng.choice(caps)
            activity = self.rng.choice(ACTIVITIES[activity_cap]).format(
                n=self.rng.choice([3, 5, 8, 12, 18, 24, 30, 42, 50, 64])
            )
            framing = "As prime contractor" if our_role == "prime" else "As subcontractor"
            scope = (
                f"{framing}, {activity} for {agency.abbrev} {subunit}. "
                f"Work spanned {', '.join(CAPABILITY_BY_ID[c].name for c in caps)}."
            )

            contract = Contract(
                id=f"k{index:03d}",
                number=self._contract_number(agency_id, start_year, index),
                name=self._contract_name(agency, subunit, caps, start_year),
                agency_id=agency_id,
                subunit=subunit,
                our_role=our_role,
                prime_id=prime_id,
                sub_ids=sub_ids,
                value_total=value_total,
                value_ours=value_ours,
                start_year=start_year,
                end_year=end_year,
                vehicle_id=vehicle_id,
                naics=naics,
                psc=psc,
                set_aside=set_aside,
                capabilities=caps,
                sub_performed=sub_performed,
                cpars=self.rng.choices(CPARS_RATINGS, weights=CPARS_WEIGHTS, k=1)[0],
                personnel_ids=self.rng.sample(
                    list(self.people), min(len(self.people), self.rng.randint(1, 3))
                ),
                place=self.rng.choice(PLACES),
                scope=scope,
                outcomes=self._outcomes(),
            )
            self.contracts[contract.id] = contract

    def _pick_contract_vehicle(self, naics: str, our_role: str, prime_id: str) -> str:
        holder = self.us if our_role == "prime" else self.companies[prime_id]
        matching = [
            v
            for v in holder.vehicles
            if not VEHICLE_BY_ID[v].naics or naics in VEHICLE_BY_ID[v].naics
        ]
        if matching:
            return self.rng.choice(matching)
        return self.rng.choice(holder.vehicles) if holder.vehicles else "open"

    def _pick_set_aside(self, our_role: str, prime_id: str) -> str:
        holder = self.us if our_role == "prime" else self.companies[prime_id]
        if holder.size == "large":
            return "Full & Open"
        if holder.socioeconomic and self.rng.random() < 0.6:
            choice = self.rng.choice(holder.socioeconomic)
            return choice if choice in SET_ASIDES else "Small Business"
        return "Small Business"

    def _contract_number(self, agency_id: str, year: int, index: int) -> str:
        prefix = {
            "cdc": "75D301", "hrsa": "75R602", "cms": "75FCMC", "nih": "75N931",
            "aspr": "75A501", "samhsa": "75S203", "ahrq": "75Q801", "fda": "75F401",
            "ihs": "75H701", "va": "36C241", "dha": "HT001", "usaid": "7200AA",
            "state_hd": "SDH", "acf": "75ACF1",
        }[agency_id]
        return f"{prefix}{str(year)[2:]}{'C' if index % 2 == 0 else 'D'}{index:05d}"

    def _contract_name(self, agency, subunit: str, caps: list[str], start_year: int) -> str:
        """Build a name that is unique across the library.

        Uniqueness is not cosmetic. Half the recall questions address a contract
        by name ("walk me through the X engagement"), so two contracts sharing a
        name produce one question with two different correct answers -- a
        contradiction the model is then trained on, and one that would be
        invisible in the loss curve.
        """
        lead = CAPABILITY_BY_ID[caps[0]].name
        # Subunits are written as "NCIRD (Immunization & Respiratory Diseases)"
        # where an acronym exists, and plainly otherwise. Take the acronym when
        # there is one; splitting on whitespace unconditionally would render
        # "Office of Information Technology" as the useless token "Office".
        code = subunit.split(" (")[0] if " (" in subunit else subunit
        suffix = self.rng.choice(
            ("Support Services", "Technical Support", "Program Support", "Services", "Support")
        )
        base = f"{agency.abbrev} {code} {lead} {suffix}"

        name = base
        attempt = 1
        while name in self._used_contract_names:
            # Fiscal year first, since that is how a real portfolio distinguishes
            # recompetes of the same scope; a counter only if that still collides.
            name = f"{base} (FY{str(start_year)[2:]})"
            if attempt > 1:
                name = f"{base} (FY{str(start_year)[2:]} #{attempt})"
            attempt += 1
        self._used_contract_names.add(name)
        return name

    def _outcomes(self) -> list[str]:
        picks = self.rng.sample(OUTCOMES, self.rng.randint(1, 3))
        rendered = []
        for template in picks:
            a = self.rng.choice([8, 12, 15, 20, 25, 30, 40, 55, 72, 85, 90])
            b = self.rng.choice([2, 3, 5, 45, 60, 95, 98, 110, 150])
            rendered.append(template.format(a=a, b=b))
        return rendered

    def _assert_unique_names(self) -> None:
        """Names and numbers must identify exactly one contract.

        Both are used as question keys, so a collision turns into contradictory
        training targets rather than a visible error.
        """
        for field in ("name", "number"):
            values = [getattr(k, field) for k in self.contracts.values()]
            if len(set(values)) != len(values):
                dupes = {v for v in values if values.count(v) > 1}
                raise AssertionError(
                    f"contract {field} collision: {sorted(dupes)[:3]}"
                )

    def _backfill_relationships(self) -> None:
        for contract in self.contracts.values():
            participants = set(contract.sub_ids)
            if contract.our_role == "sub":
                participants.add(contract.prime_id)
            for company_id in participants:
                company = self.companies.get(company_id)
                if company is None or company.is_us:
                    continue
                company.contracts_with_us.append(contract.id)
                if contract.agency_id not in company.agencies_served:
                    company.agencies_served.append(contract.agency_id)

        for company in self.companies.values():
            company.agencies_served = sorted(set(company.agencies_served))

    def _build_opportunities(self) -> None:
        """Generate pursuits, rejecting any that lack a relevance spectrum.

        An opportunity with no hard negatives still trains ranking, but it
        teaches nothing about discrimination -- and discrimination is the entire
        question the demo is asking. At smaller scales a sampled opportunity can
        easily land in a corner of the graph where every plausible partner
        happens to cover the gap, so candidates are scored and resampled until
        the spectrum is actually present.
        """
        # Imported here rather than at module scope: scoring imports World, so a
        # top-level import would close the cycle.
        from .scoring import hard_negatives, rank_partners

        for index in range(self.scale.opportunities):
            best = None
            best_traps = -1
            for _ in range(12):
                candidate = self._draft_opportunity(index)
                ranked = rank_partners(self, candidate, "teaming")
                traps = hard_negatives(ranked)
                tiers = {a.tier for a in ranked}

                if len(traps) >= 2 and {0, 2} <= tiers:
                    best = candidate
                    break
                # Otherwise keep whichever draft came closest, so a hostile
                # corner of the graph degrades the spectrum rather than the run.
                if len(traps) > best_traps:
                    best, best_traps = candidate, len(traps)

            self.opportunities[best.id] = best

    def _draft_opportunity(self, index: int) -> Opportunity:
        agency_id = self._weighted_agency()
        agency = AGENCY_BY_ID[agency_id]
        subunit = self.rng.choice(agency.subunits)

        # Every opportunity must contain at least one capability we lack --
        # otherwise there is nothing to team for and the question is hollow.
        # Two or three, never one. With a single-capability gap, coverage is
        # binary and dozens of partners tie at the top, which makes the
        # ranking arbitrary and teaches the model that it is.
        gap_pick = self.rng.sample(list(US_GAPS), self.rng.randint(2, 3))

        # The self-perform half of the requirement skews toward what this agency
        # actually buys -- the same bias contract generation already uses.
        # Drawing it uniformly left opportunities systematically uncorrelated
        # with our own portfolio, which capped how well any past performance
        # could match and made citation questions a choice among uniformly weak
        # options.
        favored_ours = [c for c in agency.favors if c in US_CAPABILITIES]
        ours_pick: set[str] = set()
        if favored_ours:
            ours_pick.update(
                self.rng.sample(
                    favored_ours, min(len(favored_ours), self.rng.randint(1, 2))
                )
            )
        ours_pick.update(self.rng.sample(list(US_CAPABILITIES), self.rng.randint(1, 2)))
        required = sorted(set(gap_pick) | ours_pick)

        family = CAPABILITY_BY_ID[gap_pick[0]].family
        naics = self.rng.choice(FAMILY_NAICS[family])
        psc = self.rng.choice(FAMILY_PSC[family])
        theme = self.rng.choice(OPPORTUNITY_THEMES[family])
        framing = self.rng.choice(OPPORTUNITY_FRAMING[family])

        activity = to_base(
            self.rng.choice(ACTIVITIES[gap_pick[0]]).format(
                n=self.rng.choice([4, 6, 10, 15, 22, 28, 36, 48])
            )
        )

        opportunity = Opportunity(
            id=f"o{index:03d}",
            name=f"{agency.abbrev} {theme} {self.rng.choice(OPPORTUNITY_SUFFIX)}",
            agency_id=agency_id,
            subunit=subunit,
            naics=naics,
            psc=psc,
            value=int(self.rng.lognormvariate(15.6, 0.9)),
            duration_years=self.rng.randint(1, 5),
            vehicle_id=self.rng.choice(
                [v.id for v in VEHICLES if not v.naics or naics in v.naics] or ["open"]
            ),
            set_aside=self.rng.choice(SET_ASIDES),
            required_capabilities=required,
            place=self.rng.choice(PLACES),
            scope=f"{framing}. The contractor will {activity}.",
            due=f"{self.rng.choice(['Q1', 'Q2', 'Q3', 'Q4'])} FY{self.rng.choice([26, 27])}",
        )
        return opportunity

    # -- query API -------------------------------------------------------

    @property
    def partners(self) -> list[Company]:
        return [c for c in self.companies.values() if not c.is_us]

    def our_capabilities(self) -> set[str]:
        return set(self.us.capabilities)

    def capability_gap(self, opportunity: Opportunity) -> list[str]:
        """Required capabilities we cannot self-perform. The decisive dimension."""
        return [c for c in opportunity.required_capabilities if c not in self.our_capabilities()]

    def covered_requirements(self, opportunity: Opportunity) -> list[str]:
        return [c for c in opportunity.required_capabilities if c in self.our_capabilities()]

    def contracts_by_agency(self, agency_id: str) -> list[Contract]:
        return [k for k in self.contracts.values() if k.agency_id == agency_id]

    def contracts_by_capability(self, capability_id: str) -> list[Contract]:
        return [k for k in self.contracts.values() if capability_id in k.capabilities]

    def contracts_with_company(self, company_id: str) -> list[Contract]:
        return [
            k
            for k in self.contracts.values()
            if company_id in k.sub_ids or (k.our_role == "sub" and k.prime_id == company_id)
        ]

    def contracts_by_vehicle(self, vehicle_id: str) -> list[Contract]:
        return [k for k in self.contracts.values() if k.vehicle_id == vehicle_id]

    def contracts_for_person(self, person_id: str) -> list[Contract]:
        return [k for k in self.contracts.values() if person_id in k.personnel_ids]

    def active_contracts(self) -> list[Contract]:
        return [k for k in self.contracts.values() if k.end_year >= CURRENT_YEAR]

    def companies_with_capability(self, capability_id: str) -> list[Company]:
        return [c for c in self.partners if capability_id in c.capabilities]

    def companies_with_vehicle(self, vehicle_id: str) -> list[Company]:
        return [c for c in self.partners if vehicle_id in c.vehicles]

    def collaboration_count(self, company_id: str) -> int:
        return len(self.companies[company_id].contracts_with_us)

    def last_collaboration_year(self, company_id: str) -> int | None:
        years = [
            self.contracts[k].end_year
            for k in self.companies[company_id].contracts_with_us
        ]
        return max(years) if years else None

    def collaboration_cpars(self, company_id: str) -> list[str]:
        return [
            self.contracts[k].cpars for k in self.companies[company_id].contracts_with_us
        ]

    def primes_we_have_subbed_to(self) -> list[Company]:
        ids = {k.prime_id for k in self.contracts.values() if k.our_role == "sub"}
        return [self.companies[i] for i in sorted(ids) if i in self.companies]

    def subs_we_have_used(self) -> list[Company]:
        ids: set[str] = set()
        for contract in self.contracts.values():
            if contract.our_role == "prime":
                ids.update(contract.sub_ids)
        return [self.companies[i] for i in sorted(ids)]

    def agency_of(self, agency_id: str):
        return AGENCY_BY_ID[agency_id]

    def stats(self) -> dict[str, int]:
        return {
            "contracts": len(self.contracts),
            "partners": len(self.partners),
            "people": len(self.people),
            "opportunities": len(self.opportunities),
            "prime_contracts": sum(1 for k in self.contracts.values() if k.our_role == "prime"),
            "sub_contracts": sum(1 for k in self.contracts.values() if k.our_role == "sub"),
            "active_contracts": len(self.active_contracts()),
            "agencies_touched": len({k.agency_id for k in self.contracts.values()}),
        }

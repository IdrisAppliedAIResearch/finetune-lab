"""Entity records for the synthetic past performance world.

Plain dataclasses rather than pydantic models: these are constructed by the
generator in tight loops and never parsed from untrusted input, so validation
would only cost time. They serialize to JSON for the library dump that ships
alongside the training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .taxonomy import AGENCY_BY_ID, CAPABILITY_BY_ID, NAICS, VEHICLE_BY_ID


@dataclass
class Person:
    id: str
    name: str
    role: str
    capabilities: list[str]
    years_experience: int
    degree: str

    def describe(self) -> str:
        return f"{self.name}, {self.role} ({self.degree}, {self.years_experience} yrs)"


@dataclass
class Company:
    id: str
    name: str
    size: str  # "small" | "large"
    socioeconomic: list[str]
    capabilities: list[str]
    vehicles: list[str]
    hq_state: str
    employees: int
    founded: int
    is_us: bool = False

    # Filled in by the graph once contracts exist, so partner profiles can be
    # answered without re-deriving the joins on every question.
    agencies_served: list[str] = field(default_factory=list)
    contracts_with_us: list[str] = field(default_factory=list)

    @property
    def size_label(self) -> str:
        socio = ", ".join(self.socioeconomic)
        base = "Small Business" if self.size == "small" else "Large Business"
        return f"{base} ({socio})" if socio else base

    def capability_names(self) -> list[str]:
        return [CAPABILITY_BY_ID[c].name for c in self.capabilities]

    def vehicle_names(self) -> list[str]:
        return [VEHICLE_BY_ID[v].name for v in self.vehicles]


@dataclass
class Contract:
    id: str
    number: str
    name: str
    agency_id: str
    subunit: str
    our_role: str  # "prime" | "sub"
    prime_id: str
    sub_ids: list[str]
    value_total: int
    value_ours: int
    start_year: int
    end_year: int
    vehicle_id: str
    naics: str
    psc: str
    set_aside: str
    capabilities: list[str]
    # Scope the contract covered that a subcontractor performed, not us. The
    # contract is still citable for it -- past performance is cited at the
    # contract level -- but a technical evaluator discounts scope you did not
    # perform, and a capture lead needs to know which is which.
    sub_performed: list[str]
    cpars: str
    personnel_ids: list[str]
    place: str
    scope: str
    outcomes: list[str]

    @property
    def agency(self) -> str:
        return AGENCY_BY_ID[self.agency_id].abbrev

    @property
    def agency_full(self) -> str:
        return AGENCY_BY_ID[self.agency_id].name

    @property
    def vehicle(self) -> str:
        return VEHICLE_BY_ID[self.vehicle_id].name

    @property
    def naics_title(self) -> str:
        return NAICS[self.naics]

    @property
    def period(self) -> str:
        return f"{self.start_year}-{self.end_year}"

    @property
    def is_active(self) -> bool:
        return self.end_year >= CURRENT_YEAR

    def capability_names(self) -> list[str]:
        return [CAPABILITY_BY_ID[c].name for c in self.capabilities]

    @property
    def self_performed(self) -> list[str]:
        return [c for c in self.capabilities if c not in self.sub_performed]

    def money(self) -> str:
        return f"${self.value_total / 1_000_000:.1f}M"

    def money_ours(self) -> str:
        return f"${self.value_ours / 1_000_000:.1f}M"


@dataclass
class Opportunity:
    """A live pursuit. In closed-book training this is the only thing supplied
    in the prompt -- everything used to answer it must come from the weights."""

    id: str
    name: str
    agency_id: str
    subunit: str
    naics: str
    psc: str
    value: int
    duration_years: int
    vehicle_id: str
    set_aside: str
    required_capabilities: list[str]
    place: str
    scope: str
    due: str

    @property
    def agency(self) -> str:
        return AGENCY_BY_ID[self.agency_id].abbrev

    @property
    def vehicle(self) -> str:
        return VEHICLE_BY_ID[self.vehicle_id].name

    def capability_names(self) -> list[str]:
        return [CAPABILITY_BY_ID[c].name for c in self.required_capabilities]

    def money(self) -> str:
        return f"${self.value / 1_000_000:.1f}M"

    def brief(self) -> str:
        """The block that gets pasted into a question prompt."""
        lines = [
            f"Opportunity: {self.name}",
            f"Agency: {AGENCY_BY_ID[self.agency_id].abbrev} / {self.subunit}",
            f"Vehicle: {self.vehicle}",
            f"NAICS: {self.naics} ({NAICS[self.naics]})",
            f"Set-aside: {self.set_aside}",
            f"Estimated value: {self.money()} over {self.duration_years} years",
            f"Place of performance: {self.place}",
            f"Required capabilities: {', '.join(self.capability_names())}",
            f"Scope: {self.scope}",
        ]
        return "\n".join(lines)


CURRENT_YEAR = 2026


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)

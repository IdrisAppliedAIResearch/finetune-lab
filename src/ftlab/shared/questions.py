"""The shared vocabulary: a question, an agency label, and the relevance tiers.

This file used to be 853 lines, most of it generators: thirteen archetypes that
expanded a graph into a couple of thousand templated training rows, plus the
blind-set generator. All of that is gone, and the reasons are worth keeping.

The generated corpus produced every recurring problem the project had. Its
answers were the output of a scoring function, so an archetype predicted its own
answer format perfectly and the first fine-tune collapsed into seven templates.
Its blind slates labelled every distractor tier 0 by fiat -- recomputing with
``tier_for`` below shows only ~22% of those labels were right. And because every
gold answer it could produce was an observed relationship in the training graph,
it taught one rule: *the answer is a firm this prime has already hired*. On the
held-out questions where no correct answer was a prior partner, the model that
learned that rule scored 0.091 against a 0.369 random floor -- worse than the
untuned base model, and worse than guessing.

What replaces it is hand-written: a curated set of relationships, examples
written one at a time over them, and a test set written the same way. Small
enough to read, which is the property the generated corpus never had.

What survives here is the vocabulary those hand-written files share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .graph import TeamingGraph

# How many picks an answer is graded on, and how many candidates a slate offers.
TOP_K = 5
CANDIDATES = 12


@dataclass
class Question:
    """One business question, its answer, and the evidence behind it."""

    question: str
    answer: str
    reasoning: str
    archetype: str
    # Companies the answer names, for entity-level grading.
    gold: list[str] = field(default_factory=list)
    # Candidate -> tier, for ranking questions. Empty for factual ones.
    tiers: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "question": self.question.strip(),
            "reasoning": self.reasoning.strip(),
            "answer": self.answer.strip(),
            "meta": {
                "archetype": self.archetype,
                "gold": self.gold,
                "tiers": self.tiers,
                **self.meta,
            },
        }


def _agency_short(agency: str) -> str:
    return {
        "Centers for Disease Control and Prevention": "CDC",
        "National Institutes of Health": "NIH",
        "Centers for Medicare and Medicaid Services": "CMS",
        "Food and Drug Administration": "FDA",
        "Health Resources and Services Administration": "HRSA",
        "Office of Assistant Secretary for Preparedness and Response": "ASPR",
        "Indian Health Service": "IHS",
        "Substance Abuse and Mental Health Services Administration": "SAMHSA",
        "Agency for Healthcare Research and Quality": "AHRQ",
    }.get(agency, agency)


def tier_for(
    graph: TeamingGraph,
    candidate: str,
    prime: str,
    agency: str,
    naics: set[str],
) -> int:
    """Where a candidate sits on the relevance spectrum, from observed records.

        tier 4  actually subbed for this prime at this agency
        tier 3  actually subbed for this prime, different agency
        tier 2  subbed at this agency for someone else, overlapping work
        tier 1  hard negative -- shares the coarse signals (NAICS, HHS scale)
                and has none of the relationships
        tier 0  no meaningful overlap

    Tier 1 is the discrimination worth testing. NAICS is close to useless on its
    own here -- 541690 "Other Scientific and Technical Consulting" holds both
    Apache targeting sights and CDC surveillance work -- so a tier-1 company
    looks correct on every structured field and wrong on every relationship.

    One limit bounds every number built on this: subaward reporting is
    incomplete, so a company absent from an award's sub list may still have
    worked it. Tier 0 and 1 mean "no reported relationship", not "no
    relationship".
    """
    company = graph.companies.get(candidate)
    if company is None:
        return 0

    with_prime = [r for r in company.as_sub if r["prime"] == prime]
    if any(r["agency"] == agency for r in with_prime):
        return 4
    if with_prime:
        return 3
    if any(r["agency"] == agency for r in company.as_sub):
        return 2

    scale = len(company.as_sub) + len(company.as_prime) + len(company.prime_awards)
    if naics & set(company.naics) and scale >= 3:
        return 1
    return 0


__all__ = ["CANDIDATES", "TOP_K", "Question", "_agency_short", "tier_for"]

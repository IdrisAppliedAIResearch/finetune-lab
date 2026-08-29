"""Business questions with a real-outcome answer key.

Every golden answer here is a fact about what companies actually did, taken from
FSRS subcontract reports. That is the property the synthetic corpus lacked: its
answers were the output of a scoring function, so any implementation of that
function scored 1.000 and the model could at best tie. Here the label belongs to
neither arm, so a rule engine, a fine-tuned model, and a base model with
retrieval can each be wrong, and the comparison means something.

The relevance spectrum is likewise observed rather than assigned:

    tier 4  actually subbed for this prime at this agency
    tier 3  actually subbed for this prime, different agency
    tier 2  subbed at this agency for someone else, overlapping work
    tier 1  hard negative -- shares the coarse signals (NAICS, HHS scale) and
            has none of the relationships
    tier 0  no meaningful overlap

Tier 1 is the whole experiment. A rule engine ranks on the structured fields it
can parse, and NAICS is close to useless here: 541690 "Other Scientific and
Technical Consulting" holds both Apache targeting sights and CDC surveillance
work. A tier-1 company looks correct on every field and wrong on every
relationship, which is the discrimination the demo claims a model can make.

One limit stated plainly, because it bounds every number below: subaward
reporting is incomplete, so a company absent from an award's sub list may
still have worked it. These labels have false negatives. Precision is
trustworthy, recall is a lower bound, and tier 0/1 means "no reported
relationship" rather than "no relationship".
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Any

from .graph import MEANINGFUL_DESCRIPTION, Company, TeamingGraph

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


def _fmt(names: list[str]) -> str:
    return "\n".join(f"{i}. {n}" for i, n in enumerate(names, start=1))


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


# ---------------------------------------------------------------------------
# relevance tiers
# ---------------------------------------------------------------------------


def tier_for(
    graph: TeamingGraph,
    candidate: str,
    prime: str,
    agency: str,
    naics: set[str],
) -> int:
    """Where a candidate sits on the spectrum, from observed relationships."""
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

    # Everything below has no relationship to this prime or agency. It is a hard
    # negative only if it still *looks* right: real HHS scale and matching
    # industry codes, which is exactly what a structured-field ranker sees.
    scale = len(company.as_sub) + len(company.as_prime) + len(company.prime_awards)
    if naics & set(company.naics) and scale >= 3:
        return 1
    return 0


def build_candidates(
    graph: TeamingGraph,
    prime: str,
    agency: str,
    naics: set[str],
    rng: random.Random,
    exclude: set[str] | None = None,
    size: int = CANDIDATES,
) -> dict[str, int]:
    """A slate spanning the spectrum, with hard negatives guaranteed present.

    Sampled per tier rather than taken from a ranking, because a slate drawn by
    the same signal the answer turns on contains no traps -- that mistake cost
    the synthetic corpus 76% of its hard negatives before it was caught.
    """
    exclude = (exclude or set()) | {prime}
    pool = collections.defaultdict(list)
    for name, company in graph.companies.items():
        if name in exclude or not (company.as_sub or company.as_prime):
            continue
        pool[tier_for(graph, name, prime, agency, naics)].append(name)

    for names in pool.values():
        names.sort()
        rng.shuffle(names)

    # Deliberate mix: enough true positives to rank, enough tier-1 traps that
    # rejecting them is measurable, and filler below.
    wanted = [(4, 4), (3, 2), (2, 2), (1, 3), (0, 1)]
    slate: dict[str, int] = {}
    for tier, count in wanted:
        for name in pool.get(tier, [])[:count]:
            slate[name] = tier
    for tier in (2, 1, 0):
        for name in pool.get(tier, []):
            if len(slate) >= size:
                break
            slate.setdefault(name, tier)
    return slate


# ---------------------------------------------------------------------------
# archetypes
# ---------------------------------------------------------------------------


def q_team_composition(graph: TeamingGraph, prime: str, agency: str) -> Question | None:
    """Who does this prime actually put on its teams at this customer?"""
    team = sorted(
        {r["sub"] for r in graph.train_subawards if r["prime"] == prime and r["agency"] == agency}
    )
    if len(team) < 3:
        return None
    short = _agency_short(agency)
    repeats = collections.Counter(
        r["sub"] for r in graph.train_subawards if r["prime"] == prime and r["agency"] == agency
    )
    named = [n for n, _ in repeats.most_common(TOP_K)]
    brought_back = [n for n, c in repeats.items() if c > 1]

    reasoning = (
        f"{prime} reports {len(team)} distinct subcontractors on {short} work. "
        f"Ranking by how often they appear, since a partner used once may be a "
        f"one-off and a partner used repeatedly is a standing relationship."
        + (
            f" {len(brought_back)} of them were brought back on more than one award."
            if brought_back
            else " None were used more than once, so none of this is a standing team."
        )
    )
    answer = (
        f"{prime}'s {short} team, most-used first:\n{_fmt(named)}\n\n"
        + (
            f"Brought back more than once: {', '.join(sorted(brought_back)[:5])}."
            if brought_back
            else "No repeat pairings on record -- this is not a settled team."
        )
    )
    return Question(
        question=f"Who does {prime} usually put on its {short} teams?",
        answer=answer,
        reasoning=reasoning,
        archetype="team_composition",
        gold=named,
        meta={"prime": prime, "agency": agency, "short": short},
    )


def q_sub_candidates(
    graph: TeamingGraph, prime: str, agency: str, naics: set[str], rng: random.Random
) -> Question | None:
    """Rank a mixed slate for a real requirement. The core ranking question."""
    slate = build_candidates(graph, prime, agency, naics, rng)
    if sum(1 for t in slate.values() if t == 4) < 2 or not any(t == 1 for t in slate.values()):
        return None
    short = _agency_short(agency)

    def evidence(name: str) -> tuple[int, int, int, str]:
        """Tier first, then how much work actually backs it.

        Within a tier the candidates are genuinely comparable, so breaking ties
        alphabetically produced answers that ranked one prior award above three.
        """
        company = graph.companies[name]
        with_prime = sum(1 for r in company.as_sub if r["prime"] == prime)
        at_agency = sum(1 for r in company.as_sub if r["agency"] == agency)
        return (-slate[name], -with_prime, -at_agency, name)

    ranked = sorted(slate, key=evidence)
    picks = ranked[:4]
    traps = [n for n, t in slate.items() if t == 1][:3]

    lines = []
    for name in picks:
        company = graph.companies[name]
        with_prime = [r for r in company.as_sub if r["prime"] == prime]
        at_agency = [r for r in company.as_sub if r["agency"] == agency]
        why = []
        if with_prime:
            why.append(f"{len(with_prime)} prior award(s) under {prime}")
        if at_agency:
            why.append(f"{len(at_agency)} {short} subcontract(s)")
        lines.append(f"- {name}: " + ("; ".join(why) or "adjacent HHS work only"))

    reasoning = (
        f"The question is who {prime} can actually put on a {short} bid, so the "
        f"test is relationships on record, not industry codes. NAICS is close to "
        f"useless here -- the same code covers unrelated work -- so the ranking "
        f"is: worked with {prime} at {short} first, then worked with {prime} "
        f"elsewhere, then {short} experience under another prime.\n\n"
        + "\n".join(lines)
        + (
            f"\n\nWorth rejecting: {', '.join(traps)}. Each matches on industry "
            f"code and has real HHS volume, which is what makes them look right, "
            f"and none has a reported subcontract with {prime} or at {short}."
            if traps
            else ""
        )
    )
    answer = (
        f"Sub candidates for {prime} on {short} work:\n{_fmt(picks)}\n\n"
        f"Not recommended, despite looking like obvious picks:\n"
        + "\n".join(f"- {t}: no reported {short} or {prime} relationship" for t in traps)
    )
    return Question(
        question=(
            f"We're teaming with {prime} on {short} work. Who should we put "
            f"forward as subcontractors?"
        ),
        answer=answer,
        reasoning=reasoning,
        archetype="sub_candidates",
        gold=picks,
        tiers=slate,
        meta={"prime": prime, "agency": agency, "short": short, "naics": sorted(naics)},
    )


def q_prime_candidates(graph: TeamingGraph, agency: str, naics: str) -> Question | None:
    """Which primes actually sub out work of this kind at this customer?"""
    rows = [r for r in graph.train_subawards if r["agency"] == agency and r["naics"] == naics]
    if len(rows) < 6:
        return None
    counts = collections.Counter(r["prime"] for r in rows)
    named = [n for n, _ in counts.most_common(TOP_K)]
    short = _agency_short(agency)
    title = next((r["naics_title"] for r in rows if r["naics_title"]), naics).title()

    reasoning = (
        f"Approaching a prime only pays if they actually subcontract this work. "
        f"Across {short} awards in NAICS {naics} ({title}) there are "
        f"{len(rows)} reported subcontracts from {len(counts)} primes; ranking "
        f"by how many subs each one has taken on is a direct measure of whether "
        f"they team at all."
    )
    answer = (
        f"Primes to approach for {short} {title} work, by subcontracting volume:\n"
        f"{_fmt([f'{n} ({counts[n]} reported subcontracts)' for n in named])}"
    )
    return Question(
        question=(
            f"We want to sub on {short} {title.lower()} work. "
            f"Which primes should we approach?"
        ),
        answer=answer,
        reasoning=reasoning,
        archetype="prime_candidates",
        gold=named,
        meta={"agency": agency, "naics": naics, "short": short, "title": title},
    )


def q_prior_relationship(
    graph: TeamingGraph, a: str, b: str, related: bool
) -> Question | None:
    """A yes/no the graph settles. Cheap to ask, and it catches confabulation."""
    company = graph.companies.get(a)
    if company is None:
        return None
    shared = [r for r in company.as_prime if r["sub"] == b] + [
        r for r in company.as_sub if r["prime"] == b
    ]
    if related != bool(shared):
        return None

    if shared:
        agencies = sorted({_agency_short(r["agency"]) for r in shared})
        answer = (
            f"Yes. {len(shared)} reported subcontract(s) between {a} and {b}, "
            f"at {', '.join(agencies)}."
        )
        reasoning = (
            f"Checking both directions, since either could have been the prime. "
            f"{len(shared)} award(s) on record connect them."
        )
    else:
        answer = (
            f"No reported subcontract between {a} and {b}. Note this means no "
            f"*reported* relationship: subaward reporting is incomplete, so this "
            f"is weaker than proof they have never worked together."
        )
        reasoning = (
            "Neither appears as the other's subcontractor in the record. The "
            "honest answer is 'nothing reported' rather than 'never happened'."
        )
    return Question(
        question=f"Have {a} and {b} ever teamed on an HHS contract?",
        answer=answer,
        reasoning=reasoning,
        archetype="prior_relationship",
        gold=[b] if shared else [],
        meta={"a": a, "b": b, "related": bool(shared)},
    )


def q_warm_intro(graph: TeamingGraph, target: str, agency: str) -> Question | None:
    """Who in our network could open a door -- a two-hop graph question."""
    company = graph.companies.get(target)
    if company is None or agency in company.agencies:
        # "Breaking into" an agency they already serve is not a question. The
        # first version picked the target's own top agency and asked how to get
        # in, which made every answer trivially "all of them".
        return None
    bridges = sorted(
        {
            partner
            for partner in company.partners
            if agency in (graph.companies[partner].agencies if partner in graph.companies else [])
        }
    )
    if not 2 <= len(bridges) < len(company.partners):
        # Also drop it when *every* partner qualifies: nothing is being selected.
        return None
    short = _agency_short(agency)
    reasoning = (
        f"{target} has {len(company.partners)} companies it has teamed with. "
        f"The useful subset is the ones already working {short}, because they can "
        f"carry {target} in as a sub rather than {target} arriving cold."
    )
    answer = (
        f"{len(bridges)} of {target}'s existing partners already work {short}:\n"
        f"{_fmt(bridges[:TOP_K])}"
    )
    return Question(
        question=f"{target} wants to break into {short}. Who in their network could bring them in?",
        answer=answer,
        reasoning=reasoning,
        archetype="warm_intro",
        gold=bridges[:TOP_K],
        meta={"target": target, "agency": agency, "short": short},
    )


def q_portfolio(graph: TeamingGraph, name: str) -> Question | None:
    """What does this company actually do? Closed-book knowledge, no ranking."""
    company = graph.companies.get(name)
    if company is None or len(company.agencies) < 2:
        return None
    work = company.descriptions(2)
    if not work:
        return None
    agencies = [_agency_short(a) for a in company.agencies]
    reasoning = (
        f"Reading {name}'s record rather than its industry codes. It appears "
        f"across {len(agencies)} HHS components, with {len(company.as_sub)} "
        f"subcontracts and {len(company.as_prime)} awards where it hired subs."
    )
    answer = (
        f"{name} works {', '.join(agencies[:4])}. "
        f"Reported role: {len(company.as_sub)} subcontracts taken, "
        f"{len(company.as_prime)} awards where they were the prime hiring subs. "
        f"Representative scope: {work[0][:220]}"
    )
    return Question(
        question=f"What does {name} actually do for HHS?",
        answer=answer,
        reasoning=reasoning,
        archetype="portfolio",
        gold=[name],
        meta={"company": name},
    )


def q_repeat_partners(graph: TeamingGraph, prime: str) -> Question | None:
    """Repeat teaming is the only performance signal available. CPARS is not."""
    counts = collections.Counter(
        r["sub"] for r in graph.train_subawards if r["prime"] == prime
    )
    repeats = [(n, c) for n, c in counts.most_common() if c > 1]
    if len(repeats) < 2:
        return None
    reasoning = (
        f"CPARS ratings are not public, so the observable proxy for 'did this go "
        f"well' is whether {prime} hired them again. Of {len(counts)} "
        f"subcontractors, {len(repeats)} appear on more than one award."
    )
    answer = (
        f"{prime} has brought back {len(repeats)} of {len(counts)} subcontractors:\n"
        f"{_fmt([f'{n} ({c} awards)' for n, c in repeats[:TOP_K]])}\n\n"
        f"Repeat use is revealed preference, not a rating -- it is the closest "
        f"public substitute for a performance record."
    )
    return Question(
        question=f"Which of {prime}'s subcontractors have they used more than once?",
        answer=answer,
        reasoning=reasoning,
        archetype="repeat_partners",
        gold=[n for n, _ in repeats[:TOP_K]],
        meta={"prime": prime},
    )


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

ARCHETYPES = (
    "team_composition",
    "sub_candidates",
    "prime_candidates",
    "prior_relationship",
    "warm_intro",
    "portfolio",
    "repeat_partners",
)


def generate(graph: TeamingGraph, seed: int = 42, limit_per: int = 60) -> list[Question]:
    """Instantiate every archetype across the training-period graph."""
    rng = random.Random(seed)
    out: list[Question] = []

    primes = graph.primes_using_subs(3)
    pairs_by_prime = collections.defaultdict(collections.Counter)
    for row in graph.train_subawards:
        pairs_by_prime[row["prime"]][row["agency"]] += 1

    for prime in primes[:limit_per]:
        for agency, _count in pairs_by_prime[prime].most_common(2):
            naics = {
                r["naics"]
                for r in graph.train_subawards
                if r["prime"] == prime and r["agency"] == agency and r["naics"]
            }
            for item in (
                q_team_composition(graph, prime, agency),
                q_sub_candidates(graph, prime, agency, naics, rng),
            ):
                if item:
                    out.append(item)
        repeat = q_repeat_partners(graph, prime)
        if repeat:
            out.append(repeat)

    seen_pc = set()
    for row in graph.train_subawards:
        key = (row["agency"], row["naics"])
        if key in seen_pc or not row["naics"]:
            continue
        seen_pc.add(key)
        item = q_prime_candidates(graph, row["agency"], row["naics"])
        if item:
            out.append(item)

    rich = [
        name
        for name, c in graph.companies.items()
        if len(c.partners) >= 3 and len(c.agencies) >= 2
    ]
    rich.sort()
    rng.shuffle(rich)
    for name in rich[:limit_per]:
        item = q_portfolio(graph, name)
        if item:
            out.append(item)
        company = graph.companies[name]
        elsewhere = [
            a
            for a in sorted({r["agency"] for r in graph.train_subawards if r["agency"]})
            if a not in company.agencies
        ]
        for agency in elsewhere[:2]:
            item = q_warm_intro(graph, name, agency)
            if item:
                out.append(item)
                break
        partners = sorted(company.partners)
        if partners:
            item = q_prior_relationship(graph, name, partners[0], related=True)
            if item:
                out.append(item)
        strangers = [n for n in rich if n not in company.partners and n != name]
        if strangers:
            item = q_prior_relationship(
                graph, name, rng.choice(strangers), related=False
            )
            if item:
                out.append(item)

    return out


def describe(questions: list[Question]) -> dict[str, Any]:
    by = collections.Counter(q.archetype for q in questions)
    ranked = [q for q in questions if q.tiers]
    traps = sum(1 for q in ranked if any(t == 1 for t in q.tiers.values()))
    return {
        "total": len(questions),
        "by_archetype": dict(by.most_common()),
        "ranking_questions": len(ranked),
        "ranking_questions_with_hard_negatives": traps,
    }


__all__ = [
    "CANDIDATES",
    "MEANINGFUL_DESCRIPTION",
    "TOP_K",
    "ARCHETYPES",
    "Company",
    "Question",
    "build_candidates",
    "describe",
    "generate",
    "tier_for",
]


# ---------------------------------------------------------------------------
# blind holdout
# ---------------------------------------------------------------------------


def generate_blind(graph: TeamingGraph, seed: int = 1729, count: int = 200) -> list[Question]:
    """Questions whose answers live entirely in the sealed period.

    This is the answer to "how do you know you did not train on the answer".
    Everything a model or a retriever can see is the training-period graph;
    every gold entity comes from subcontracts dated after the cutoff. Of the
    blind teaming pairs, 71% appear nowhere in training, so the answer cannot be
    reached by recalling a training example -- it has to be inferred from who a
    prime works with and what they do.

    Two forms, both real forecasts:

    * *next team* -- given a prime's history, who did they actually bring on
      after the cutoff?
    * *new entrant* -- which companies started subbing at an agency in the
      sealed period, having no reported work there before?

    ``count`` defaults high deliberately: the sealed period supports 51
    questions and the first run used 18 of them, which left the whole comparison
    resting on ten items per archetype. Differences of a few points were noise
    at that size and had to be reported as such. Nothing is gained by capping a
    held-out set below what the data allows.

    Graded on precision and recall against the observed outcome. Recall is a
    lower bound: incomplete FSRS reporting means a correct name can be missing
    from the key, so a model is never penalised as harshly as the number looks.
    """
    rng = random.Random(seed)
    out: list[Question] = []

    blind_by_prime = collections.defaultdict(set)
    for row in graph.blind_subawards:
        blind_by_prime[row["prime"]].add(row["sub"])

    train_pairs = set(graph.pairs())
    candidates = sorted(
        (p for p, subs in blind_by_prime.items() if len(subs) >= 3 and p in graph.companies),
        key=lambda p: -len(blind_by_prime[p]),
    )

    for prime in candidates:
        company = graph.companies[prime]
        if not company.as_prime:
            continue
        actual = blind_by_prime[prime]
        fresh = sorted(s for s in actual if (prime, s) not in train_pairs)
        if len(fresh) < 2:
            continue

        # A slate, not an open-ended guess. Asking which of 1,700 companies a
        # prime will hire next is not a task any arm can do -- for one prime,
        # 28 of 32 subsequent partners were first-time pairings. Ranking a
        # dozen named candidates, some of whom did go on to work with them, is
        # the same judgement in a form that can actually be scored.
        # Sampled, not the alphabetical head. Taking sorted(actual)[:5] put every
        # correct answer in the first five slots of an alphabetically presented
        # slate, so "name items 1-5" scored perfectly without reading anything.
        picks_pool = sorted(actual)
        rng.shuffle(picks_pool)
        positives = sorted(picks_pool[:5])
        pool = [
            name
            for name, other in graph.companies.items()
            if name not in actual
            and name != prime
            and len(other.as_sub) >= 2
            and set(other.naics) & set(company.naics)
        ]
        pool.sort()
        rng.shuffle(pool)
        slate = {n: 4 for n in positives}
        for name in pool[: CANDIDATES - len(positives)]:
            slate[name] = 0
        if len(slate) < 8:
            continue

        # Presentation order must carry no signal: shuffled once, deterministically.
        shown = sorted(slate)
        rng.shuffle(shown)

        known = sorted(graph.team_of(prime))
        agencies = sorted({_agency_short(a) for a in company.agencies})[:3]
        out.append(
            Question(
                question=(
                    f"{prime} is standing up new HHS work. Of these companies, "
                    f"which are they most likely to bring on as subcontractors?\n"
                    + _fmt(shown)
                ),
                answer=(
                    f"Most likely: {', '.join(positives)}.\n\n"
                    f"They went on to use {len(actual)} subcontractors in this "
                    f"period, {len(fresh)} of them first-time pairings -- prior "
                    f"teaming is the base rate, not the whole answer."
                ),
                reasoning=(
                    f"{prime} has {len(known)} subcontractors on record across "
                    f"{', '.join(agencies)}. The candidates worth backing are the "
                    f"ones whose own work matches what {prime} subcontracts, not "
                    f"simply the largest firms on the list."
                ),
                archetype="blind_next_team",
                gold=positives,
                tiers=slate,
                meta={
                    "prime": prime,
                    "known_partners": known,
                    "all_actual": sorted(actual),
                    "unseen_in_training": fresh,
                    "blind": True,
                },
            )
        )
        if len(out) >= count // 2:
            break

    seen_at_agency = collections.defaultdict(set)
    for row in graph.train_subawards:
        seen_at_agency[row["agency"]].add(row["sub"])

    by_agency = collections.defaultdict(set)
    for row in graph.blind_subawards:
        if row["sub"] not in seen_at_agency[row["agency"]]:
            by_agency[row["agency"]].add(row["sub"])

    for agency, newcomers in sorted(by_agency.items(), key=lambda kv: -len(kv[1])):
        established = sorted(seen_at_agency[agency])
        if len(newcomers) < 3 or len(established) < 4:
            continue
        short = _agency_short(agency)

        # A slate, for the same reason blind_next_team needs one. Asked openly
        # -- "which companies are new to NIH?" -- the only way to supply useful
        # context is to supply the newcomers, and then the context *is* the
        # answer: measured, eight of eight records offered were gold and the
        # untuned base model scored 1.000 by reading the names back. Mixing
        # newcomers with established incumbents makes it a judgement instead.
        fresh = sorted(newcomers)
        rng.shuffle(fresh)
        old_hands = list(established)
        rng.shuffle(old_hands)
        positives = sorted(fresh[:4])
        slate = {n: 4 for n in positives}
        for name in old_hands:
            if len(slate) >= CANDIDATES:
                break
            slate.setdefault(name, 0)
        if len(slate) < 8 or len(positives) < 3:
            continue

        shown = sorted(slate)
        rng.shuffle(shown)
        out.append(
            Question(
                question=(
                    f"Which of these companies are new to {short} -- subcontracting "
                    f"there now with no prior reported work?\n" + _fmt(shown)
                ),
                answer=f"New to {short}: {', '.join(positives)}.",
                reasoning=(
                    f"A new entrant has {short} subcontracts now and none before. "
                    f"The rest of this list have {short} history already, which is "
                    f"what makes them the wrong answer rather than obviously so."
                ),
                archetype="blind_new_entrant",
                gold=positives,
                tiers=slate,
                meta={"agency": agency, "blind": True},
            )
        )
        if len(out) >= count:
            break

    return out


# ---------------------------------------------------------------------------
# paraphrases
# ---------------------------------------------------------------------------

# Alternative phrasings per archetype. The fact is identical; only the way it is
# asked changes. Knowledge has to survive rewording to be worth anything, and
# the earlier closed-book run showed why: asking training questions back
# verbatim scored 35% against 29% on reworded ones, so the model had learned
# neither the phrasing nor the fact. More phrasings per fact is the lever that
# addresses the second half of that.
#
# Templates read from a question's own meta, so a variant can never disagree
# with the answer it inherits.
PARAPHRASES: dict[str, tuple[str, ...]] = {
    "team_composition": (
        "Who does {prime} usually put on its {short} teams?",
        "What does a typical {prime} team look like at {short}?",
        "Which subcontractors show up most often on {prime}'s {short} work?",
        "If we're bidding against {prime} at {short}, who is likely on their team?",
    ),
    "sub_candidates": (
        "We're teaming with {prime} on {short} work. Who should we put forward "
        "as subcontractors?",
        "{prime} is priming a {short} bid. Which of these companies belong on "
        "the team, and which should we drop?",
        "Build me a subcontractor slate for {prime} at {short}.",
        "Who are the credible subs for {prime} on this {short} requirement?",
    ),
    "prime_candidates": (
        "We want to sub on {short} {title_lower} work. Which primes should we "
        "approach?",
        "Which primes actually subcontract {title_lower} at {short}?",
        "Who should we call about getting on a {short} {title_lower} team?",
    ),
    "prior_relationship": (
        "Have {a} and {b} ever teamed on an HHS contract?",
        "Is there any past performance connecting {a} and {b}?",
        "Do we have evidence {a} has worked with {b}?",
    ),
    "warm_intro": (
        "{target} wants to break into {short}. Who in their network could bring "
        "them in?",
        "Which of {target}'s existing partners could open a door at {short}?",
        "{target} has no {short} work. Who could team them in?",
    ),
    "portfolio": (
        "What does {company} actually do for HHS?",
        "Give me a read on {company} -- where do they work and what do they do?",
        "Is {company} worth a teaming conversation? What's their footprint?",
    ),
    "repeat_partners": (
        "Which of {prime}'s subcontractors have they used more than once?",
        "Who does {prime} keep going back to?",
        "Which of {prime}'s teaming relationships have actually stuck?",
    ),
}


def expand_paraphrases(
    questions: list[Question], per_question: int = 3, seed: int = 42
) -> list[Question]:
    """Restate each question a few ways, keeping variants tied together.

    ``fact_key`` is the load-bearing part. Variants of one fact must land on the
    same side of a train/eval split -- otherwise the eval set is asking about a
    fact the model was directly taught, in different words, and the split stops
    measuring generalisation.
    """
    rng = random.Random(seed)
    out: list[Question] = []

    for index, item in enumerate(questions):
        templates = PARAPHRASES.get(item.archetype, ())
        fields = dict(item.meta)
        title = fields.get("title")
        fields["title_lower"] = title.lower() if isinstance(title, str) else ""
        key = f"{item.archetype}:{index}"

        variants = [item.question]
        for template in templates:
            try:
                text = template.format(**fields)
            except (KeyError, IndexError):
                continue
            if text not in variants:
                variants.append(text)

        rng.shuffle(variants[1:])
        for text in variants[: max(1, per_question)]:
            clone = Question(
                question=text,
                answer=item.answer,
                reasoning=item.reasoning,
                archetype=item.archetype,
                gold=list(item.gold),
                tiers=dict(item.tiers),
                meta={**item.meta, "fact_key": key},
            )
            out.append(clone)
    return out

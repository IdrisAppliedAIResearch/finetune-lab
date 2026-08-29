"""Six more question types, added after the first fine-tune collapsed.

Trained on seven archetypes, the model learned to recognise an archetype and
emit its template: eighteen of eighteen blind answers took one of the seven
shapes, and the blind question type absent from training scored zero where
random scored 0.36. Asked which companies were new to NIH, it returned a
well-formed list of primes to approach.

Widening the set is half the fix. The other half is that answers now vary in
shape within a type (``ftlab.real.variety``), so recognising the type no longer
predicts the format and the content is left as the only thing worth learning.

Two of these earn their place beyond mere variety:

* ``unsupported`` has no answer. A model that has only ever seen answerable
  questions answers unanswerable ones anyway, in whichever template fits best,
  which is exactly the observed failure.
* ``capability_match`` can only be done by reading scope prose. Industry codes
  cannot separate its candidates -- NAICS 541690 covers both Apache targeting
  sights and CDC surveillance -- so it is the archetype where a language model
  should beat a structured-field rule engine if it beats it anywhere.

Kept separate from ``questions.py`` so that module needs no import from here;
``build`` combines the two generators.
"""

from __future__ import annotations

import collections
import random

from .graph import TeamingGraph
from .questions import CANDIDATES, TOP_K, Question, _agency_short, _fmt
from .variety import phrase, render_list

# Scope terms that appear in real award descriptions often enough to build a
# question from, and that industry codes do not distinguish.
CAPABILITY_TERMS = (
    "surveillance", "clinical", "data management", "training", "laboratory",
    "evaluation", "informatics", "statistical", "vaccine", "outbreak",
    "registry", "cybersecurity", "cloud", "modernization", "analytics",
    "genomic", "behavioral health", "quality measurement", "emergency",
    "biostatistics", "epidemiologic", "health equity", "workforce",
)


def _oxford(items: list[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def q_new_at_agency(
    graph: TeamingGraph, agency: str, rng: random.Random
) -> Question | None:
    """Which of these are new to this customer? A slate mixing both kinds."""
    established = sorted({r["sub"] for r in graph.train_subawards if r["agency"] == agency})
    elsewhere = sorted(
        {r["sub"] for r in graph.train_subawards if r["agency"] != agency}
        - set(established)
    )
    if len(established) < 5 or len(elsewhere) < 5:
        return None

    newcomers = list(elsewhere)
    rng.shuffle(newcomers)
    incumbents = list(established)
    rng.shuffle(incumbents)

    positives = sorted(newcomers[:4])
    slate = {n: 4 for n in positives}
    for name in incumbents:
        if len(slate) >= CANDIDATES:
            break
        slate.setdefault(name, 0)
    if len(slate) < 8:
        return None

    shown = sorted(slate)
    rng.shuffle(shown)
    short = _agency_short(agency)
    style = rng.choice(("numbered", "bulleted", "inline"))
    return Question(
        question=f"Which of these have no reported {short} work yet?\n" + _fmt(shown),
        answer=(
            f"No {short} record: {render_list(positives, style)}\n\n"
            f"The others already subcontract there."
        ),
        reasoning=(
            f"Checking each against {short} specifically rather than against HHS "
            f"as a whole. A company can be busy across the department and still "
            f"be new to this customer, and that distinction is the question."
        ),
        archetype="new_at_agency",
        gold=positives,
        tiers=slate,
        meta={"agency": agency, "short": short},
    )


def q_capability_match(
    graph: TeamingGraph, term: str, rng: random.Random
) -> Question | None:
    """Who has done work described this way? Answerable only from scope text."""
    matches = sorted(
        name
        for name, company in graph.companies.items()
        if any(term.lower() in d.lower() for d in company.descriptions(4))
    )
    # Loose on the upper bound. A term matching many companies is still a
    # usable question -- four of them go on the slate either way -- and the
    # tight window that looked principled produced three questions from eight
    # terms, starving the one archetype that can only be answered by reading.
    if len(matches) < 4 or len(matches) > len(graph.companies) * 0.6:
        return None

    others = sorted(set(graph.companies) - set(matches))
    rng.shuffle(others)
    picked = list(matches)
    rng.shuffle(picked)
    positives = sorted(picked[:4])

    slate = {n: 4 for n in positives}
    for name in others:
        if len(slate) >= CANDIDATES:
            break
        slate.setdefault(name, 0)
    if len(slate) < 8:
        return None

    shown = sorted(slate)
    rng.shuffle(shown)
    style = rng.choice(("numbered", "bulleted", "inline"))
    return Question(
        question=(
            f"Which of these have actually done {term} work? Go by what their "
            f"scope says, not their industry code.\n" + _fmt(shown)
        ),
        answer=render_list(positives, style),
        reasoning=(
            f"Industry codes are too coarse for this -- a single NAICS code "
            f"covers unrelated work -- so the test is whether {term} appears in "
            f"what these companies actually delivered."
        ),
        archetype="capability_match",
        gold=positives,
        tiers=slate,
        meta={"term": term},
    )


def q_agency_bridge(
    graph: TeamingGraph, first: str, second: str, rng: random.Random
) -> Question | None:
    """Who works both customers? Two conditions, only one of them discriminating."""
    both = sorted(
        name
        for name, company in graph.companies.items()
        if first in company.agencies and second in company.agencies
    )
    one_only = sorted(
        name
        for name, company in graph.companies.items()
        if first in company.agencies and second not in company.agencies
    )
    if len(both) < 4 or len(one_only) < 5:
        return None

    a_short, b_short = _agency_short(first), _agency_short(second)
    rng.shuffle(one_only)
    chosen = list(both)
    rng.shuffle(chosen)
    positives = sorted(chosen[:4])

    slate = {n: 4 for n in positives}
    for name in one_only:
        if len(slate) >= CANDIDATES:
            break
        slate.setdefault(name, 0)
    if len(slate) < 8:
        return None

    shown = sorted(slate)
    rng.shuffle(shown)
    return Question(
        question=f"Which of these work both {a_short} and {b_short}?\n" + _fmt(shown),
        answer=(
            f"Both: {render_list(positives, rng.choice(('inline', 'bulleted')))}\n\n"
            f"The rest appear at {a_short} only."
        ),
        reasoning=(
            f"Two conditions, and only one of them separates anybody. Everyone "
            f"listed has {a_short} work, so {a_short} experience is not the "
            f"filter -- {b_short} is."
        ),
        archetype="agency_bridge",
        gold=positives,
        tiers=slate,
        meta={"agency": first, "second": second},
    )


def q_bench_depth(graph: TeamingGraph, prime: str, agency: str) -> Question | None:
    """How deep is the bench? An aggregate answered with numbers, not a list."""
    rows = [
        r for r in graph.train_subawards if r["prime"] == prime and r["agency"] == agency
    ]
    if len(rows) < 4:
        return None
    subs = collections.Counter(r["sub"] for r in rows)
    repeats = sorted(n for n, c in subs.items() if c > 1)
    short = _agency_short(agency)
    return Question(
        question=f"How deep is {prime}'s subcontractor bench at {short}?",
        answer=(
            f"{len(subs)} distinct subcontractors across {len(rows)} reported "
            f"awards, {len(repeats)} of them used more than once. "
            + (
                f"The repeats are {_oxford(repeats[:3])}, which is what makes it a "
                f"bench rather than a list."
                if repeats
                else "Nobody was used twice, so this reads as a series of one-off "
                "pairings rather than a standing team."
            )
        ),
        reasoning=(
            "Depth is two numbers rather than one: how many partners, and how "
            "many they went back to. A long list of single-use subcontractors is "
            "shallower than a short list of repeats."
        ),
        archetype="bench_depth",
        gold=repeats[:TOP_K],
        meta={"prime": prime, "agency": agency, "short": short},
    )


def q_unsupported(
    graph: TeamingGraph, name: str, agency: str, rng: random.Random
) -> Question | None:
    """A question the records cannot settle. Saying so is the answer.

    Without this the model has never been shown that declining is permitted, and
    it will reach for whichever template fits -- the observed failure.
    """
    company = graph.companies.get(name)
    if company is None or agency in company.agencies:
        return None
    short = _agency_short(agency)
    asked = rng.choice(
        (
            f"What CPARS rating did {name} get on their {short} work?",
            f"How did {name}'s {short} contract perform?",
            f"Why did {name} lose their {short} recompete?",
        )
    )
    known = ", ".join(_agency_short(a) for a in company.agencies[:3]) or "nothing"
    return Question(
        question=asked,
        answer=(
            f"{phrase('no_record', rng)} {name} has no reported {short} work to "
            f"rate, and performance ratings are not public in any case -- CPARS is "
            f"source-selection information. What is on record for them: {known}."
        ),
        reasoning=(
            f"Two independent reasons this has no answer. There is no {short} work "
            f"by {name} on record, and CPARS ratings are not released publicly "
            f"even where the work exists. Answering would mean inventing both."
        ),
        archetype="unsupported",
        gold=[],
        meta={"company": name, "agency": agency, "short": short},
    )


def generate_extra(graph: TeamingGraph, seed: int = 7, limit: int = 40) -> list[Question]:
    """Instantiate the widened archetypes across the training-period graph."""
    rng = random.Random(seed)
    out: list[Question] = []

    agencies = sorted({r["agency"] for r in graph.train_subawards if r["agency"]})
    for agency in agencies:
        item = q_new_at_agency(graph, agency, rng)
        if item:
            out.append(item)
        for other in agencies:
            if other == agency:
                continue
            bridge = q_agency_bridge(graph, agency, other, rng)
            if bridge:
                out.append(bridge)

    # Several draws per term: the slate is sampled, so each draw is a different
    # question about the same capability rather than a duplicate.
    for term in CAPABILITY_TERMS:
        for _ in range(3):
            item = q_capability_match(graph, term, rng)
            if item:
                out.append(item)

    primes = graph.primes_using_subs(3)[:limit]
    for prime in primes:
        for agency in agencies:
            item = q_bench_depth(graph, prime, agency)
            if item:
                out.append(item)

    candidates = sorted(
        name for name, c in graph.companies.items() if c.as_sub and len(c.agencies) <= 2
    )
    rng.shuffle(candidates)
    for name in candidates[:limit]:
        for agency in agencies:
            item = q_unsupported(graph, name, agency, rng)
            if item:
                out.append(item)
                break

    return out

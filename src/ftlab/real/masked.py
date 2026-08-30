"""The masked-sub evaluation: hide a real subcontract and ask who filled it.

Every question this project has been graded on until now had a key someone
wrote. That is the flaw underneath most of its history -- a hand-authored key
measures agreement with the author, and a generated key measures agreement with
the function that produced it. The first fine-tune scored 1.000 on a set whose
retrieval had been handed the answer, and that went unnoticed for a week.

Here the key is what actually happened. Take a subaward from the blind window,
hide the subcontractor, present the prime with a slate of candidates, and score
whether the true sub is named. No opinion is involved and no authoring is
needed, so the set is as large as the record allows rather than as large as
someone had time to write.

What that buys, concretely:

* **Size.** 216 gradeable instances over 67 primes, against the 51 the authored
  set reached. At n=51 nothing below a 0.14 difference was detectable and every
  comparison the project ran came back "indistinguishable".
* **A floor that is not a guess.** The groupby -- rank the prime's most-used
  prior partners -- is computed from the same rows in ``rule_ranking``, needs no
  GPU, and scores hit@1 0.264 against a random draw's 0.079.

The set splits in two and the halves are different tasks. ``scoreboard``
reports them separately for that reason:

* **128 prior pairings**, where the prime had used this firm before. The
  groupby gets hit@1 0.445 and hit@5 0.758. Little headroom, and a strong arm
  here has mostly demonstrated it can read a partner list.
* **88 new pairings**, where the prime had never used this firm. The groupby
  scores **exactly 0 by construction**, and random scores 0.091 at hit@1. This
  is the subset that separates reasoning from counting, and it is the only
  number in this project that can answer the question it was started for.

Beware hit@5 on a twelve-name slate: a blind draw scores 0.394, so the metric
flatters everything. hit@1 and MRR are the ones to read.

What it does not buy, and these bound every number that comes out of it:

* **A knowable ceiling.** Real teaming turns on price, set-asides and
  incumbency that no record here reports. Some fraction of these rows is
  unpredictable from the evidence supplied, and nothing distinguishes a model
  reasoning badly from a model reasoning well about a decision made on grounds
  it cannot see. Treat the score as comparative between arms, never absolute.
* **Complete reporting.** Subaward reporting is incomplete, so a candidate
  marked wrong may have done the work unreported. That noise is one-directional
  and it depresses every arm equally.

Rows whose true sub has no retrievable record are dropped rather than scored.
Keeping them would grade every arm on a question with no answer available in
the context -- 53% of the blind window, concentrated in the largest primes,
whose subs are frequently one-award entries and individual physicians.
"""

from __future__ import annotations

import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .build import CONTEXT_K, build_index, context_for
from .graph import TeamingGraph, build_graph
from .ingest import load_slice
from .questions import CANDIDATES, Question, _agency_short, tier_for

# A company needs this many records before it is worth putting on a slate. At
# one award the record is a name and a date, which is not something to reason
# over in either direction -- as the answer or as a distractor.
MIN_RECORD = 2

ARCHETYPE = "masked_sub"


@dataclass(frozen=True)
class Instance:
    """One hidden subcontract, with the slate it will be recovered from."""

    prime: str
    agency: str
    date: str
    true_sub: str
    slate: tuple[str, ...]
    # Was this pairing already present in the training window? The single most
    # important split in the results: on prior pairings the groupby is a
    # legitimate strategy, and on new ones it cannot work at all.
    is_new: bool


def _retrievable(graph: TeamingGraph, name: str) -> bool:
    company = graph.companies.get(name)
    if company is None:
        return False
    records = len(company.as_sub) + len(company.as_prime) + len(company.prime_awards)
    return records >= MIN_RECORD


def distractor_pool(
    graph: TeamingGraph,
    prime: str,
    agency: str,
    exclude: set[str],
) -> list[str]:
    """Candidates that are plausible and verifiably not the answer.

    Three sources, and the mix is the whole design. Drawn purely at random the
    slate is trivial -- a CDC laboratory beside a Medicare mailroom vendor is
    not a discrimination anyone needs a model for. Drawn only from the prime's
    own bench it is worse than trivial, because then the groupby is not a
    baseline but the correct answer by construction.

    So: the prime's prior partners, who are the honest traps whenever the true
    sub is new; other firms working the same agency, plausible on every coarse
    signal; and firms sharing the prime's NAICS, which look right on the
    structured fields and carry no relationship at all.
    """
    prior: collections.Counter[str] = collections.Counter()
    at_agency: collections.Counter[str] = collections.Counter()
    for row in graph.train_subawards:
        if row["prime"] == prime:
            prior[row["sub"]] += 1
        if row["agency"] == agency:
            at_agency[row["sub"]] += 1

    prime_company = graph.companies.get(prime)
    naics = set(prime_company.naics[:3]) if prime_company else set()

    pool: list[str] = []
    seen = set(exclude) | {prime}

    def take(names: list[str], limit: int) -> None:
        added = 0
        for name in names:
            if added >= limit:
                return
            if name in seen or not _retrievable(graph, name):
                continue
            seen.add(name)
            pool.append(name)
            added += 1

    take([n for n, _ in prior.most_common()], 6)
    take([n for n, _ in at_agency.most_common()], 8)
    if naics:
        matches = sorted(
            (n for n, c in graph.companies.items() if naics & set(c.naics)),
            key=lambda n: -(
                len(graph.companies[n].as_sub) + len(graph.companies[n].as_prime)
            ),
        )
        take(matches, 6)

    # Last resort so a row is never thrown away for want of distractors. Capping
    # each source at five dropped 40 of 217 rows -- a fifth of the set lost to an
    # arbitrary constant rather than to anything about the data. These are weaker
    # distractors, drawn only on scale, and they sit at the end of the pool so
    # they are used only when the plausible sources run short.
    if len(pool) < CANDIDATES:
        by_scale = sorted(
            graph.companies,
            key=lambda n: -(
                len(graph.companies[n].as_sub)
                + len(graph.companies[n].as_prime)
                + len(graph.companies[n].prime_awards)
            ),
        )
        take(by_scale, CANDIDATES)
    return pool


def _collides(name: str, chosen: list[str]) -> bool:
    """Whether one name contains another, in either direction.

    The grader finds companies by literal substring, so a slate holding both
    ``VANDERBILT UNIVERSITY`` and ``VANDERBILT UNIVERSITY MEDICAL CENTER``
    cannot be scored: naming the longer one credits the shorter one too. This
    already produced a wrong number once, when ``KFORCE 3`` matched the company
    ``FORCE 3``, and it affected 30 of 177 instances here before the filter.
    """
    return any(name in other or other in name for other in chosen)


def instances(
    graph: TeamingGraph,
    seed: int = 0,
    candidates: int = CANDIDATES,
) -> list[Instance]:
    """Every gradeable hidden subcontract in the blind window."""
    rng = random.Random(seed)

    # Every blind partner of a prime is a correct answer to "who did they add",
    # so none of them may appear as a distractor on any of that prime's rows.
    # Without this a sibling true sub sits on the slate scored as wrong, and an
    # arm is penalised for an answer the record supports.
    blind_partners: dict[str, set[str]] = collections.defaultdict(set)
    for row in graph.blind_subawards:
        blind_partners[row["prime"]].add(row["sub"])

    prior_pairs = set(graph.pairs())

    out: list[Instance] = []
    seen_rows: set[tuple[str, str, str]] = set()
    for row in graph.blind_subawards:
        prime, sub, agency = row["prime"], row["sub"], row["agency"]
        if not agency or not _retrievable(graph, sub):
            continue
        # The same pairing recurs across mods; one instance each.
        key = (prime, sub, agency)
        if key in seen_rows:
            continue
        seen_rows.add(key)

        # A true sub whose name sits inside its own prime's is unscoreable in
        # either direction, so the row goes rather than the check.
        if _collides(sub, [prime]):
            continue

        pool = distractor_pool(graph, prime, agency, exclude=blind_partners[prime])
        slate = [sub]
        for name in pool:
            if len(slate) >= candidates:
                break
            if not _collides(name, [*slate, prime]):
                slate.append(name)
        if len(slate) < candidates:
            continue
        # Shuffled, never positives-first. The corpus builder had exactly this
        # leak twice -- once in the question text and once again in the record
        # ordering -- and both times it inflated a headline number.
        rng.shuffle(slate)

        out.append(
            Instance(
                prime=prime,
                agency=agency,
                date=row.get("date", ""),
                true_sub=sub,
                slate=tuple(slate),
                is_new=(prime, sub) not in prior_pairs,
            )
        )
    return out


def to_question(graph: TeamingGraph, item: Instance) -> Question:
    """Phrased so the ask gives nothing away.

    No mention of size, capability or incumbency: any adjective here is a hint,
    and a hint is a leak. The model gets the prime, the component, the slate and
    the records, which is what a person would have.
    """
    short = _agency_short(item.agency)
    slate = "\n".join(f"{i}. {n}" for i, n in enumerate(item.slate, start=1))
    question = (
        f"{item.prime} took on a new subcontractor for {short} work. "
        f"Which of these was it? Rank your top five, most likely first.\n{slate}"
    )
    prime_company = graph.companies.get(item.prime)
    naics = set(prime_company.naics[:3]) if prime_company else set()
    tiers = {
        name: tier_for(graph, name, item.prime, item.agency, naics)
        for name in item.slate
    }
    return Question(
        question=question,
        answer="",
        reasoning="",
        archetype=ARCHETYPE,
        gold=[item.true_sub],
        tiers=tiers,
        meta={
            "prime": item.prime,
            "agency": item.agency,
            "date": item.date,
            "is_new": item.is_new,
        },
    )


def rule_ranking(graph: TeamingGraph, item: Instance) -> list[str]:
    """The groupby's answer: this prime's slate candidates, most-used first."""
    prior: collections.Counter[str] = collections.Counter()
    for row in graph.train_subawards:
        if row["prime"] == item.prime:
            prior[row["sub"]] += 1
    return [n for n, _ in prior.most_common() if n in item.slate]


def rule_recovery(graph: TeamingGraph, items: list[Instance], top_k: int = 5) -> float:
    """The groupby's hit rate at ``top_k``. Kept for the headline comparison."""
    if not items:
        return 0.0
    hits = sum(
        1 for i in items if i.true_sub in rule_ranking(graph, i)[:top_k]
    )
    return hits / len(items)


def scoreboard(graph: TeamingGraph, items: list[Instance]) -> dict[str, Any]:
    """Rule and random, on the metrics this task should actually be judged by.

    ``hit@5`` on a twelve-candidate slate is nearly meaningless: picking five
    names blind scores 0.417, and the rule scores 0.449. Reported alone it makes
    the groupby look like it does nothing and would make any arm look strong.

    ``hit@1`` and MRR are rank-sensitive, so being right in first place is worth
    more than surviving a shortlist, which is also how a shortlist is used.
    Every one of these is reported beside its random-draw value, because a score
    without its floor is not a claim -- the lesson that mattered most on the
    previous corpus, where a 6x result turned out to be the floor moving.
    """
    n = len(items)
    if not n:
        return {}

    def mrr(ranked_of) -> float:
        total = 0.0
        for item in items:
            ranked = ranked_of(item)
            if item.true_sub in ranked:
                total += 1.0 / (ranked.index(item.true_sub) + 1)
        return total / n

    def hit(ranked_of, k: int) -> float:
        return sum(1 for i in items if i.true_sub in ranked_of(i)[:k]) / n

    rule = lambda i: rule_ranking(graph, i)  # noqa: E731
    # Random draws from the same slate, which is what every arm sees.
    rng = random.Random(0)
    shuffled = {
        id(i): rng.sample(list(i.slate), len(i.slate)) for i in items
    }
    chance = lambda i: shuffled[id(i)]  # noqa: E731

    return {
        "n": n,
        "rule_hit@1": round(hit(rule, 1), 4),
        "rule_hit@3": round(hit(rule, 3), 4),
        "rule_hit@5": round(hit(rule, 5), 4),
        "rule_mrr": round(mrr(rule), 4),
        "random_hit@1": round(hit(chance, 1), 4),
        "random_hit@3": round(hit(chance, 3), 4),
        "random_hit@5": round(hit(chance, 5), 4),
        "random_mrr": round(mrr(chance), 4),
    }


def build(
    data_dir: str | Path = "data/real",
    out_path: str | Path = "data/real_corpus/masked_sub.jsonl",
    seed: int = 0,
) -> dict[str, Any]:
    slice_ = load_slice(data_dir)
    graph = build_graph(slice_.prime_awards, slice_.subawards)
    index = build_index(graph)

    items = instances(graph, seed=seed)
    records = []
    for item in items:
        question = to_question(graph, item)
        record = question.to_record()
        record["context"] = context_for(graph, question, index, k=CONTEXT_K)
        records.append(record)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    new = [i for i in items if i.is_new]
    prior = [i for i in items if not i.is_new]
    return {
        "path": str(out),
        "instances": len(items),
        "distinct_primes": len({i.prime for i in items}),
        # Split every time. The two halves are different tasks: on prior
        # pairings the groupby is a legitimate strategy and nearly solves it,
        # on new ones it cannot score at all. A single blended number hides
        # exactly the thing the set was built to expose.
        "all": scoreboard(graph, items),
        "new_pairings": scoreboard(graph, new),
        "prior_pairings": scoreboard(graph, prior),
    }


__all__ = [
    "ARCHETYPE",
    "Instance",
    "build",
    "distractor_pool",
    "instances",
    "rule_ranking",
    "rule_recovery",
    "scoreboard",
    "to_question",
]

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
someone had time to write: 872 instances over 162 primes, against the 51 the
authored set reached.

The set splits in two and the halves are different tasks. ``scoreboard``
reports them separately for that reason:

* **587 prior pairings**, where the prime had used this firm before. The
  groupby gets hit@1 0.487 and hit@5 0.840. Little headroom, and a strong arm
  here has mostly demonstrated it can read a partner list.
* **285 new pairings** over 121 primes, where these two firms have no history in
  either direction. The groupby scores **exactly 0 by construction**. This is
  the subset that separates reasoning from counting, and it is the only number
  in this project that can answer the question it was started for.

``build`` writes the whole set and a prime-disjoint train/eval split beside
it: 338 training rows over 84 primes and 534 evaluation rows over 78, sharing no
prime. 156 of the evaluation rows are new pairings.

285 is enough to compare two arms with, which 60 was not. A paired McNemar test
detects a +8 point gain with 0.99 power and a +5 point gain with 0.79; at the
old cutoff those were 0.40 and 0.17. That is what moving ``TRAIN_UNTIL`` to
2024-06-30 bought, and it only became possible once the corpus was refilled --
before that, moving the cut back left nothing behind it but 2015.

**Three floors, and an arm has to clear all of them.** Reporting only the
groupby is what hid the worst defect this set has had. Distractors were drawn
from the head of an activity ranking, so every one of them was a large firm and
the answer was the small one, and sorting a slate by the record size printed in
its own prompt scored hit@1 0.282 / MRR 0.459 -- above the groupby, and above it
on the new-pairing half specifically, the half being advertised as the one no
counting strategy could touch. ``build_slate`` now matches candidates on size
and draws the answer's rank among them, which takes that sort down to hit@1
0.09-0.10 against chance's 0.083. The residual is intrinsic -- raising
``MIN_RECORD`` to 3, 4 or 5 leaves it unchanged at 0.091 while costing a quarter
of the set -- and sorting the other way round scores 0.079, below chance, so
there is no inverse of it to exploit either. ``size_ranking`` stays on the
scoreboard so a regression cannot hide again. ``name_ranking`` is there for the
same reason: research foundations have long names and take a lot of
subcontracts, so name length is a real correlate of the answer and sits at hit@1
0.107.

Beware hit@5 on a twelve-name slate: a blind draw scores 0.417, so the metric
flatters everything. hit@1 and MRR are the ones to read, and the random floor
beside every figure is computed in closed form rather than sampled -- one
permutation per item put the prior-pairings floor 1.8 standard deviations low
and turned a true 5.3x rule-over-random ratio into a reported 11.4x.

What the set does not buy, and these bound every number that comes out of it:

* **A knowable ceiling.** Real teaming turns on price, set-asides and
  incumbency that no record here reports. Some fraction of these rows is
  unpredictable from the evidence supplied, and nothing distinguishes a model
  reasoning badly from a model reasoning well about a decision made on grounds
  it cannot see. Treat the score as comparative between arms, never absolute.
* **Complete reporting.** Subaward reporting is incomplete, so a candidate
  marked wrong may have done the work unreported. That noise is one-directional
  and it depresses every arm equally.

Rows are dropped for a true sub with no retrievable record, for being the same
pairing recurring across mods, for naming a company too short for the grader to
see, and -- on the new half only -- for having the pairing already visible in a
record. Unretrievable subs were 63% of the blind window before the corpus was
refilled to cover 2016-2023; they are 28% now, which is most of why the set
grew.
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

# grade.known_companies drops any name of four characters or fewer, so a
# shorter name can never be returned by an arm however clearly it names it.
MIN_NAME = 4

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
    if name not in graph.companies:
        return False
    return record_size(graph, name) >= MIN_RECORD


def record_size(graph: TeamingGraph, name: str) -> int:
    """How much history a company has, counted the way the record prints it.

    This is the number the slate has to be matched on. ``company_record`` puts
    it in the prompt verbatim -- ``Subcontracts taken: 29; awards where prime:
    0`` -- so any correlation between it and the answer is a shortcut a model
    can read off rather than reason to.
    """
    company = graph.companies.get(name)
    if company is None:
        return 0
    return len(company.as_sub) + len(company.as_prime) + len(company.prime_awards)


# Suffixes USASpending spells inconsistently for the same firm. Collapsing them
# is what stops ``THE JOHNS HOPKINS UNIVERSITY`` and ``JOHNS HOPKINS
# UNIVERSITY, THE`` counting as two companies -- which they did, three ways, and
# it put a prior pairing in the new-pairing half.
_SUFFIXES = frozenset(
    {
        "INC", "INCORPORATED", "LLC", "LLP", "LP", "CORP", "CORPORATION",
        "CO", "COMPANY", "LTD", "LIMITED", "THE", "USA", "PC", "PA",
    }
)


def normalise(name: str) -> str:
    """One spelling per firm, for identity checks only -- never for display."""
    text = "".join(c if c.isalnum() else " " for c in name.upper())
    return " ".join(w for w in text.split() if w not in _SUFFIXES)


def _plausible(graph: TeamingGraph, prime: str, agency: str) -> list[tuple[int, str]]:
    """Every firm it would not be silly to name, tagged by how good a trap it is.

    The mix is the design. Drawn from the whole roster the slate is trivial -- a
    CDC laboratory beside a Medicare mailroom vendor is not a discrimination
    anyone needs a model for. Drawn only from the prime's own bench it is worse
    than trivial, because then the groupby is not a baseline but the correct
    answer by construction.

    So, in order of how hard they are to dismiss: the prime's prior partners,
    who are the honest traps whenever the true sub is new; other firms working
    the same agency, plausible on every coarse signal; and firms sharing the
    prime's NAICS, which look right on the structured fields and carry no
    relationship at all.

    The rank is a preference, not an ordering of the slate. Sorting the pool by
    activity is what produced the size shortcut -- every distractor a large firm
    and the answer the small one, so a sort by record size scored hit@1 0.282
    and MRR 0.459 without reading anything.
    """
    prior: set[str] = set()
    at_agency: set[str] = set()
    for row in graph.train_subawards:
        if row["prime"] == prime:
            prior.add(row["sub"])
        if row["agency"] == agency:
            at_agency.add(row["sub"])

    prime_company = graph.companies.get(prime)
    naics = set(prime_company.naics[:3]) if prime_company else set()
    by_naics = (
        {n for n, c in graph.companies.items() if naics & set(c.naics)}
        if naics
        else set()
    )

    ranked: dict[str, int] = {}
    for rank, group in enumerate((prior, at_agency, by_naics)):
        for name in group:
            if _retrievable(graph, name):
                ranked.setdefault(name, rank)
    return sorted((rank, name) for name, rank in ranked.items())


def _collides(name: str, chosen: list[str]) -> bool:
    """Whether one name contains another, or is the same firm spelled twice.

    The grader finds companies by literal substring, so a slate holding both
    ``VANDERBILT UNIVERSITY`` and ``VANDERBILT UNIVERSITY MEDICAL CENTER``
    cannot be scored: naming the longer one credits the shorter one too. This
    already produced a wrong number once, when ``KFORCE 3`` matched the company
    ``FORCE 3``, and it affected 30 of 177 instances before the filter.

    The normalised check catches the other half of the same problem, where two
    names do not contain each other but denote one firm -- one slate carried
    ``THE UNIVERSITY OF CHICAGO`` and ``UNIVERSITY OF CHICAGO, THE`` as separate
    candidates, so two of its twelve options were the same answer.
    """
    if any(name in other or other in name for other in chosen):
        return True
    key = normalise(name)
    return any(key == normalise(other) for other in chosen)


# Ranks past the plausible sources, used only to keep a size band from running
# dry. A firm with no tie to the prime or the agency is a weak distractor, but a
# weak distractor on the correct side of the answer costs less than a slate
# whose size order gives the answer away.
_ROSTER_RANK = 3

# Multiplicative half-widths for the size band, tried in order. Tight enough
# that the slate looks uniform on the number the prompt prints; widened only
# when a firm is too big or too isolated to have twelve near-peers.
_BANDS = (1.35, 2.0, 3.0, 5.0, 10.0, float("inf"))


def _pairing_visible(graph: TeamingGraph, prime: str, sub: str) -> bool:
    """Whether a record already reports this teaming under some other name.

    Asked only of new pairings. ``is_new`` guarantees no training edge between
    these two exact names, which is not the same as the prompt staying quiet. ``MBO PARTNERS`` lists
    ``GUIDEHOUSE DIGITAL`` in its partner line and the prime is ``GUIDEHOUSE``;
    ``KAMBRIAN`` lists ``PERATON ENTERPRISE SOLUTIONS`` and the prime is
    ``PERATON``. Four rows leaked this way, all of them past the index where the
    old test stopped sampling.

    Checked in both directions, because one leak was in the prime's record
    rather than the answer's and the test only ever looked at the answer's.
    """
    prime_key, sub_key = normalise(prime), normalise(sub)

    def touches(name: str, other_key: str) -> bool:
        key = normalise(name)
        return key.startswith(other_key) or other_key.startswith(key)

    for row in graph.train_subawards:
        if row["prime"] == prime and touches(row["sub"], sub_key):
            return True
        if row["sub"] == prime and touches(row["prime"], sub_key):
            return True
        if row["prime"] == sub and touches(row["sub"], prime_key):
            return True
        if row["sub"] == sub and touches(row["prime"], prime_key):
            return True
    return False


def build_slate(
    graph: TeamingGraph,
    prime: str,
    agency: str,
    true_sub: str,
    exclude: set[str],
    rng: random.Random,
    candidates: int = CANDIDATES,
) -> list[str] | None:
    """A slate whose record sizes say nothing about which name is right.

    ``Subcontracts taken: 29; awards where prime: 0`` is printed in the prompt
    for every candidate, so any correlation between that number and the answer
    is a shortcut an arm can read rather than reason to. Two mechanisms keep it
    at chance, and it took both:

    * **A band.** Candidates are drawn from near-peers of the answer's own
      record size, so the twelve names look alike on the printed number. The
      band widens until it can fill a slate, so a very large or very isolated
      firm still gets one.
    * **A drawn rank.** How many candidates fall below the answer is drawn
      uniformly before any of them are chosen. Inside a band alone the draw
      still skewed, because the roster holds far more small firms than large
      and near-peers are therefore mostly smaller.

    Neither was enough by itself, and each failed in the opposite direction --
    worth keeping in view before anyone simplifies this. The original draw took
    the head of each plausible source sorted by activity: every distractor came
    out large and the answer small, and sorting by size scored hit@1 0.282 / MRR
    0.459, above the groupby, on the half of the set advertised as clean. A
    drawn rank on its own left 0.124 against chance's 0.083, because a quarter
    of the answers sit at the floor of the roster with nothing beneath them. A
    band on its own gave 0.034 -- below chance, which is the same leak upside
    down and just as usable.

    Inside the band, better traps come first: the prime's prior partners, then
    firms at the same agency, then NAICS matches, then the roster. Sampled
    rather than taken in order, so two rows sharing a prime and agency get
    different slates -- when they shared one, the answer was the single name its
    sibling rows' slates lacked, recoverable by set difference on 169 of 216
    instances.

    Returns ``None`` when even the whole roster cannot fill a slate.
    """
    gold_size = record_size(graph, true_sub)
    banned = set(exclude) | {prime, true_sub}

    pool: list[tuple[int, str]] = _plausible(graph, prime, agency)
    pool += [
        (_ROSTER_RANK, n) for n in sorted(graph.companies) if _retrievable(graph, n)
    ]
    seen: set[str] = set()
    ranked: list[tuple[int, float, str, int]] = []
    for rank, name in pool:
        if name in seen or name in banned or len(name) < MIN_NAME:
            continue
        seen.add(name)
        # Shuffled within a preference rank, so the slate varies row to row
        # while the good traps still come first. Size is carried, never sorted
        # on: ordering by it inside the band picks the smallest near-peers every
        # time and makes the answer the largest name on the slate.
        ranked.append((rank, rng.random(), name, record_size(graph, name)))
    ranked.sort()

    # How many candidates sit strictly below the answer. A tie is neutral -- the
    # order inside a tie group is arbitrary -- so a tied name serves either side.
    want_smaller = rng.randrange(candidates)

    for width in _BANDS:
        low, high = gold_size / width, gold_size * width
        smaller, larger, tied = [], [], []
        for _, _, name, size in ranked:
            if not low <= size <= high:
                continue
            group = smaller if size < gold_size else larger if size > gold_size else tied
            group.append(name)

        slate = [true_sub]

        def take(names: list[str], count: int, slate: list[str] = slate) -> int:
            taken = 0
            for name in names:
                if taken >= count:
                    break
                if name in slate or _collides(name, [*slate, prime]):
                    continue
                slate.append(name)
                taken += 1
            return taken

        got = take(smaller, want_smaller)
        take(tied, want_smaller - got)
        got = take(larger, candidates - 1 - want_smaller)
        take(tied, candidates - 1 - want_smaller - got)
        for group in (smaller, larger, tied):
            if len(slate) >= candidates:
                break
            take(group, candidates - len(slate))

        if len(slate) >= candidates:
            rng.shuffle(slate)
            return slate
    return None


def size_ranking(
    graph: TeamingGraph, item: Instance, rng: random.Random | None = None
) -> list[str]:
    """The content-free baseline: smallest firm first.

    Both integers this sorts on are printed in the prompt -- ``Subcontracts
    taken: 29; awards where prime: 0`` -- so it is a strategy an arm can execute
    by reading rather than reasoning. It is reported on every split for that
    reason. If this line climbs above chance, the slate construction has
    regressed and no other number on the board means anything.

    Ties are broken at random rather than alphabetically. Band matching leaves
    most of a slate tied, and an alphabetical tie-break would measure the
    alphabet instead of the shortcut.
    """
    rng = rng or random.Random(0)
    return [
        n
        for _, _, n in sorted(
            (record_size(graph, n), rng.random(), n) for n in item.slate
        )
    ]


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

    # Undirected and normalised. A pairing is only new if these two firms have
    # no history at all: ``GDIT -> PERATON`` is not a new relationship when GDIT
    # subbed for Peraton last year, and neither is a pairing that already exists
    # under a second spelling of one of the two names.
    prior_pairs: set[frozenset[str]] = {
        frozenset({normalise(row["prime"]), normalise(row["sub"])})
        for row in graph.train_subawards
    }

    out: list[Instance] = []
    seen_rows: set[tuple[str, str, str]] = set()
    # A slate is only unrecoverable while no sibling row shares it: with the
    # same eleven distractors twice, the answer is the name the other one
    # lacks. The band leaves a handful of primes with barely twelve eligible
    # peers, so this still happens and the later row goes.
    seen_slates: set[tuple[str, ...]] = set()
    for row in graph.blind_subawards:
        prime, sub, agency = row["prime"], row["sub"], row["agency"]
        if not agency or not _retrievable(graph, sub):
            continue
        # A name the grader cannot see is unwinnable for every arm: it ignores
        # anything shorter than MIN_NAME, so FCN as an answer scored zero for
        # everyone and HP sat unscoreable on 79 slates.
        if len(sub) < MIN_NAME:
            continue
        # The same pairing recurs across mods and under both spellings.
        key = (normalise(prime), normalise(sub), agency)
        if key in seen_rows:
            continue
        seen_rows.add(key)

        if _collides(sub, [prime]):
            continue
        is_new = frozenset({normalise(prime), normalise(sub)}) not in prior_pairs
        # Only on the new half. A prior pairing is legitimately on the record
        # and is the groupby signal every arm is entitled to; silencing it
        # there would delete the prior half of the set entirely.
        if is_new and _pairing_visible(graph, prime, sub):
            continue

        slate = build_slate(
            graph, prime, agency, sub, blind_partners[prime], rng, candidates
        )
        if slate is None:
            continue
        distractors = tuple(sorted(n for n in slate if n != sub))
        if distractors in seen_slates:
            continue
        seen_slates.add(distractors)

        out.append(
            Instance(
                prime=prime,
                agency=agency,
                date=row.get("date", ""),
                true_sub=sub,
                slate=tuple(slate),
                is_new=is_new,
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


def name_ranking(
    graph: TeamingGraph, item: Instance, rng: random.Random | None = None
) -> list[str]:
    """The other content-free baseline: longest name first.

    Research foundations and university medical centres have long names and take
    a lot of subcontracts, so name length is a real correlate of being the
    answer and no slate construction removes it entirely. It sits at hit@1 0.100
    against chance 0.083 and is reported rather than engineered away, because
    matching distractors on name length as well as record size would make the
    slate an artefact. An arm has to clear this too.
    """
    rng = rng or random.Random(0)
    return [n for _, _, n in sorted((-len(n), rng.random(), n) for n in item.slate)]


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
    hits = sum(1 for i in items if i.true_sub in rule_ranking(graph, i)[:top_k])
    return hits / len(items)


def scoreboard(graph: TeamingGraph, items: list[Instance]) -> dict[str, Any]:
    """Every baseline that needs no model, on the metrics worth judging by.

    ``hit@5`` on a twelve-candidate slate is nearly meaningless: picking five
    names blind scores 0.417. ``hit@1`` and MRR are rank-sensitive, so being
    right first is worth more than surviving a shortlist, which is also how a
    shortlist gets used.

    Three floors, not one. ``rule`` is the groupby, ``size`` sorts by the
    record size printed in the prompt, and ``name`` sorts by name length; an
    arm has to clear all three before its score is evidence of anything.
    Reporting only the groupby is what hid the size shortcut for a week, and
    the size shortcut beat the groupby.

    The random floor is computed in closed form rather than sampled. Drawing one
    permutation per item and calling it the floor put the prior-pairings figure
    1.8 standard deviations low, and turned a true 5.3x rule-over-random ratio
    into a reported 11.4x -- the same failure as the last corpus, where a 6x
    result turned out to be the floor moving.
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

    def random_hit(k: int) -> float:
        return sum(min(k, len(i.slate)) / len(i.slate) for i in items) / n

    def random_mrr() -> float:
        return sum(
            sum(1 / r for r in range(1, len(i.slate) + 1)) / len(i.slate) for i in items
        ) / n

    rule = lambda i: rule_ranking(graph, i)  # noqa: E731
    size = lambda i: size_ranking(graph, i)  # noqa: E731
    name = lambda i: name_ranking(graph, i)  # noqa: E731

    out: dict[str, Any] = {"n": n}
    for label, ranked_of in (("rule", rule), ("size", size), ("name", name)):
        for k in (1, 3, 5):
            out[f"{label}_hit@{k}"] = round(hit(ranked_of, k), 4)
        out[f"{label}_mrr"] = round(mrr(ranked_of), 4)
    for k in (1, 3, 5):
        out[f"random_hit@{k}"] = round(random_hit(k), 4)
    out["random_mrr"] = round(random_mrr(), 4)
    return out


# How many new pairings the eval half needs. A paired comparison between two
# arms wants 126 to detect an eight-point gain at 80% power; this leaves a
# margin for rows lost to a reingest without handing the training half so little
# that there is nothing to learn from.
EVAL_NEW_TARGET = 147


def split_by_prime(
    items: list[Instance], seed: int = 0, eval_new_target: int = EVAL_NEW_TARGET
) -> tuple[list[Instance], list[Instance]]:
    """Divide the set into training and evaluation halves, sharing no prime.

    Split by prime rather than by row, which is the whole point. A prime appears
    on a median of three rows here and up to thirteen, and its bench is the same
    on all of them: with the same prime on both sides, a policy can learn who
    RESEARCH TRIANGLE INSTITUTE hires from the training rows and then score on
    the evaluation rows without generalising anything. That is the same class of
    leak as every other one this set has had, and it would look like a result.

    Primes are shuffled and taken for evaluation until it holds
    ``eval_new_target`` new pairings; the rest train. Both halves carry prior
    pairings too, because a policy that never sees one has not been taught that
    incumbency counts at all -- it is a real signal, just not the only one, and
    ``scoreboard`` reports the two kinds separately on whichever half is being
    read.

    Returns ``(train, eval)``.
    """
    by_prime: dict[str, list[Instance]] = collections.defaultdict(list)
    for item in items:
        by_prime[item.prime].append(item)

    primes = sorted(by_prime)
    random.Random(seed).shuffle(primes)

    held: list[Instance] = []
    train: list[Instance] = []
    new_held = 0
    for prime in primes:
        rows = by_prime[prime]
        if new_held < eval_new_target:
            held.extend(rows)
            new_held += sum(1 for r in rows if r.is_new)
        else:
            train.extend(rows)
    return train, held


def build(
    data_dir: str | Path = "data/real",
    out_path: str | Path = "data/real_corpus/masked_sub.jsonl",
    seed: int = 0,
) -> dict[str, Any]:
    """Write the whole set and the two prime-disjoint halves beside it.

    Three files, because the whole set is the thing to read when characterising
    the task and the halves are the thing to run against. ``masked_sub.jsonl``
    is everything; ``masked_sub.train.jsonl`` and ``masked_sub.eval.jsonl``
    share no prime, and nothing may be trained on the second.
    """
    slice_ = load_slice(data_dir)
    graph = build_graph(slice_.prime_awards, slice_.subawards)
    index = build_index(graph)

    items = instances(graph, seed=seed)

    def record_for(item: Instance) -> dict[str, Any]:
        question = to_question(graph, item)
        record = question.to_record()
        record["context"] = context_for(graph, question, index, k=CONTEXT_K)
        return record

    records = {id(i): record_for(i) for i in items}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[Instance]) -> None:
        path.write_text(
            "\n".join(json.dumps(records[id(r)], ensure_ascii=False) for r in rows)
            + "\n",
            encoding="utf-8",
        )

    train, held = split_by_prime(items, seed=seed)
    write(out, items)
    train_path = out.with_suffix(".train.jsonl")
    eval_path = out.with_suffix(".eval.jsonl")
    write(train_path, train)
    write(eval_path, held)

    def report(rows: list[Instance]) -> dict[str, Any]:
        new = [i for i in rows if i.is_new]
        prior = [i for i in rows if not i.is_new]
        return {
            "instances": len(rows),
            "distinct_primes": len({i.prime for i in rows}),
            # Split every time. The two kinds are different tasks: on prior
            # pairings the groupby is a legitimate strategy and nearly solves
            # it, on new ones it cannot score at all. A single blended number
            # hides exactly the thing the set was built to expose.
            "all": scoreboard(graph, rows),
            "new_pairings": scoreboard(graph, new),
            "prior_pairings": scoreboard(graph, prior),
        }

    shared = {i.prime for i in train} & {i.prime for i in held}
    return {
        "path": str(out),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        # Asserted rather than reported: a shared prime is the leak this split
        # exists to prevent, and it must never be something to notice later in
        # a stats blob.
        "primes_on_both_sides": len(shared),
        "whole_set": report(items),
        "train": report(train),
        "eval": report(held),
    }


__all__ = [
    "ARCHETYPE",
    "Instance",
    "EVAL_NEW_TARGET",
    "build",
    "build_slate",
    "instances",
    "name_ranking",
    "normalise",
    "record_size",
    "rule_ranking",
    "rule_recovery",
    "scoreboard",
    "size_ranking",
    "split_by_prime",
    "to_question",
]

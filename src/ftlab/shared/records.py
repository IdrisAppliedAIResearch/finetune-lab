"""Company records and the retrieval that puts them in a prompt.

Shared because both halves of the project read the same records: the
supervised corpus builder writes them into training examples, and the
masked-sub evaluation attaches them as context. If these two ever
disagreed about what a company record says, every comparison between a
tuned model and a baseline would be measuring the disagreement.
"""

from __future__ import annotations

import collections
from typing import Any

from .graph import TeamingGraph
from .questions import Question, _agency_short

# Records shown when context is supplied. Large enough to cover a full
# candidate slate: showing 8 records for 12 candidates leaves four that the
# model is asked to rank and given nothing to rank them on.
CONTEXT_K = 13


PARTNERS_SHOWN = 8
AGENCIES_SHOWN = 5


def _partner_lines(company: Any, limit: int = PARTNERS_SHOWN) -> tuple[int, str]:
    """Partners with how often, most-used first, each with where.

    The previous version listed six partner names in alphabetical order and no
    counts at all, which had two consequences worth stating because they shaped
    a whole batch of training data.

    The truncation was uninformative: Perspecta has 33 partners and the six
    shown started at ANACAPA and stopped at CISCO, so its most-used supplier by
    a factor of five -- HP, 27 awards -- did not appear in its own record. And
    no per-pair count existed anywhere in retrieval, so every answer in the
    corpus citing "27 reported awards" was asking the model to produce a number
    the prompt does not contain. The grader reads company names and not numbers,
    so a model learning to invent them scored exactly as well as one that did
    not.

    Ordering by use rather than alphabet is safe here for the same reason the
    counts are: ``Company`` is built from training-period rows only, so nothing
    in this line can carry the sealed period's answer. It reflects who a prime
    has hired before, which is the base rate the blind answer key names
    explicitly, and it reaches every arm through the same context.

    98% of (prime, sub) pairs sit at a single agency, so the agency rides along
    in brackets at almost no cost; the handful that span components list their
    top two.
    """
    counts: collections.Counter[str] = collections.Counter()
    where: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in company.as_sub:  # this company was the subcontractor
        counts[row["prime"]] += 1
        where[row["prime"]][row["agency"]] += 1
    for row in company.as_prime:  # this company hired the subcontractor
        counts[row["sub"]] += 1
        where[row["sub"]][row["agency"]] += 1

    # Most used first, alphabetical within a tie so a rebuild is deterministic.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    parts = []
    for partner, total in ranked:
        agencies = where[partner].most_common(2)
        if len(where[partner]) == 1:
            detail = _agency_short(agencies[0][0])
        else:
            detail = ", ".join(f"{_agency_short(a)} {n}" for a, n in agencies)
            if len(where[partner]) > 2:
                detail += ", ..."
        parts.append(f"{partner} {total} ({detail})")
    return len(counts), "; ".join(parts)


def company_record(graph: TeamingGraph, name: str) -> str:
    """One company as retrievable text, from training-period evidence only."""
    company = graph.companies.get(name)
    if company is None:
        return f"{name}\nNo record."

    # Awards and distinct counterparties per component, so "how much do they do
    # here" and "how widely do they team here" are both readable. The bare list
    # of agency names could answer neither, and several question types turn on
    # exactly that comparison across a prime's components.
    per_agency: dict[str, list[int]] = {}
    partners_at: dict[str, set[str]] = collections.defaultdict(set)
    for row in company.as_sub:
        partners_at[row["agency"]].add(row["prime"])
    for row in company.as_prime:
        partners_at[row["agency"]].add(row["sub"])
    awards_at: collections.Counter[str] = collections.Counter()
    for row in (*company.prime_awards, *company.as_sub, *company.as_prime):
        if row.get("agency"):
            awards_at[row["agency"]] += 1
    for agency in company.agencies[:AGENCIES_SHOWN]:
        per_agency[agency] = [awards_at.get(agency, 0), len(partners_at.get(agency, ()))]

    if per_agency:
        agencies = ", ".join(
            f"{agency} ({awards} award{'s' if awards != 1 else ''}, "
            f"{partners} partner{'s' if partners != 1 else ''})"
            for agency, (awards, partners) in per_agency.items()
        )
    else:
        agencies = "none on record"

    naics = ", ".join(company.naics[:5]) or "none on record"
    total, listed = _partner_lines(company)
    lines = [
        name,
        f"Agencies: {agencies}",
        f"NAICS: {naics}",
        f"Subcontracts taken: {len(company.as_sub)}; awards where prime: {len(company.as_prime)}",
        f"Teamed with ({total}), most used first: {listed}"
        if total
        else "Teamed with: none on record",
    ]
    work = company.descriptions(1)
    if work:
        lines.append(f"Representative scope: {work[0][:150]}")
    return "\n".join(lines)


def build_index(graph: TeamingGraph) -> Any:
    """BM25 over company records, so context can be chosen without the answer."""
    from .retrieve import BM25Index, Document

    docs = [
        Document(id=name, kind="partner", title=name, text=company_record(graph, name))
        for name in sorted(graph.companies)
    ]
    return docs, BM25Index(docs)


def context_for(
    graph: TeamingGraph,
    question: Question,
    index: tuple[Any, Any],
    k: int = CONTEXT_K,
) -> str:
    """The library records a prompt carries -- chosen without consulting the key.

    The first version of this built context from the question's own gold list,
    which meant retrieval retrieved the answer. Measured on the blind set: for
    "which companies are new to NIH", eight of the eight records supplied were
    gold, and the untuned base model scored a perfect 1.000 simply by reading
    the record names back. That is not a benchmark, it is an echo.

    So context comes from three answer-independent sources, in order:

    * the entities the question names outright -- a question about a prime is
      entitled to that prime's record
    * the slate, when the question presents one, since those names are in the
      question text already
    * BM25 over the remaining company records, which is what a retrieval system
      would actually return and cannot see the key

    Gold is never injected. When it appears it is because retrieval found it.
    """
    docs, bm25 = index
    names: list[str] = []
    meta = question.meta

    for key in ("prime", "target", "company", "a", "b"):
        value = meta.get(key)
        if isinstance(value, str) and value in graph.companies and value not in names:
            names.append(value)

    # Sorted, not in slate order. The slate dict is built positives-first, so
    # iterating it put every correct answer in the opening context slots -- the
    # same positional leak already fixed in the question text, reappearing in
    # the records. Alphabetical is arbitrary with respect to the key.
    for candidate in sorted(question.tiers):
        if candidate not in names:
            names.append(candidate)

    if len(names) < k:
        scored = bm25.scores(question.question)
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], docs[kv[0]].id))
        for position, _score in ranked:
            name = docs[position].id
            if name not in names:
                names.append(name)
            if len(names) >= k:
                break

    return "\n\n".join(
        f"[{i}] {company_record(graph, name)}"
        for i, name in enumerate(names[:k], start=1)
    )




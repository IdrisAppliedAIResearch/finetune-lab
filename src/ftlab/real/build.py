"""Assemble the real corpus: train, eval, and the sealed blind set.

Three files, and the third is the point. ``blind.jsonl`` is generated from
subcontracts dated after the training cutoff, so nothing that writes a training
example has read it. Of the teaming pairs in that period, 71% appear nowhere in
training -- the answers cannot be reached by recalling a training example.

Context is attached in two flavours because the benchmark has three arms:

    A  fine-tuned model + retrieval    trained with context, served with context
    B  fine-tuned model alone          trained with context sometimes absent
    C  base model + retrieval          no training at all

Arm B is why ``context_dropout`` exists. A model trained only on prompts that
carry library records meets, when the records are withheld, a prompt shape it
has never seen, and fails for a reason that has nothing to do with what it knows.
Dropping the context on a fraction of training examples teaches it both modes,
so a single training run serves arms A and B and the comparison between them is
about knowledge rather than formatting.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

from .archetypes import generate_extra
from .authored import authored_examples
from .authored_context import context_examples
from .authored_profiles import profile_examples
from .graph import TeamingGraph, build_graph
from .ingest import load_slice
from .questions import (
    Question,
    _agency_short,
    describe,
    expand_paraphrases,
    generate,
    generate_blind,
)
from .variety import general_examples

# Fraction of training examples with the retrieved records withheld, so the same
# adapter can answer closed-book. Two in five is enough to establish the mode
# without starving the open-book behaviour that arm A depends on.
CONTEXT_DROPOUT = 0.4

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
    from ..retrieve import BM25Index, Document

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


def build(
    data_dir: str | Path = "data/real",
    out_dir: str | Path = "data/real_corpus",
    seed: int = 42,
    eval_ratio: float = 0.15,
    dropout: float = CONTEXT_DROPOUT,
    paraphrases: int = 3,
    general: int = 150,
    authored_repeat: int = 8,
    authored_context_repeat: int = 8,
    authored_profile_repeat: int = 8,
) -> dict[str, Any]:
    slice_ = load_slice(data_dir)
    graph = build_graph(slice_.prime_awards, slice_.subawards)

    search_index = build_index(graph)
    rng = random.Random(seed)
    # Thirteen archetypes, not seven. The first fine-tune learned to recognise
    # an archetype and emit its template -- 18 of 18 blind answers took one of
    # seven shapes -- so a wider set is half the remedy and varied answer shapes
    # within a type are the other half.
    base = generate(graph) + generate_extra(graph, seed=seed + 7)
    questions = expand_paraphrases(base, per_question=paraphrases, seed=seed)

    # Split on the fact, not the sentence. Splitting rows would scatter
    # paraphrases of one fact across train and eval, and the eval set would then
    # be asking about facts the model was taught outright -- measuring recall of
    # a rewording rather than generalisation.
    groups: dict[str, list[Question]] = {}
    for item in questions:
        groups.setdefault(item.meta.get("fact_key", item.question), []).append(item)
    keys = sorted(groups)
    rng.shuffle(keys)

    n_eval = max(1, int(len(keys) * eval_ratio))
    eval_qs = [q for key in keys[:n_eval] for q in groups[key]]
    train_qs = [q for key in keys[n_eval:] for q in groups[key]]
    rng.shuffle(eval_qs)
    rng.shuffle(train_qs)
    blind_qs = generate_blind(graph)

    def render(items: list[Question], *, drop: bool) -> list[dict[str, Any]]:
        rows = []
        for position, item in enumerate(items):
            record = item.to_record()
            # Deterministic by position, so a rerun produces the same corpus.
            withhold = drop and (position % 100) < int(dropout * 100)
            record["context"] = (
                "" if withhold else context_for(graph, item, search_index)
            )
            record["meta"]["closed_book"] = withhold
            rows.append(record)
        return rows

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "train.jsonl": render(train_qs, drop=True),
        "eval.jsonl": render(eval_qs, drop=True),
        # The blind set is served both ways so arms A and B are asked the same
        # questions; the arm decides whether the context is used.
        "blind.jsonl": render(blind_qs, drop=False),
    }
    # Hand-written examples, repeated. Few in number and carrying the style the
    # generated bulk cannot: answers that vary in shape, that decline when the
    # records do not settle a question, and that correct a question's premise.
    # Repetition is the cheap way to give fifteen good examples weight against
    # a thousand templated ones -- and unlike a repeated template, each copy is
    # of prose that was actually reasoned.
    if authored_repeat:
        written["train.jsonl"] = [
            *written["train.jsonl"],
            *authored_examples(repeat=authored_repeat),
        ]

    # The same thing again with the records attached. Those above are all
    # closed-book, which aimed every hand-written row at arm B -- the arm that
    # scored below the random floor -- and left arm A learning open-book
    # behaviour from generated templates alone. These carry a real context
    # block and reason over it, including the case the generated set cannot
    # produce: a record that is in the library and not on the slate.
    if authored_context_repeat:
        written["train.jsonl"] = [
            *written["train.jsonl"],
            *context_examples(graph, search_index, repeat=authored_context_repeat),
        ]

    # Record-reading profiles. The slate examples teach choosing between named
    # candidates; these teach what the fields in a retrieved record mean, which
    # is the only part that transfers to a firm the model has never seen -- and
    # 71% of the blind set's true pairs are exactly that.
    if authored_profile_repeat:
        written["train.jsonl"] = [
            *written["train.jsonl"],
            *profile_examples(graph, search_index, repeat=authored_profile_repeat),
        ]

    # General instruction data, closed-book and off-domain. A model fine-tuned
    # only on domain templates has no reason to keep the instruction-following
    # it started with, and that capability is exactly what let the untuned base
    # model beat both fine-tuned arms on an unfamiliar question type.
    if general:
        written["train.jsonl"] = [
            *written["train.jsonl"],
            *general_examples(general, seed=seed),
        ]
        rng.shuffle(written["train.jsonl"])

    for filename, rows in written.items():
        (out / filename).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )

    train_facts = {q.meta.get("fact_key") for q in train_qs}
    eval_facts = {q.meta.get("fact_key") for q in eval_qs}
    stats = {
        "graph": graph.stats(),
        "questions": describe(questions),
        "base_questions": len(base),
        "paraphrases_per_question": paraphrases,
        "facts_train": len(train_facts),
        "facts_eval": len(eval_facts),
        "facts_in_both": len(train_facts & eval_facts),
        "train": len(written["train.jsonl"]),
        "eval": len(written["eval.jsonl"]),
        "blind": len(written["blind.jsonl"]),
        "general_examples": general,
        "authored_examples": len(authored_examples(repeat=authored_repeat)),
        "authored_profile_examples": len(
            profile_examples(graph, search_index, repeat=authored_profile_repeat)
        ),
        "authored_context_examples": len(
            context_examples(graph, search_index, repeat=authored_context_repeat)
        ),
        "closed_book_share": round(
            sum(1 for r in written["train.jsonl"] if r["meta"]["closed_book"])
            / max(1, len(written["train.jsonl"])),
            3,
        ),
    }
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats

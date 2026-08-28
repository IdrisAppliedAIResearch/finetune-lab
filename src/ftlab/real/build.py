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

import json
import random
from pathlib import Path
from typing import Any

from .graph import TeamingGraph, build_graph
from .ingest import load_slice
from .questions import Question, describe, generate, generate_blind

# Fraction of training examples with the retrieved records withheld, so the same
# adapter can answer closed-book. Two in five is enough to establish the mode
# without starving the open-book behaviour that arm A depends on.
CONTEXT_DROPOUT = 0.4

# Records shown when context is supplied.
CONTEXT_K = 8


def company_record(graph: TeamingGraph, name: str) -> str:
    """One company as retrievable text, from training-period evidence only."""
    company = graph.companies.get(name)
    if company is None:
        return f"{name}\nNo record."
    agencies = ", ".join(company.agencies[:5]) or "none on record"
    naics = ", ".join(company.naics[:5]) or "none on record"
    partners = sorted(company.partners)
    lines = [
        name,
        f"Agencies: {agencies}",
        f"NAICS: {naics}",
        f"Subcontracts taken: {len(company.as_sub)}; awards where prime: {len(company.as_prime)}",
        f"Teamed with ({len(partners)}): {', '.join(partners[:8])}"
        if partners
        else "Teamed with: none on record",
    ]
    work = company.descriptions(1)
    if work:
        lines.append(f"Representative scope: {work[0][:240]}")
    return "\n".join(lines)


def context_for(graph: TeamingGraph, question: Question, k: int = CONTEXT_K) -> str:
    """The library records a question needs, as a numbered block.

    Assembled from the entities the question is about rather than by lexical
    search: these questions name their subjects, so retrieval would be a
    round-trip to the same answer. The retrieval layer in ftlab.retrieve is what
    arm C uses at serve time; this is the training-side equivalent.
    """
    names: list[str] = []
    meta = question.meta
    for key in ("prime", "target", "company", "a", "b"):
        value = meta.get(key)
        if isinstance(value, str) and value not in names:
            names.append(value)
    for candidate in question.tiers:
        if candidate not in names:
            names.append(candidate)
    for name in question.gold:
        if name not in names:
            names.append(name)

    blocks = [
        f"[{i}] {company_record(graph, name)}"
        for i, name in enumerate(names[:k], start=1)
    ]
    return "\n\n".join(blocks)


def build(
    data_dir: str | Path = "data/real",
    out_dir: str | Path = "data/real_corpus",
    seed: int = 42,
    eval_ratio: float = 0.15,
    dropout: float = CONTEXT_DROPOUT,
) -> dict[str, Any]:
    slice_ = load_slice(data_dir)
    graph = build_graph(slice_.prime_awards, slice_.subawards)

    rng = random.Random(seed)
    questions = generate(graph)
    rng.shuffle(questions)

    n_eval = max(1, int(len(questions) * eval_ratio))
    eval_qs, train_qs = questions[:n_eval], questions[n_eval:]
    blind_qs = generate_blind(graph)

    def render(items: list[Question], *, drop: bool) -> list[dict[str, Any]]:
        rows = []
        for index, item in enumerate(items):
            record = item.to_record()
            # Deterministic by position, so a rerun produces the same corpus.
            withhold = drop and (index % 100) < int(dropout * 100)
            record["context"] = "" if withhold else context_for(graph, item)
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
    for filename, rows in written.items():
        (out / filename).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )

    stats = {
        "graph": graph.stats(),
        "questions": describe(questions),
        "train": len(written["train.jsonl"]),
        "eval": len(written["eval.jsonl"]),
        "blind": len(written["blind.jsonl"]),
        "closed_book_share": round(
            sum(1 for r in written["train.jsonl"] if r["meta"]["closed_book"])
            / max(1, len(written["train.jsonl"])),
            3,
        ),
    }
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats

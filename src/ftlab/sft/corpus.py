"""Assemble the supervised corpus: train and eval from hand-written examples.

The retrieval half of this module moved to ``shared.records`` -- the
records themselves are read by the reinforcement-learning side too, and
one definition of a company record is the only way the two halves stay
comparable.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

from ..shared.graph import build_graph
from ..shared.ingest import load_slice
from ..shared.records import build_index
from .authored import authored_examples
from .authored_context import context_examples
from .authored_profiles import profile_examples

# Fraction of training examples with the retrieved records withheld, so the
# same adapter can answer closed-book. Two in five is enough to establish
# the mode without starving the open-book behaviour that arm A depends on.
CONTEXT_DROPOUT = 0.15


def build(
    data_dir: str | Path = "data/real",
    out_dir: str | Path = "data/real_corpus",
    seed: int = 42,
    eval_examples: int = 8,
    dropout: float = CONTEXT_DROPOUT,
    authored_repeat: int = 3,
) -> dict[str, Any]:
    """Assemble the corpus from hand-written examples only.

    This used to expand thirteen archetypes into ~1,900 templated rows and split
    them into train and eval. Two things were wrong with that and both are fixed
    by deleting it.

    The eval split contained *zero* hand-written rows -- the authored families
    were appended to train after the split -- so eval loss measured how well the
    model reproduced templates, and checkpoint selection was steered by it. Here
    the split is over distinct authored examples, so eval measures the behaviour
    the project is actually trying to teach.

    And every generated gold answer was an observed relationship in the training
    graph, which taught one rule: the answer is a firm this prime has already
    hired. Hand-written examples can teach the case that rule gets wrong.

    Splitting on the *example*, not the row: an example is repeated
    ``authored_repeat`` times, and letting copies straddle the split would put
    the eval answer verbatim in train.
    """
    slice_ = load_slice(data_dir)
    graph = build_graph(slice_.prime_awards, slice_.subawards)
    search_index = build_index(graph)
    rng = random.Random(seed)

    # One copy of each distinct example, tagged so the split can group them.
    singles: list[dict[str, Any]] = [
        *authored_examples(repeat=1),
        *context_examples(graph, search_index, repeat=1),
        *profile_examples(graph, search_index, repeat=1),
    ]
    for row in singles:
        row["meta"]["example_id"] = row["question"][:80]

    keys = sorted({r["meta"]["example_id"] for r in singles})
    rng.shuffle(keys)
    held_out = set(keys[:eval_examples])

    def expand(rows: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _ in range(repeat):
            out.extend(json.loads(json.dumps(r)) for r in rows)
        return out

    train_rows = expand([r for r in singles if r["meta"]["example_id"] not in held_out],
                        authored_repeat)
    eval_rows = [r for r in singles if r["meta"]["example_id"] in held_out]

    # Context dropout, deterministic by position so a rebuild is reproducible.
    # Far lower than the 0.4 this used to run at: closed-book answering measured
    # 0.279 against a 0.369 random floor, so 43% of the corpus was training a
    # mode that performs worse than guessing. Enough is kept for arm B to exist
    # as a control, and no more.
    for position, row in enumerate(train_rows):
        if row["context"] and (position % 100) < int(dropout * 100):
            row["context"] = ""
            row["meta"]["closed_book"] = True

    rng.shuffle(train_rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("train.jsonl", train_rows), ("eval.jsonl", eval_rows)):
        (out / filename).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )

    kinds = collections.Counter(r["meta"].get("archetype") for r in train_rows)
    stats = {
        "graph": graph.stats(),
        "distinct_examples": len(keys),
        "held_out_examples": sorted(held_out),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "authored_repeat": authored_repeat,
        "train_by_kind": dict(kinds),
        "closed_book_share": round(
            sum(1 for r in train_rows if not r["context"]) / max(1, len(train_rows)), 3
        ),
    }
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats

"""Run the arms over the same questions and compare them.

    A  fine-tuned + retrieval   adapter, records supplied
    B  fine-tuned alone         adapter, records withheld
    C  base + retrieval         no adapter, records supplied
    D  prior-teaming rule       no model at all

A, B and C differ in exactly two switches -- adapter or not, context or not --
and share the questions, the decoding settings and the grader. Anything else
that differed between them would be a confound, and the comparison is the whole
deliverable.

Arm D is the one to beat, and it was missing until it was pointed out that a
``Counter`` over the training subawards outscored every model arm on the v2
generations. Arm C only tells you whether fine-tuning earned its keep against
the base model; arm D tells you whether *either* earned its keep against not
using a language model. Without it the benchmark cannot answer the question the
project is named after, and a result that beats C while losing to D is a result
that says: use the groupby.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from .grade import (
    TOP_K,
    aggregate,
    collapse_report,
    grade_one,
    known_companies,
    load_items,
    random_floor,
    render,
)

ARMS = {
    "a": ("fine-tuned + retrieval", True, True),
    "b": ("fine-tuned, no retrieval", True, False),
    "c": ("base model + retrieval", False, True),
    "d": ("prior-teaming rule, no model", False, False),
}

# Arm D uses no model at all, so it never reaches run_arm.
RULE_ARMS = frozenset({"d"})


@dataclass
class ArmResult:
    arm: str
    label: str
    summary: dict[str, Any]
    generations: list[str]


def run_arm(
    cfg: Config,
    arm: str,
    items: list[dict[str, Any]],
    known: list[str],
    adapter: str | Path | None,
    max_new_tokens: int = 900,
    batch_size: int = 8,
) -> ArmResult:
    from ..infer import generate_many, load_for_inference

    label, use_adapter, use_context = ARMS[arm]
    model, tokenizer = load_for_inference(cfg, adapter if use_adapter else None)

    pairs = [
        (item["question"], item.get("context", "") if use_context else "")
        for item in items
    ]
    generated = generate_many(
        model,
        tokenizer,
        pairs,
        cfg,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        batch_size=batch_size,
    )
    graded = [grade_one(i, g, known) for i, g in zip(items, generated, strict=True)]
    summary = aggregate(graded)
    # Recorded per arm because it is the failure the ranking metrics cannot see:
    # a model can score respectably while fluently answering a different
    # question in a shape it was trained on.
    summary["collapse"] = collapse_report(items, generated)
    return ArmResult(arm=arm, label=label, summary=summary, generations=generated)


def run_rule_arm(
    arm: str,
    items: list[dict[str, Any]],
    known: list[str],
    graph: Any,
) -> ArmResult:
    """Arm D: rank the slate by how often this prime has already hired each name.

    The null hypothesis the project exists to reject, and it was missing from
    the benchmark. ``questions.py`` argues that a rule engine ranks on
    structured fields and that NAICS is close to useless here -- both true, and
    beside the point, because the discriminating field is not NAICS. It is prior
    teaming, it sits in the same table, and one ``Counter`` over the training
    subawards extracts it.

    Scored through ``grade_one`` like every other arm, so it is answering the
    same questions against the same key with the same parser. It emits the
    corpus's own "Most likely: ..." shape for exactly that reason.

    Ties -- and most of the slate ties at zero prior awards -- break
    alphabetically. That is deliberately signal-free: ranking a blind slate
    alphabetically scores at the random floor, so the tie-break contributes
    nothing and the arm's score is the prior-teaming signal alone.
    """
    import collections

    prior: collections.Counter[tuple[str, str]] = collections.Counter(
        (row["prime"], row["sub"]) for row in graph.train_subawards
    )

    generated: list[str] = []
    for item in items:
        meta = item["meta"]
        prime = meta.get("prime")
        tiers: dict[str, int] = meta.get("tiers") or {}
        if not prime or not tiers:
            generated.append("No ranking: this question does not present a slate.")
            continue
        ranked = sorted(tiers, key=lambda name: (-prior[(prime, name)], name))
        picks = ranked[:TOP_K]
        generated.append(
            f"Most likely: {', '.join(picks)}.\n\n"
            "Ranked by how many times this prime has already subcontracted to "
            "each candidate in the training period."
        )

    graded = [grade_one(i, g, known) for i, g in zip(items, generated, strict=True)]
    summary = aggregate(graded)
    summary["collapse"] = collapse_report(items, generated)
    return ArmResult(arm=arm, label=ARMS[arm][0], summary=summary, generations=generated)


def compare(results: list[ArmResult], floor: dict[str, Any]) -> str:
    """One table, arms side by side, every number against the same floor."""
    from .grade import LABELS

    order = [r.arm for r in results]
    head = "".join(f"{('arm ' + a.upper()):>12}" for a in order)
    lines = [
        f"=== {len(results)} arm{'s' if len(results) != 1 else ''} ===",
        "",
        *[f"  arm {r.arm.upper()}  {r.label}" for r in results],
        "",
        f"{'metric':<40}{'floor':>9}{head}",
        "-" * (49 + 12 * len(order)),
    ]
    for key, label in LABELS:
        if not any(key in r.summary["overall"] for r in results):
            continue
        base = floor["overall"].get(key, 0.0)
        row = f"{label:<40}{base:>9.3f}"
        for result in results:
            value = result.summary["overall"].get(key)
            row += f"{value:>12.3f}" if value is not None else f"{'--':>12}"
        lines.append(row)

    # Below the rule because it is a different kind of number: not how well the
    # arm did, but whether it answered the question it was asked. The first run
    # scored like an ordinary underperformer while forcing every blind answer
    # into a trained shape, and none of the rows above could show that.
    lines.append("-" * (49 + 12 * len(order)))
    for key, label in (
        ("answers_in_a_template", "answers in a trained template"),
        ("answers_in_the_wrong_template", "...in the WRONG template"),
    ):
        row = f"{label:<40}{0.0:>9.3f}"
        for result in results:
            row += f"{result.summary['collapse'][key]:>12.3f}"
        lines.append(row)
    return "\n".join(lines)


def run(
    cfg: Config,
    split_path: str | Path,
    out_dir: str | Path,
    adapter: str | Path | None = None,
    arms: tuple[str, ...] = ("d", "c", "b", "a"),
    data_dir: str | Path = "data/real",
    **kwargs: Any,
) -> dict[str, Any]:
    items = load_items(split_path)
    known = known_companies(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = None
    if RULE_ARMS & set(arms):
        from .graph import build_graph
        from .ingest import load_slice

        slice_ = load_slice(data_dir)
        graph = build_graph(slice_.prime_awards, slice_.subawards)

    results: list[ArmResult] = []
    for arm in arms:
        print(f"\n[arms] running arm {arm.upper()} -- {ARMS[arm][0]} ({len(items)} items)")
        if arm in RULE_ARMS:
            result = run_rule_arm(arm, items, known, graph)
        else:
            result = run_arm(cfg, arm, items, known, adapter, **kwargs)
        results.append(result)

        (out / f"arm_{arm}_generations.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"question": i["question"], "generated": g, "meta": i["meta"]},
                    ensure_ascii=False,
                )
                for i, g in zip(items, result.generations, strict=True)
            ),
            encoding="utf-8",
        )
        print()
        print(render(result.summary, f"arm {arm.upper()} -- {result.label}"))

    floor = random_floor(items, known)
    oracle = aggregate([grade_one(i, i["answer"], known) for i in items])
    table = compare(results, floor)
    print("\n" + table)
    print(f"\n(oracle replay of the answer key: precision@4 "
          f"{oracle['overall'].get('precision_at_k', 0):.3f} -- the grader agrees "
          f"with the corpus)")

    report = {
        "floor": floor,
        "oracle": oracle,
        "arms": {r.arm: {"label": r.label, "summary": r.summary} for r in results},
    }
    (out / "arms.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "arms_report.txt").write_text(table, encoding="utf-8")
    return report

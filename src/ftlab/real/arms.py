"""Run the three arms over the same questions and compare them.

    A  fine-tuned + retrieval   adapter, records supplied
    B  fine-tuned alone         adapter, records withheld
    C  base + retrieval         no adapter, records supplied

The arms differ in exactly two switches -- adapter or not, context or not -- and
share the questions, the decoding settings and the grader. Anything else that
differed between them would be a confound, and the comparison is the whole
deliverable.

Arm C is the one to beat. If A and C land together, retrieval is doing the work
and the fine-tune is decoration. If B is far below both, the graph did not make
it into the weights and the honest answer is that this task wants retrieval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from .grade import aggregate, grade_one, known_companies, load_items, random_floor, render

ARMS = {
    "a": ("fine-tuned + retrieval", True, True),
    "b": ("fine-tuned, no retrieval", True, False),
    "c": ("base model + retrieval", False, True),
}


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
    max_new_tokens: int = 420,
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
    return ArmResult(arm=arm, label=label, summary=aggregate(graded), generations=generated)


def compare(results: list[ArmResult], floor: dict[str, Any]) -> str:
    """One table, arms side by side, every number against the same floor."""
    from .grade import LABELS

    order = [r.arm for r in results]
    head = "".join(f"{('arm ' + a.upper()):>12}" for a in order)
    lines = [
        "=== three arms ===",
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
    return "\n".join(lines)


def run(
    cfg: Config,
    split_path: str | Path,
    out_dir: str | Path,
    adapter: str | Path | None = None,
    arms: tuple[str, ...] = ("c", "b", "a"),
    data_dir: str | Path = "data/real",
    **kwargs: Any,
) -> dict[str, Any]:
    items = load_items(split_path)
    known = known_companies(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[ArmResult] = []
    for arm in arms:
        print(f"\n[arms] running arm {arm.upper()} -- {ARMS[arm][0]} ({len(items)} items)")
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

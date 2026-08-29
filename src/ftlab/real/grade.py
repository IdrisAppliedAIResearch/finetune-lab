"""Grade answers against the real-outcome answer key.

Entity matching, same as the synthetic grader and for the same reason: every
company name in the corpus is known, so finding which ones an answer names is
exact and needs no judge model. What changes is where the key comes from --
observed subcontracts rather than a scoring function -- so a perfect score here
is not available to any arm by construction.

Every headline number is reported beside the score a **random pick from the same
slate** achieves. That correction mattered more than anything else measured on
the previous corpus: floored against the whole 1,700-company roster, picking
blind out of the dozen candidates already in the prompt looked like a 6x result.
The floor is what turns a number into a claim.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOP_K = 4

# Headings the corpus uses to separate picks from rejections. An answer with no
# heading is read as recommending everything it names, which is the
# conservative reading: it cannot be credited with a rejection it never made.
REJECT_MARKERS = (
    "Not recommended, despite looking like obvious picks:",
    "Not recommended",
    "Worth rejecting",
    "Do not",
)
THINK_CLOSE = "</think>"


# Phrases a model uses when it stops working and starts concluding. Needed
# because a verbose model enumerates every candidate before choosing, and
# reading the first names it mentions grades its *analysis order* rather than
# its answer. Measured: the base model discussed all twelve candidates in slate
# order, so "first four named" was very nearly a random draw -- which is exactly
# the score it received, and the reason that score meant nothing.
CONCLUSION = re.compile(
    r"(?im)^.*\b(most likely|recommend(?:ation|ed)?|in summary|conclusion|"
    r"therefore|final answer|answer)\b.*$"
)


def conclusion_of(text: str, known: list[str] | None = None) -> str:
    """The part of an answer that states a choice, if the model states one.

    Falls back to the whole text. A model that never concludes is graded on what
    it did produce, not excused for producing nothing.

    The tail is only trusted when it actually names a company. A first attempt
    guarded on tail length instead and rejected a correct three-word conclusion
    -- "Most likely: GAMMA INC and DELTA GROUP." is 39 characters. Length is a
    proxy for content and a bad one; whether the tail names anything is the
    thing actually being asked.
    """
    matches = list(CONCLUSION.finditer(text))
    if not matches:
        return text
    tail = text[matches[-1].start():]
    if known is None:
        return tail
    return tail if any(name in tail for name in known) else text


def looks_truncated(text: str) -> bool:
    """Did generation stop mid-thought rather than finish?

    Worth a metric of its own: 16 of 18 base-model answers in the first run were
    cut off by the token budget while still enumerating candidates, so the arm
    was scored on an answer it had not finished writing.
    """
    stripped = text.rstrip()
    return bool(stripped) and not stripped.endswith((".", "!", "?", ":", ")", '"'))


def split_answer(text: str) -> tuple[str, str]:
    """(recommended, rejected) halves of a generated answer."""
    body = text.split(THINK_CLOSE, 1)[-1] if THINK_CLOSE in text else text
    for marker in REJECT_MARKERS:
        if marker in body:
            head, _, tail = body.partition(marker)
            return head, tail
    return body, ""


def find_companies(text: str, known: list[str]) -> list[str]:
    """Known company names present, in order of appearance.

    Longest first so that a company whose name contains another's does not let
    the shorter one match on the same span.
    """
    hits: list[tuple[int, str]] = []
    claimed: list[tuple[int, int]] = []
    for name in known:
        position = text.find(name)
        if position < 0:
            continue
        if any(start <= position < end for start, end in claimed):
            continue
        claimed.append((position, position + len(name)))
        hits.append((position, name))
    return [name for _, name in sorted(hits)]


@dataclass
class Graded:
    archetype: str
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def grade_one(item: dict[str, Any], generated: str, known: list[str]) -> Graded:
    meta = item["meta"]
    tiers: dict[str, int] = meta.get("tiers") or {}
    gold = set(meta.get("gold") or [])

    # The question's own subject is not a pick. Answers name the prime in their
    # header -- "Sub candidates for X on CDC work" -- and counting X as a
    # recommendation charged the oracle with naming one off-slate company on
    # every ranking question.
    subject = {
        meta[key]
        for key in ("prime", "target", "company", "a")
        if isinstance(meta.get(key), str)
    }

    recommended, rejected = split_answer(conclusion_of(generated, known))
    picked = [n for n in find_companies(recommended, known) if n not in subject][:TOP_K]
    turned_down = set(find_companies(rejected, known)) - subject

    scores: dict[str, float] = {
        "named_any": float(bool(picked)),
        "truncated": float(looks_truncated(generated)),
    }
    notes: dict[str, Any] = {"picked": picked}

    # Precision needs something to be precise about. A "have A and B ever
    # teamed" question answered correctly with "no" has an empty key, and
    # scoring it 0.0 would punish the right answer.
    if picked and gold:
        scores["precision_at_k"] = len(set(picked) & gold) / len(picked)
    if picked:
        if tiers:
            # Tier 3+ means a real relationship with this prime; that is what a
            # correct pick looks like even when it is not one of the five names
            # the answer key happens to list.
            scores["tier_hit_rate"] = sum(
                1 for n in picked if tiers.get(n, 0) >= 3
            ) / len(picked)
            scores["mean_tier"] = statistics.mean(tiers.get(n, 0) for n in picked)
            # Ungrounded: named a company the prompt never offered.
            scores["off_slate"] = float(sum(1 for n in picked if n not in tiers))

    if gold:
        scores["recall"] = len(set(picked) & gold) / len(gold)

    if tiers:
        traps = {n for n, t in tiers.items() if t <= 1}
        scores["traps_recommended"] = float(len(set(picked) & traps))
        scores["trap_rejection_recall"] = (
            len(turned_down & traps) / len(traps) if traps else 0.0
        )
        notes["traps"] = sorted(traps)

    return Graded(archetype=meta.get("archetype", "?"), scores=scores, notes=notes)


def aggregate(graded: list[Graded]) -> dict[str, Any]:
    buckets: dict[str, list[Graded]] = {}
    for item in graded:
        buckets.setdefault(item.archetype, []).append(item)

    def means(rows: list[Graded]) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for row in rows:
            for key, value in row.scores.items():
                totals.setdefault(key, []).append(value)
        return {k: statistics.mean(v) for k, v in totals.items() if v}

    return {
        "n": len(graded),
        "overall": means(graded),
        "by_archetype": {k: {"n": len(v), **means(v)} for k, v in sorted(buckets.items())},
    }


def random_floor(
    items: list[dict[str, Any]], known: list[str], trials: int = 5, seed: int = 0
) -> dict[str, Any]:
    """What a no-knowledge answer scores, drawing from each question's own slate.

    Not from the full roster. Sampling the roster measures how hard the
    companies are to find, which retrieval already did, and credits the model
    for work it did not do.
    """
    per_trial = []
    for trial in range(trials):
        rng = random.Random(seed + trial)
        generated = []
        for item in items:
            slate = sorted(item["meta"].get("tiers") or {})
            pool = slate or known
            picks = rng.sample(pool, min(TOP_K, len(pool)))
            generated.append(
                "\n".join(f"{i + 1}. {n}" for i, n in enumerate(picks))
            )
        per_trial.append(
            aggregate([grade_one(i, g, known) for i, g in zip(items, generated, strict=True)])
        )

    keys = per_trial[0]["overall"]
    return {
        "overall": {
            k: statistics.mean(t["overall"].get(k, 0.0) for t in per_trial) for k in keys
        }
    }


LABELS = (
    ("precision_at_k", "precision@4 vs answer key"),
    ("tier_hit_rate", "picks with a real relationship"),
    ("recall", "recall of the answer key"),
    ("mean_tier", "mean tier of picks (0-4)"),
    ("traps_recommended", "hard negatives recommended (lower better)"),
    ("trap_rejection_recall", "hard negatives rejected"),
    ("off_slate", "companies named that were not offered"),
    ("named_any", "answers naming anything"),
    ("truncated", "answers cut off by the token budget"),
)


def render(summary: dict[str, Any], title: str, floor: dict[str, Any] | None = None) -> str:
    lines = [f"=== {title} ===", f"items graded: {summary['n']}", ""]
    if floor:
        lines += [
            "floor = random picks from each question's own slate",
            "",
        ]
    overall = summary["overall"]
    for key, label in LABELS:
        if key not in overall:
            continue
        value = overall[key]
        row = f"  {label:<42} {value:.3f}"
        if floor and key in floor["overall"]:
            base = floor["overall"][key]
            row += (
                f"   (floor {base:.3f}, {value / base:.1f}x)"
                if base > 0.001
                else f"   (floor {base:.3f})"
            )
        lines.append(row)
    return "\n".join(lines)


def load_items(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def known_companies(data_dir: str | Path = "data/real") -> list[str]:
    """Every company name in the slice, longest first for safe matching."""
    subawards = json.loads(
        (Path(data_dir) / "subawards.json").read_text(encoding="utf-8")
    )
    names = {r["prime"] for r in subawards} | {r["sub"] for r in subawards}
    primes = json.loads(
        (Path(data_dir) / "prime_awards.json").read_text(encoding="utf-8")
    )
    names |= {r["recipient"] for r in primes}
    return sorted((n for n in names if len(n) > 3), key=len, reverse=True)


# ---------------------------------------------------------------------------
# template collapse
# ---------------------------------------------------------------------------

# Openings that only a corpus-trained model produces, one per generated
# archetype. A base model asked these questions writes none of them.
#
# This is a first-class metric because it was the whole finding of the first
# run and it was invisible in every number reported: precision, recall and
# tier-hit all looked like ordinary underperformance, while the actual failure
# was that eighteen of eighteen answers had been forced into one of seven
# shapes. A model can score respectably on the ranking metrics while answering
# a different question fluently, and only this catches that.
TEMPLATE_MARKERS: tuple[tuple[str, str], ...] = (
    ("prime_candidates", "Primes to approach"),
    ("team_composition", "team, most-used first"),
    ("sub_candidates", "Sub candidates for"),
    ("repeat_partners", "has brought back"),
    ("warm_intro", "existing partners already work"),
    ("portfolio", "Reported role:"),
    ("prior_relationship", "reported subcontract between"),
    ("bench_depth", "distinct subcontractors across"),
    ("new_at_agency", "No .* record:"),
)


def template_used(generated: str) -> str | None:
    """Which training archetype's shape this answer fell into, if any."""
    for name, marker in TEMPLATE_MARKERS:
        if marker.startswith("No ") and marker.endswith("record:"):
            if re.search(marker, generated):
                return name
        elif marker in generated:
            return name
    return None


def collapse_report(items: list[dict[str, Any]], generated: list[str]) -> dict[str, Any]:
    """How often answers took a trained shape, and whether it was the right one.

    Emitting a template is not itself wrong -- a sub-candidates question should
    produce a sub-candidates answer. What is wrong is emitting one that does not
    match the question, which is what "answered a different question fluently"
    looks like in a number.
    """
    used = 0
    mismatched = 0
    for item, text in zip(items, generated, strict=True):
        template = template_used(text)
        if template is None:
            continue
        used += 1
        if template != item["meta"].get("archetype"):
            mismatched += 1
    total = max(1, len(items))
    return {
        "answers_in_a_template": used / total,
        "answers_in_the_wrong_template": mismatched / total,
    }

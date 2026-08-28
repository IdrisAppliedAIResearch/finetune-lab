"""Corpus assembly: world -> QRA triples -> train/eval/probe files.

The split is the part worth reading carefully. A random shuffle would put a
paraphrase of the same question on both sides and report a score that mostly
measures leakage.

Instead, two different holdouts, because two different things are being tested:

* **Opportunities are held out whole.** Every recommendation question about a
  held-out opportunity goes to eval and none to train. The model has never seen
  that pursuit, so answering it requires applying the reasoning pattern to the
  library rather than recalling a memorized pairing. This is the real test.

* **Recall questions are split by paraphrase.** The facts themselves must be in
  training -- closed-book means the library lives in the weights -- but one
  phrasing of each fact is held back, so eval measures whether the fact is
  retrievable through a question the model never saw, not whether a string was
  memorized.

A third file, ``eval_probes.jsonl``, asks for single exact values -- contract
numbers, dollar figures, end years. Parametric recall degrades on precisely
these, and the probes make that degradation measurable instead of anecdotal.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .graph import SCALES, World
from .items import QRAItem
from .recall import (
    build_agency_recall,
    build_capability_partners,
    build_capability_recall,
    build_contract_recall,
    build_multihop,
    build_partner_recall,
    build_person_recall,
    build_portfolio_questions,
    build_teaming_history,
    build_vehicle_questions,
)
from .recommend import (
    build_bid_decision,
    build_citations,
    build_gap_analysis,
    build_prime_candidates,
    build_sub_candidates,
    build_teaming,
)

# Fraction of one-of-a-kind questions (multi-hop joins, focused single-fact
# lookups) routed to eval instead of train.
SINGLETON_HOLDOUT = 0.15


@dataclass
class Multipliers:
    """Paraphrases per fact. Recall multipliers carry the closed-book burden:
    a fact seen once in one phrasing rarely survives into the weights."""

    contract_detail: int = 8
    partner_profile: int = 4
    agency_portfolio: int = 4
    capability_experience: int = 4
    person_profile: int = 2
    teaming_history: int = 3
    capability_partners: int = 2
    vehicle_coverage: int = 2
    multihop: int = 45

    teaming: int = 5
    prime_candidates: int = 4
    sub_candidates: int = 3
    citations: int = 4
    gap_analysis: int = 3
    bid_decision: int = 3


@dataclass
class BuildResult:
    world: World
    train: list[QRAItem] = field(default_factory=list)
    eval: list[QRAItem] = field(default_factory=list)
    probes: list[QRAItem] = field(default_factory=list)

    def stats(self) -> dict[str, Any]:
        def breakdown(items: list[QRAItem]) -> dict[str, Any]:
            return {
                "total": len(items),
                "by_layer": dict(Counter(i.layer for i in items).most_common()),
                "by_archetype": dict(Counter(i.archetype for i in items).most_common()),
            }

        return {
            "world": self.world.stats(),
            "seed": self.world.seed,
            "scale": self.world.scale_name,
            "train": breakdown(self.train),
            "eval": breakdown(self.eval),
            "probes": {"total": len(self.probes)},
        }


def generate(
    seed: int = 42,
    scale: str = "demo",
    mult: Multipliers | None = None,
    holdout_ratio: float = 0.2,
) -> BuildResult:
    mult = mult or Multipliers()
    world = World(seed=seed, scale=scale)
    rng = random.Random(seed + 1)

    # --- decide the opportunity holdout before generating anything ---
    opportunity_ids = sorted(world.opportunities)
    n_holdout = max(1, int(len(opportunity_ids) * holdout_ratio))
    held_out = set(rng.sample(opportunity_ids, n_holdout))

    recall_items: list[QRAItem] = []
    recall_items += build_contract_recall(world, rng, mult.contract_detail)
    recall_items += build_partner_recall(world, rng, mult.partner_profile)
    recall_items += build_agency_recall(world, rng, mult.agency_portfolio)
    recall_items += build_capability_recall(world, rng, mult.capability_experience)
    recall_items += build_person_recall(world, rng, mult.person_profile)
    recall_items += build_teaming_history(world, rng, mult.teaming_history)
    recall_items += build_capability_partners(world, rng, mult.capability_partners)
    recall_items += build_vehicle_questions(world, rng, mult.vehicle_coverage)
    recall_items += build_portfolio_questions(world, rng)
    recall_items += build_multihop(world, rng, mult.multihop)

    rec_items: list[QRAItem] = []
    rec_items += build_teaming(world, rng, mult.teaming)
    rec_items += build_prime_candidates(world, rng, mult.prime_candidates)
    rec_items += build_sub_candidates(world, rng, mult.sub_candidates)
    rec_items += build_citations(world, rng, mult.citations)
    rec_items += build_gap_analysis(world, rng, mult.gap_analysis)
    rec_items += build_bid_decision(world, rng, mult.bid_decision)

    result = BuildResult(world=world)

    # Recommendation items follow their opportunity, whole.
    for item in rec_items:
        target = result.eval if item.meta.get("opportunity") in held_out else result.train
        item.meta["held_out_opportunity"] = item.meta.get("opportunity") in held_out
        target.append(item)

    # Recall items split by paraphrase: group on the fact, hold one phrasing back.
    grouped: dict[tuple[str, str], list[QRAItem]] = {}
    for item in recall_items:
        key = (item.archetype, _fact_key(item))
        grouped.setdefault(key, []).append(item)

    for group in grouped.values():
        if len(group) >= 3:
            rng.shuffle(group)
            result.eval.append(group[0])
            result.train.extend(group[1:])
        elif rng.random() < SINGLETON_HOLDOUT:
            # Multi-hop joins and focused single-fact questions are mostly
            # one-of-a-kind, so the paraphrase rule never reaches them and the
            # eval set ends up with almost none. Holding a slice of them out
            # whole is a fair test: the underlying facts are still taught by the
            # full-record items, so answering requires performing the join
            # rather than recalling this exact pairing.
            result.eval.extend(group)
        else:
            result.train.extend(group)

    result.probes = _build_probes(world, rng)
    _drop_leaked(result)

    rng.shuffle(result.train)
    rng.shuffle(result.eval)
    return result


def _drop_leaked(result: BuildResult) -> None:
    """Remove eval items whose question text also appears in training.

    Generators that sample combinations can emit the same question twice, and a
    split that puts one copy on each side reports memorization as
    generalization. Training keeps its copy -- closed-book needs the fact -- and
    eval loses the duplicate. This is a backstop, not the primary defence: the
    generators dedupe their own combinations first.
    """
    train_questions = {item.question for item in result.train}
    kept = [item for item in result.eval if item.question not in train_questions]
    dropped = len(result.eval) - len(kept)
    if dropped:
        print(f"[synth] dropped {dropped} leaked eval item(s)")
    result.eval = kept


def _fact_key(item: QRAItem) -> str:
    """What underlying fact a recall item is about, for paraphrase grouping."""
    for field_name in ("contract", "company", "agency", "capability", "person", "vehicle"):
        if field_name in item.meta:
            return f"{field_name}:{item.meta[field_name]}"
    return item.answer[:80]


def _build_probes(world: World, rng: random.Random) -> list[QRAItem]:
    """Single-value questions. These are where parametric recall frays first,
    and answering them with one short string makes grading unambiguous."""
    probes: list[QRAItem] = []
    for contract in world.contracts.values():
        specs = [
            (
                f"What is the contract number for {contract.name}?",
                contract.number,
                "contract_number",
            ),
            (
                f"What was the total value of {contract.name} ({contract.number})?",
                f"${contract.value_total / 1_000_000:.1f}M",
                "contract_value",
            ),
            (
                f"What year did {contract.name} ({contract.number}) end?",
                str(contract.end_year),
                "contract_end_year",
            ),
            (
                f"What CPARS rating did we receive on {contract.number}?",
                contract.cpars,
                "contract_cpars",
            ),
        ]
        question, answer, kind = rng.choice(specs)
        probes.append(
            QRAItem(
                question=question,
                reasoning=f"Exact-value lookup against the library record for {contract.number}.",
                answer=answer,
                archetype=kind,
                layer="probe",
                meta={"contract": contract.id, "exact": True},
            )
        )
    return probes


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, items: list[QRAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_record(), ensure_ascii=False) + "\n")


def dump_library(world: World, path: Path) -> None:
    """The graph itself, as the human-readable reference and grading key."""
    payload = {
        "seed": world.seed,
        "scale": world.scale_name,
        "us": asdict(world.us),
        "companies": [asdict(c) for c in world.partners],
        "people": [asdict(p) for p in world.people.values()],
        "contracts": [asdict(k) for k in world.contracts.values()],
        "opportunities": [asdict(o) for o in world.opportunities.values()],
        "stats": world.stats(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write(result: BuildResult, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    _write_jsonl(out / "train.jsonl", result.train)
    _write_jsonl(out / "eval.jsonl", result.eval)
    _write_jsonl(out / "eval_probes.jsonl", result.probes)
    dump_library(result.world, out / "library.json")

    stats = result.stats()
    (out / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def build_and_write(
    out_dir: str | Path,
    seed: int = 42,
    scale: str = "demo",
    holdout_ratio: float = 0.2,
) -> dict[str, Any]:
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {sorted(SCALES)}")
    result = generate(seed=seed, scale=scale, holdout_ratio=holdout_ratio)
    return write(result, out_dir)

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

# Our own contracts shown to a citation question. CITE_K is 3, so the pool has
# to be several times that before choosing between them means anything.
CITATION_CONTRACTS = 8


@dataclass
class Multipliers:
    """Paraphrases per fact, weighted toward reasoning over lookup.

    The closed-book design needed high recall multipliers because a fact seen
    once in one phrasing rarely survived into the weights. Retrieval removes
    that burden entirely -- the record is in the prompt now -- and the measured
    result of trying to carry it in weights was 29% fact recall against 34-54x
    floor on relationships. So the lookup archetypes drop to the minimum that
    still teaches the model to read a record correctly, and the budget moves to
    the questions that require joining records together.

    Recall items are not removed outright. Reading a retrieved record and
    answering from it is a skill in its own right, and it is the one that keeps
    the model from ignoring context it was given.
    """

    # Lookup: enough to teach reading a record, not to memorise it.
    contract_detail: int = 2
    partner_profile: int = 2
    agency_portfolio: int = 1
    capability_experience: int = 2
    person_profile: int = 1
    vehicle_coverage: int = 1

    # Relational: the reason the model exists.
    teaming_history: int = 5
    capability_partners: int = 4
    multihop: int = 90

    teaming: int = 8
    prime_candidates: int = 7
    sub_candidates: int = 6
    citations: int = 5
    gap_analysis: int = 6
    bid_decision: int = 6


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
    context_k: int = 5,
    context_alpha: float = 0.5,
    context_anchors: int = 2,
    context_partners: int = 14,
) -> BuildResult:
    mult = mult or Multipliers()
    world = World(seed=seed, scale=scale)
    rng = random.Random(seed + 1)

    # --- decide the opportunity holdout before generating anything ---
    opportunity_ids = sorted(world.opportunities)
    n_holdout = max(1, int(len(opportunity_ids) * holdout_ratio))
    held_out = set(rng.sample(opportunity_ids, n_holdout))

    # Retrieval is planned before generation, because the recommendation golden
    # answers are computed over the candidates the prompt will actually show.
    from ..retrieve import plan_opportunity_context

    # Two plans, because the two tasks need different pools. A teaming question
    # needs many partners and barely any of our own contracts; a past-performance
    # citation needs the opposite, and asking it to pick three citations from a
    # pool of two is not a task at all -- it is a formatting exercise that would
    # have scored near 1.0 and meant nothing.
    plan = (
        plan_opportunity_context(
            world,
            contracts=context_anchors,
            partners=context_partners,
            alpha=context_alpha,
        )
        if context_k
        else {}
    )
    citation_plan = (
        plan_opportunity_context(
            world, contracts=CITATION_CONTRACTS, partners=3, alpha=context_alpha
        )
        if context_k
        else {}
    )

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
    rec_items += build_teaming(world, rng, mult.teaming, plan)
    rec_items += build_prime_candidates(world, rng, mult.prime_candidates, plan)
    rec_items += build_sub_candidates(world, rng, mult.sub_candidates, plan)
    rec_items += build_citations(world, rng, mult.citations, citation_plan)
    rec_items += build_gap_analysis(world, rng, mult.gap_analysis, plan)
    rec_items += build_bid_decision(world, rng, mult.bid_decision, plan)

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

    # Probes must avoid any terse pairing that ended up in training, so they are
    # built after the split is known -- an item held back to eval still counts
    # as "not trained" and is fair game.
    result.probes = _build_probes(world, rng, _trained_facets(result.train))
    _drop_leaked(result)

    attach_context(result, plan, citation_plan, k=context_k, alpha=context_alpha)

    rng.shuffle(result.train)
    rng.shuffle(result.eval)
    return result


def attach_context(
    result: BuildResult,
    plan: dict[str, Any],
    citation_plan: dict[str, Any] | None = None,
    k: int = 5,
    alpha: float = 0.5,
) -> None:
    """Give every item the library records its prompt will carry.

    Two paths, because the two kinds of question need different retrieval.

    A recommendation question is anchored on an opportunity, and its golden
    answer was already computed over the candidates in ``plan`` -- so it takes
    that same context verbatim. Anything else re-retrieves on its own question
    text, which is safe there because a lookup or relational question names its
    subject and recall@1 on this corpus is 100%.

    Deliberately no injection of the gold record on the second path. Training on
    a context that always contains the answer teaches the model that it always
    will, and the first miss at inference then produces a confident answer from
    records that do not support it.
    """
    from ..retrieve import Retriever

    if not k:
        return

    retriever = Retriever.from_world(result.world, alpha=alpha)
    for item in [*result.train, *result.eval, *result.probes]:
        source = (
            citation_plan
            if item.archetype == "pp_citation" and citation_plan
            else plan
        )
        shown = source.get(item.meta.get("opportunity", ""))
        if not shown:
            item.context = retriever.context(item.question, k=k)
            continue
        item.context = shown.context
        # The grader has to rank over the same candidates the corpus did, or it
        # marks a correct answer wrong. Carrying the ids on the item is exact
        # and costs nothing; re-deriving them from the context text would be a
        # second implementation of the same fact, free to drift from this one.
        item.meta["candidate_partners"] = sorted(shown.partner_ids)
        item.meta["candidate_contracts"] = sorted(shown.contract_ids)


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


def _build_probes(
    world: World, rng: random.Random, trained_facets: set[tuple[str, str]]
) -> list[QRAItem]:
    """Single-value questions, for measuring how parametric recall degrades.

    Two things make these a fair measurement rather than a trick question.

    First, a probe is only built for a (contract, facet) pair whose terse
    question was *not* trained. The fact is still taught -- it sits inside the
    full contract record the model saw many times -- but this exact short-form
    pairing is not, so the probe measures retrieval rather than recall of a
    memorized pair.

    Second, the answer is a short sentence in the same shape the terse training
    items use, not a bare token. Training answers average around a thousand
    characters; grading a model against an eight-character target would mostly
    measure whether it guessed the output format, and would report a format
    mismatch as lost knowledge. ``meta.exact_value`` carries the bare value so
    grading can test containment and stay unambiguous either way.
    """
    probes: list[QRAItem] = []
    for contract in world.contracts.values():
        specs = {
            "number": (
                f"What's the contract number for {contract.name}?",
                contract.number,
                f"{contract.number}. That is {contract.name}, {contract.agency} "
                f"{contract.subunit}, {contract.period}.",
            ),
            "value": (
                f"What was the total value of {contract.name}?",
                f"${contract.value_total / 1_000_000:.1f}M",
                f"{contract.name} ({contract.number}) was "
                f"${contract.value_total / 1_000_000:.1f}M total; our share was "
                f"${contract.value_ours / 1_000_000:.1f}M as {contract.our_role}.",
            ),
            "end_year": (
                f"When did {contract.name} finish?",
                str(contract.end_year),
                f"{contract.end_year}. {contract.name} ({contract.number}) ran "
                f"{contract.period}.",
            ),
            "cpars": (
                f"What CPARS rating did we receive on {contract.name}?",
                contract.cpars,
                f"{contract.cpars} on {contract.name} ({contract.number}), "
                f"{contract.agency}, {contract.period}.",
            ),
        }

        available = [
            facet
            for facet in specs
            if (contract.id, facet) not in trained_facets
        ]
        if not available:
            continue

        facet = rng.choice(available)
        question, exact, answer = specs[facet]
        probes.append(
            QRAItem(
                question=question,
                reasoning=(
                    f"Exact-value lookup against the library record for {contract.number}."
                ),
                answer=answer,
                archetype=f"contract_{facet}",
                layer="probe",
                meta={
                    "contract": contract.id,
                    "exact": True,
                    "exact_value": exact,
                    "facet": facet,
                },
            )
        )
    return probes


def _trained_facets(items: list[QRAItem]) -> set[tuple[str, str]]:
    """(contract, facet) pairs that already have a trained terse answer."""
    pairs: set[tuple[str, str]] = set()
    for item in items:
        if item.archetype.startswith("contract_") and item.archetype != "contract_detail":
            facet = item.archetype.removeprefix("contract_")
            if "contract" in item.meta:
                pairs.add((item.meta["contract"], facet))
    return pairs


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
    context_k: int = 5,
    context_alpha: float = 0.5,
    context_contracts: int = 2,
    context_partners: int = 14,
) -> dict[str, Any]:
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {sorted(SCALES)}")
    result = generate(
        seed=seed, scale=scale, holdout_ratio=holdout_ratio,
        context_k=context_k, context_alpha=context_alpha,
        context_anchors=context_contracts, context_partners=context_partners,
    )
    return write(result, out_dir)

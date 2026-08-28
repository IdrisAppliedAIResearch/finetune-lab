"""Grade generated answers against the graph that produced the questions.

Eval loss says a run converged. It does not say whether the model recommends the
right partners, and on this corpus that is the only question worth asking. Since
the golden answers were computed from a graph we still hold, grading can be
deterministic: no LLM judge, no embedding similarity, no human pass.

The method is entity matching. Every company, contract number and contract name
in the world is known, so finding which of them appear in a generated answer --
and in what order -- is exact. From there the interesting measures fall out:

* **Hard negatives recommended.** The corpus is built so that every opportunity
  has partners who look right and are not. Whether the model recommends them
  anyway is the thesis of the whole demo, and it is the one number to read first.
* **Hallucinated partners.** Closed-book models invent plausible names. Every
  real name in this world ends in a known suffix, so a name with the right shape
  that is not in the library is, unambiguously, invented.
* **Gap coverage.** Whether the partners it named can actually do the work we
  cannot, which is the criterion the reasoning traces are built around.

One subtlety drives the parsing: the training answers name their rejected
candidates out loud, under a "Not recommended" heading. Counting entities across
the whole answer would therefore score a correct rejection as a bad
recommendation. The answer is split at that heading and the two halves are
scored differently -- a trap in the first half is a failure, a trap in the second
is exactly right.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .synth.graph import World
from .synth.scoring import hard_negatives, rank_partners, rank_past_performance
from .synth.taxonomy import (
    AGENCIES,
    CAPABILITY_BY_ID,
    NAICS,
    NAME_TAILS,
    PSC,
    VEHICLES,
)

# Headings that separate what the model recommends from what it rejects.
REJECT_MARKERS = (
    "Not recommended, despite looking like obvious picks:",
    "Not recommended",
    "Do not cite:",
    "Worth rejecting",
)
THINK_CLOSE = "</think>"

TOP_K = 4
CITE_K = 3
# Mirrors synth.recommend.REJECT_N -- how many traps an answer names.
REJECT_N = 3


# ---------------------------------------------------------------------------
# text segmentation
# ---------------------------------------------------------------------------


def split_reasoning(text: str) -> tuple[str, str]:
    """Return (reasoning, answer). Untagged output counts entirely as answer."""
    if THINK_CLOSE in text:
        head, _, tail = text.partition(THINK_CLOSE)
        return head.replace("<think>", "").strip(), tail.strip()
    return "", text.strip()


def split_recommendations(answer: str) -> tuple[str, str]:
    """Return (recommended, rejected) halves of an answer.

    If the model never produced a rejection heading, everything counts as
    recommended. That is the conservative reading: it cannot then be credited
    with a rejection it did not make.
    """
    for marker in REJECT_MARKERS:
        if marker in answer:
            head, _, tail = answer.partition(marker)
            return head, tail
    return answer, ""


# ---------------------------------------------------------------------------
# entity index
# ---------------------------------------------------------------------------


class EntityIndex:
    """Finds known library entities in free text, in order of appearance."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.companies = {c.name: c for c in world.partners}
        self.contracts_by_name = {k.name: k for k in world.contracts.values()}
        self.contracts_by_number = {k.number: k for k in world.contracts.values()}
        self.capabilities = {
            spec.name: cap_id for cap_id, spec in CAPABILITY_BY_ID.items()
        }

        # Every generated company name in this world ends in one of a known set
        # of suffixes, so a phrase with that shape which is not in the library
        # was invented by the model rather than recalled.
        # Longest first so a removal never leaves a fragment that then
        # shape-matches on its own.
        # Vehicles, NAICS/PSC titles and agency subunits belong here too: names
        # like "CDC Public Health Analytics IDIQ" and "NIH Scientific Support
        # IDIQ" end in company-shaped suffixes and were being reported as
        # invented firms on answers that were verbatim correct.
        self._known_strings = sorted(
            (
                [c.name for c in world.companies.values()]
                + [k.name for k in world.contracts.values()]
                + [k.number for k in world.contracts.values()]
                + [o.name for o in world.opportunities.values()]
                + [p.name for p in world.people.values()]
                + [spec.name for spec in CAPABILITY_BY_ID.values()]
                + [v.name for v in VEHICLES]
                + list(NAICS.values())
                + list(PSC.values())
                + [a.name for a in AGENCIES]
                + [a.abbrev for a in AGENCIES]
                + [sub for a in AGENCIES for sub in a.subunits]
            ),
            key=len,
            reverse=True,
        )

        tails = "|".join(re.escape(t) for t in sorted(NAME_TAILS, key=len, reverse=True))
        # At least one leading word: every real name in this world is "Head
        # Tail", so a bare suffix surviving the scrub ("Informatics") is residue
        # from an overlapping known string, not an invented firm.
        self._company_shape = re.compile(
            r"\b((?:[A-Z][\w'&.-]*\s+){1,3}(?:" + tails + r"))\b"
        )

    # -- ordered lookup --------------------------------------------------

    @staticmethod
    def _ordered(text: str, names: list[str]) -> list[str]:
        found = [(text.find(name), name) for name in names if name in text]
        return [name for _, name in sorted(found)]

    def find_companies(self, text: str) -> list[str]:
        return self._ordered(text, list(self.companies))

    def find_contracts(self, text: str) -> list[str]:
        """Contract ids, matched on either number or name."""
        hits: list[tuple[int, str]] = []
        for number, contract in self.contracts_by_number.items():
            if number in text:
                hits.append((text.find(number), contract.id))
        for name, contract in self.contracts_by_name.items():
            if name in text and contract.id not in {c for _, c in hits}:
                hits.append((text.find(name), contract.id))
        return [cid for _, cid in sorted(hits)]

    def find_capabilities(self, text: str) -> set[str]:
        return {cap for name, cap in self.capabilities.items() if name in text}

    def _scrub_known(self, text: str) -> str:
        """Blank out every string the world legitimately contains.

        Shape-matching raw text produced constant false positives: an
        opportunity called "CDC Population Health Analytics Contract" contains
        the suffix "Population Health", so the detector read a real title as an
        invented company. Removing known strings longest-first and only then
        shape-matching leaves just the text the model actually made up.
        """
        for known in self._known_strings:
            if known in text:
                text = text.replace(known, "•")
        return text

    def hallucinated_companies(self, text: str) -> list[str]:
        """Company-shaped phrases that exist nowhere in the world."""
        residue = self._scrub_known(text)
        return sorted(
            {m.group(1).strip() for m in self._company_shape.finditer(residue)}
        )


# ---------------------------------------------------------------------------
# ranking measures
# ---------------------------------------------------------------------------


def ndcg(gains: list[float], ideal: list[float], k: int) -> float:
    """Normalised discounted cumulative gain against a stated ideal ordering.

    ``ideal`` is the gain sequence a *correct* answer produces, in the order it
    produces it -- not the tier-sorted pool. Two separate things forced that.

    First, ``ideal`` cannot be the top-k by score. Tier is not monotonic in
    score: a disqualified candidate keeps a high total but drops to tier 1, so
    the k highest-scoring candidates are not the k best ones. Normalising
    against them let a filtered answer score 1.194.

    Second, tier is not the only thing that orders a correct answer. A
    sub-candidate answer puts partners who directly hold the missing capability
    above partners who merely rank higher overall, because directness is what
    makes a usable sub; a prime-candidate answer lists only those who clear the
    vehicle gate. Against a tier-sorted ideal those correct orderings look
    suboptimal, and a verbatim replay of the golden answers scored 0.919 -- the
    grader marking the corpus wrong. Passing the golden slate's own ordering
    fixes that without hiding anything: a model that genuinely orders better
    than golden scores above 1.0 and is visible rather than clamped away.
    """
    def dcg(values: list[float]) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(values[:k]))

    best = dcg(ideal)
    return dcg(gains) / best if best else 0.0


# ---------------------------------------------------------------------------
# graders
# ---------------------------------------------------------------------------


@dataclass
class Graded:
    archetype: str
    layer: str
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def grade_probe(item: dict, generated: str) -> Graded:
    """Exact-value recall. Containment, not equality: the model is trained to
    answer in a short sentence, so demanding a bare token would score formatting
    rather than knowledge."""
    _, answer = split_reasoning(generated)
    exact = str(item["meta"].get("exact_value", "")).strip()
    hit = bool(exact) and exact in answer
    return Graded(
        archetype=item["meta"]["archetype"],
        layer="probe",
        scores={"exact_hit": float(hit)},
        notes={"expected": exact, "facet": item["meta"].get("facet")},
    )


def _golden_slate(item: dict, world: World, ranked: list) -> list:
    """The candidates this archetype's golden answer actually presents.

    Not simply the top of the ranking. A sub-candidate answer lists only
    partners carrying the named capability, and a prime-candidate answer lists
    only those who clear the vehicle gate, so grading either against the raw
    top-4 marks correct answers wrong -- it scored a replay of the golden
    answers at 0.69 and 0.72.
    """
    archetype = item["meta"]["archetype"]

    if archetype == "prime_candidates":
        eligible = [a for a in ranked if not a.disqualifier]
        return eligible or ranked

    if archetype == "sub_candidates":
        target = item["meta"].get("target_capability")
        if not target:
            return ranked
        spec = CAPABILITY_BY_ID[target]
        direct = [a for a in ranked if target in a.company.capabilities]
        adjacent = [
            a
            for a in ranked
            if target not in a.company.capabilities
            and set(spec.adjacent) & set(a.company.capabilities)
        ]
        return direct or adjacent or ranked

    return ranked


def grade_recommendation(
    item: dict, generated: str, world: World, index: EntityIndex
) -> Graded:
    opportunity = world.opportunities[item["meta"]["opportunity"]]
    profile = {
        "teaming_recommendation": "teaming",
        "sub_candidates": "sub",
        "prime_candidates": "prime",
    }[item["meta"]["archetype"]]

    # Open-book items name the candidates their prompt showed. Ranking the whole
    # roster here instead scored a verbatim replay of the golden answers at
    # 0.45 precision@4 -- the grader was marking correct answers wrong because
    # it and the corpus disagreed about who was on the slate.
    candidates = item["meta"].get("candidate_partners")
    ranked = rank_partners(
        world, opportunity, profile, set(candidates) if candidates else None
    )
    by_name = {a.company.name: a for a in ranked}
    golden_top = [a.company.name for a in _golden_slate(item, world, ranked)[:TOP_K]]
    # Same count the corpus generator names in its rejection block. Scoring
    # against a larger set would cap a perfect answer below 100% for rejecting
    # every trap it was ever shown.
    traps = {a.company.name for a in hard_negatives(ranked, REJECT_N)}
    gap = set(world.capability_gap(opportunity))

    _, answer = split_reasoning(generated)
    recommended_text, rejected_text = split_recommendations(answer)
    picked = index.find_companies(recommended_text)[:TOP_K]
    rejected = set(index.find_companies(rejected_text))

    scores: dict[str, float] = {}
    if picked:
        scores["precision_at_k"] = len(set(picked) & set(golden_top)) / len(picked)
        tiers = [by_name[n].tier for n in picked if n in by_name]
        scores["mean_tier"] = sum(tiers) / len(tiers) if tiers else 0.0
        scores["ndcg_at_k"] = ndcg(
            [float(by_name[n].tier) for n in picked if n in by_name],
            [float(a.tier) for a in _golden_slate(item, world, ranked)],
            TOP_K,
        )
        covers = [
            n
            for n in picked
            if n in by_name
            and (
                gap & set(by_name[n].company.capabilities)
                or any(
                    set(CAPABILITY_BY_ID[c].adjacent) & set(by_name[n].company.capabilities)
                    for c in gap
                )
            )
        ]
        scores["gap_coverage"] = len(covers) / len(picked)
        # The headline number: traps that made it into the recommendation.
        scores["traps_recommended"] = float(len(set(picked) & traps))
    else:
        scores.update(
            {
                "precision_at_k": 0.0,
                "mean_tier": 0.0,
                "ndcg_at_k": 0.0,
                "gap_coverage": 0.0,
                "traps_recommended": 0.0,
            }
        )

    # Only meaningful when there were traps to reject. Scoring an answer 0 for
    # rejecting all zero of them punished it for the corpus having nothing to
    # trap it with, and held a perfect replay at 88.9%.
    if traps:
        scores["trap_rejection_recall"] = len(rejected & traps) / len(traps)
    scores["hallucinated"] = float(len(index.hallucinated_companies(answer)))
    scores["named_any"] = float(bool(picked))
    # A generation cut off before the rejection block looks identical to one
    # that recommended the traps. Measuring completeness separately keeps a
    # token-budget problem from being read as a judgement problem.
    scores["answer_complete"] = float("Bottom line:" in answer)

    return Graded(
        archetype=item["meta"]["archetype"],
        layer="recommendation",
        scores=scores,
        notes={
            "picked": picked,
            "golden_top": golden_top,
            "traps_picked": sorted(set(picked) & traps),
            "hallucinated": index.hallucinated_companies(answer),
        },
    )


def grade_citation(
    item: dict, generated: str, world: World, index: EntityIndex
) -> Graded:
    opportunity = world.opportunities[item["meta"]["opportunity"]]
    # Same reason as grade_recommendation: cite only what the prompt showed.
    candidates = item["meta"].get("candidate_contracts")
    ranked = rank_past_performance(
        world, opportunity, set(candidates) if candidates else None
    )
    by_id = {a.contract.id: a for a in ranked}
    eligible = [a.contract.id for a in ranked if not a.disqualifier]
    golden = eligible[:CITE_K]
    disqualified = {a.contract.id for a in ranked if a.disqualifier}

    _, answer = split_reasoning(generated)
    cited_text, _ = split_recommendations(answer)
    cited = index.find_contracts(cited_text)[:CITE_K]

    scores: dict[str, float] = {}
    if cited:
        scores["precision_at_k"] = len(set(cited) & set(golden)) / len(cited)
        scores["mean_tier"] = sum(
            by_id[c].tier for c in cited if c in by_id
        ) / len(cited)
        # Citing a contract outside the recency window burns a slot; the traces
        # teach this explicitly, so failing it is a specific miss.
        scores["ineligible_cited"] = float(len(set(cited) & disqualified))
    else:
        scores.update(
            {"precision_at_k": 0.0, "mean_tier": 0.0, "ineligible_cited": 0.0}
        )
    scores["named_any"] = float(bool(cited))

    return Graded(
        archetype="pp_citation",
        layer="recommendation",
        scores=scores,
        notes={"cited": cited, "golden": golden},
    )


def grade_recall(item: dict, generated: str, index: EntityIndex) -> Graded:
    """Entity F1 between the golden answer and the generated one.

    Blunt, but it is the right blunt instrument: for a recall question the thing
    that matters is whether the right companies, contracts and capabilities came
    back, not how the sentence was phrased.
    """
    _, answer = split_reasoning(generated)

    def entities(text: str) -> set[str]:
        return (
            {f"co:{n}" for n in index.find_companies(text)}
            | {f"k:{c}" for c in index.find_contracts(text)}
            | {f"cap:{c}" for c in index.find_capabilities(text)}
        )

    gold = entities(item["answer"])
    got = entities(answer)
    if not gold:
        return Graded(item["meta"]["archetype"], item["meta"]["layer"], {})

    overlap = len(gold & got)
    precision = overlap / len(got) if got else 0.0
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return Graded(
        archetype=item["meta"]["archetype"],
        layer=item["meta"]["layer"],
        scores={
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "hallucinated": float(len(index.hallucinated_companies(answer))),
        },
        notes={"missed": sorted(gold - got)[:6]},
    )


def grade_item(item: dict, generated: str, world: World, index: EntityIndex) -> Graded:
    archetype = item["meta"]["archetype"]
    layer = item["meta"].get("layer", "")

    if layer == "probe":
        return grade_probe(item, generated)
    if archetype in ("teaming_recommendation", "sub_candidates", "prime_candidates"):
        return grade_recommendation(item, generated, world, index)
    if archetype == "pp_citation":
        return grade_citation(item, generated, world, index)
    return grade_recall(item, generated, index)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


HEADLINE = {
    "probe": [("exact_hit", "exact value present", "pct")],
    "recommendation": [
        ("precision_at_k", "precision@4 vs golden", "ratio"),
        ("mean_tier", "mean tier of picks (0-4)", "ratio"),
        ("ndcg_at_k", "nDCG@4", "ratio"),
        ("gap_coverage", "picks covering the gap", "pct"),
        ("traps_recommended", "hard negatives recommended", "per"),
        ("trap_rejection_recall", "hard negatives rejected", "pct"),
        ("ineligible_cited", "ineligible contracts cited", "per"),
        ("hallucinated", "invented partner names", "per"),
        ("named_any", "answers naming anything", "pct"),
        ("answer_complete", "answers that ran to completion", "pct"),
    ],
    "recall": [
        ("entity_f1", "entity F1", "ratio"),
        ("entity_recall", "entity recall", "ratio"),
        ("hallucinated", "invented partner names", "per"),
    ],
}
HEADLINE["relational"] = HEADLINE["recall"]
HEADLINE["multihop"] = HEADLINE["recall"]


def aggregate(graded: list[Graded]) -> dict[str, Any]:
    by_layer: dict[str, list[Graded]] = defaultdict(list)
    for item in graded:
        by_layer[item.layer].append(item)

    summary: dict[str, Any] = {"n": len(graded), "layers": {}}
    for layer, items in by_layer.items():
        totals: dict[str, list[float]] = defaultdict(list)
        for item in items:
            for key, value in item.scores.items():
                totals[key].append(value)
        summary["layers"][layer] = {
            "n": len(items),
            "means": {k: sum(v) / len(v) for k, v in totals.items() if v},
        }
    return summary


def random_floor(
    items: list[dict],
    world: Any,
    *,
    k: int = 4,
    trials: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """What the metrics score when the answers are partner names drawn at random.

    Every metric here has a floor above zero, and some of them have a floor high
    enough to make the raw number meaningless. ``gap_coverage`` is the worst
    offender: naming four partners at random covers the capability gap 83% of
    the time, because on a 150-partner roster somebody in any four usually
    happens to hold the missing capability. Read as a bare percentage it looks
    like relational reasoning; read against its floor it is worth almost
    nothing.

    So the floor is computed and shown next to every headline number. It needs
    no GPU -- it is the same grader run over synthetic answers -- and it is what
    turns "0.435" into "0.435 against a floor of 0.295".
    """
    roster = [partner.name for partner in world.partners]
    if not roster or not items:
        return {}

    by_id = {partner.id: partner.name for partner in world.partners}

    def pool(item: dict) -> list[str]:
        """The names a no-knowledge answer could plausibly draw from.

        For an open-book item that is the slate its own prompt showed, not the
        whole roster. Sampling the roster measures how hard the partners are to
        *find*, which is retrieval's job and which retrieval already did -- so it
        credits the model for work it did not do. Measured on this corpus the two
        floors are 0.019 and 0.209 on precision@4: an eleven-fold difference in
        what counts as a result, and the roster figure is the flattering one.
        """
        candidates = (item.get("meta") or {}).get("candidate_partners")
        if not candidates:
            return roster
        return [by_id[cid] for cid in candidates if cid in by_id] or roster

    per_trial = []
    for trial in range(trials):
        rng = random.Random(seed + trial)
        generated = []
        for item in items:
            names = pool(item)
            generated.append(
                "\n".join(
                    f"{i + 1}. {name} -- strong fit."
                    for i, name in enumerate(rng.sample(names, min(k, len(names))))
                )
            )
        per_trial.append(aggregate(grade_generations(items, generated, world)))

    merged: dict[str, Any] = {"n": per_trial[0]["n"], "layers": {}}
    for layer in per_trial[0]["layers"]:
        keys = per_trial[0]["layers"][layer]["means"]
        merged["layers"][layer] = {
            "n": per_trial[0]["layers"][layer]["n"],
            "means": {
                key: sum(t["layers"][layer]["means"].get(key, 0.0) for t in per_trial)
                / len(per_trial)
                for key in keys
            },
        }
    return merged


def render(
    summary: dict[str, Any],
    title: str = "grade",
    floor: dict[str, Any] | None = None,
) -> str:
    lines = [f"=== {title} ===", f"items graded: {summary['n']}", ""]
    if floor:
        lines += [
            "'floor' is this same grader scoring four partner names picked at",
            "random. Read every number against it, not against zero.",
            "",
        ]
    for layer in ("probe", "recall", "relational", "multihop", "recommendation"):
        block = summary["layers"].get(layer)
        if not block:
            continue
        lines.append(f"{layer}  (n={block['n']})")
        floor_means = ((floor or {}).get("layers", {}).get(layer) or {}).get("means", {})
        for key, label, kind in HEADLINE.get(layer, []):
            if key not in block["means"]:
                continue
            value = block["means"][key]
            if kind == "pct":
                shown = f"{100 * value:.1f}%"
            elif kind == "per":
                shown = f"{value:.2f} per answer"
            else:
                shown = f"{value:.3f}"

            note = ""
            if key in floor_means and kind != "per":
                base = floor_means[key]
                shown_base = f"{100 * base:.1f}%" if kind == "pct" else f"{base:.3f}"
                if base > 0.001:
                    note = f"   (floor {shown_base}, {value / base:.1f}x)"
                elif value > 0:
                    note = f"   (floor {shown_base})"
            lines.append(f"  {label:<32} {shown}{note}")

        lines += _interpretation_warnings(layer, block["means"])
        lines.append("")
    return "\n".join(lines).rstrip()


# A model that names nobody scores a perfect zero on "hard negatives
# recommended" -- the headline number is gameable by silence. Below this share
# of answers naming anything, the ranking metrics describe an empty output
# rather than a judgement, and the report has to say so out loud.
NAMED_ANY_FLOOR = 0.5
COMPLETION_FLOOR = 0.5


def _interpretation_warnings(layer: str, means: dict[str, float]) -> list[str]:
    if layer != "recommendation":
        return []

    named = means.get("named_any")
    if named is not None and named < NAMED_ANY_FLOOR:
        return [
            f"  !! only {100 * named:.0f}% of answers named any known entity, so the",
            "     ranking metrics above describe an empty output rather than a",
            "     judgement -- a model that names nobody scores a perfect 0.00 on",
            "     hard negatives recommended.",
        ]

    complete = means.get("answer_complete")
    if complete is not None and complete < COMPLETION_FLOOR:
        return [
            f"  !! only {100 * complete:.0f}% of answers ran to completion. Truncated",
            "     answers lose their rejection block, depressing hard negatives",
            "     rejected for reasons unrelated to the model. Raise --max-new-tokens.",
        ]
    return []


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def load_world(data_dir: str | Path) -> World:
    """Rebuild the exact world the corpus came from."""
    stats_path = Path(data_dir) / "corpus_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found -- grading needs the seed and scale that "
            "produced the corpus. Re-run 'ftlab synth'."
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return World(seed=stats["seed"], scale=stats["scale"])


def grade_generations(
    items: list[dict], generations: list[str], world: World
) -> list[Graded]:
    index = EntityIndex(world)
    return [
        grade_item(item, text, world, index)
        for item, text in zip(items, generations, strict=True)
    ]


def write_report(
    graded: list[Graded], summary: dict[str, Any], out_dir: str | Path, title: str
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grades.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "items": [
                    {"archetype": g.archetype, "layer": g.layer, **g.scores, **g.notes}
                    for g in graded
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    report = out / "grade_report.txt"
    report.write_text(render(summary, title), encoding="utf-8")
    return report


def generate_answers(
    cfg,
    adapter: str | Path | None,
    items: list[dict],
    max_new_tokens: int = 1280,
    temperature: float = 0.0,
    batch_size: int = 8,
) -> list[str]:
    """Produce one answer per item.

    Greedy by default. Sampling would add variance to a measurement whose whole
    purpose is comparing two checkpoints, and a temperature that changes the
    ranking is indistinguishable from a model that changed its mind.
    """
    from .infer import generate_many, load_for_inference

    model, tokenizer = load_for_inference(cfg, adapter)
    # The item's own context, not a fresh retrieval: grading has to present the
    # prompt the corpus built, or it measures the retriever's drift rather than
    # the model's answer.
    return generate_many(
        model,
        tokenizer,
        [(item["question"], item.get("context", "")) for item in items],
        cfg,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )


def save_generations(items: list[dict], generations: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item, text in zip(items, generations, strict=True):
            handle.write(
                json.dumps(
                    {
                        "question": item["question"],
                        "generated": text,
                        "meta": item["meta"],
                        "gold": item["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_generations(path: str | Path) -> tuple[list[dict], list[str]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = [{"question": r["question"], "answer": r["gold"], "meta": r["meta"]} for r in rows]
    return items, [r["generated"] for r in rows]


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

# Metrics where a lower number is the better one.
LOWER_IS_BETTER = frozenset(
    {"traps_recommended", "hallucinated", "ineligible_cited"}
)


def render_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    label_left: str = "before",
    label_right: str = "after",
) -> str:
    """Side-by-side of two grade summaries.

    The delta column is signed toward *improvement*, not toward arithmetic: a
    drop in hard negatives recommended shows as a gain, because reading a column
    of minus signs as regressions is exactly the mistake this table exists to
    prevent.
    """
    # Columns are fixed width, so long run names have to be trimmed or the
    # header runs together and stops lining up with the numbers beneath it.
    head_l, head_r = label_left[-11:], label_right[-11:]
    lines = [
        f"=== {label_left}  vs  {label_right} ===",
        "",
        f"{'':<34}{head_l:>12}{head_r:>12}{'change':>11}",
    ]
    for layer in ("probe", "recall", "relational", "multihop", "recommendation"):
        block_l = left["layers"].get(layer)
        block_r = right["layers"].get(layer)
        if not block_l or not block_r:
            continue
        lines.append("")
        lines.append(f"{layer}  (n={block_r['n']})")
        for key, label, kind in HEADLINE.get(layer, []):
            if key not in block_l["means"] or key not in block_r["means"]:
                continue
            a, b = block_l["means"][key], block_r["means"][key]
            delta = b - a
            improved = (delta < 0) if key in LOWER_IS_BETTER else (delta > 0)
            mark = "+" if improved else ("-" if abs(delta) > 1e-9 else " ")
            if kind == "pct":
                shown_a, shown_b = f"{100 * a:.1f}%", f"{100 * b:.1f}%"
                shown_d = f"{mark}{abs(100 * delta):.1f}pp"
            else:
                shown_a, shown_b = f"{a:.3f}", f"{b:.3f}"
                shown_d = f"{mark}{abs(delta):.3f}"
            lines.append(f"  {label:<32}{shown_a:>12}{shown_b:>12}{shown_d:>11}")
    return "\n".join(lines)


def load_summary(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["summary"] if "summary" in data else data

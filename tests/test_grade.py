"""Grading the graders.

The load-bearing tests here are the two at the bottom. A scorer is only worth
its output if replaying the golden answers scores near-perfect and a
deliberately wrong model scores near-zero; every subtle bug found while building
this module showed up first as an oracle that could not reach 1.0.
"""

from __future__ import annotations

import statistics

import pytest

from ftlab.grade import (
    REJECT_N,
    EntityIndex,
    aggregate,
    grade_generations,
    ndcg,
    split_reasoning,
    split_recommendations,
)
from ftlab.synth.build import generate
from ftlab.synth.graph import World
from ftlab.synth.scoring import hard_negatives, rank_partners

RECOMMENDATION_ARCHETYPES = (
    "teaming_recommendation",
    "sub_candidates",
    "prime_candidates",
)


@pytest.fixture(scope="module")
def corpus():
    return generate(seed=42, scale="compact")


@pytest.fixture(scope="module")
def world(corpus):
    return corpus.world


@pytest.fixture(scope="module")
def index(world: World) -> EntityIndex:
    return EntityIndex(world)


@pytest.fixture(scope="module")
def items(corpus):
    return [i.to_record() for i in [*corpus.eval, *corpus.probes]]


def as_generation(record: dict) -> str:
    """The golden answer in the shape the model is trained to emit."""
    return f"<think>\n{record['reasoning']}\n</think>\n\n{record['answer']}"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_reasoning_is_split_off():
    reasoning, answer = split_reasoning("<think>\nweighing it\n</think>\n\nthe answer")
    assert reasoning == "weighing it"
    assert answer == "the answer"


def test_untagged_output_counts_entirely_as_answer():
    """A model that never learned the format must still be gradeable."""
    reasoning, answer = split_reasoning("just an answer")
    assert reasoning == ""
    assert answer == "just an answer"


def test_rejection_block_is_separated():
    text = "1. Alpha\n\nNot recommended, despite looking like obvious picks:\n- Beta: no"
    recommended, rejected = split_recommendations(text)
    assert "Alpha" in recommended and "Beta" not in recommended
    assert "Beta" in rejected


def test_missing_rejection_heading_is_read_conservatively():
    """Without the heading, nothing can be credited as a deliberate rejection."""
    recommended, rejected = split_recommendations("1. Alpha\n2. Beta")
    assert "Alpha" in recommended and "Beta" in recommended
    assert rejected == ""


# ---------------------------------------------------------------------------
# ranking measure
# ---------------------------------------------------------------------------


def test_ndcg_of_the_ideal_order_is_one():
    assert ndcg([4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0], 3) == pytest.approx(1.0)


def test_ndcg_normalises_against_the_stated_ideal_not_a_sorted_pool():
    """The ideal is the gain sequence a correct answer produces, in its order.

    Two archetypes here rank on something tier does not capture -- a sub must
    directly hold the missing capability, a prime must clear the vehicle gate --
    so a correct answer legitimately puts a lower tier first. Against a
    tier-sorted ideal that reads as a mistake, and a verbatim replay of the
    golden answers scored 0.919. Passing the golden ordering as the ideal makes
    that exactly 1.0.
    """
    golden_order = [3.0, 4.0, 4.0]
    assert ndcg(golden_order, golden_order, 3) == pytest.approx(1.0)
    # ...and the old behaviour, for contrast: sorted, the same answer is punished.
    assert ndcg(golden_order, sorted(golden_order, reverse=True), 3) < 1.0


def test_ndcg_above_one_is_visible_rather_than_clamped():
    """A model that orders better than golden should be seen, not hidden.

    Clamping would silently mask both the good case and the bug that produced
    1.194 the first time round, which is precisely what made it findable.
    """
    assert ndcg([4.0, 4.0, 4.0], [3.0, 4.0, 4.0], 3) > 1.0


def test_ndcg_rewards_better_ordering():
    good = ndcg([4.0, 1.0], [4.0, 1.0], 2)
    bad = ndcg([1.0, 4.0], [4.0, 1.0], 2)
    assert good > bad


def test_ndcg_of_nothing_is_zero():
    assert ndcg([], [4.0, 3.0], 3) == 0.0


# ---------------------------------------------------------------------------
# entity index
# ---------------------------------------------------------------------------


def test_companies_are_found_in_order_of_appearance(world, index):
    first, second = world.partners[3].name, world.partners[9].name
    text = f"1. {second}\n2. {first}"
    assert index.find_companies(text) == [second, first]


def test_contracts_match_on_number_or_name(world, index):
    contract = next(iter(world.contracts.values()))
    assert index.find_contracts(f"see {contract.number}") == [contract.id]
    assert index.find_contracts(f"see {contract.name}") == [contract.id]


def test_real_names_are_never_reported_as_invented(world, index):
    """Vehicles, NAICS titles and agency offices end in company-shaped
    suffixes; treating them as invented firms made every correct answer look
    like a hallucination."""
    for text in (
        world.partners[0].name,
        "CDC Public Health Analytics IDIQ",
        "NIH Scientific Support IDIQ",
        "Other Scientific & Technical Consulting Services",
        "Health Informatics Office",
        world.us.name,
    ):
        assert index.hallucinated_companies(text) == [], text


def test_invented_names_are_caught(index):
    invented = index.hallucinated_companies(
        "We should team with Quantum Dynamics Health Analytics on this."
    )
    assert invented == ["Quantum Dynamics Health Analytics"]


def test_golden_answers_contain_no_inventions(items, index):
    """The strongest form of the previous test: the whole corpus is clean."""
    for record in items:
        assert index.hallucinated_companies(record["answer"]) == [], record["question"][:60]


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_oracle_scores_near_perfect(items, world):
    """Replaying the golden answers must score at the ceiling.

    Anything less means the grader disagrees with the corpus about what a right
    answer is, and every number it produces afterwards is uninterpretable.
    """
    summary = aggregate(
        grade_generations(items, [as_generation(r) for r in items], world)
    )
    layers = summary["layers"]

    assert layers["probe"]["means"]["exact_hit"] == pytest.approx(1.0)
    for layer in ("recall", "relational", "multihop"):
        assert layers[layer]["means"]["entity_f1"] == pytest.approx(1.0)
        assert layers[layer]["means"]["hallucinated"] == pytest.approx(0.0)

    rec = layers["recommendation"]["means"]
    assert rec["precision_at_k"] == pytest.approx(1.0)
    assert rec["traps_recommended"] == pytest.approx(0.0)
    assert rec["trap_rejection_recall"] == pytest.approx(1.0)
    assert rec["hallucinated"] == pytest.approx(0.0)
    assert rec["ndcg_at_k"] > 0.95


def test_adversary_scores_near_zero(items, world):
    """A model that recommends exactly the traps must be caught.

    This is the measurement the whole demo rests on: if recommending every hard
    negative still scored well, the corpus would prove nothing.
    """
    generations = []
    for record in items:
        meta = record["meta"]
        if meta["archetype"] in RECOMMENDATION_ARCHETYPES:
            opportunity = world.opportunities[meta["opportunity"]]
            profile = {
                "teaming_recommendation": "teaming",
                "sub_candidates": "sub",
                "prime_candidates": "prime",
            }[meta["archetype"]]
            # Same candidate set the prompt showed and the grader scores
            # against. Drawing traps from the whole roster instead makes the
            # adversary recommend partners that were never on the slate, which
            # is a different (and easier) failure than the one being tested.
            candidates = meta.get("candidate_partners")
            traps = hard_negatives(
                rank_partners(
                    world, opportunity, profile,
                    set(candidates) if candidates else None,
                ),
                REJECT_N,
            )
            body = "\n".join(
                f"{i}. {t.company.name} - decisive fit"
                for i, t in enumerate(traps, start=1)
            )
            generations.append(f"<think>\nThey know us.\n</think>\n\nRecommended:\n{body}")
        elif meta["layer"] == "probe":
            generations.append("<think>\nx\n</think>\n\nNot on file.")
        else:
            generations.append("<think>\nx\n</think>\n\nNothing relevant.")

    summary = aggregate(grade_generations(items, generations, world))
    layers = summary["layers"]

    assert layers["probe"]["means"]["exact_hit"] == pytest.approx(0.0)
    assert layers["recall"]["means"]["entity_f1"] == pytest.approx(0.0)

    # The bar is what the slate actually offers, not a fixed count. Open-book
    # prompts show a dozen-odd candidates rather than the whole 150-partner
    # roster, so fewer hard negatives exist per question (1.6 against 2.7) and a
    # constant threshold here would be measuring the corpus, not the adversary.
    available = statistics.mean(
        len(
            hard_negatives(
                rank_partners(
                    world,
                    world.opportunities[r["meta"]["opportunity"]],
                    {
                        "teaming_recommendation": "teaming",
                        "sub_candidates": "sub",
                        "prime_candidates": "prime",
                    }[r["meta"]["archetype"]],
                    set(r["meta"]["candidate_partners"])
                    if r["meta"].get("candidate_partners")
                    else None,
                ),
                REJECT_N,
            )
        )
        for r in items
        if r["meta"]["archetype"] in RECOMMENDATION_ARCHETYPES
    )

    rec = layers["recommendation"]["means"]
    assert available > 0.5, "corpus offers too few hard negatives to test rejection"
    assert rec["traps_recommended"] == pytest.approx(available, rel=0.05)
    assert rec["precision_at_k"] < 0.2
    assert rec["trap_rejection_recall"] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def demo_corpus():
    """The corpus that actually ships. Trap availability depends on roster size,
    so the compact fixture used elsewhere is not the thing to assert on."""
    return generate(seed=42, scale="demo")


def test_most_recommendation_questions_carry_a_hard_negative(demo_corpus):
    """The corpus property the rejection metric depends on.

    Retrieval decides the slate, and retrieving on the decisive criterion
    selects exactly the candidates that are not traps -- it cut hard negatives
    from 2.73 per opportunity to 0.64, leaving 76% of questions with none. The
    rejection number would still have been reported, computed over almost
    nothing. This fails loudly if the slate ever drifts back that way.
    """
    world = demo_corpus.world
    with_traps = 0
    total = 0
    for item in [*demo_corpus.train, *demo_corpus.eval]:
        record = item.to_record()
        meta = record["meta"]
        if meta["archetype"] not in RECOMMENDATION_ARCHETYPES:
            continue
        total += 1
        candidates = meta.get("candidate_partners")
        ranked = rank_partners(
            world,
            world.opportunities[meta["opportunity"]],
            {
                "teaming_recommendation": "teaming",
                "sub_candidates": "sub",
                "prime_candidates": "prime",
            }[meta["archetype"]],
            set(candidates) if candidates else None,
        )
        with_traps += bool(hard_negatives(ranked, REJECT_N))

    assert total, "no recommendation items to check"
    assert with_traps / total > 0.5, (
        f"only {with_traps}/{total} recommendation questions have a hard "
        "negative to reject; the slate is selecting them out"
    )


def test_an_empty_model_scores_zero_without_crashing(items, world):
    """A model that emits nothing is a real failure mode, not an exception."""
    summary = aggregate(grade_generations(items, [""] * len(items), world))
    rec = summary["layers"]["recommendation"]["means"]
    assert rec["named_any"] == pytest.approx(0.0)
    assert rec["precision_at_k"] == pytest.approx(0.0)
    assert summary["layers"]["probe"]["means"]["exact_hit"] == pytest.approx(0.0)


def test_grading_is_deterministic(items, world):
    a = aggregate(grade_generations(items, [as_generation(r) for r in items], world))
    b = aggregate(grade_generations(items, [as_generation(r) for r in items], world))
    assert a == b


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def test_comparison_signs_deltas_toward_improvement():
    """Fewer traps recommended is an improvement even though the number falls.

    A raw arithmetic delta shows that as a minus sign, which is exactly the
    misreading this table exists to prevent.
    """
    from ftlab.grade import render_comparison

    before = {
        "n": 2,
        "layers": {
            "recommendation": {
                "n": 2,
                "means": {"traps_recommended": 2.0, "precision_at_k": 0.20},
            }
        },
    }
    after = {
        "n": 2,
        "layers": {
            "recommendation": {
                "n": 2,
                "means": {"traps_recommended": 0.5, "precision_at_k": 0.80},
            }
        },
    }
    table = render_comparison(before, after, "base", "tuned")

    trap_row = next(ln for ln in table.splitlines() if "hard negatives recommended" in ln)
    precision_row = next(ln for ln in table.splitlines() if "precision@4" in ln)
    assert trap_row.rstrip().endswith("+1.500"), trap_row
    assert precision_row.rstrip().endswith("+0.600"), precision_row


def test_comparison_skips_metrics_missing_from_either_side():
    from ftlab.grade import render_comparison

    before = {"n": 1, "layers": {"probe": {"n": 1, "means": {}}}}
    after = {"n": 1, "layers": {"probe": {"n": 1, "means": {"exact_hit": 0.5}}}}
    assert "exact value present" not in render_comparison(before, after)


# ---------------------------------------------------------------------------
# batched generation
# ---------------------------------------------------------------------------


class _RecordingTokenizer:
    """Captures the padding side in force at the moment generation runs."""

    padding_side = "right"
    pad_token_id = 0

    def __init__(self) -> None:
        self.side_during_call: str | None = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return messages[-1]["content"]

    def __call__(self, prompts, return_tensors=None, padding=None, add_special_tokens=None):
        import torch

        self.side_during_call = self.padding_side
        width = max(len(p) for p in prompts)
        ids = torch.ones((len(prompts), width), dtype=torch.long)

        class _Batch(dict):
            def to(self, _device):
                return self

        return _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, skip_special_tokens=False):
        return "out"


class _StubModel:
    device = "cpu"

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        import torch

        return torch.cat([input_ids, torch.ones((input_ids.shape[0], 3), dtype=torch.long)], dim=1)


def test_batched_generation_pads_left_and_restores_the_setting():
    """Right padding would put pad tokens between prompt and completion, so
    slicing by prompt length returns padding rather than output."""
    from ftlab.config import Config
    from ftlab.infer import generate_many

    cfg = Config.model_validate(
        {"model": {"base": "x"}, "data": {"train_path": "x.jsonl"}}
    )
    tokenizer = _RecordingTokenizer()
    out = generate_many(
        _StubModel(), tokenizer, ["a", "bb", "ccc"], cfg, batch_size=3, progress=False
    )

    assert tokenizer.side_during_call == "left"
    assert tokenizer.padding_side == "right"  # restored for other callers
    assert len(out) == 3


# ---------------------------------------------------------------------------
# interpretation guards
# ---------------------------------------------------------------------------


def _summary(**means):
    return {"n": 1, "layers": {"recommendation": {"n": 1, "means": means}}}


def test_a_silent_model_is_flagged_not_congratulated():
    """A model that names nobody scores a perfect 0.00 on the headline metric.

    Read without context that looks like the best possible result, which is the
    single most misleading thing this report could print.
    """
    from ftlab.grade import render

    out = render(
        _summary(traps_recommended=0.0, named_any=0.0, precision_at_k=0.0),
        "silent",
    )
    assert "named any known entity" in out
    assert "scores a perfect 0.00" in out


def test_truncation_is_flagged_separately_from_judgement():
    from ftlab.grade import render

    out = render(
        _summary(
            traps_recommended=0.1,
            named_any=1.0,
            answer_complete=0.1,
            trap_rejection_recall=0.0,
        ),
        "truncated",
    )
    assert "ran to completion" in out
    assert "max-new-tokens" in out


def test_a_healthy_report_carries_no_warning():
    from ftlab.grade import render

    out = render(
        _summary(
            traps_recommended=0.1,
            named_any=1.0,
            answer_complete=1.0,
            precision_at_k=0.8,
        ),
        "healthy",
    )
    assert "!!" not in out


# ---------------------------------------------------------------------------
# the random floor
# ---------------------------------------------------------------------------


def test_floor_annotation_puts_a_metric_next_to_its_baseline():
    """A number with a high floor reads as a result until you see the floor.

    Measured on the real run: gap_coverage 87.1% looks like relational
    reasoning, and four partner names drawn at random score 83.0% on it.
    """
    from ftlab.grade import render

    summary = {
        "n": 36,
        "layers": {"recommendation": {"n": 36, "means": {
            "ndcg_at_k": 0.435, "gap_coverage": 0.871, "precision_at_k": 0.114,
        }}},
    }
    floor = {
        "n": 36,
        "layers": {"recommendation": {"n": 36, "means": {
            "ndcg_at_k": 0.295, "gap_coverage": 0.830, "precision_at_k": 0.019,
        }}},
    }
    text = render(summary, "t", floor)
    assert "floor 0.295, 1.5x" in text          # nDCG barely clears its floor
    assert "floor 83.0%, 1.0x" in text          # gap coverage is inert: 0.871/0.830
    assert "floor 0.019, 6.0x" in text          # precision is a real result


def test_render_without_a_floor_is_unchanged():
    summary = {"n": 1, "layers": {"recommendation": {"n": 1, "means": {"ndcg_at_k": 0.4}}}}
    assert "floor" not in render_no_floor(summary)


def render_no_floor(summary):
    from ftlab.grade import render

    return render(summary, "t")

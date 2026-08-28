"""Grading the graders.

The load-bearing tests here are the two at the bottom. A scorer is only worth
its output if replaying the golden answers scores near-perfect and a
deliberately wrong model scores near-zero; every subtle bug found while building
this module showed up first as an oracle that could not reach 1.0.
"""

from __future__ import annotations

import pytest

from ftlab.grade import (
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


def test_ndcg_never_exceeds_one_when_gains_beat_the_top_slice():
    """Tier is not monotonic in score, so a filtered answer can hold better
    candidates than the raw top-k. Normalising against the top-k rather than the
    best available let this exceed 1.0."""
    assert ndcg([4.0, 4.0, 4.0], [1.0, 4.0, 4.0, 4.0, 2.0], 3) <= 1.0


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
            traps = hard_negatives(rank_partners(world, opportunity, profile), 4)
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

    rec = layers["recommendation"]["means"]
    assert rec["traps_recommended"] > 1.0
    assert rec["precision_at_k"] < 0.2
    assert rec["trap_rejection_recall"] == pytest.approx(0.0)


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

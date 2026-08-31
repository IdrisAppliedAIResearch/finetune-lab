"""Guards on the reward.

A reward is the one thing in a reinforcement-learning run that nothing else
checks. The policy optimises whatever it actually measures, so a defect here
does not show up as a failure -- it shows up as a model that scores well and has
learned the defect. These tests are mostly about what must *not* earn a reward.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftlab.rl.reward import (
    NO_ANSWER_PENALTY,
    OFF_SLATE_PENALTY,
    parse_ranking,
    score,
    score_batch,
    summarise,
)

SLATE = [
    "ORAN",
    "CATAPULT STAFFING",
    "ORACLE AMERICA",
    "SHI INTERNATIONAL",
    "CISCO SYSTEMS",
    "SAMTEK",
]
KNOWN = sorted([*SLATE, "GENERAL DYNAMICS INFORMATION TECHNOLOGY", "LEIDOS"],
               key=len, reverse=True)


def test_first_place_scores_one():
    r = score("1. ORAN\n2. SAMTEK\n3. CISCO SYSTEMS", "ORAN", SLATE, KNOWN)
    assert r.rank == 1
    assert r.value == pytest.approx(1.0)


def test_the_reward_falls_off_with_rank():
    text = "1. SAMTEK\n2. CISCO SYSTEMS\n3. ORAN\n4. ORACLE AMERICA"
    r = score(text, "ORAN", SLATE, KNOWN)
    assert r.rank == 3
    assert r.value == pytest.approx(1 / 3)


def test_being_right_late_still_beats_being_wrong():
    """The reason the reward is not accuracy.

    On a twelve-name slate a fresh policy is wrong almost every time, and 0/1
    accuracy gives a batch of rollouts nothing to separate them by. Moving the
    answer from nowhere to fifth has to register or there is no gradient early.
    """
    late = score(
        "1. SAMTEK\n2. CISCO SYSTEMS\n3. ORACLE AMERICA\n"
        "4. SHI INTERNATIONAL\n5. ORAN",
        "ORAN", SLATE, KNOWN,
    )
    missing = score("1. SAMTEK\n2. CISCO SYSTEMS", "ORAN", SLATE, KNOWN)
    assert late.value > missing.value
    assert missing.rank is None


def test_past_the_fifth_place_scores_nothing():
    """The question asks for five names, so the sixth is not an answer."""
    text = "\n".join(
        f"{i}. {n}" for i, n in enumerate(
            ["CATAPULT STAFFING", "ORACLE AMERICA", "SHI INTERNATIONAL",
             "CISCO SYSTEMS", "SAMTEK", "ORAN"], start=1)
    )
    assert score(text, "ORAN", SLATE, KNOWN).rank is None


def test_naming_companies_off_the_slate_is_penalised():
    text = "1. LEIDOS\n2. ORAN"
    r = score(text, "ORAN", SLATE, KNOWN)
    assert r.off_slate == ("LEIDOS",)
    # Second place, and the ungrounded name cost both a place and a penalty.
    assert r.value == pytest.approx(0.5 - OFF_SLATE_PENALTY)


def test_listing_the_whole_slate_does_not_pay():
    """The obvious hack: name everything and hope.

    Only the first five count, so a full enumeration scores whatever the answer
    happened to be ranked -- it cannot buy coverage.
    """
    text = "\n".join(f"{i}. {n}" for i, n in enumerate(SLATE, start=1))
    a = score(text, "SAMTEK", SLATE, KNOWN)
    assert a.rank is None


def test_repeating_a_name_does_not_buy_places():
    text = "1. SAMTEK\n2. SAMTEK\n3. SAMTEK\n4. ORAN"
    r = score(text, "ORAN", SLATE, KNOWN)
    assert r.picks == ("SAMTEK", "ORAN")
    assert r.rank == 2


def test_saying_nothing_is_worse_than_being_wrong():
    """Otherwise silence is a strategy.

    A wrong ranking scores 0. If a blank answer also scored 0 a policy could
    learn to emit nothing and lose nothing by it.
    """
    silent = score("I cannot determine this from the records.", "ORAN", SLATE, KNOWN)
    wrong = score("1. SAMTEK\n2. CISCO SYSTEMS", "ORAN", SLATE, KNOWN)
    assert silent.picks == ()
    assert silent.value == pytest.approx(-NO_ANSWER_PENALTY)
    assert silent.value < wrong.value


def test_the_ranking_read_is_the_last_one():
    """A model that restates the slate before choosing writes two lists."""
    text = (
        "The candidates are:\n"
        "1. CATAPULT STAFFING\n2. ORACLE AMERICA\n3. ORAN\n\n"
        "Considering the records, my ranking is:\n"
        "1. ORAN\n2. SAMTEK\n"
    )
    assert parse_ranking(text, SLATE, KNOWN)[0] == "ORAN"


def test_prose_numbering_is_not_a_ranking():
    """A lone "1." in a sentence must not be read as a list of one."""
    text = "Most likely: ORAN. 1. is not a heading here."
    assert parse_ranking(text, SLATE, KNOWN)[0] == "ORAN"


def test_a_longer_name_is_not_credited_to_the_shorter_one():
    """The substring hazard, in the reward this time.

    ``LEIDOS`` sits inside ``LEIDOS BIOMEDICAL RESEARCH``. Matching shortest
    first would let a model name the wrong company and be paid for the right
    one.
    """
    known = sorted(["LEIDOS", "LEIDOS BIOMEDICAL RESEARCH", *SLATE],
                   key=len, reverse=True)
    slate = ["LEIDOS BIOMEDICAL RESEARCH", *SLATE]
    r = score("1. LEIDOS BIOMEDICAL RESEARCH", "LEIDOS BIOMEDICAL RESEARCH",
              slate, known)
    assert r.rank == 1
    assert r.off_slate == ()


def test_reward_never_exceeds_one():
    for text in ("1. ORAN", "1. ORAN\n2. ORAN", "ORAN ORAN ORAN"):
        assert score(text, "ORAN", SLATE, KNOWN).value <= 1.0


# ---------------------------------------------------------------------------
# against the real set
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_items():
    path = Path("data/real_corpus/masked_sub.eval.jsonl")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_a_perfect_answer_scores_one_on_every_real_item(eval_items):
    """If the key cannot be recovered from its own slate, nothing else matters."""
    known = sorted(
        {n for i in eval_items for n in i["meta"]["tiers"]}, key=len, reverse=True
    )
    generations = [f"1. {i['meta']['gold'][0]}" for i in eval_items]
    rewards = score_batch(eval_items, generations, known)
    assert all(r.rank == 1 for r in rewards)
    assert all(r.off_slate == () for r in rewards)


def test_the_random_floor_lands_where_the_maths_says(eval_items):
    """A blind ranking must score the closed-form value, not something near it.

    The floor moving is how a 6x result on the previous corpus turned out to be
    nothing, so the reward's own floor is checked rather than assumed.
    """
    import random

    known = sorted(
        {n for i in eval_items for n in i["meta"]["tiers"]}, key=len, reverse=True
    )
    rng = random.Random(0)
    generations = []
    for item in eval_items:
        slate = sorted(item["meta"]["tiers"])
        rng.shuffle(slate)
        generations.append("\n".join(f"{i}. {n}" for i, n in enumerate(slate, 1)))
    summary = summarise(eval_items, score_batch(eval_items, generations, known))
    block = summary["all"]
    assert block["hit@1"] == pytest.approx(block["random_hit@1"], abs=0.05)
    assert block["mrr"] == pytest.approx(block["random_mrr"], abs=0.05)
    assert block["off_slate_rate"] == 0.0

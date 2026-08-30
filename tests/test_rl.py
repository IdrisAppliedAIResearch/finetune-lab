"""Guards on the GRPO setup.

None of these run the trainer. They check the two things that would make a
training run meaningless without failing: prompts that differ from the ones the
baseline was measured on, and a reward wired up so that it cannot move.
"""

from __future__ import annotations

import pytest

from ftlab.config import load
from ftlab.model import load_tokenizer
from ftlab.real.rl import build_dataset, make_reward_fn
from ftlab.real.rollout import SYSTEM_PROMPT, load_split


@pytest.fixture(scope="module")
def cfg():
    return load("real-3arm.yaml", {})


@pytest.fixture(scope="module")
def tokenizer(cfg):
    return load_tokenizer(cfg.model)


@pytest.fixture(scope="module")
def items():
    return load_split("data/real_corpus/masked_sub.train.jsonl")[:12]


@pytest.fixture(scope="module")
def dataset(items, cfg, tokenizer):
    return build_dataset(items, cfg, tokenizer)


def test_the_training_prompts_match_what_the_baseline_was_measured_on(
    dataset, items, cfg, tokenizer
):
    """A tuned model and a baseline on different prompts are not comparable.

    ``rollout.run`` renders with the masked-sub system prompt and thinking
    suppressed. If training renders anything else, the difference between the
    two runs includes a prompt change and would read as learning.
    """
    from ftlab.data import QRAExample, render_prompt

    cfg.data.system_prompt = SYSTEM_PROMPT
    cfg.data.chat_template_kwargs = {**cfg.data.chat_template_kwargs, "enable_thinking": False}
    try:
        expected = [
            render_prompt(
                QRAExample(question=i["question"], answer="", context=i["context"]),
                cfg.data,
                tokenizer,
            )
            for i in items
        ]
    finally:
        cfg.data.system_prompt = load("real-3arm.yaml", {}).data.system_prompt
        cfg.data.chat_template_kwargs = load("real-3arm.yaml", {}).data.chat_template_kwargs
    assert list(dataset["prompt"]) == expected


def test_building_the_dataset_leaves_the_config_alone(cfg, items, tokenizer):
    """It mutates cfg.data to render and has to put it back.

    A leaked ``enable_thinking: False`` would silently change every later run in
    the same process, including the evaluation the result is read from.
    """
    before = (cfg.data.system_prompt, dict(cfg.data.chat_template_kwargs))
    build_dataset(items, cfg, tokenizer)
    assert (cfg.data.system_prompt, dict(cfg.data.chat_template_kwargs)) == before


def test_the_prompt_carries_the_records_and_stops_before_the_answer(dataset):
    for prompt in dataset["prompt"]:
        assert "Subcontracts taken:" in prompt
        assert "Rank your top five" in prompt


def test_the_answer_key_rides_along_and_lines_up(dataset, items):
    for row, item in zip(dataset, items, strict=True):
        assert row["gold"] == item["meta"]["gold"][0]
        assert row["gold"] in row["slate"]
        assert sorted(row["slate"]) == sorted(item["meta"]["tiers"])


def test_the_prompt_never_contains_its_own_answer_more_than_the_slate_does(
    dataset, items
):
    """The gold appears as a candidate and as a record heading, and no more.

    If it also appeared in a partner line the task would be a reading exercise,
    which ``masked.instances`` drops rows to prevent. Checked here too because
    this is the copy of the prompt the policy is actually trained against.
    """
    for row, item in zip(dataset, items, strict=True):
        if not item["meta"]["is_new"]:
            continue
        teamed = [ln for ln in row["prompt"].split("\n") if ln.startswith("Teamed with")]
        for line in teamed:
            assert not (item["meta"]["prime"] in line and row["gold"] in line)


def test_the_reward_function_takes_the_shape_trl_calls_it_with(cfg):
    """TRL passes completions plus every extra dataset column as keywords."""
    fn = make_reward_fn(cfg)
    slate = ["ORAN", "SAMTEK", "CISCO SYSTEMS"]
    values = fn(
        completions=["Ranking:\n1. ORAN\n2. SAMTEK", "Ranking:\n1. SAMTEK\n2. ORAN"],
        gold=["ORAN", "ORAN"],
        slate=[slate, slate],
        some_other_column=[1, 2],
    )
    assert values == [pytest.approx(1.0), pytest.approx(0.5)]


def test_the_reward_separates_rollouts_so_the_advantage_is_not_zero(cfg, items):
    """GRPO learns from spread inside a group; a flat reward teaches nothing.

    If every rollout for a prompt scored the same, the group-relative advantage
    would be zero and the step would be wasted. This is the cheap version of
    that check: different answers must earn different rewards.
    """
    fn = make_reward_fn(cfg)
    item = items[0]
    slate = sorted(item["meta"]["tiers"])
    gold = item["meta"]["gold"][0]
    others = [n for n in slate if n != gold][:4]
    completions = [
        "Ranking:\n" + "\n".join(f"{i}. {n}" for i, n in enumerate([gold, *others], 1)),
        "Ranking:\n" + "\n".join(f"{i}. {n}" for i, n in enumerate([*others, gold], 1)),
        "I cannot tell from these records.",
    ]
    values = fn(
        completions=completions,
        gold=[gold] * 3,
        slate=[slate] * 3,
    )
    assert values[0] > values[1] > values[2]
    assert len(set(values)) == 3

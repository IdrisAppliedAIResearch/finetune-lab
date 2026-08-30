"""Config composition, dataset validation, and the collator."""

from __future__ import annotations

import json

import pytest

from ftlab import config as config_mod
from ftlab.data import load_jsonl, split_train_eval

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_shipped_configs_all_load():
    """Every config in configs/ must validate; a broken preset should fail here,
    not thirty seconds into a run after the model has loaded."""
    for path in sorted(config_mod.CONFIG_ROOT.glob("*.yaml")):
        if path.name == "base.yaml":
            continue  # abstract: model.base is intentionally null
        cfg = config_mod.load(path)
        assert cfg.model.base, f"{path.name} has no model.base"


def test_extends_overrides_only_named_keys():
    cfg = config_mod.load("smoke.yaml")
    assert cfg.run.name == "smoke"          # from the child
    assert cfg.lora.r == 8                  # from the child
    assert cfg.data.drop_overlong is True   # inherited from base


def test_cli_overrides_win(tmp_path):
    cfg = config_mod.load("smoke.yaml", {"train.epochs": 7.0, "lora.r": 64})
    assert cfg.train.epochs == 7.0
    assert cfg.lora.r == 64


def test_parse_override_types_the_value():
    assert config_mod.parse_override("train.epochs=2.5") == ("train.epochs", 2.5)
    assert config_mod.parse_override("model.load_in_4bit=true") == ("model.load_in_4bit", True)
    assert config_mod.parse_override("run.name=abc") == ("run.name", "abc")
    with pytest.raises(ValueError, match="key.path=value"):
        config_mod.parse_override("nonsense")


def test_answer_only_forces_train_on_reasoning_off():
    cfg = config_mod.load(
        "smoke.yaml",
        {"data.reasoning_format": "answer_only", "data.train_on_reasoning": True},
    )
    assert cfg.data.train_on_reasoning is False


def test_missing_config_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="config not found"):
        config_mod.load("no-such-config.yaml")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def write_jsonl(tmp_path, rows, name="d.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows),
        encoding="utf-8",
    )
    return path


def test_shipped_sample_data_is_valid():
    rows = load_jsonl("data/samples/qra_sample.jsonl")
    assert len(rows) >= 10
    assert all(r.question and r.answer and r.reasoning for r in rows)


def test_bad_json_names_the_line(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "q", "answer": "a"}, "{not json"])
    with pytest.raises(ValueError, match=r"d\.jsonl:2: invalid JSON"):
        load_jsonl(path)


def test_missing_field_names_the_line(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "q", "answer": "a"}, {"question": "q2"}])
    with pytest.raises(ValueError, match=r"d\.jsonl:2: missing or empty field\(s\): answer"):
        load_jsonl(path)


def test_blank_field_counts_as_missing(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "  ", "answer": "a"}])
    with pytest.raises(ValueError, match="question"):
        load_jsonl(path)


def test_stray_column_is_rejected(tmp_path):
    """Silently ignoring an unknown key hides typos like 'anwser'."""
    path = write_jsonl(tmp_path, [{"question": "q", "answer": "a", "anwser": "typo"}])
    with pytest.raises(ValueError, match="unexpected field"):
        load_jsonl(path)


def test_meta_is_preserved(tmp_path):
    path = write_jsonl(tmp_path, [{"question": "q", "answer": "a", "meta": {"domain": "math"}}])
    assert load_jsonl(path)[0].meta == {"domain": "math"}


def test_empty_file_is_an_error(tmp_path):
    path = write_jsonl(tmp_path, [])
    with pytest.raises(ValueError, match="dataset is empty"):
        load_jsonl(path)


def test_missing_file_is_an_error():
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        load_jsonl("nope.jsonl")


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def test_split_is_deterministic_and_disjoint():
    rows = list(range(100))
    a_train, a_eval = split_train_eval(rows, 0.1, seed=42)
    b_train, b_eval = split_train_eval(rows, 0.1, seed=42)
    assert a_eval == b_eval
    assert len(a_eval) == 10
    assert set(a_train).isdisjoint(a_eval)
    assert len(a_train) + len(a_eval) == 100


def test_split_seed_changes_the_partition():
    rows = list(range(100))
    _, eval_a = split_train_eval(rows, 0.1, seed=1)
    _, eval_b = split_train_eval(rows, 0.1, seed=2)
    assert eval_a != eval_b


def test_zero_ratio_keeps_everything_in_train():
    train, held = split_train_eval(list(range(10)), 0.0, seed=42)
    assert len(train) == 10 and held == []


def test_split_never_empties_train():
    train, held = split_train_eval([1, 2], 0.99, seed=42)
    assert len(train) >= 1 and len(held) >= 1


# ---------------------------------------------------------------------------
# collator
# ---------------------------------------------------------------------------


def test_collator_pads_labels_with_ignore_index():
    from ftlab.collate import PaddedCollator
    from ftlab.data import IGNORE_INDEX

    collator = PaddedCollator(pad_token_id=99, pad_to_multiple_of=1)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4], "labels": [4], "attention_mask": [1]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 99, 99]]
    # Padding labels with the pad id instead of IGNORE_INDEX would train the
    # model to emit padding.
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_collator_rounds_up_to_multiple_of_eight():
    from ftlab.collate import PaddedCollator

    batch = PaddedCollator(pad_token_id=0, pad_to_multiple_of=8)(
        [{"input_ids": [1] * 3, "labels": [1] * 3, "attention_mask": [1] * 3}]
    )
    assert batch["input_ids"].shape[-1] == 8


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def test_warmup_steps_derived_from_ratio():
    """transformers 5 removed warmup_ratio, so ftlab computes the step count."""
    from ftlab.train import compute_schedule

    cfg = config_mod.load(
        "smoke.yaml",
        {
            "train.max_steps": -1,
            "train.epochs": 2.0,
            "train.per_device_batch_size": 2,
            "train.grad_accum": 4,
            "train.warmup_ratio": 0.1,
        },
    )
    schedule = compute_schedule(cfg, n_train=800)
    assert schedule["effective_batch"] == 8
    assert schedule["steps_per_epoch"] == 100
    assert schedule["total_steps"] == 200
    assert schedule["warmup_steps"] == 20


def test_max_steps_caps_the_schedule():
    from ftlab.train import compute_schedule

    cfg = config_mod.load("smoke.yaml", {"train.max_steps": 6})
    assert compute_schedule(cfg, n_train=10_000)["total_steps"] == 6


def test_save_steps_must_align_with_eval_steps():
    """A best checkpoint on an unsaved step is a best checkpoint you cannot serve.

    The v2 run evaluated every 25 steps, found its best at 175, and saved only
    at 115 and 230. Both fine-tuned arms then ran on step 230 -- worse on eval,
    and carrying the memorisation gap the gate had already flagged.
    """
    import pytest

    from ftlab.config import Config

    base = {
        "model": {"base": "unused"},
        "data": {"train_path": "unused.jsonl"},
        "train": {"eval_steps": 25, "save_steps": 115},
    }
    with pytest.raises(ValueError, match="multiple"):
        Config.model_validate(base)

    base["train"]["save_steps"] = 25
    Config.model_validate(base)

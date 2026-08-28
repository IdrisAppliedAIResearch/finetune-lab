"""Checkpoint inspection and config validation.

Built against a synthetic checkpoint shaped like Gemma 4: a language tower, a
couple of perception projections, norms and an embedding. The case that matters
is the first one under "config checks" -- an exclude pattern matching nothing is
invisible to PEFT, which is how the shipped Gemma preset stayed wrong.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from ftlab.config import Config
from ftlab.modelinfo import check_against_config, inspect_model, render


@pytest.fixture
def checkpoint(tmp_path):
    """A miniature multimodal checkpoint with Gemma-4-shaped module names."""
    path = tmp_path / "ckpt"
    path.mkdir()

    tensors = {
        "model.language_model.embed_tokens.weight": torch.zeros(64, 8),
        "model.language_model.norm.weight": torch.zeros(8),
    }
    for layer in range(2):
        prefix = f"model.language_model.layers.{layer}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{prefix}.self_attn.{proj}.weight"] = torch.zeros(8, 8)
        tensors[f"{prefix}.mlp.down_proj.weight"] = torch.zeros(8, 16)
        tensors[f"{prefix}.input_layernorm.weight"] = torch.zeros(8)
    # The three perception linears, named as Gemma 4 actually names them.
    tensors["model.vision_embedder.patch_dense.weight"] = torch.zeros(8, 16)
    tensors["model.vision_embedder.patch_ln1.weight"] = torch.zeros(16)
    tensors["model.embed_vision.embedding_projection.weight"] = torch.zeros(8, 8)
    tensors["model.embed_audio.embedding_projection.weight"] = torch.zeros(8, 4)

    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "model_type": "gemma4_unified",
                "dtype": "bfloat16",
                "text_config": {"hidden_size": 8},
                "vision_config": {"hidden_size": 8},
                "audio_config": {"hidden_size": 4},
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(**lora):
    return Config.model_validate(
        {
            "model": {"base": "x"},
            "data": {"train_path": "x.jsonl"},
            "lora": {"target_modules": "auto", **lora},
        }
    )


def _add_tokenizer(path, chat_template=None):
    payload = {"tokenizer_class": "GemmaTokenizer", "bos_token": "<bos>"}
    if chat_template:
        payload["chat_template"] = chat_template
    (path / "tokenizer_config.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------


def test_reads_architecture_without_loading_weights(checkpoint):
    info = inspect_model(checkpoint)
    assert info.architectures == ["Gemma4UnifiedForConditionalGeneration"]
    assert info.model_type == "gemma4_unified"
    assert info.dtype == "bfloat16"
    assert info.is_multimodal


def test_groups_modules_by_top_level_prefix(checkpoint):
    info = inspect_model(checkpoint)
    assert "model.language_model" in info.top_prefixes
    assert "model.vision_embedder" in info.top_prefixes


def test_norms_and_embeddings_are_not_counted_as_linears(checkpoint):
    """'all-linear' targets projections, not norms or the token embedding.

    Miscounting these would make the scope check meaningless.
    """
    info = inspect_model(checkpoint)
    assert not any("layernorm" in m for m in info.linear_modules)
    assert not any("embed_tokens" in m for m in info.linear_modules)
    assert not any(m.endswith(".norm") for m in info.linear_modules)


def test_finds_exactly_the_perception_linears(checkpoint):
    info = inspect_model(checkpoint)
    outside = [m for m in info.linear_modules if "language_model" not in m]
    assert sorted(outside) == [
        "model.embed_audio.embedding_projection",
        "model.embed_vision.embedding_projection",
        "model.vision_embedder.patch_dense",
    ]


# ---------------------------------------------------------------------------
# config checks
# ---------------------------------------------------------------------------


def test_an_exclude_that_matches_nothing_is_reported(checkpoint):
    """The bug this module exists for.

    PEFT does not warn when an exclusion hits zero modules, so a wrong name is
    silent everywhere: the run trains, the loss falls, and the comment claiming
    the towers are protected stays wrong.
    """
    info = inspect_model(checkpoint)
    checks = check_against_config(info, _config(exclude_modules=["vision_tower"]))

    failed = [c for c in checks if c.name == "exclude 'vision_tower'"]
    assert failed and not failed[0].ok
    assert "MATCHES NOTHING" in failed[0].detail


def test_correct_excludes_pass_and_clear_the_scope_check(checkpoint):
    info = inspect_model(checkpoint)
    checks = check_against_config(
        info,
        _config(exclude_modules=["vision_embedder", "embed_vision", "embed_audio"]),
    )
    by_name = {c.name: c for c in checks}

    assert all(by_name[f"exclude '{p}'"].ok for p in
               ("vision_embedder", "embed_vision", "embed_audio"))
    scope = by_name["all-linear scope"]
    assert scope.ok
    assert "0 non-text linears" in scope.detail


def test_empty_excludes_on_a_multimodal_checkpoint_is_flagged(checkpoint):
    info = inspect_model(checkpoint)
    checks = check_against_config(info, _config(exclude_modules=[]))
    flagged = [c for c in checks if c.name == "exclude_modules"]
    assert flagged and not flagged[0].ok


def test_missing_chat_template_is_flagged_as_a_base_model(checkpoint):
    """QRA training renders chat turns; a base checkpoint cannot supply them."""
    info = inspect_model(checkpoint)
    check = next(c for c in check_against_config(info, _config()) if c.name == "chat template")
    assert not check.ok
    assert "base model" in check.detail


def test_present_chat_template_passes(checkpoint):
    _add_tokenizer(checkpoint, chat_template="{{ messages }}")
    info = inspect_model(checkpoint)
    check = next(c for c in check_against_config(info, _config()) if c.name == "chat template")
    assert check.ok


def test_a_sidecar_template_file_also_counts(checkpoint):
    """Newer transformers writes chat_template.jinja beside tokenizer_config."""
    _add_tokenizer(checkpoint)
    (checkpoint / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    assert inspect_model(checkpoint).chat_template


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_report_explains_the_silent_failure_only_when_relevant(checkpoint):
    info = inspect_model(checkpoint)
    bad = render(info, check_against_config(info, _config(exclude_modules=["nope"])))
    assert "PEFT does not" in bad

    good = render(
        info,
        check_against_config(
            info, _config(exclude_modules=["vision_embedder", "embed_vision", "embed_audio"])
        ),
    )
    # The chat template still fails here, but that is not an exclude problem.
    assert "PEFT does not" not in good

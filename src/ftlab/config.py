"""Typed, YAML-backed experiment configuration.

Configs compose by inheritance: a file may declare ``extends: base.yaml`` and
override any subset of keys. This keeps per-experiment files down to the handful
of values that actually differ, so a diff between two runs is readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


class RunConfig(BaseModel):
    name: str = "unnamed"
    output_dir: str = "outputs"
    seed: int = 42
    # Directory for TensorBoard event files; resolved under output_dir/run_name.
    report_to: Literal["tensorboard", "none"] = "tensorboard"


class ModelConfig(BaseModel):
    # HF repo id or a local path. Anything transformers can load.
    base: str
    trust_remote_code: bool = False
    # "sdpa" is the right default on native Windows: flash-attention has no
    # official Windows wheels, and PyTorch's fused SDPA covers most of the win.
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    # --- 4-bit (QLoRA) ---
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # Gradient checkpointing trades ~30% step time for a large activation saving.
    gradient_checkpointing: bool = True


class LoraConfig_(BaseModel):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    # "auto" lets PEFT pick every linear layer it can find, which is the
    # right default for an unfamiliar architecture. Name modules explicitly
    # once you know the model and want to trim the adapter.
    target_modules: list[str] | Literal["auto"] = "auto"
    # Substrings or a regex naming modules to keep adapters OFF. This is how a
    # multimodal base gets trained text-only: Gemma 4, for instance, carries
    # vision and audio towers whose linears "all-linear" would otherwise wrap in
    # adapters that no text batch can ever produce a gradient for.
    exclude_modules: list[str] = Field(default_factory=list)

    # Layers to train fully and save alongside the adapter (e.g. "embed_tokens"
    # when you add special tokens). Empty is the common case.
    modules_to_save: list[str] = Field(default_factory=list)
    bias: Literal["none", "all", "lora_only"] = "none"

    # Rank-stabilised scaling: alpha/sqrt(r) instead of alpha/r. Worth enabling
    # at r >= 64, where the plain ratio makes updates unhelpfully small.
    use_rslora: bool = False


class DataConfig(BaseModel):
    train_path: str
    eval_path: str | None = None
    # Used only when eval_path is unset: fraction of train held out for eval.
    eval_split_ratio: float = 0.05

    max_seq_len: int = 2048
    # Drop examples that exceed max_seq_len instead of truncating them. A
    # truncated reasoning trace teaches the model to stop mid-thought.
    drop_overlong: bool = True

    system_prompt: str | None = None

    # How a (question, reasoning, answer) triple becomes assistant text.
    #   think_tags  -> "<think>\n{reasoning}\n</think>\n\n{answer}"
    #   labeled     -> "Reasoning:\n{reasoning}\n\nAnswer:\n{answer}"
    #   answer_only -> "{answer}"   (ablation: does the trace actually help?)
    reasoning_format: Literal["think_tags", "labeled", "answer_only"] = "think_tags"
    think_open: str = "<think>"
    think_close: str = "</think>"

    # Train on the reasoning tokens as well as the answer. Turning this off
    # keeps the trace in the prompt-side context but zeroes its loss.
    train_on_reasoning: bool = True


class TrainConfig(BaseModel):
    epochs: float = 3.0
    # Real batch = per_device * grad_accum. Raise grad_accum, not per_device,
    # when you hit OOM.
    per_device_batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"

    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3

    # Windows: keep this at 0. The spawn-based multiprocessing start method
    # re-imports the training module in every worker, which is both slow and a
    # reliable source of confusing crashes.
    dataloader_num_workers: int = 0

    # Cap steps for smoke tests; -1 means "run the full schedule".
    max_steps: int = -1


class Config(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig
    lora: LoraConfig_ = Field(default_factory=LoraConfig_)
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _check_coherent(self) -> Config:
        if self.model.load_in_4bit and self.model.dtype == "float32":
            raise ValueError(
                "load_in_4bit with dtype=float32 wastes the memory you saved; "
                "use bfloat16."
            )
        if not 0.0 <= self.data.eval_split_ratio < 1.0:
            raise ValueError("data.eval_split_ratio must be in [0, 1)")
        if self.data.reasoning_format == "answer_only" and self.data.train_on_reasoning:
            # Not an error -- just meaningless, since there is no trace in the
            # target. Normalise it so the run log tells the truth.
            self.data.train_on_reasoning = False
        return self

    @property
    def run_dir(self) -> Path:
        return Path(self.run.output_dir) / self.run.name


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` on ``base``; scalars and lists replace."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_raw(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config, resolving the ``extends`` chain."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_ROOT / path
    path = path.resolve()

    if path in _seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data

    parent_path = (path.parent / parent_name).resolve()
    parent = load_raw(parent_path, _seen | {path})
    return _deep_merge(parent, data)


def load(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a config file and apply dotted-key CLI overrides on top."""
    raw = load_raw(path)
    if overrides:
        for dotted, value in overrides.items():
            cursor = raw
            *parents, leaf = dotted.split(".")
            for part in parents:
                cursor = cursor.setdefault(part, {})
            cursor[leaf] = value
    return Config.model_validate(raw)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key.path=value`` CLI override, YAML-typing the value."""
    if "=" not in text:
        raise ValueError(f"override must look like key.path=value, got: {text!r}")
    key, _, value = text.partition("=")
    return key.strip(), yaml.safe_load(value)

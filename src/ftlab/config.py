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
    #   native      -> reasoning passed as a separate 'reasoning' key on the
    #                  message, letting the model's own template place it.
    #
    # 'native' exists because Gemma 4 renders reasoning into a dedicated
    # "<|channel>thought ... <channel|>" span taken from a separate message
    # field, and its generation prompt emits an *empty, closed* channel.
    # Embedding "<think>" in the content there yields a prompt that is not a
    # prefix of the full conversation, which makes label masking underivable --
    # ftlab refuses to train on that rather than guess where the boundary is.
    reasoning_format: Literal["think_tags", "labeled", "answer_only", "native"] = "think_tags"
    think_open: str = "<think>"
    think_close: str = "</think>"

    # Extra keyword arguments for tokenizer.apply_chat_template, applied
    # identically to the prompt and to the full conversation. Gemma 4 needs
    # {enable_thinking: true} for those two renderings to stay consistent.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Marker whose end separates the reasoning span from the answer under
    # 'native'. Only consulted when train_on_reasoning is false.
    native_reasoning_close: str = "<channel|>"

    # Train on the reasoning tokens as well as the answer. Turning this off
    # keeps the trace in the prompt-side context but zeroes its loss.
    train_on_reasoning: bool = True


class GateConfig(BaseModel):
    """Rule deciding whether one more epoch is earned.

    The point of writing it down as thresholds is that the decision gets made
    the same way whether or not anyone is watching the curve, and can be argued
    with before the run rather than rationalised after it.
    """

    enabled: bool = True
    extra_epochs: float = 1.0

    # The last epoch must have bought at least this much relative eval loss for
    # another one to be worth its wall time. 0.5% is deliberately low: the test
    # is "still learning at all", not "still learning fast".
    min_rel_improvement: float = 0.005

    # ...and the final measurement must still be sitting at the floor. A curve
    # that improved on average but has already turned up is done, whatever the
    # epoch-over-epoch delta says.
    overfit_tolerance: float = 0.002

    # The extra epoch is a warm restart with its own cosine decay, not a
    # continuation of the finished one, so it gets its own (lower) peak.
    lr_scale: float = 0.5


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

    gate: GateConfig = Field(default_factory=GateConfig)

    # Continue training from an existing adapter instead of a fresh one. Set by
    # 'ftlab train --resume-adapter'; this is how the gate's extra epoch runs.
    resume_adapter: str | None = None


class MetricsConfig(BaseModel):
    enabled: bool = True
    # nvidia-smi poll interval. Five seconds is far below any step time and the
    # subprocess cost is irrelevant next to it.
    power_sample_seconds: float = 5.0

    # --- prices: inputs, not measurements ---
    electricity_usd_per_kwh: float = 0.17
    # Everything in the box that is not the GPU. A constant is honest; metering
    # it properly needs a wall plug.
    system_overhead_watts: float = 120.0
    # What an equivalent rented GPU-hour would cost. Zero disables the
    # comparison, because a wrong number here is worse than no number.
    cloud_usd_per_hour: float = 0.0


class Config(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig
    lora: LoraConfig_ = Field(default_factory=LoraConfig_)
    data: DataConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

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

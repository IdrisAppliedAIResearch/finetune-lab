"""Model and tokenizer construction: quantization, LoRA attachment, dtype policy."""

from __future__ import annotations

from typing import Any

import torch

from .config import Config, ModelConfig

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    return DTYPES[name]


def load_tokenizer(cfg: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base,
        trust_remote_code=cfg.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        # Never reuse EOS as PAD without also masking it: the model would learn
        # that EOS is a filler token and stop terminating. We only ever pad
        # positions whose label is IGNORE_INDEX, so aliasing is safe here, but
        # prefer a dedicated pad token when the vocabulary already has one.
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise ValueError(
            f"{cfg.base} has no chat_template. QRA training renders a chat turn, "
            "so set tokenizer.chat_template before training, or point data."
            "reasoning_format at a plain-text scheme and supply your own template."
        )

    return tokenizer


def build_quantization_config(cfg: ModelConfig) -> Any | None:
    if not cfg.load_in_4bit:
        return None

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=resolve_dtype(cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def load_base_model(cfg: ModelConfig) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(cfg.dtype),
        "attn_implementation": cfg.attn_implementation,
        "trust_remote_code": cfg.trust_remote_code,
        # Single GPU: pin everything to cuda:0 rather than letting accelerate
        # spill layers to CPU, which silently turns a slow run into a stalled one.
        "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
    }

    quant = build_quantization_config(cfg)
    if quant is not None:
        kwargs["quantization_config"] = quant

    model = AutoModelForCausalLM.from_pretrained(cfg.base, **kwargs)

    # Cache and gradient checkpointing are mutually exclusive; leaving the cache
    # on emits a warning every step and wastes memory during training.
    model.config.use_cache = not cfg.gradient_checkpointing
    return model


def attach_lora(model: Any, cfg: Config) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.model.gradient_checkpointing,
        )

    if cfg.train.resume_adapter:
        # Warm restart: keep the weights the previous phase learned and carry on
        # training them. is_trainable is load-bearing -- PEFT loads adapters
        # frozen by default, and a silently frozen adapter trains nothing while
        # the loss curve still moves (the base model is stochastic under
        # dropout), so this would look like a working run producing no update.
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(cfg.train.resume_adapter), is_trainable=True
        )
        if cfg.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        return model

    lora = cfg.lora
    target = "all-linear" if lora.target_modules == "auto" else list(lora.target_modules)

    peft_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type="CAUSAL_LM",
        target_modules=target,
        exclude_modules=list(lora.exclude_modules) or None,
        modules_to_save=list(lora.modules_to_save) or None,
        use_rslora=lora.use_rslora,
    )

    model = get_peft_model(model, peft_config)

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # With checkpointing the inputs to the first block have no grad path
        # unless we explicitly ask for one; without this LoRA sees zero grads.
        model.enable_input_require_grads()

    return model


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percent": round(100 * trainable / total, 4) if total else 0.0,
    }


def build(cfg: Config) -> tuple[Any, Any]:
    """Load tokenizer + LoRA-wrapped model for a run."""
    tokenizer = load_tokenizer(cfg.model)
    model = load_base_model(cfg.model)
    model = attach_lora(model, cfg)
    return model, tokenizer

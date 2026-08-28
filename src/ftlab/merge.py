"""Merge a LoRA adapter into the base weights, producing a standalone model.

Required before GGUF conversion: llama.cpp consumes full weights, not adapters.

The merge deliberately reloads the base model in the dtype you intend to ship
rather than reusing a 4-bit training-time model. Merging into dequantized 4-bit
weights bakes the quantization error into the output, and you then quantize a
second time on the way to GGUF -- two lossy steps where one would do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config


def merge_adapter(
    cfg: Config,
    adapter_dir: str | Path,
    output_dir: str | Path,
    dtype: str = "bfloat16",
) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from .model import DTYPES

    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")

    print(f"[ftlab] loading base {cfg.model.base} in {dtype} (no quantization)")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.base,
        dtype=DTYPES[dtype],
        trust_remote_code=cfg.model.trust_remote_code,
        # CPU keeps a 27B merge off the GPU entirely; it is a one-off and the
        # box has far more system RAM than VRAM.
        device_map="cpu",
    )

    print(f"[ftlab] applying adapter {adapter_dir}")
    merged = PeftModel.from_pretrained(base, str(adapter_dir), dtype=DTYPES[dtype])
    merged = merged.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ftlab] writing merged weights -> {output_dir}")
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    # The tokenizer must travel with the weights, and it must be the *adapter's*
    # tokenizer: training may have added or resized tokens.
    tokenizer = _tokenizer_for_merge(adapter_dir, cfg)
    tokenizer.save_pretrained(str(output_dir))

    del merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[ftlab] merged model ready at {output_dir}")
    return output_dir


def _tokenizer_for_merge(adapter_dir: Path, cfg: Config) -> Any:
    from transformers import AutoTokenizer

    if (adapter_dir / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(str(adapter_dir))
    from .model import load_tokenizer

    return load_tokenizer(cfg.model)

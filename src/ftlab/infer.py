"""Generate from a trained adapter -- the fastest way to see what changed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .data import QRAExample, build_messages, iter_jsonl


def load_for_inference(cfg: Config, adapter_dir: str | Path | None) -> tuple[Any, Any]:
    """Load the base model with the adapter applied, in eval mode."""
    from peft import PeftModel

    from . import model as model_mod

    tokenizer = model_mod.load_tokenizer(cfg.model)
    model = model_mod.load_base_model(cfg.model)

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print(f"[ftlab] adapter loaded from {adapter_dir}")
    else:
        print("[ftlab] no adapter -- generating from the base model")

    model.eval()
    # Undo the training-time setting; generation without a KV cache is glacial.
    model.config.use_cache = True
    return model, tokenizer


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    question: str,
    cfg: Config,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    example = QRAExample(question=question, answer="")
    prompt = tokenizer.apply_chat_template(
        build_messages(example, cfg.data),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Slice off the prompt so we return only what the model actually produced.
    completion = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def run(
    cfg: Config,
    adapter_dir: str | Path | None,
    questions: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict[str, str]]:
    model, tokenizer = load_for_inference(cfg, adapter_dir)

    results = []
    for i, question in enumerate(questions, start=1):
        print(f"\n{'=' * 70}\n[{i}/{len(questions)}] {question}\n{'-' * 70}")
        answer = generate(model, tokenizer, question, cfg, max_new_tokens, temperature)
        print(answer)
        results.append({"question": question, "generated": answer})
    return results


def questions_from(path: str | Path, limit: int | None = None) -> list[str]:
    """Pull the question field out of a QRA file, for eyeballing held-out data."""
    questions = [example.question for example in iter_jsonl(path)]
    return questions[:limit] if limit else questions


def save(results: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[ftlab] wrote {len(results)} generations -> {path}")

"""Generate from a trained adapter -- the fastest way to see what changed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .data import QRAExample, iter_jsonl, render_prompt


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


# Records retrieved for a live question. Matches the corpus's own budget
# closely enough that a query and a training example look alike to the model.
RETRIEVE_K = 12

# Appended to an answer that hit the token budget instead of finishing. A
# sentinel rather than a separate return value so it survives being written to
# and read back from a generations file, which is where grading happens.
TRUNCATION_MARK = "…[TRUNCATED]"


def _decode(tokenizer: Any, completion: Any, preserve: tuple[str, ...] = ()) -> str:
    """Decoded text, optionally keeping some special tokens.

    ``skip_special_tokens=True`` is right for almost everything and was wrong
    for one thing that mattered. Gemma 4 separates reasoning from the answer
    with channel tokens -- the reply is ``<|channel>thought ... <channel|>``
    followed by the answer -- and stripping them deleted the only boundary
    there was. Every downstream reader then saw one undivided blob of the
    model's thinking and scored it as the answer, which is not a parsing
    inconvenience: it means a benchmark measured the reasoning trace and
    reported it as a ranking.

    Named tokens are preserved and the rest are still dropped, so a caller asks
    for the boundary it needs rather than getting all the scaffolding back.
    """
    if not preserve:
        return tokenizer.decode(completion, skip_special_tokens=True).strip()
    text = tokenizer.decode(completion, skip_special_tokens=False)
    for token in tokenizer.all_special_tokens:
        if token not in preserve:
            text = text.replace(token, "")
    return text.strip()


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    question: str,
    cfg: Config,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    context: str = "",
) -> str:
    example = QRAExample(question=question, answer="", context=context)
    prompt = render_prompt(example, cfg.data, tokenizer)
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


def questions_from(
    path: str | Path, limit: int | None = None
) -> list[tuple[str, str]]:
    """Pull questions and their context out of a QRA file.

    The context comes with them deliberately. Re-retrieving would answer a
    different question than the corpus posed, and any difference between the two
    would show up as a model error.
    """
    pairs = [(example.question, example.context) for example in iter_jsonl(path)]
    return pairs[:limit] if limit else pairs


def save(results: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[ftlab] wrote {len(results)} generations -> {path}")


@torch.no_grad()
def generate_many(
    model: Any,
    tokenizer: Any,
    questions: list[str] | list[tuple[str, str]],
    cfg: Config,
    max_new_tokens: int = 1280,
    temperature: float = 0.0,
    batch_size: int = 8,
    progress: bool = True,
    preserve_special: tuple[str, ...] = (),
) -> list[str]:
    """Generate for many questions at once.

    Each item is either a question or a (question, context) pair. Context is
    not optional in practice: the prompt the model was trained on carries the
    retrieved library records, and generating without them presents a format the
    model has never seen and asks it to answer from weights that were never
    taught to hold the facts.

    Left padding is not a preference here, it is a correctness requirement. With
    right padding, a short prompt is followed by pad tokens and *then* the
    generated text, so slicing the prompt off by length returns padding and the
    model has attended across a gap it never saw in training. Left padding puts
    every prompt's final token at the same index, which makes one slice correct
    for the whole batch.
    """
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(questions), batch_size):
            chunk = questions[start : start + batch_size]
            if progress:
                print(f"[ftlab] generating {start + len(chunk)}/{len(questions)}")

            prompts = [
                render_prompt(
                    QRAExample(
                        question=item[0] if isinstance(item, tuple) else item,
                        answer="",
                        context=item[1] if isinstance(item, tuple) else "",
                    ),
                    cfg.data,
                    tokenizer,
                )
                for item in chunk
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)

            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = encoded["input_ids"].shape[-1]
            for row in generated:
                completion = row[prompt_len:]
                text = _decode(tokenizer, completion, preserve_special)
                # Mark answers that ran out of budget rather than finishing.
                # Inferring this from the text is unreliable: a complete
                # bulleted list ends without punctuation, which a
                # "does it end in a full stop" heuristic reads as truncated --
                # it reported 61% truncation on an arm whose longest answer used
                # 60% of the budget. Whether generation stopped on an
                # end-of-sequence token is the only signal that actually knows.
                eos = getattr(tokenizer, "eos_token_id", None)
                stopped_on_eos = eos is not None and bool((completion == eos).any())
                if not stopped_on_eos and len(completion) >= max_new_tokens:
                    text += TRUNCATION_MARK
                outputs.append(text)
    finally:
        tokenizer.padding_side = original_side
    return outputs

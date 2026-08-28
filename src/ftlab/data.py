"""Question-Reasoning-Answer dataset loading, rendering, and loss masking.

A QRA record is one JSON object per line:

    {"question": "...", "reasoning": "...", "answer": "...", "meta": {...}}

``meta`` is optional and carried through untouched so you can filter or slice on
it later. The interesting work here is turning that triple into token ids whose
labels are masked everywhere the model should not be scored: the prompt always,
and the reasoning trace too when ``data.train_on_reasoning`` is off.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, DataConfig

REQUIRED_FIELDS = ("question", "answer")
IGNORE_INDEX = -100


@dataclass
class QRAExample:
    question: str
    answer: str
    reasoning: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: dict[str, Any], *, where: str) -> QRAExample:
        missing = [f for f in REQUIRED_FIELDS if not str(obj.get(f, "")).strip()]
        if missing:
            raise ValueError(f"{where}: missing or empty field(s): {', '.join(missing)}")
        unknown = set(obj) - {"question", "answer", "reasoning", "meta"}
        if unknown:
            raise ValueError(
                f"{where}: unexpected field(s): {', '.join(sorted(unknown))}. "
                "Put extra columns under 'meta'."
            )
        return cls(
            question=str(obj["question"]).strip(),
            answer=str(obj["answer"]).strip(),
            reasoning=str(obj.get("reasoning", "")).strip(),
            meta=obj.get("meta") or {},
        )


def iter_jsonl(path: str | Path) -> Iterator[QRAExample]:
    """Stream a JSONL file, reporting the offending line number on bad input."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            where = f"{path.name}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: expected a JSON object, got {type(obj).__name__}")
            yield QRAExample.from_obj(obj, where=where)


def load_jsonl(path: str | Path) -> list[QRAExample]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


# ---------------------------------------------------------------------------
# Rendering: triple -> (assistant text, offset where the answer starts)
# ---------------------------------------------------------------------------


def render_assistant(example: QRAExample, cfg: DataConfig) -> tuple[str, int]:
    """Render the assistant turn, and the character offset where the answer begins.

    The offset is what lets us mask the reasoning trace separately from the
    answer. It always lands on a newline boundary, which keeps the later
    tokenizer split away from any byte-pair merge.
    """
    fmt = cfg.reasoning_format
    if fmt == "answer_only" or not example.reasoning:
        return example.answer, 0

    if fmt == "think_tags":
        prefix = f"{cfg.think_open}\n{example.reasoning}\n{cfg.think_close}\n\n"
    elif fmt == "labeled":
        prefix = f"Reasoning:\n{example.reasoning}\n\nAnswer:\n"
    else:  # pragma: no cover - guarded by the config Literal
        raise ValueError(f"unknown reasoning_format: {fmt}")

    return prefix + example.answer, len(prefix)


def build_assistant_message(
    example: QRAExample, cfg: DataConfig
) -> tuple[dict[str, str], str, int]:
    """The assistant turn, as (message, rendered text, answer offset).

    Under 'native' the reasoning travels as its own message field and the
    template decides where it goes, so the offset is not knowable here -- it is
    recovered from the rendered completion instead.
    """
    if cfg.reasoning_format == "native":
        message = {"role": "assistant", "content": example.answer}
        if example.reasoning:
            message["reasoning"] = example.reasoning
        return message, example.answer, 0

    text, offset = render_assistant(example, cfg)
    return {"role": "assistant", "content": text}, text, offset


def build_messages(example: QRAExample, cfg: DataConfig) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if cfg.system_prompt:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({"role": "user", "content": example.question})
    return messages


def render_prompt(example: QRAExample, cfg: DataConfig, tokenizer: Any) -> str:
    """The full prompt string, including the template's generation cue.

    Must stay byte-identical to what training produced, template kwargs
    included, or the model meets a format at inference it never saw.
    """
    return tokenizer.apply_chat_template(
        build_messages(example, cfg),
        tokenize=False,
        add_generation_prompt=True,
        **dict(cfg.chat_template_kwargs),
    )


# ---------------------------------------------------------------------------
# Tokenization + masking
# ---------------------------------------------------------------------------


def encode(example: QRAExample, cfg: DataConfig, tokenizer: Any) -> dict[str, list[int]]:
    """Encode one triple into input_ids / labels / attention_mask.

    Segments are tokenized independently and concatenated rather than being
    tokenized as one string and sliced. That guarantees the label boundary sits
    exactly where we think it does: a tokenizer is free to merge characters
    across a slice point, and a mask that is one token off silently trains the
    model on its own prompt.
    """
    assistant_message, assistant_text, answer_offset = build_assistant_message(example, cfg)
    messages = build_messages(example, cfg)

    # Identical kwargs on both renders. Gemma 4's prompt emits an empty, closed
    # thought channel unless enable_thinking is set, so rendering the two sides
    # with different arguments silently destroys prefix-stability.
    template_kwargs = dict(cfg.chat_template_kwargs)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, assistant_message], tokenize=False, **template_kwargs
    )

    if not full_text.startswith(prompt_text):
        raise ValueError(
            "chat template is not prefix-stable: rendering with "
            "add_generation_prompt=True did not produce a prefix of the rendered "
            "conversation including the assistant turn, so loss masking cannot be "
            "derived safely.\n"
            "If this is a reasoning model, its generation prompt may open a "
            "thinking span the full conversation fills differently. Try "
            "data.reasoning_format: native with the template flag that enables "
            "thinking, e.g. data.chat_template_kwargs: {enable_thinking: true}."
        )
    completion_text = full_text[len(prompt_text) :]

    if cfg.reasoning_format == "native":
        # The template owns the reasoning span, so the answer starts after its
        # closing marker rather than at a known offset into our own string.
        marker = cfg.native_reasoning_close
        found = completion_text.find(marker)
        split_ok = found >= 0
        answer_offset = found + len(marker) if split_ok else 0
    else:
        # The template wraps the content, so the assistant text must survive
        # intact at the head of the completion for the split to be valid.
        split_ok = completion_text.startswith(assistant_text)

    def ids(text: str) -> list[int]:
        # add_special_tokens=False throughout: the chat template already emits
        # BOS itself, and adding a second one is a classic silent regression.
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prompt_ids = ids(prompt_text)

    if cfg.train_on_reasoning or answer_offset == 0 or not split_ok:
        completion_ids = ids(completion_text)
        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    else:
        # Mask the trace, score only the answer plus the template's closing tokens.
        trace_ids = ids(completion_text[:answer_offset])
        answer_ids = ids(completion_text[answer_offset:])
        input_ids = prompt_ids + trace_ids + answer_ids
        labels = [IGNORE_INDEX] * (len(prompt_ids) + len(trace_ids)) + list(answer_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def encode_all(
    examples: list[QRAExample],
    cfg: DataConfig,
    tokenizer: Any,
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    """Encode a list of triples, applying the length policy.

    Returns the encoded rows plus a small stats dict for the run log.
    """
    kept: list[dict[str, list[int]]] = []
    dropped = 0
    truncated = 0

    for example in examples:
        row = encode(example, cfg, tokenizer)
        if len(row["input_ids"]) > cfg.max_seq_len:
            if cfg.drop_overlong:
                dropped += 1
                continue
            for key in ("input_ids", "labels", "attention_mask"):
                row[key] = row[key][: cfg.max_seq_len]
            truncated += 1

        # An example whose labels are entirely masked contributes no gradient
        # and makes the loss curve lie about the effective batch size.
        if all(label == IGNORE_INDEX for label in row["labels"]):
            dropped += 1
            continue
        kept.append(row)

    if not kept:
        raise ValueError(
            "every example was dropped -- check data.max_seq_len against your "
            "reasoning trace lengths."
        )

    lengths = [len(row["input_ids"]) for row in kept]
    stats = {
        "kept": len(kept),
        "dropped": dropped,
        "truncated": truncated,
        "len_min": min(lengths),
        "len_mean": round(sum(lengths) / len(lengths)),
        "len_max": max(lengths),
    }
    return kept, stats


def split_train_eval(rows: list[Any], ratio: float, seed: int) -> tuple[list[Any], list[Any]]:
    """Deterministic held-out split. Returns (train, eval); eval may be empty."""
    if ratio <= 0 or len(rows) < 2:
        return rows, []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * ratio))
    n_eval = min(n_eval, len(shuffled) - 1)
    return shuffled[n_eval:], shuffled[:n_eval]


def build_datasets(cfg: Config, tokenizer: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Load, split, and encode into HF Datasets ready for the trainer."""
    from datasets import Dataset

    train_raw = load_jsonl(cfg.data.train_path)

    if cfg.data.eval_path:
        eval_raw = load_jsonl(cfg.data.eval_path)
    else:
        train_raw, eval_raw = split_train_eval(
            train_raw, cfg.data.eval_split_ratio, cfg.run.seed
        )

    train_rows, train_stats = encode_all(train_raw, cfg.data, tokenizer)
    stats: dict[str, Any] = {"train": train_stats}

    eval_ds = None
    if eval_raw:
        eval_rows, eval_stats = encode_all(eval_raw, cfg.data, tokenizer)
        stats["eval"] = eval_stats
        eval_ds = Dataset.from_list(eval_rows)

    return Dataset.from_list(train_rows), eval_ds, stats

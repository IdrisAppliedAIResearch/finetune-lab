"""Shared fixtures.

The fake tokenizer is char-level on purpose. Because every character is exactly
one token, concatenating separately-encoded segments is guaranteed to equal
encoding the joined string, so a masking test that fails here is a real bug in
the masking logic rather than a tokenizer-merge artefact.
"""

from __future__ import annotations

import pytest

BOS = "\x02"
TURN_START = "\x03"
TURN_END = "\x04"


class FakeTokenizer:
    """Minimal stand-in with a prefix-stable chat template."""

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = None
    chat_template = "fake"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [BOS]
        for message in messages:
            parts.append(f"{TURN_START}{message['role']}\n{message['content']}{TURN_END}\n")
        if add_generation_prompt:
            parts.append(f"{TURN_START}assistant\n")
        return "".join(parts)

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [ord(ch) for ch in text]
        if add_special_tokens:
            ids = [ord(BOS)] + ids
        return {"input_ids": ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


class NonPrefixTokenizer(FakeTokenizer):
    """A template that rewrites earlier turns when the assistant turn is added.

    Real templates occasionally do this. Masking cannot be derived safely from
    such a template, and ftlab must refuse rather than train on a bad mask.
    """

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        if any(m["role"] == "assistant" for m in messages):
            return "REWRITTEN" + super().apply_chat_template(messages, add_generation_prompt=False)
        return super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt)


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def data_cfg():
    from ftlab.config import DataConfig

    return DataConfig(train_path="unused.jsonl", max_seq_len=100_000)

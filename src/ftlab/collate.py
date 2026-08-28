"""Padding collator for pre-tokenized, label-masked QRA rows.

Written out rather than reused from transformers so that the padding of
``labels`` is visibly IGNORE_INDEX and not the pad token id -- padding labels
with the pad id is the single most common way to poison an SFT loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data import IGNORE_INDEX


@dataclass
class PaddedCollator:
    pad_token_id: int
    # Tensor cores prefer a multiple of 8; rounding up costs a few pad tokens
    # and buys a measurably faster matmul.
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of > 1:
            remainder = longest % self.pad_to_multiple_of
            if remainder:
                longest += self.pad_to_multiple_of - remainder

        input_ids, labels, attention = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            lab = list(feature["labels"])
            mask = list(feature.get("attention_mask") or [1] * len(ids))
            gap = longest - len(ids)

            # Right padding. Left padding is for batched generation, not training:
            # with a causal mask, right padding keeps position ids aligned.
            input_ids.append(ids + [self.pad_token_id] * gap)
            labels.append(lab + [IGNORE_INDEX] * gap)
            attention.append(mask + [0] * gap)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

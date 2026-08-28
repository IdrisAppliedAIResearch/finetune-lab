"""The unit of output: one Question-Reasoning-Answer triple plus provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QRAItem:
    question: str
    reasoning: str
    answer: str
    archetype: str
    layer: str  # "recall" | "relational" | "multihop" | "recommendation"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """The on-disk shape ftlab.data expects."""
        return {
            "question": self.question.strip(),
            "reasoning": self.reasoning.strip(),
            "answer": self.answer.strip(),
            "meta": {"archetype": self.archetype, "layer": self.layer, **self.meta},
        }

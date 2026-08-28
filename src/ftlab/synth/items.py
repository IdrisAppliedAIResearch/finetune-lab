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

    # Library records retrieved for this question, as the numbered block the
    # prompt will carry. Filled in at build time by the same retriever that runs
    # at inference, so training and serving see the same context for the same
    # question rather than two different renderings of it.
    context: str = ""

    def to_record(self) -> dict[str, Any]:
        """The on-disk shape ftlab.data expects."""
        return {
            "question": self.question.strip(),
            "reasoning": self.reasoning.strip(),
            "answer": self.answer.strip(),
            "context": self.context.strip(),
            "meta": {"archetype": self.archetype, "layer": self.layer, **self.meta},
        }


@dataclass
class Retrieved:
    """The library records a prompt will show for one opportunity.

    Carries the ids as well as the rendered text, because the golden answer has
    to be computed over exactly these candidates. Ranking the whole roster and
    then showing a slice of it produces answers that name partners the prompt
    never mentioned -- which trains the model to invent names instead of
    choosing between what it was given.
    """

    context: str
    partner_ids: set[str] = field(default_factory=set)
    contract_ids: set[str] = field(default_factory=set)

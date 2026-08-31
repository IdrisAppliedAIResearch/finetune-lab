"""Generate answers for masked-sub questions and score them.

Shared by the baseline measurement and the training loop on purpose. If they
used different generation settings or different parsing, the "before" and the
"after" would not be comparable, and the comparison is the whole deliverable.
Anything that differs between a baseline run and a training rollout is a
confound, so there is one code path and it takes arguments.

The system prompt here is not the corpus one. That prompt was written for
hand-authored questions of nine different shapes and tells the model to "rank
recommendations and give the evidence behind each one", which invites an essay.
This task has one shape and a machine reads the output, so the format is stated
flatly. That matters for fairness rather than tidiness: an untrained model that
reasons well and formats badly would score as reasoning badly, and the first
version of this project's benchmark made exactly that mistake in reverse -- the
tuned arms emitted a trained heading the parser knew and the base model did not.

The brevity instruction is load-bearing, not style. Told only to "think about
which candidate fits", the base model spent 2,500 tokens re-reading the records
and second-guessing itself -- "Wait, let me re-read the prompt" -- and hit the
budget on 8 of 8 answers without ever ranking anything. Raising the budget does
not fix a model that loops; asking it to conclude does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shared.config import Config
from ..shared.grade import known_companies
from ..shared.questions import TOP_K
from .reward import Reward, score_batch, summarise

SYSTEM_PROMPT = (
    "You are a past performance and teaming analyst for a public health "
    "government contractor. You are shown a prime contractor, an agency, a "
    "numbered list of candidate subcontractors, and library records for each "
    "candidate. Exactly one candidate is the firm the prime actually hired.\n\n"
    "Give your reasoning in at most four short sentences, then stop reasoning "
    "and answer. End your reply with the line 'Ranking:' followed by a numbered "
    f"list of exactly {TOP_K} candidate names, most likely first, one per line, "
    "using the names exactly as they appear in the candidate list. Name only "
    "candidates from that list. Do not restate the records."
)


@dataclass
class Rollouts:
    """Generations and their rewards for one arm over one split."""

    label: str
    items: list[dict[str, Any]]
    generations: list[str]
    # The final channel only. Kept beside the raw text rather than replacing it,
    # because the thinking is what a later run will want to look at when a
    # number moves and nobody can say why.
    answers: list[str]
    rewards: list[Reward]

    @property
    def summary(self) -> dict[str, Any]:
        return summarise(self.items, self.rewards)

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "prime": item["meta"]["prime"],
                "agency": item["meta"]["agency"],
                "gold": item["meta"]["gold"][0],
                "is_new": item["meta"]["is_new"],
                "generated": text,
                "answer": answer,
                **reward.as_dict(),
            }
            for item, text, answer, reward in zip(
                self.items, self.generations, self.answers, self.rewards, strict=True
            )
        ]


def answer_of(text: str, close: str) -> str:
    """The part of a reply that is the answer rather than the thinking.

    Gemma 4 emits ``<|channel>thought ... <channel|>`` and then answers, and
    ``real-3arm.yaml`` inherits ``enable_thinking: true``. Decoded with special
    tokens stripped, those two spans arrive as one blob and the reasoning reads
    like an answer -- which is how a smoke run came back with 100% truncation
    and rankings that were really the model listing the candidates to itself.
    Splitting on the close marker fixes it at the source, and it holds whether
    or not thinking is switched on: with it off the marker is in the prompt and
    the reply has none, so the whole reply is the answer.

    The truncation mark is moved onto the answer, because a reply cut off inside
    its own thinking has no answer at all and must not be read as a silent one.
    """
    from ..shared.infer import TRUNCATION_MARK

    truncated = text.endswith(TRUNCATION_MARK)
    body = text.removesuffix(TRUNCATION_MARK)
    answer = body.rsplit(close, 1)[-1].strip() if close in body else body.strip()
    return answer + TRUNCATION_MARK if truncated else answer


def load_split(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(
    cfg: Config,
    items: list[dict[str, Any]],
    *,
    label: str = "base + retrieval",
    adapter: str | Path | None = None,
    data_dir: str | Path = "data/real",
    max_new_tokens: int = 900,
    temperature: float = 0.0,
    batch_size: int = 8,
    thinking: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> Rollouts:
    """One generation per item, scored.

    ``thinking`` is off by default and that is a measured choice, not a default
    inherited by accident. ``real-3arm.yaml`` sets ``enable_thinking: true``, and
    with it on this base model opens ``<|channel>thought`` and does not close it:
    14 of 16 answers hit a 1,600-token budget still deliberating, re-reading the
    records and second-guessing -- "Wait, none of these are in the candidate
    list. Let me re-examine." Raising the budget does not fix a trace that does
    not converge. With thinking suppressed the same model answers the same
    questions in under 700 tokens, in the format the prompt asks for.

    What is lost is real and worth stating: this measures the model's ranking,
    not its reasoning, and a policy trained to reason might well need the
    channel. Turn it on for that, budget accordingly, and watch
    ``truncated_rate`` -- but keep the setting identical across arms, because a
    baseline and a trained model on different prompts are not comparable.

    ``model`` and ``tokenizer`` may be passed in so a training loop does not
    reload eleven gigabytes of weights to measure itself between epochs.
    """
    from ..shared.infer import generate_many, load_for_inference

    if model is None or tokenizer is None:
        model, tokenizer = load_for_inference(cfg, adapter)

    # Set on the config rather than threaded through generate_many, which reads
    # it from cfg.data. Restored so a caller holding one Config for several
    # arms does not silently inherit this prompt.
    close = cfg.data.native_reasoning_close
    original = cfg.data.system_prompt
    original_kwargs = dict(cfg.data.chat_template_kwargs)
    cfg.data.system_prompt = SYSTEM_PROMPT
    cfg.data.chat_template_kwargs = {**original_kwargs, "enable_thinking": thinking}
    try:
        generations = generate_many(
            model,
            tokenizer,
            [(item["question"], item.get("context", "")) for item in items],
            cfg,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            preserve_special=(close,),
        )
    finally:
        cfg.data.system_prompt = original
        cfg.data.chat_template_kwargs = original_kwargs

    known = known_companies(data_dir)
    answers = [answer_of(text, close) for text in generations]
    return Rollouts(
        label=label,
        items=items,
        generations=generations,
        answers=answers,
        rewards=score_batch(items, answers, known),
    )


def save(rollouts: Rollouts, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rollouts.records()) + "\n",
        encoding="utf-8",
    )
    summary = {"label": rollouts.label, **rollouts.summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = ["SYSTEM_PROMPT", "Rollouts", "answer_of", "load_split", "run", "save"]

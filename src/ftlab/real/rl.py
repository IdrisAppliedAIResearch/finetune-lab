"""GRPO on the masked-sub task: reinforce against what actually happened.

The reward is verified rather than modelled -- ``reward.score`` checks the
ranking against a subcontract that was really reported after the blind cut -- so
there is no learned reward model to overfit and no judge whose taste becomes the
target. That is the whole reason this task is worth doing with RL rather than
supervised fine-tuning: nobody wrote the key.

Group-relative advantage suits it. Each prompt is sampled several times and the
advantage is each sample's reward against its own group's mean, so a hard slate
where every rollout fails contributes nothing rather than a uniform penalty, and
an easy one where every rollout succeeds contributes nothing rather than a
uniform reward. With one gold in twelve names and a policy that starts near
chance, that is most of what makes early training stable.

Three choices worth knowing before reading a result off this:

* **Prompts are rendered once, here, in exactly the form the baseline saw.**
  Pre-rendered text rather than message dicts, so the trainer cannot quietly
  apply a different template than the measurement did. A tuned model and a
  baseline on different prompts are not comparable and the difference would look
  like learning.
* **Truncated rollouts are penalised, not masked.** TRL can drop completions
  that hit the budget; here a rollout that rambles past it has failed to answer
  and the reward says so. Masking would teach the policy nothing about finishing,
  and this base model's failure mode is precisely not finishing.
* **Nothing selects on the evaluation half.** Checkpoints are chosen on training
  reward. The eval split is run once, at the end, by ``masked-run``. Choosing a
  step by its eval score is how a held-out set stops being held out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config
from .grade import known_companies
from .questions import TOP_K
from .reward import score
from .rollout import SYSTEM_PROMPT, answer_of, load_split


def build_dataset(items: list[dict[str, Any]], cfg: Config, tokenizer: Any) -> Any:
    """Prompts as text, with the answer key carried alongside.

    ``gold`` and ``slate`` ride on the dataset because TRL forwards any extra
    column to the reward function. The alternative -- looking the answer up from
    the prompt inside the reward -- would mean parsing the prompt twice and
    getting a chance to disagree with itself.
    """
    from datasets import Dataset

    from ..data import QRAExample, render_prompt

    original_prompt = cfg.data.system_prompt
    original_kwargs = dict(cfg.data.chat_template_kwargs)
    cfg.data.system_prompt = SYSTEM_PROMPT
    # Matches rollout.run's default. With thinking on, this base model does not
    # close its reasoning channel and every rollout scores as a non-answer, so
    # the policy would be optimising a reward that is constant at -0.2.
    cfg.data.chat_template_kwargs = {**original_kwargs, "enable_thinking": False}
    try:
        prompts = [
            render_prompt(
                QRAExample(
                    question=item["question"], answer="", context=item.get("context", "")
                ),
                cfg.data,
                tokenizer,
            )
            for item in items
        ]
    finally:
        cfg.data.system_prompt = original_prompt
        cfg.data.chat_template_kwargs = original_kwargs

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "gold": [item["meta"]["gold"][0] for item in items],
            "slate": [sorted(item["meta"]["tiers"]) for item in items],
        }
    )


def make_reward_fn(cfg: Config, data_dir: str | Path = "data/real") -> Any:
    """The reward, in the shape TRL calls it with.

    Named ``masked_sub_rank`` because TRL logs reward functions by name and an
    unnamed lambda in a training log is worth nothing six weeks later.
    """
    known = known_companies(data_dir)
    close = cfg.data.native_reasoning_close

    def masked_sub_rank(
        completions: list[str], gold: list[str], slate: list[list[str]], **_: Any
    ) -> list[float]:
        return [
            score(answer_of(text, close), g, s, known, top_k=TOP_K).value
            for text, g, s in zip(completions, gold, slate, strict=True)
        ]

    return masked_sub_rank


def train(
    cfg: Config,
    *,
    split_path: str | Path = "data/real_corpus/masked_sub.train.jsonl",
    out_dir: str | Path | None = None,
    epochs: float = 2.0,
    num_generations: int = 8,
    learning_rate: float = 1e-6,
    beta: float = 0.0,
    temperature: float = 1.0,
    max_completion_length: int = 700,
    batch_size: int = 8,
    grad_accum: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run GRPO and write the adapter.

    ``beta`` defaults to 0, which is TRL's default and also what keeps this
    inside 32 GB: a non-zero KL penalty needs a reference model resident
    alongside the policy. Raise it if the policy starts producing degenerate
    text, and watch that VRAM.
    """
    import torch
    from trl import GRPOConfig, GRPOTrainer

    from ..model import build_lora_config, build_quantization_config, load_tokenizer

    tokenizer = load_tokenizer(cfg.model)
    items = load_split(split_path)
    if limit:
        items = items[:limit]
    dataset = build_dataset(items, cfg, tokenizer)

    run_dir = Path(out_dir or (cfg.run_dir / "grpo"))
    run_dir.mkdir(parents=True, exist_ok=True)

    args = GRPOConfig(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        num_generations=num_generations,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        beta=beta,
        temperature=temperature,
        max_completion_length=max_completion_length,
        # A rollout that runs out of budget has not answered, and the reward
        # scores it as silence. Masking it would hide the one failure this base
        # model actually has.
        mask_truncated_completions=False,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_steps=20,
        save_total_limit=3,
        log_completions=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=cfg.model.base,
        reward_funcs=make_reward_fn(cfg),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        quantization_config=build_quantization_config(cfg.model),
        peft_config=build_lora_config(cfg.lora),
    )
    result = trainer.train()

    adapter = run_dir / "adapter"
    trainer.save_model(str(adapter))

    stats = {
        "adapter": str(adapter),
        "prompts": len(dataset),
        "num_generations": num_generations,
        "epochs": epochs,
        "metrics": {
            k: (round(v, 5) if isinstance(v, float) else v)
            for k, v in result.metrics.items()
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
    }
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


__all__ = ["build_dataset", "make_reward_fn", "train"]

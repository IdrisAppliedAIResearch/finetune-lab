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
    num_generations: int = 4,
    learning_rate: float = 1e-6,
    beta: float = 0.0,
    temperature: float = 1.0,
    # The answer is four sentences and five names; rollouts here average 215
    # tokens and every one of the 534 baseline replies finished well inside
    # this. The budget is paid on every rollout of every step, so it is not a
    # free margin: at 700 a single optimiser step did not finish in 22 minutes.
    max_completion_length: int = 320,
    batch_size: int = 1,
    grad_accum: int = 8,
    generation_batch_size: int | None = None,
    limit: int | None = None,
    log_completions: bool = False,
    vram_fraction: float = 0.92,
) -> dict[str, Any]:
    """Run GRPO and write the adapter.

    **The resident rollout count is the thing to watch, not the totals.** TRL
    derives ``generation_batch_size`` from ``gradient_accumulation_steps``, so
    raising accumulation to get a stable effective batch also multiplies how
    many rollouts sit in VRAM at once. That is how the first run here reached
    32.1 GB of a 32.6 GB card and took 499 s per step -- against 37 s for the
    same work at half the resident rollouts and 19 GB. Doubling the work made it
    13x slower, which is not a compute curve.

    It is the failure ``train.py`` documents from the supervised runs, in a new
    place: 12.8 s/step to 204 s/step, power draw collapsing to 141 W at a
    reported 97% utilisation, and nothing in the log saying "out of memory".
    So ``generation_batch_size`` is pinned to ``num_generations`` here -- the
    smallest legal value, one group at a time -- and accumulation is free to be
    whatever the effective batch wants.

    ``beta`` defaults to 0, which is TRL's default and also what keeps this
    inside 32 GB: a non-zero KL penalty needs a reference model resident
    alongside the policy.
    """
    import torch
    from trl import GRPOConfig, GRPOTrainer

    from ..model import build_lora_config, build_quantization_config, load_tokenizer
    from ..train import build_memory_guard, cap_memory_fraction

    tokenizer = load_tokenizer(cfg.model)
    items = load_split(split_path)
    if limit:
        items = items[:limit]
    dataset = build_dataset(items, cfg, tokenizer)

    run_dir = Path(out_dir or (cfg.run_dir / "grpo"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Loud beats slow. Without a cap, exhausting VRAM on Windows degrades into
    # system-memory spilling and the run just gets slower and slower.
    cap_memory_fraction(vram_fraction)

    args = GRPOConfig(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        num_generations=num_generations,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        # One group resident at a time. Left to default this is
        # batch_size * grad_accum, which quietly ties memory to a knob that has
        # nothing to do with memory.
        generation_batch_size=generation_batch_size or num_generations,
        # LoRA optimiser state is small, but paging it is free at this size and
        # it is the same optimiser the supervised path uses.
        optim="paged_adamw_8bit",
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
        # TRL prints sampled completions through rich, and on Windows a
        # redirected stdout is cp1252: the first table containing a character
        # outside it raised UnicodeEncodeError and killed the run at the end of
        # step 1, after eight minutes of rollouts. The completions are not worth
        # a run, and ``masked-run`` shows what the policy produces anyway.
        log_completions=log_completions,
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
        # Every step, not every 25: a GRPO step is minutes, so the cost of
        # releasing cached blocks is noise and the fragmentation it
        # prevents is what turned 37 s into 499 s.
        callbacks=[build_memory_guard(every=1)],
    )
    result = trainer.train()

    adapter = run_dir / "adapter"
    trainer.save_model(str(adapter))

    stats = {
        "adapter": str(adapter),
        "prompts": len(dataset),
        "num_generations": num_generations,
        "generation_batch_size": args.generation_batch_size,
        "steps_per_generation": args.steps_per_generation,
        "epochs": epochs,
        "metrics": {
            k: (round(v, 5) if isinstance(v, float) else v)
            for k, v in result.metrics.items()
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
        # The number that says whether a slow step is slow because of memory.
        # A retry is the allocator failing, freeing cached blocks and trying
        # again -- invisible in wall-clock terms except that everything is
        # slower. Generation alone peaks at 16.6 GB with none of these; the
        # optimisation half is what fills the card.
        "alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0)
        if torch.cuda.is_available()
        else None,
    }
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


__all__ = ["build_dataset", "make_reward_fn", "train"]

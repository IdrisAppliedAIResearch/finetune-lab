"""Training entry point: QRA triples -> LoRA/QLoRA adapter.

Uses ``transformers.Trainer`` directly rather than TRL's SFTTrainer. All the
QRA-specific work -- chat rendering, the reasoning/answer split, and label
masking -- already happened in ``ftlab.data``, so a higher-level trainer would
only add dataset preprocessing we would then have to switch off.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from ..shared.collate import PaddedCollator
from ..shared.config import Config
from ..shared.memory import (
    build_memory_guard,
    cap_memory_fraction,
)


def compute_schedule(cfg: Config, n_train: int) -> dict[str, int]:
    """Work out total optimizer steps and the warmup step count.

    transformers 5 dropped ``warmup_ratio``, so the ratio in our config has to
    be turned into an absolute step count here.
    """
    effective_batch = max(1, cfg.train.per_device_batch_size * cfg.train.grad_accum)
    steps_per_epoch = max(1, math.ceil(n_train / effective_batch))

    if cfg.train.max_steps and cfg.train.max_steps > 0:
        total_steps = cfg.train.max_steps
    else:
        total_steps = max(1, math.ceil(steps_per_epoch * cfg.train.epochs))

    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    return {
        "effective_batch": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }


def build_training_args(cfg: Config, schedule: dict[str, int], has_eval: bool) -> Any:
    from transformers import TrainingArguments

    run_dir = cfg.run_dir
    use_bf16 = cfg.model.dtype == "bfloat16"

    return TrainingArguments(
        output_dir=str(run_dir),
        seed=cfg.run.seed,
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps if cfg.train.max_steps > 0 else -1,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_steps=schedule["warmup_steps"],
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        bf16=use_bf16,
        fp16=cfg.model.dtype == "float16",
        logging_steps=cfg.train.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.train.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        # The v2 run evaluated every 25 steps, found its best model at step 175,
        # and saved checkpoints only at 115 and 230 -- so the best model was
        # never written to disk and both fine-tuned arms were served by the
        # final, worse one. Keeping the best requires that a checkpoint exist at
        # the step it was measured, which is why the config asserts
        # save_steps % eval_steps == 0, and that the Trainer be told to reload
        # it, which is what these three lines do.
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        report_to=[cfg.run.report_to] if cfg.run.report_to != "none" else [],
        # Our rows are already exactly the tensors the model consumes; letting
        # the Trainer prune columns by signature only risks dropping labels.
        remove_unused_columns=False,
        gradient_checkpointing=False,  # applied on the model in ftlab.model
    )


def write_run_metadata(cfg: Config, extra: dict[str, Any]) -> Path:
    """Snapshot everything needed to explain or reproduce this run."""
    import importlib.metadata

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    meta = {
        "config": cfg.model_dump(),
        "versions": versions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }

    path = run_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def train(cfg: Config) -> Path:
    from transformers import Trainer, set_seed

    from . import model as model_mod
    from ..shared.data import build_datasets

    set_seed(cfg.run.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ftlab] run '{cfg.run.name}' -> {run_dir}")
    print(f"[ftlab] base model: {cfg.model.base}")

    # Tokenizer first: the dataset cannot be rendered without its chat template.
    tokenizer = model_mod.load_tokenizer(cfg.model)
    train_ds, eval_ds, stats = build_datasets(cfg, tokenizer)
    print(f"[ftlab] data: {json.dumps(stats)}")

    model = model_mod.load_base_model(cfg.model)
    model = model_mod.attach_lora(model, cfg)
    params = model_mod.trainable_parameter_summary(model)
    print(
        f"[ftlab] trainable: {params['trainable']:,} / {params['total']:,} "
        f"({params['percent']}%)"
    )

    schedule = compute_schedule(cfg, len(train_ds))
    print(
        f"[ftlab] schedule: {schedule['total_steps']} steps "
        f"(effective batch {schedule['effective_batch']}, "
        f"{schedule['warmup_steps']} warmup)"
    )

    args = build_training_args(cfg, schedule, has_eval=eval_ds is not None)
    collator = PaddedCollator(pad_token_id=tokenizer.pad_token_id)

    cap_memory_fraction()
    callbacks: list[Any] = [build_memory_guard()]
    metrics_cb = None
    if cfg.metrics.enabled:
        from .metrics import CostConfig, build_callback

        metrics_cb = build_callback(
            run_name=cfg.run.name,
            out_dir=run_dir,
            cost=CostConfig(
                electricity_usd_per_kwh=cfg.metrics.electricity_usd_per_kwh,
                system_overhead_watts=cfg.metrics.system_overhead_watts,
                cloud_usd_per_hour=cfg.metrics.cloud_usd_per_hour,
            ),
            collator=collator,
            sample_seconds=cfg.metrics.power_sample_seconds,
        )
        metrics_cb.metrics.train_examples = len(train_ds)
        callbacks.append(metrics_cb)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    meta_path = write_run_metadata(
        cfg,
        {"data_stats": stats, "schedule": schedule, "parameters": params},
    )
    print(f"[ftlab] wrote {meta_path}")

    result = trainer.train()

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = dict(result.metrics)
    if eval_ds is not None:
        metrics.update(trainer.evaluate())
    # Deliberately NOT metrics.json: the metrics callback owns that filename and
    # writes it during on_train_end, so writing here would silently clobber the
    # richer report a moment after it was produced.
    (run_dir / "trainer_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[ftlab] done. adapter -> {adapter_dir}")
    if metrics_cb is not None:
        print()
        print(metrics_cb.metrics.report())
        print(f"\n[ftlab] metrics -> {run_dir / 'metrics.json'}")
    else:
        print(f"[ftlab] metrics: {json.dumps(metrics, indent=2, default=str)}")
    return adapter_dir

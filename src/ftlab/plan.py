"""Pre-run projection: steps, tokens, VRAM, wall time, energy, cost.

The numbers that matter here cannot be derived from parameter counts with any
useful accuracy, so this measures instead of guessing: it runs a handful of real
micro-batches on the real data and extrapolates from what the machine actually
did.

Three things about that calibration, the first two learned the hard way:

* **Training time is projected from seconds per step, not tokens per second.**
  The reverse seemed more principled -- normalise for sequence length, multiply
  by total tokens -- and it was measurably wrong. Calibration deliberately runs
  the longest examples in the corpus, so it clocks a high token rate at an
  ordinary step time; real mixed-length steps cost about the same wall time
  while carrying far fewer tokens. Against a real 1,266-step run the token-rate
  model under-predicted by 2.3x while step time predicted it within 7%. At these
  sequence lengths per-step overhead dominates, so step time is the stable
  quantity.

* **Peak VRAM is measured on the longest examples in the corpus**, not on a
  random sample. With dynamic padding, a short calibration can easily miss the
  batch that would have OOMed, and a memory estimate that is only sometimes
  right is worse than none.

* **Calibration runs at gradient accumulation 1 and multiplies back.** Timing
  whole optimizer steps at the config's accumulation made '--calibrate 8' mean
  128 forward+backward passes on the longest examples in the corpus -- on a 12B
  model, a coffee break rather than a pre-flight check. Accumulation multiplies
  wall time but not peak memory, since it repeats the same forward/backward
  before one update, so measuring one micro-batch and scaling costs nothing in
  accuracy.

Because the measurements come from worst-case batches, the projection is an
upper bound on step time and therefore errs long. That is the right direction to
err in.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .metrics import (
    CostConfig,
    GpuSampler,
    estimate_cost,
    gpu_memory_snapshot,
    human_duration,
    render_section,
    reset_peak_memory,
)
from .train import compute_schedule


@dataclass
class TokenPlan:
    examples: int
    real_tokens: int
    padded_tokens_per_epoch: int
    p50: int
    p90: int
    p99: int
    longest: int
    max_seq_len: int
    over_cap: int

    # Knowledge-injection corpora repeat each fact deliberately, so epochs
    # multiply on top of repetition that is already there. Tracking it stops the
    # epoch count being chosen as if every example were unique.
    distinct_answers: int = 0
    median_repetitions: int = 1
    max_repetitions: int = 1

    eval_examples: int = 0
    eval_padded_tokens: int = 0

    @property
    def padding_waste_pct(self) -> float:
        if not self.padded_tokens_per_epoch:
            return 0.0
        return 100.0 * (1 - self.real_tokens / self.padded_tokens_per_epoch)


@dataclass
class Calibration:
    """Timing and memory from a short real run.

    ``steps`` counts micro-batches, not optimizer steps. Calibration forces
    gradient accumulation to 1 and multiplies back, because accumulation does
    not change peak memory -- it is the same forward/backward repeated before
    one update -- while it multiplies the wall time by grad_accum. Measuring
    whole optimizer steps at the config's accumulation turned '--calibrate 8'
    into 128 forward+backward passes on the longest examples in the corpus,
    which is not the quick pre-flight the flag promises.
    """

    steps: int
    seconds: float
    padded_tokens: int
    peak_allocated_gb: float
    peak_reserved_gb: float
    device_total_gb: float
    device_name: str
    mean_gpu_watts: float
    # Accumulation the real run will use; calibration itself runs at 1.
    grad_accum: int = 1
    # Measured separately: an evaluation pass is forward-only, so it runs several
    # times faster than a training step and cannot be projected from one.
    eval_seconds: float = 0.0
    eval_padded_tokens: int = 0

    @property
    def eval_tokens_per_second(self) -> float:
        return self.eval_padded_tokens / self.eval_seconds if self.eval_seconds else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.padded_tokens / self.seconds if self.seconds else 0.0

    @property
    def seconds_per_micro_batch(self) -> float:
        return self.seconds / self.steps if self.steps else 0.0

    @property
    def seconds_per_step(self) -> float:
        """One optimizer step: grad_accum micro-batches plus an update.

        Slightly conservative -- the optimizer update is measured once per
        micro-batch here rather than once per step -- which errs long, the
        safe direction for a time estimate.
        """
        return self.seconds_per_micro_batch * self.grad_accum

    @property
    def headroom_gb(self) -> float:
        return self.device_total_gb - self.peak_reserved_gb


@dataclass
class PlanReport:
    run_name: str
    model: str
    tokens: TokenPlan
    schedule: dict[str, int]
    epochs: float
    calibration: Calibration | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    projected_seconds: float | None = None
    eval_steps: int = 0

    def render(self) -> str:
        t = self.tokens
        blocks = [
            f"=== plan: {self.run_name} ===",
            f"    {self.model}",
            "",
            render_section(
                "Data",
                [
                    ("train examples", t.examples),
                    ("seq len p50/p90/p99", f"{t.p50} / {t.p90} / {t.p99}"),
                    ("longest example", t.longest),
                    ("max_seq_len", t.max_seq_len),
                    ("over the cap", t.over_cap),
                    ("tokens / epoch", t.padded_tokens_per_epoch),
                    ("padding waste", f"{t.padding_waste_pct:.1f}%"),
                    ("distinct answers", t.distinct_answers),
                    (
                        "repetition p50/max",
                        f"{t.median_repetitions}x / {t.max_repetitions}x",
                    ),
                ],
                width=22,
            ),
            "",
            render_section(
                "Schedule",
                [
                    ("epochs", self.epochs),
                    ("effective batch", self.schedule["effective_batch"]),
                    ("steps / epoch", self.schedule["steps_per_epoch"]),
                    ("total steps", self.schedule["total_steps"]),
                    ("warmup steps", self.schedule["warmup_steps"]),
                    ("tokens (whole run)", self.total_padded_tokens),
                    (
                        "verbatim exposures",
                        f"{self.exposures_p50}x typical, {self.exposures_max}x worst",
                    ),
                ],
                width=22,
            ),
        ]

        if self.calibration is None:
            blocks += [
                "",
                "No calibration run -- pass --calibrate to measure throughput,",
                "peak VRAM, and cost on this machine.",
            ]
            return "\n".join(blocks)

        c = self.calibration
        # Reserved memory creeps up over a long run as the caching allocator
        # fragments -- a 1,266-step run reserved 12% more than an 8-step
        # calibration predicted -- so anything under a couple of GB spare is not
        # really spare.
        if c.headroom_gb > 3.0:
            verdict = f"fits, {c.headroom_gb:.1f} GB spare"
        elif c.headroom_gb > 1.0:
            verdict = (
                f"{c.headroom_gb:.1f} GB spare -- thin; allocator growth over a "
                "long run can consume this"
            )
        else:
            verdict = "TIGHT -- expect OOM; raise grad_accum or lower max_seq_len"
        blocks += [
            "",
            render_section(
                f"Calibration ({c.steps} micro-batches, worst-case lengths)",
                [
                    ("sec / micro-batch", round(c.seconds_per_micro_batch, 2)),
                    (
                        "sec / step",
                        f"{c.seconds_per_step:.2f}  ({c.grad_accum} micro-batches)",
                    ),
                    ("peak allocated", f"{c.peak_allocated_gb:.2f} GB"),
                    ("peak reserved", f"{c.peak_reserved_gb:.2f} GB"),
                    ("device total", f"{c.device_total_gb:.2f} GB"),
                    ("verdict", verdict),
                ],
                width=22,
            ),
            "",
            render_section(
                "Projection",
                [
                    ("training", human_duration(self.train_seconds)),
                    (
                        "evaluation",
                        f"{human_duration(self.eval_seconds)} "
                        f"({self.eval_passes} passes over "
                        f"{self.tokens.eval_examples:,} examples)",
                    ),
                    ("wall time", human_duration(self.projected_seconds or 0.0)),
                    ("finishes around", self.finish_time()),
                ],
                width=22,
            ),
        ]

        if self.eval_share > 0.25:
            blocks += [
                "",
                f"WARNING: evaluation is {self.eval_share:.0%} of this run. "
                f"train.eval_steps={self.eval_steps} means the eval set is scored "
                f"{self.eval_passes} times.",
                "         Raise eval_steps -- evaluating every few hundred steps "
                "tells you the same thing for a fraction of the time.",
            ]

        if self.cost:
            blocks += ["", "Cost", *[f"  {line}" for line in self.cost["lines"]]]
        blocks += [
            "",
            "Projection assumes steady-state throughput; the first steps and",
            "checkpoint writes are slower, so treat it as a floor.",
        ]
        return "\n".join(blocks)

    @property
    def total_padded_tokens(self) -> int:
        return int(self.tokens.padded_tokens_per_epoch * self.epochs)

    @property
    def train_seconds(self) -> float:
        """Steps times measured step time -- see the module docstring."""
        if self.calibration is None or self.calibration.seconds_per_step <= 0:
            return 0.0
        return self.schedule["total_steps"] * self.calibration.seconds_per_step

    @property
    def eval_share(self) -> float:
        total = (self.projected_seconds or 0.0)
        return self.eval_seconds / total if total else 0.0

    @property
    def eval_passes(self) -> int:
        """How many times the eval set is scored over the run."""
        if not self.tokens.eval_examples or self.eval_steps <= 0:
            return 0
        return max(1, self.schedule["total_steps"] // self.eval_steps)

    @property
    def eval_seconds(self) -> float:
        if self.calibration is None or self.calibration.eval_tokens_per_second <= 0:
            return 0.0
        per_pass = (
            self.tokens.eval_padded_tokens / self.calibration.eval_tokens_per_second
        )
        return per_pass * self.eval_passes

    @property
    def exposures_p50(self) -> int:
        """How often the model sees a typical answer verbatim across the run."""
        return round(self.tokens.median_repetitions * self.epochs)

    @property
    def exposures_max(self) -> int:
        return round(self.tokens.max_repetitions * self.epochs)

    def finish_time(self) -> str:
        if not self.projected_seconds:
            return "n/a"
        return time.strftime("%H:%M on %d %b", time.localtime(time.time() + self.projected_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "model": self.model,
            "tokens": self.tokens.__dict__,
            "schedule": self.schedule,
            "epochs": self.epochs,
            "total_padded_tokens": self.total_padded_tokens,
            "calibration": self.calibration.__dict__ if self.calibration else None,
            "projected_seconds": self.projected_seconds,
            "cost": self.cost,
        }


# ---------------------------------------------------------------------------
# token accounting
# ---------------------------------------------------------------------------


def _padded_tokens_for_epoch(lengths: list[int], batch_size: int, seed: int) -> int:
    """Simulate one epoch's batching to get the padded token count.

    Padding is a property of how examples land together, so with a batch size
    above one it has to be simulated rather than summed.
    """
    order = list(lengths)
    random.Random(seed).shuffle(order)
    total = 0
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        longest = max(batch)
        longest = math.ceil(longest / 8) * 8
        total += longest * len(batch)
    return total


def token_plan(cfg: Config) -> tuple[TokenPlan, list[dict[str, list[int]]]]:
    from .data import encode_all, load_jsonl, split_train_eval
    from .model import load_tokenizer

    tokenizer = load_tokenizer(cfg.model)
    rows = load_jsonl(cfg.data.train_path)
    eval_rows: list = []
    if cfg.data.eval_path:
        eval_rows = load_jsonl(cfg.data.eval_path)
    else:
        rows, eval_rows = split_train_eval(
            rows, cfg.data.eval_split_ratio, cfg.run.seed
        )

    # Encode with the cap lifted so the report can say how many examples the
    # configured cap would actually drop.
    uncapped = cfg.data.model_copy(update={"max_seq_len": 10**9, "drop_overlong": False})
    encoded, _ = encode_all(rows, uncapped, tokenizer)

    lengths = sorted(len(e["input_ids"]) for e in encoded)
    over = sum(1 for n in lengths if n > cfg.data.max_seq_len)

    kept, _ = encode_all(rows, cfg.data, tokenizer)
    kept_lengths = [len(e["input_ids"]) for e in kept]

    eval_examples = 0
    eval_padded = 0
    if eval_rows:
        eval_encoded, _ = encode_all(eval_rows, cfg.data, tokenizer)
        eval_lengths = [len(e["input_ids"]) for e in eval_encoded]
        eval_examples = len(eval_encoded)
        eval_padded = _padded_tokens_for_epoch(
            eval_lengths, cfg.train.per_device_batch_size, cfg.run.seed
        )

    def pct(p: float) -> int:
        return lengths[max(0, int(len(lengths) * p) - 1)]

    answer_counts = Counter(row.answer for row in rows)
    repetitions = sorted(answer_counts.values())

    plan = TokenPlan(
        distinct_answers=len(answer_counts),
        median_repetitions=repetitions[len(repetitions) // 2],
        max_repetitions=repetitions[-1],
        examples=len(kept),
        real_tokens=sum(kept_lengths),
        padded_tokens_per_epoch=_padded_tokens_for_epoch(
            kept_lengths, cfg.train.per_device_batch_size, cfg.run.seed
        ),
        p50=pct(0.50),
        p90=pct(0.90),
        p99=pct(0.99),
        longest=lengths[-1],
        max_seq_len=cfg.data.max_seq_len,
        over_cap=over,
        eval_examples=eval_examples,
        eval_padded_tokens=eval_padded,
    )
    return plan, kept


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def calibrate(cfg: Config, encoded: list[dict[str, list[int]]], steps: int = 8) -> Calibration:
    """Run a few real optimizer steps and measure what the machine does."""
    from datasets import Dataset
    from transformers import Trainer, set_seed

    from . import model as model_mod
    from .collate import PaddedCollator
    from .train import build_training_args

    set_seed(cfg.run.seed)

    # Worst-case batches: the longest examples in the corpus, so peak VRAM is an
    # upper bound rather than a lucky draw.
    ordered = sorted(encoded, key=lambda e: -len(e["input_ids"]))
    needed = steps * cfg.train.per_device_batch_size
    sample = ordered[: max(needed, cfg.train.per_device_batch_size)]
    dataset = Dataset.from_list(sample)

    tokenizer = model_mod.load_tokenizer(cfg.model)
    model = model_mod.load_base_model(cfg.model)
    model = model_mod.attach_lora(model, cfg)

    tmp = Path(tempfile.mkdtemp(prefix="ftlab-calib-"))
    try:
        calib_cfg = cfg.model_copy(deep=True)
        calib_cfg.train.max_steps = steps
        # One micro-batch per step here; the real accumulation is multiplied back
        # in Calibration.seconds_per_step. Peak memory is unaffected, since
        # accumulation repeats the same forward/backward rather than enlarging it.
        calib_cfg.train.grad_accum = 1
        calib_cfg.run.report_to = "none"
        calib_cfg.train.save_steps = 10**9
        calib_cfg.run.output_dir = str(tmp)

        args = build_training_args(
            calib_cfg, compute_schedule(calib_cfg, len(sample)), has_eval=False
        )
        args.output_dir = str(tmp)
        args.save_strategy = "no"

        collator = PaddedCollator(pad_token_id=tokenizer.pad_token_id)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )

        sampler = GpuSampler(2.0)
        reset_peak_memory()
        sampler.start()
        started = time.time()
        trainer.train()
        elapsed = time.time() - started
        # Snapshot now: the evaluation pass below runs through the same collator,
        # and reading the counter afterwards would fold eval tokens into the
        # training throughput and overstate it.
        train_tokens = collator.padded_tokens

        # Evaluation is forward-only and several times faster than a training
        # step, so projecting it from the training rate would overstate it
        # badly. Measure it instead.
        eval_sample = Dataset.from_list(ordered[: min(24, len(ordered))])
        tokens_before = collator.padded_tokens
        eval_started = time.time()
        trainer.evaluate(eval_dataset=eval_sample)
        eval_seconds = time.time() - eval_started
        eval_tokens = collator.padded_tokens - tokens_before

        sampler.stop()

        snapshot = gpu_memory_snapshot()
        return Calibration(
            steps=steps,
            seconds=elapsed,
            padded_tokens=train_tokens,
            peak_allocated_gb=snapshot.get("peak_allocated_gb", 0.0),
            peak_reserved_gb=snapshot.get("peak_reserved_gb", 0.0),
            device_total_gb=snapshot.get("device_total_gb", 0.0),
            device_name=snapshot.get("device_name", ""),
            mean_gpu_watts=sampler.mean_power_w,
            grad_accum=cfg.train.grad_accum,
            eval_seconds=eval_seconds,
            eval_padded_tokens=eval_tokens,
        )
    finally:
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def plan(cfg: Config, calibrate_steps: int = 0) -> PlanReport:
    tokens, encoded = token_plan(cfg)
    schedule = compute_schedule(cfg, tokens.examples)
    epochs = (
        cfg.train.epochs
        if cfg.train.max_steps <= 0
        else schedule["total_steps"] / max(1, schedule["steps_per_epoch"])
    )

    report = PlanReport(
        run_name=cfg.run.name,
        model=cfg.model.base,
        tokens=tokens,
        schedule=schedule,
        epochs=round(epochs, 3),
    )

    if tokens.eval_examples:
        report.eval_steps = cfg.train.eval_steps

    if calibrate_steps > 0:
        measurement = calibrate(cfg, encoded, calibrate_steps)
        report.calibration = measurement
        if measurement.seconds_per_step > 0:
            report.projected_seconds = report.train_seconds + report.eval_seconds
        # Only project cost when GPU draw was actually sampled; deriving energy
        # from an unmeasured zero would advertise a free run.
        if report.projected_seconds and measurement.mean_gpu_watts > 0:
            estimate = estimate_cost(
                CostConfig(
                    electricity_usd_per_kwh=cfg.metrics.electricity_usd_per_kwh,
                    system_overhead_watts=cfg.metrics.system_overhead_watts,
                    cloud_usd_per_hour=cfg.metrics.cloud_usd_per_hour,
                ),
                report.projected_seconds / 3600.0,
                measurement.mean_gpu_watts,
            )
            report.cost = {**estimate.__dict__, "lines": estimate.lines()}

    return report


def write_plan(report: PlanReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path

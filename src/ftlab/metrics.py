"""Run instrumentation: throughput, memory, energy, and cost.

Everything here is measured rather than modelled. Power comes from nvidia-smi,
memory from torch's own allocator counters, tokens from the collator that
actually built the batches. The only inputs that are *assumptions* are the two
prices in ``CostConfig`` -- an electricity rate and an optional cloud
comparison -- and they are reported alongside their values so nobody mistakes a
default for a measurement.

The one thing worth knowing about GPU power: nvidia-smi reports instantaneous
board draw, so a mean over 5-second samples is a good estimate of energy but
will not match a wall meter exactly. It excludes the rest of the machine, which
is why ``system_overhead_watts`` exists as a separate, explicitly-guessed term.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

NVIDIA_SMI_QUERY = "power.draw,utilization.gpu,memory.used,temperature.gpu"


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


@dataclass
class CostConfig:
    """Prices. These are inputs, not measurements -- set them for your site."""

    electricity_usd_per_kwh: float = 0.17
    # Everything in the box that is not the GPU: CPU, RAM, fans, PSU losses.
    # A rough constant is honest here; metering it properly needs a wall plug.
    system_overhead_watts: float = 120.0
    # Optional yardstick: what an equivalent rented GPU-hour would cost you.
    # Left at zero because a wrong number is worse than no number.
    cloud_usd_per_hour: float = 0.0


@dataclass
class CostEstimate:
    hours: float
    mean_gpu_watts: float
    system_overhead_watts: float
    energy_kwh: float
    electricity_usd: float
    cloud_usd: float | None
    usd_per_kwh: float

    def lines(self) -> list[str]:
        out = [
            f"mean GPU draw      {self.mean_gpu_watts:,.0f} W",
            f"system overhead    {self.system_overhead_watts:,.0f} W (assumed)",
            f"energy             {self.energy_kwh:.3f} kWh",
            f"electricity        ${self.electricity_usd:.2f} "
            f"(at ${self.usd_per_kwh:.3f}/kWh, assumed)",
        ]
        if self.cloud_usd is not None:
            out.append(f"cloud equivalent   ${self.cloud_usd:.2f}")
        else:
            out.append("cloud equivalent   not set (metrics.cloud_usd_per_hour)")
        return out


def estimate_cost(cfg: CostConfig, hours: float, mean_gpu_watts: float) -> CostEstimate:
    total_watts = mean_gpu_watts + cfg.system_overhead_watts
    energy_kwh = total_watts * hours / 1000.0
    return CostEstimate(
        hours=hours,
        mean_gpu_watts=mean_gpu_watts,
        system_overhead_watts=cfg.system_overhead_watts,
        energy_kwh=energy_kwh,
        electricity_usd=energy_kwh * cfg.electricity_usd_per_kwh,
        cloud_usd=(
            cfg.cloud_usd_per_hour * hours if cfg.cloud_usd_per_hour > 0 else None
        ),
        usd_per_kwh=cfg.electricity_usd_per_kwh,
    )


# ---------------------------------------------------------------------------
# GPU sampling
# ---------------------------------------------------------------------------


@dataclass
class GpuSample:
    at: float
    power_w: float
    util_pct: float
    mem_used_mb: float
    temp_c: float


class GpuSampler:
    """Polls nvidia-smi on a background thread.

    Shelling out rather than binding pynvml keeps this dependency-free and works
    the same on Windows, which is where this runs. At a five-second interval the
    subprocess cost is irrelevant next to a training step.
    """

    def __init__(self, interval_seconds: float = 5.0, device_index: int = 0) -> None:
        self.interval = max(0.5, interval_seconds)
        self.device_index = device_index
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = self._probe() is not None

    def _probe(self) -> tuple[float, float, float, float] | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={NVIDIA_SMI_QUERY}",
                    "--format=csv,noheader,nounits",
                    f"--id={self.device_index}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip().splitlines()[0]
            power, util, mem, temp = (p.strip() for p in out.split(","))
            return float(power), float(util), float(mem), float(temp)
        except Exception:  # noqa: BLE001 - absence of nvidia-smi is not an error
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            reading = self._probe()
            if reading is not None:
                power, util, mem, temp = reading
                self.samples.append(GpuSample(time.time(), power, util, mem, temp))
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)
        self._thread = None

    # -- summaries -------------------------------------------------------

    @property
    def mean_power_w(self) -> float:
        active = [s.power_w for s in self.samples if s.util_pct > 5]
        pool = active or [s.power_w for s in self.samples]
        return sum(pool) / len(pool) if pool else 0.0

    @property
    def peak_power_w(self) -> float:
        return max((s.power_w for s in self.samples), default=0.0)

    @property
    def mean_util_pct(self) -> float:
        return (
            sum(s.util_pct for s in self.samples) / len(self.samples)
            if self.samples
            else 0.0
        )

    @property
    def peak_mem_mb(self) -> float:
        return max((s.mem_used_mb for s in self.samples), default=0.0)

    @property
    def peak_temp_c(self) -> float:
        return max((s.temp_c for s in self.samples), default=0.0)


# ---------------------------------------------------------------------------
# torch memory
# ---------------------------------------------------------------------------


def gpu_memory_snapshot() -> dict[str, float]:
    """Peak allocator figures in GB, plus what the card actually has."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "peak_allocated_gb": torch.cuda.max_memory_allocated(index) / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(index) / 1024**3,
            "device_total_gb": props.total_memory / 1024**3,
            "device_name": props.name,
        }
    except Exception:  # noqa: BLE001
        return {}


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_section(title: str, rows: list[tuple[str, Any]], width: int = 20) -> str:
    lines = [title]
    for label, value in rows:
        lines.append(f"  {label:<{width}} {_fmt(value)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the callback
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Everything measured about one run."""

    run_name: str = ""
    started_at: str = ""
    wall_seconds: float = 0.0

    total_steps: int = 0
    epochs_completed: float = 0.0
    effective_batch: int = 0

    train_tokens: int = 0
    padded_tokens: int = 0
    train_examples: int = 0

    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    device_total_gb: float = 0.0
    device_name: str = ""

    mean_gpu_watts: float = 0.0
    peak_gpu_watts: float = 0.0
    mean_gpu_util_pct: float = 0.0
    peak_temp_c: float = 0.0
    power_samples: int = 0

    first_train_loss: float | None = None
    last_train_loss: float | None = None
    first_eval_loss: float | None = None
    last_eval_loss: float | None = None
    best_eval_loss: float | None = None
    best_eval_step: int | None = None

    cost: dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------

    @property
    def hours(self) -> float:
        return self.wall_seconds / 3600.0

    @property
    def tokens_per_second(self) -> float:
        return self.padded_tokens / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def seconds_per_step(self) -> float:
        return self.wall_seconds / self.total_steps if self.total_steps else 0.0

    @property
    def padding_waste_pct(self) -> float:
        if not self.padded_tokens:
            return 0.0
        return 100.0 * (1 - self.train_tokens / self.padded_tokens)

    @property
    def headroom_gb(self) -> float:
        return max(0.0, self.device_total_gb - self.peak_reserved_gb)

    def report(self) -> str:
        blocks = [
            f"=== run: {self.run_name} ===",
            "",
            render_section(
                "Schedule",
                [
                    ("epochs", round(self.epochs_completed, 3)),
                    ("optimizer steps", self.total_steps),
                    ("effective batch", self.effective_batch),
                    ("train examples", self.train_examples),
                ],
            ),
            "",
            render_section(
                "Throughput",
                [
                    ("wall time", human_duration(self.wall_seconds)),
                    ("sec / step", round(self.seconds_per_step, 2)),
                    ("tokens / sec", round(self.tokens_per_second)),
                    ("tokens (real)", self.train_tokens),
                    ("tokens (padded)", self.padded_tokens),
                    ("padding waste", f"{self.padding_waste_pct:.1f}%"),
                ],
            ),
            "",
            render_section(
                "Memory",
                [
                    ("device", self.device_name or "n/a"),
                    ("peak allocated", f"{self.peak_allocated_gb:.2f} GB"),
                    ("peak reserved", f"{self.peak_reserved_gb:.2f} GB"),
                    ("device total", f"{self.device_total_gb:.2f} GB"),
                    ("headroom", f"{self.headroom_gb:.2f} GB"),
                ],
            ),
        ]

        if self.power_samples:
            blocks += [
                "",
                render_section(
                    "Power",
                    [
                        ("mean draw", f"{self.mean_gpu_watts:,.0f} W"),
                        ("peak draw", f"{self.peak_gpu_watts:,.0f} W"),
                        ("mean utilisation", f"{self.mean_gpu_util_pct:.0f}%"),
                        ("peak temperature", f"{self.peak_temp_c:.0f} C"),
                        ("samples", self.power_samples),
                    ],
                ),
            ]
        if self.cost:
            blocks += ["", "Cost", *[f"  {line}" for line in self.cost["lines"]]]

        loss_rows: list[tuple[str, Any]] = []
        if self.first_train_loss is not None:
            loss_rows.append(
                (
                    "train first -> last",
                    f"{self.first_train_loss:.4f} -> {self.last_train_loss:.4f}",
                )
            )
        if self.first_eval_loss is not None:
            loss_rows.append(
                ("eval first -> last", f"{self.first_eval_loss:.4f} -> {self.last_eval_loss:.4f}")
            )
        if self.best_eval_loss is not None:
            loss_rows.append(
                ("best eval", f"{self.best_eval_loss:.4f} @ step {self.best_eval_step}")
            )
        if loss_rows:
            blocks += ["", render_section("Loss", loss_rows)]

        return "\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "hours": self.hours,
                "tokens_per_second": self.tokens_per_second,
                "seconds_per_step": self.seconds_per_step,
                "padding_waste_pct": self.padding_waste_pct,
                "headroom_gb": self.headroom_gb,
            }
        )
        return data


class MetricsCallback:
    """Records throughput, memory, power and loss for one training run.

    Not subclassed from TrainerCallback at import time so this module stays
    importable without transformers -- ``ftlab plan`` uses the cost model and
    the sampler without ever constructing a Trainer.
    """

    def __init__(
        self,
        run_name: str,
        out_dir: Path,
        cost: CostConfig,
        collator: Any = None,
        sample_seconds: float = 5.0,
    ) -> None:
        self.metrics = TrainingMetrics(run_name=run_name)
        self.out_dir = Path(out_dir)
        self.cost_cfg = cost
        self.collator = collator
        self.sampler = GpuSampler(sample_seconds)
        self.timeline: list[dict[str, Any]] = []
        self._t0 = 0.0

    # -- hooks -----------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        reset_peak_memory()
        self._t0 = time.time()
        self.metrics.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metrics.effective_batch = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        self.sampler.start()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return control
        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0.0, 4),
            "elapsed_s": round(time.time() - self._t0, 2),
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        if self.sampler.samples:
            entry["gpu_w"] = round(self.sampler.samples[-1].power_w, 1)
        self.timeline.append(entry)

        if "loss" in logs:
            value = float(logs["loss"])
            if self.metrics.first_train_loss is None:
                self.metrics.first_train_loss = value
            self.metrics.last_train_loss = value
        if "eval_loss" in logs:
            value = float(logs["eval_loss"])
            if self.metrics.first_eval_loss is None:
                self.metrics.first_eval_loss = value
            self.metrics.last_eval_loss = value
            if self.metrics.best_eval_loss is None or value < self.metrics.best_eval_loss:
                self.metrics.best_eval_loss = value
                self.metrics.best_eval_step = state.global_step
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self.sampler.stop()
        m = self.metrics
        m.wall_seconds = time.time() - self._t0
        m.total_steps = state.global_step
        m.epochs_completed = float(state.epoch or 0.0)

        if self.collator is not None:
            m.train_tokens = getattr(self.collator, "real_tokens", 0)
            m.padded_tokens = getattr(self.collator, "padded_tokens", 0)

        snapshot = gpu_memory_snapshot()
        m.peak_allocated_gb = snapshot.get("peak_allocated_gb", 0.0)
        m.peak_reserved_gb = snapshot.get("peak_reserved_gb", 0.0)
        m.device_total_gb = snapshot.get("device_total_gb", 0.0)
        m.device_name = snapshot.get("device_name", "")

        m.mean_gpu_watts = self.sampler.mean_power_w
        m.peak_gpu_watts = self.sampler.peak_power_w
        m.mean_gpu_util_pct = self.sampler.mean_util_pct
        m.peak_temp_c = self.sampler.peak_temp_c
        m.power_samples = len(self.sampler.samples)

        if m.power_samples:
            estimate = estimate_cost(self.cost_cfg, m.hours, m.mean_gpu_watts)
            m.cost = {**asdict(estimate), "lines": estimate.lines()}
        else:
            # Without power samples the GPU term is zero, and an energy figure
            # built from that reads as a cheap run rather than as a missing
            # measurement. Say what happened instead.
            m.cost = {
                "lines": [
                    "unavailable -- nvidia-smi returned no samples, so GPU draw "
                    "was never measured",
                    f"elapsed           {human_duration(m.wall_seconds)}",
                ]
            }

        self.write()
        return control

    # -- output ----------------------------------------------------------

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "metrics.json").write_text(
            json.dumps(self.metrics.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        with (self.out_dir / "metrics_timeline.jsonl").open("w", encoding="utf-8") as fh:
            for entry in self.timeline:
                fh.write(json.dumps(entry) + "\n")
        (self.out_dir / "metrics_report.txt").write_text(
            self.metrics.report(), encoding="utf-8"
        )


def build_callback(*args: Any, **kwargs: Any):
    """Return a MetricsCallback that transformers will accept as a callback.

    The mixin is applied here rather than at class definition so importing this
    module never requires transformers.
    """
    from transformers import TrainerCallback

    class _Callback(MetricsCallback, TrainerCallback):
        pass

    return _Callback(*args, **kwargs)


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))

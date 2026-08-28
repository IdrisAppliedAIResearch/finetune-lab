"""Detect a training run that has gone wrong without saying so.

Written after a run that never raised, never logged a warning, and quietly went
from 12.8 s/step to 204 s/step -- turning a 39-minute job into a five-hour one.
The failure that costs the most is not the one that crashes; it is the one that
keeps producing plausible output slowly enough that nobody looks.

Three things can go wrong quietly, and a guard that catches only crashes catches
none of them:

* **Slowdown.** Step time drifts up as the allocator fragments and the driver
  starts backing reservations with system RAM. Detected by comparing current
  step time against this run's *own* early baseline, because the healthy rate
  depends on the corpus and the hardware and cannot be hardcoded.
* **Hang.** The step counter stops advancing. A log that stops growing looks
  exactly like a log that is being written slowly, so this needs a clock rather
  than a pattern.
* **Crash.** The ordinary case, and the only one most watchdogs check.

Silence is not health. The scan returns an explicit verdict every time, so
"nothing reported" can never be mistaken for "running fine".
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

# tqdm writes "  45%|####5     | 76/168 [1:15:11<5:13:39, 204.56s/it]", and the
# rate flips to it/s once a step takes under a second -- both forms have to
# parse or a fast run looks like no run at all.
PROGRESS = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[\d:]+)<(?P<remaining>[\d:?]+),\s*"
    r"(?P<rate>[\d.]+)(?P<unit>s/it|it/s)\]"
)
FAILURE = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|RuntimeError|"
    r"torch\.cuda\.OutOfMemoryError|Killed",
)
DONE = re.compile(r"\[ftlab\] done\.")

# How much slower than its own baseline a run may get before it is called out.
# The observed failure was 16x, so 2.5x is far inside it while leaving room for
# ordinary variation between a short batch and a long one.
SLOW_FACTOR = 2.5
# Steps used to establish the baseline. Enough to average over batch-length
# variation, few enough to be established early.
BASELINE_STEPS = 8
# No new step for this long counts as hung rather than slow.
HANG_SECONDS = 900


@dataclass
class Health:
    state: str  # ok | slow | hang | crashed | done | starting
    detail: str
    step: int = 0
    total: int = 0
    seconds_per_step: float = 0.0
    baseline: float = 0.0

    @property
    def alarming(self) -> bool:
        return self.state in {"slow", "hang", "crashed"}

    def line(self) -> str:
        where = f"step {self.step}/{self.total}" if self.total else "no steps yet"
        return f"[{self.state.upper()}] {where} -- {self.detail}"


def _rates(text: str, total_steps: int | None) -> list[tuple[int, float]]:
    """(step, seconds-per-step) for every progress report in the log.

    Filtered to the training bar. Evaluation writes its own progress with a
    different total, and averaging the two together would hide a training
    slowdown behind fast eval batches.
    """
    out: list[tuple[int, float]] = []
    for match in PROGRESS.finditer(text):
        total = int(match["total"])
        if total_steps and total != total_steps:
            continue
        rate = float(match["rate"])
        seconds = rate if match["unit"] == "s/it" else (1.0 / rate if rate else 0.0)
        out.append((int(match["done"]), seconds))
    return out


def total_steps_of(text: str) -> int | None:
    match = re.search(r"\[ftlab\] schedule: (\d+) steps", text)
    return int(match.group(1)) if match else None


def scan(
    log_path: str | Path,
    *,
    slow_factor: float = SLOW_FACTOR,
    hang_seconds: int = HANG_SECONDS,
    now: float | None = None,
) -> Health:
    """Read a training log and say plainly what state the run is in."""
    path = Path(log_path)
    if not path.exists():
        return Health("starting", f"no log at {path} yet")

    text = path.read_text(encoding="utf-8", errors="ignore")

    if DONE.search(text):
        return Health("done", "run completed")
    if FAILURE.search(text):
        excerpt = FAILURE.search(text).group(0)
        return Health("crashed", f"log contains {excerpt!r}")

    total = total_steps_of(text)
    rates = _rates(text, total)
    if len(rates) < 2:
        return Health("starting", "no training steps reported yet", total=total or 0)

    step, current = rates[-1]
    baseline_pool = [s for _, s in rates[:BASELINE_STEPS] if s > 0]
    baseline = statistics.median(baseline_pool) if baseline_pool else 0.0

    age = (now or time.time()) - path.stat().st_mtime
    if age > hang_seconds:
        return Health(
            "hang",
            f"log has not been written for {age / 60:.0f} min",
            step=step, total=total or 0, seconds_per_step=current, baseline=baseline,
        )

    if baseline and current > slow_factor * baseline:
        return Health(
            "slow",
            f"{current:.1f} s/step against a {baseline:.1f} s/step baseline "
            f"({current / baseline:.1f}x) -- check VRAM pressure",
            step=step, total=total or 0, seconds_per_step=current, baseline=baseline,
        )

    remaining = ((total or step) - step) * current
    return Health(
        "ok",
        f"{current:.1f} s/step (baseline {baseline:.1f}), "
        f"~{remaining / 60:.0f} min left",
        step=step, total=total or 0, seconds_per_step=current, baseline=baseline,
    )

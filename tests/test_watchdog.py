"""The watchdog: does it catch a run that fails without saying so?"""

from __future__ import annotations

import time

from ftlab.watchdog import scan

HEADER = "[ftlab] schedule: 168 steps (effective batch 16, 5 warmup)\n"


def bar(done: int, total: int, rate: float, unit: str = "s/it") -> str:
    return f"  1%|1  | {done}/{total} [00:30<10:00, {rate:.2f}{unit}]\n"


def write(tmp_path, body: str, *, age: float = 0.0):
    path = tmp_path / "train.log"
    path.write_text(body, encoding="utf-8")
    if age:
        stamp = time.time() - age
        import os

        os.utime(path, (stamp, stamp))
    return path


def healthy(steps: int = 12, rate: float = 13.0) -> str:
    return HEADER + "".join(bar(i, 168, rate) for i in range(1, steps + 1))


# ---------------------------------------------------------------------------
# the failure that prompted this
# ---------------------------------------------------------------------------


def test_catches_a_run_that_silently_got_slower(tmp_path):
    """The real incident: 12.8 s/step drifting to 204 s/step, no error logged.

    Nothing raised, nothing warned, and the run was still producing progress
    lines -- so every check that looks for a crash or for silence reported a
    healthy run for four hours.
    """
    # Sustained, because that is what happened: seventy-odd steps at 204 s,
    # not one slow reading. A single spike is noise and must not fire -- see
    # test_an_evaluation_pause_is_not_a_stall for why that matters.
    log = healthy(steps=10, rate=12.8) + "".join(
        bar(i, 168, 204.6) for i in range(66, 71)
    )
    health = scan(write(tmp_path, log))
    assert health.state == "slow"
    assert health.alarming
    assert "16.0x" in health.detail
    assert health.step == 70


def test_ordinary_variation_is_not_an_alarm(tmp_path):
    """Batches differ in length, so step time wobbles. That is not a failure."""
    log = healthy(steps=10, rate=13.0) + bar(40, 168, 22.0)
    assert scan(write(tmp_path, log)).state == "ok"


def test_baseline_comes_from_the_run_itself(tmp_path):
    """A slow machine is not a broken one.

    The healthy rate depends on corpus and hardware, so a hardcoded threshold
    would either miss a regression on a fast box or cry wolf on a slow one.
    """
    slow_box = healthy(steps=10, rate=60.0) + "".join(
        bar(i, 168, 90.0) for i in range(36, 41)
    )
    assert scan(write(tmp_path, slow_box)).state == "ok"

    fast_box = healthy(steps=10, rate=2.0) + "".join(
        bar(i, 168, 30.0) for i in range(36, 41)
    )
    assert scan(write(tmp_path, fast_box)).state == "slow"


# ---------------------------------------------------------------------------
# the other two silent failures
# ---------------------------------------------------------------------------


def test_catches_a_hang(tmp_path):
    """A log that stopped growing reads exactly like one being written slowly."""
    health = scan(write(tmp_path, healthy(), age=3600), hang_seconds=900)
    assert health.state == "hang"
    assert "60 min" in health.detail


def test_a_recent_log_is_not_a_hang(tmp_path):
    assert scan(write(tmp_path, healthy(), age=60), hang_seconds=900).state == "ok"


def test_catches_a_crash(tmp_path):
    log = healthy() + "Traceback (most recent call last):\n  ...\n"
    assert scan(write(tmp_path, log)).state == "crashed"


def test_catches_a_cuda_oom(tmp_path):
    log = healthy() + "torch.cuda.OutOfMemoryError: CUDA out of memory.\n"
    assert scan(write(tmp_path, log)).state == "crashed"


def test_completion_beats_everything(tmp_path):
    """A finished run must not be reported as hung because it stopped writing."""
    log = healthy() + "[ftlab] done. adapter -> outputs/x/adapter\n"
    assert scan(write(tmp_path, log, age=99999)).state == "done"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_eval_progress_does_not_mask_a_training_slowdown(tmp_path):
    """Evaluation writes its own bar with its own total.

    Averaging the two hides a training stall behind fast eval batches, which is
    the shape of the bug this whole module exists to catch.
    """
    log = (
        healthy(steps=10, rate=12.8)
        + "  69%|## | 159/231 [00:40<00:16,  4.49it/s]\n"
        + "".join(bar(i, 168, 204.6) for i in range(66, 71))
    )
    health = scan(write(tmp_path, log))
    assert health.state == "slow"
    assert health.total == 168


def test_sub_second_rates_parse(tmp_path):
    """tqdm flips to it/s when a step is fast; a run must not look absent."""
    log = HEADER + "".join(
        f"  1%|1 | {i}/168 [00:05<02:00,  4.00it/s]\n" for i in range(1, 10)
    )
    health = scan(write(tmp_path, log))
    assert health.state == "ok"
    assert abs(health.seconds_per_step - 0.25) < 1e-6


def test_a_run_that_has_not_started_says_so(tmp_path):
    assert scan(write(tmp_path, HEADER)).state == "starting"
    assert scan(tmp_path / "missing.log").state == "starting"


def test_every_scan_returns_a_verdict(tmp_path):
    """Silence must never be mistaken for health.

    A watchdog that only speaks up on failure is indistinguishable from a
    watchdog that has died, which is how the original incident ran for hours.
    """
    for body, age in ((HEADER, 0), (healthy(), 0), (healthy(), 99999)):
        health = scan(write(tmp_path, body, age=age))
        assert health.state
        assert health.line()


def test_an_evaluation_pause_is_not_a_stall(tmp_path):
    """Measured false positive: tqdm smears the eval pause into the next steps.

    An 84-second evaluation every 25 steps turned a steady 18 s/step into a
    reported 43.23 then 36.00 before decaying back. The first version compared
    the last reading to the baseline and fired every time, and an alarm that
    goes off on every eval is one nobody reads.
    """
    header = "[ftlab] schedule: 230 steps (effective batch 16, 6 warmup)\n"
    log = header + "".join(bar(i, 230, 16.3) for i in range(1, 12))
    log += "".join(bar(i, 230, r) for i, r in
                   ((148, 19.96), (149, 18.12), (150, 18.43), (151, 43.23), (152, 36.00)))
    health = scan(write(tmp_path, log))
    assert health.state == "ok", health.detail


def test_a_sustained_stall_still_fires(tmp_path):
    """The real incident, which must survive the fix for the false positive.

    12.8 s/step drifting to 204 and staying there for seventy steps. Taking a
    median over recent steps must not blunt this.
    """
    log = HEADER + "".join(bar(i, 168, 12.8) for i in range(1, 12))
    log += "".join(bar(i, 168, 204.6) for i in range(60, 70))
    health = scan(write(tmp_path, log))
    assert health.state == "slow"
    assert "16.0x" in health.detail

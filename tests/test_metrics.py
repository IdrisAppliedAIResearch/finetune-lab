"""Metrics, cost arithmetic, and plan projection.

No GPU or network needed: the sampler degrades to empty summaries when
nvidia-smi is unavailable, which is also the path CI takes.
"""

from __future__ import annotations

import math

import pytest

from ftlab.collate import PaddedCollator
from ftlab.metrics import (
    CostConfig,
    GpuSample,
    GpuSampler,
    TrainingMetrics,
    estimate_cost,
    human_duration,
)
from ftlab.plan import _padded_tokens_for_epoch

# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


def test_energy_is_watts_times_hours():
    """400 W GPU + 100 W system for 2 h = 1 kWh."""
    estimate = estimate_cost(
        CostConfig(electricity_usd_per_kwh=0.20, system_overhead_watts=100.0),
        hours=2.0,
        mean_gpu_watts=400.0,
    )
    assert estimate.energy_kwh == pytest.approx(1.0)
    assert estimate.electricity_usd == pytest.approx(0.20)


def test_system_overhead_is_included():
    """Ignoring everything but the GPU understates a long run materially."""
    with_overhead = estimate_cost(CostConfig(system_overhead_watts=150.0), 1.0, 300.0)
    without = estimate_cost(CostConfig(system_overhead_watts=0.0), 1.0, 300.0)
    assert with_overhead.energy_kwh > without.energy_kwh
    assert with_overhead.energy_kwh == pytest.approx(0.45)


def test_cloud_comparison_is_opt_in():
    """A wrong rented-GPU price is worse than none, so zero means silent."""
    off = estimate_cost(CostConfig(cloud_usd_per_hour=0.0), 3.0, 400.0)
    assert off.cloud_usd is None
    assert any("not set" in line for line in off.lines())

    on = estimate_cost(CostConfig(cloud_usd_per_hour=2.5), 3.0, 400.0)
    assert on.cloud_usd == pytest.approx(7.5)


def test_cost_lines_label_assumptions():
    """Prices are inputs; the report must not let them read as measurements."""
    lines = " ".join(estimate_cost(CostConfig(), 1.0, 400.0).lines())
    assert lines.count("assumed") >= 2


# ---------------------------------------------------------------------------
# token counting
# ---------------------------------------------------------------------------


def test_collator_counts_real_and_padded_tokens():
    collator = PaddedCollator(pad_token_id=0, pad_to_multiple_of=1)
    batch = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4], "labels": [4], "attention_mask": [1]},
    ]
    collator(batch)
    assert collator.real_tokens == 4      # 3 + 1
    assert collator.padded_tokens == 6    # 3 wide, 2 rows


def test_collator_counters_accumulate_across_batches():
    collator = PaddedCollator(pad_token_id=0, pad_to_multiple_of=1)
    row = [{"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]}]
    collator(row)
    collator(row)
    assert collator.real_tokens == 4
    assert collator.padded_tokens == 4


def test_padded_token_estimate_matches_batch_size_one():
    """At batch size 1 each example pads only to the multiple of 8."""
    lengths = [10, 20, 33]
    expected = sum(math.ceil(n / 8) * 8 for n in lengths)
    assert _padded_tokens_for_epoch(lengths, batch_size=1, seed=1) == expected


def test_larger_batches_pad_to_the_longest_member():
    """Batching mixed lengths costs padding; the estimate must reflect that."""
    lengths = [8, 8, 64, 64]
    at_one = _padded_tokens_for_epoch(lengths, batch_size=1, seed=1)
    at_four = _padded_tokens_for_epoch(lengths, batch_size=4, seed=1)
    assert at_four > at_one
    assert at_four == 64 * 4


def test_padded_token_estimate_is_deterministic():
    lengths = list(range(1, 200))
    a = _padded_tokens_for_epoch(lengths, 4, seed=7)
    b = _padded_tokens_for_epoch(lengths, 4, seed=7)
    assert a == b


# ---------------------------------------------------------------------------
# derived metrics
# ---------------------------------------------------------------------------


def test_derived_throughput_and_waste():
    metrics = TrainingMetrics(
        run_name="t",
        wall_seconds=100.0,
        total_steps=50,
        train_tokens=9_000,
        padded_tokens=10_000,
        peak_reserved_gb=20.0,
        device_total_gb=32.0,
    )
    assert metrics.tokens_per_second == pytest.approx(100.0)
    assert metrics.seconds_per_step == pytest.approx(2.0)
    assert metrics.padding_waste_pct == pytest.approx(10.0)
    assert metrics.headroom_gb == pytest.approx(12.0)


def test_derived_metrics_survive_a_zero_length_run():
    """A run that dies immediately must not divide by zero on the way out."""
    metrics = TrainingMetrics(run_name="t")
    assert metrics.tokens_per_second == 0.0
    assert metrics.seconds_per_step == 0.0
    assert metrics.padding_waste_pct == 0.0
    assert metrics.headroom_gb == 0.0
    assert "run: t" in metrics.report()


def test_report_round_trips_through_dict():
    metrics = TrainingMetrics(
        run_name="r", wall_seconds=10.0, total_steps=5, padded_tokens=100
    )
    data = metrics.to_dict()
    known = set(TrainingMetrics().__dict__)
    restored = TrainingMetrics(**{k: v for k, v in data.items() if k in known})
    assert restored.report() == metrics.report()


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------


def test_sampler_summaries_are_safe_when_empty():
    sampler = GpuSampler.__new__(GpuSampler)
    sampler.samples = []
    assert sampler.mean_power_w == 0.0
    assert sampler.peak_power_w == 0.0
    assert sampler.mean_util_pct == 0.0
    assert sampler.peak_temp_c == 0.0


def test_mean_power_prefers_samples_taken_under_load():
    """Idle samples between steps would drag the mean below the real draw."""
    sampler = GpuSampler.__new__(GpuSampler)
    sampler.samples = [
        GpuSample(0.0, 60.0, 0.0, 500.0, 35.0),    # idle
        GpuSample(1.0, 400.0, 95.0, 20000.0, 70.0),
        GpuSample(2.0, 420.0, 98.0, 20000.0, 72.0),
    ]
    assert sampler.mean_power_w == pytest.approx(410.0)
    assert sampler.peak_power_w == pytest.approx(420.0)


def test_mean_power_falls_back_when_nothing_was_under_load():
    sampler = GpuSampler.__new__(GpuSampler)
    sampler.samples = [GpuSample(0.0, 60.0, 0.0, 500.0, 35.0)]
    assert sampler.mean_power_w == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0.0s"),
        (45.2, "45.2s"),
        (95.0, "1m 35s"),
        (3600.0, "1h 00m 00s"),
        (8045.0, "2h 14m 05s"),
        (-5.0, "0.0s"),
    ],
)
def test_human_duration(seconds, expected):
    assert human_duration(seconds) == expected


def test_metrics_filenames_do_not_collide():
    """The callback and the trainer both want to write a metrics file.

    They used to want the same name, and since the callback writes during
    on_train_end and the trainer writes just after, the richer report was
    silently overwritten and 'ftlab report' failed on every finished run.
    """
    import inspect

    from ftlab import train as train_mod
    from ftlab.metrics import MetricsCallback

    trainer_source = inspect.getsource(train_mod.train)
    callback_source = inspect.getsource(MetricsCallback.write)

    assert '"metrics.json"' in callback_source
    assert '"metrics.json"' not in trainer_source
    assert "trainer_metrics.json" in trainer_source

"""Calibration sampling: the memory question and the time question differ."""

from __future__ import annotations

import statistics

from ftlab.plan import representative_sample


def corpus(lengths: list[int]) -> list[dict[str, list[int]]]:
    return [{"input_ids": [0] * n} for n in lengths]


def lengths(rows) -> list[int]:
    return [len(r["input_ids"]) for r in rows]


def skewed() -> list[dict[str, list[int]]]:
    """A length distribution shaped like the real one this was measured on.

    The demo corpus under Gemma 4's tokenizer runs min 128, mean 530, max 1934 --
    a short body with a long thin tail, which is what makes head-of-the-sorted-
    list sampling so badly unrepresentative. This reproduces that shape.
    """
    values = [128 + int(1806 * (i / 999) ** 4) for i in range(1000)]
    return sorted(corpus(values), key=lambda e: -len(e["input_ids"]))


def test_sample_tracks_the_corpus_mean_not_its_tail():
    """The bug this guards: timing the longest examples and calling it an estimate.

    Taking the head of a length-sorted corpus measured a step time 2.3x the real
    one on the 12B run, which projected 4h41m for a job that ran in well under
    half that. The sample's mean length has to look like the corpus's.
    """
    rows = skewed()
    corpus_mean = statistics.mean(lengths(rows))
    head_mean = statistics.mean(lengths(rows[:8]))
    sample_mean = statistics.mean(lengths(representative_sample(rows, 8)))

    assert head_mean > 3 * corpus_mean          # the old behaviour, quantified
    assert abs(sample_mean - corpus_mean) / corpus_mean < 0.15


def test_sample_spans_the_whole_range():
    """Quantiles, not a slice: the sample must contain short and long alike."""
    rows = skewed()
    picked = lengths(representative_sample(rows, 8))
    assert max(picked) > statistics.mean(lengths(rows))
    assert min(picked) < statistics.median(lengths(rows))
    assert picked == sorted(picked, reverse=True)


def test_sample_is_deterministic():
    rows = skewed()
    assert lengths(representative_sample(rows, 8)) == lengths(representative_sample(rows, 8))


def test_asking_for_more_than_exists_returns_everything():
    rows = corpus([100, 200, 300])
    assert len(representative_sample(rows, 8)) == 3


def test_single_batch_request_is_in_range():
    rows = skewed()
    assert len(representative_sample(rows, 1)) == 1


def test_never_indexes_past_the_end():
    for size in range(1, 40):
        rows = corpus(list(range(1, size + 1)))
        for count in range(1, size + 1):
            assert len(representative_sample(rows, count)) == count


# ---------------------------------------------------------------------------
# projecting evaluation
# ---------------------------------------------------------------------------


def test_eval_projects_from_seconds_per_example_not_tokens_per_second():
    """The measured failure: 106s projected against 156s actual, under by 47%.

    A rate in tokens/second, taken from long sequences, does not transfer to a
    set of ordinary ones -- per-example overhead dominates at these lengths.
    This pins the projection to the quantity an eval pass actually scales with.
    """
    from ftlab.plan import Calibration

    # 24 examples averaging 1000 padded tokens, scored in 12s.
    c = Calibration(
        steps=8, seconds=8.0, padded_tokens=8000,
        peak_allocated_gb=1.0, peak_reserved_gb=1.0, device_total_gb=32.0,
        device_name="test", mean_gpu_watts=100.0,
        eval_seconds=12.0, eval_padded_tokens=24_000, eval_examples=24,
    )
    assert c.eval_seconds_per_example == 0.5
    # 643 examples * 0.5s = 321.5s per pass, whatever their token count.
    assert abs(c.eval_seconds_per_example * 643 - 321.5) < 1e-6


def test_eval_projection_is_blind_to_sequence_length():
    """Two calibrations, same time and example count, very different token counts.

    They must project the same eval time. Under the old token-rate model the
    long-sequence one would have claimed to be several times faster.
    """
    from ftlab.plan import Calibration

    def cal(tokens: int) -> Calibration:
        return Calibration(
            steps=8, seconds=8.0, padded_tokens=8000,
            peak_allocated_gb=1.0, peak_reserved_gb=1.0, device_total_gb=32.0,
            device_name="test", mean_gpu_watts=100.0,
            eval_seconds=12.0, eval_padded_tokens=tokens, eval_examples=24,
        )

    short, long = cal(6_000), cal(48_000)
    assert short.eval_tokens_per_second * 8 == long.eval_tokens_per_second
    assert short.eval_seconds_per_example == long.eval_seconds_per_example

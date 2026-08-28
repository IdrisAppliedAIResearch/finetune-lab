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

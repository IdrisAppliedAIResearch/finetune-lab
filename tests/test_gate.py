"""The epoch gate: reading a run's eval curve, and the rule over it."""

from __future__ import annotations

import json

from ftlab.config import GateConfig
from ftlab.gate import EvalPoint, decide, read_curve, render, run_gate

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def curve(*pairs: tuple[int, float, float]) -> list[EvalPoint]:
    return [EvalPoint(step=s, epoch=e, loss=loss) for s, e, loss in pairs]


def write_run(
    tmp_path,
    log_evals: list[tuple[int, float, float]],
    final: dict | None = None,
    epochs: float = 2.0,
    total_steps: int = 318,
):
    """Lay out the files a finished run leaves behind."""
    ckpt = tmp_path / f"checkpoint-{log_evals[-1][0] if log_evals else 0}"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"loss": 1.0, "step": 5, "epoch": 0.03},  # train-only, ignored
                    *[
                        {"eval_loss": loss, "step": s, "epoch": e}
                        for s, e, loss in log_evals
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {"config": {"train": {"epochs": epochs}}, "schedule": {"total_steps": total_steps}}
        ),
        encoding="utf-8",
    )
    if final is not None:
        (tmp_path / "trainer_metrics.json").write_text(json.dumps(final), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# reading the curve
# ---------------------------------------------------------------------------


def test_curve_merges_log_history_with_the_final_evaluate(tmp_path):
    """The end-of-training eval appears in no log_history.

    ftlab.train calls trainer.evaluate() after trainer.train() returns, so the
    single most important measurement -- the one at the last step -- is only in
    trainer_metrics.json. A gate that read log_history alone would judge the run
    on a measurement taken 18 steps before it ended.
    """
    run = write_run(
        tmp_path,
        [(100, 0.63, 1.20), (200, 1.26, 1.10), (300, 1.89, 1.05)],
        final={"epoch": 2.0, "eval_loss": 1.01},
    )
    points = read_curve(run)
    assert [p.step for p in points] == [100, 200, 300, 318]
    assert points[-1].loss == 1.01
    assert points[-1].source == "trainer_metrics"


def test_final_eval_is_not_double_counted(tmp_path):
    """A run whose last step coincides with a scheduled eval reports it twice."""
    run = write_run(
        tmp_path,
        [(100, 1.0, 1.20), (318, 2.0, 1.05)],
        final={"epoch": 2.0, "eval_loss": 1.05},
    )
    assert [p.step for p in read_curve(run)] == [100, 318]


def test_curve_survives_a_run_with_no_checkpoints(tmp_path):
    """save_total_limit can leave nothing behind on a short run."""
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"config": {"train": {"epochs": 1.0}}, "schedule": {"total_steps": 40}}),
        encoding="utf-8",
    )
    (tmp_path / "trainer_metrics.json").write_text(
        json.dumps({"epoch": 1.0, "eval_loss": 1.5}), encoding="utf-8"
    )
    points = read_curve(tmp_path)
    assert len(points) == 1 and points[0].step == 40


def test_latest_checkpoint_wins_and_is_ordered_numerically(tmp_path):
    """checkpoint-1000 sorts before checkpoint-300 as a string, not as a run."""
    for step, evals in ((300, [(300, 1.9, 1.10)]), (1000, [(300, 1.9, 1.10), (1000, 6.3, 0.90)])):
        ckpt = tmp_path / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(
            json.dumps(
                {"log_history": [{"eval_loss": ll, "step": s, "epoch": e} for s, e, ll in evals]}
            ),
            encoding="utf-8",
        )
    assert [p.step for p in read_curve(tmp_path)] == [300, 1000]


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


GATE = GateConfig()


def test_continues_while_the_curve_is_still_falling():
    decision = decide(
        curve((50, 0.31, 1.40), (150, 0.94, 1.20), (250, 1.57, 1.10), (318, 2.0, 1.05)),
        GATE,
        total_epochs=2.0,
    )
    assert decision.should_continue
    assert all(c.passed for c in decision.checks)


def test_stops_when_the_last_epoch_bought_almost_nothing():
    """Flat is the common case, and the one the gate exists to catch."""
    decision = decide(
        curve((50, 0.31, 1.40), (150, 0.94, 1.100), (250, 1.57, 1.099), (318, 2.0, 1.098)),
        GATE,
        total_epochs=2.0,
    )
    assert not decision.should_continue
    assert decision.checks[0].passed is False   # still learning
    assert decision.checks[1].passed is True    # but still at the floor


def test_stops_when_the_average_improved_but_the_last_point_turned_up():
    """The two checks disagree here, and the turn wins.

    Epoch 2 as a whole beat epoch 1 comfortably, so an improvement test alone
    says continue -- but the run has already passed its minimum and is climbing
    back out. This is exactly the overfitting onset the gate has to catch, and
    the reason the rule is a conjunction rather than a single threshold.
    """
    decision = decide(
        curve((50, 0.31, 1.40), (150, 0.94, 1.30), (250, 1.57, 1.00), (318, 2.0, 1.12)),
        GATE,
        total_epochs=2.0,
    )
    assert not decision.should_continue
    assert decision.checks[0].passed is True    # still learning, on average
    assert decision.checks[1].passed is False   # but off the floor
    assert "off the" in decision.reason


def test_noise_sized_uptick_does_not_trip_the_floor_check():
    """The tolerance exists so eval noise is not read as overfitting."""
    decision = decide(
        curve((50, 0.31, 1.40), (150, 0.94, 1.30), (250, 1.57, 1.0000), (318, 2.0, 1.0015)),
        GATE,
        total_epochs=2.0,
    )
    assert decision.should_continue


def test_too_few_points_stops_rather_than_guesses():
    assert not decide(curve((318, 2.0, 1.05)), GATE, total_epochs=2.0).should_continue


def test_disabled_gate_never_continues():
    decision = decide(
        curve((50, 0.31, 1.40), (150, 0.94, 1.20), (318, 2.0, 1.00)),
        GateConfig(enabled=False),
        total_epochs=2.0,
    )
    assert not decision.should_continue
    assert "disabled" in decision.reason


def test_single_epoch_run_refuses_without_a_baseline():
    """A continuation run has no prior epoch of its own to compare against."""
    points = curve((50, 0.31, 1.10), (100, 0.63, 1.05), (159, 1.0, 1.00))
    assert not decide(points, GATE, total_epochs=1.0).should_continue

    # ...but the previous phase's final loss supplies exactly that history.
    assert decide(points, GATE, total_epochs=1.0, baseline=1.30).should_continue
    assert not decide(points, GATE, total_epochs=1.0, baseline=1.002).should_continue


def test_baseline_never_weakens_the_bar():
    """A baseline worse than the run's own history must not make a stop a go.

    Passing --baseline on a multi-epoch run should tighten the comparison or
    leave it alone, never hand the gate an easier target to beat.
    """
    points = curve((50, 0.31, 1.10), (150, 0.94, 1.00), (250, 1.57, 0.999), (318, 2.0, 0.998))
    assert not decide(points, GATE, total_epochs=2.0).should_continue
    assert not decide(points, GATE, total_epochs=2.0, baseline=5.0).should_continue


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_run_gate_writes_a_readable_record(tmp_path):
    run = write_run(
        tmp_path,
        [(50, 0.31, 1.40), (150, 0.94, 1.20), (250, 1.57, 1.10)],
        final={"epoch": 2.0, "eval_loss": 1.05},
    )
    decision = run_gate(run, GATE)
    assert decision.should_continue

    saved = json.loads((run / "gate.json").read_text(encoding="utf-8"))
    assert saved["should_continue"] is True
    assert len(saved["curve"]) == 4
    assert len(saved["checks"]) == 2

    text = render(decision)
    assert "CONTINUE" in text and "final epoch" in text and "earlier" in text

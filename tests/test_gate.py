"""The epoch gate: reading a run's eval curve, and the rule over it."""

from __future__ import annotations

import json

from ftlab.config import GateConfig
from ftlab.gate import EvalPoint, decide, read_curve, render, run_gate

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def curve(*pairs: tuple[int, float, float]) -> list[EvalPoint]:
    """Points with train == eval, so the gap check never confounds a loss test."""
    return [EvalPoint(step=s, epoch=e, loss=loss, train_loss=loss) for s, e, loss in pairs]


def check(decision, name: str):
    """By name, not by index -- indices shift whenever a check is added."""
    return next(c for c in decision.checks if c.name == name)


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
    assert check(decision, "still learning").passed is False
    assert check(decision, "still at the floor").passed is True


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
    assert check(decision, "still learning").passed is True     # on average
    assert check(decision, "still at the floor").passed is False  # but off the floor
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
    assert {c["name"] for c in saved["checks"]} >= {
        "still learning", "not memorising", "still at the floor"
    }

    text = render(decision)
    assert "CONTINUE" in text and "final epoch" in text and "earlier" in text


# ---------------------------------------------------------------------------
# the generalisation gap
# ---------------------------------------------------------------------------


def gapped(train_last: float) -> list[EvalPoint]:
    """The real 2-epoch Gemma 4 12B run, replayed, with the last train loss free.

    Measured, not invented: this is the curve gemma4-12b-qra actually produced,
    which is what showed the loss-only checks to be weak instruments.
    """
    return [
        EvalPoint(50, 0.32, 0.3303, train_loss=0.3165),
        EvalPoint(100, 0.63, 0.2498, train_loss=0.2612),
        EvalPoint(150, 0.95, 0.2190, train_loss=0.2045),
        EvalPoint(200, 1.26, 0.1880, train_loss=0.1524),
        EvalPoint(250, 1.58, 0.1615, train_loss=0.1364),
        EvalPoint(300, 1.89, 0.1529, train_loss=0.1358),
        EvalPoint(318, 2.00, 0.1526, train_loss=train_last),
    ]


def test_widening_gap_stops_a_curve_that_is_otherwise_still_falling():
    """The real 2-epoch run: +30% epoch-over-epoch, last point the best, and
    yet eval sitting 18% above train. The first two checks both pass and the
    model is still memorising."""
    decision = decide(gapped(0.1255), GATE, total_epochs=2.0)
    # ~+30% epoch-over-epoch, against +0.20% between the last two points.
    assert "+30.3" in check(decision, "still learning").detail
    assert check(decision, "still learning").passed is True
    assert check(decision, "still at the floor").passed is True
    assert check(decision, "not memorising").passed is False
    assert not decision.should_continue
    assert "fitting the training" in decision.reason


def test_a_narrow_gap_leaves_the_decision_to_the_loss_checks():
    decision = decide(gapped(0.1450), GATE, total_epochs=2.0)   # ~5% gap
    assert check(decision, "not memorising").passed is True
    assert decision.should_continue


def test_gap_check_is_skipped_when_no_train_loss_was_logged():
    """Older runs and hand-built curves must not be failed for missing data."""
    points = [EvalPoint(s, e, loss) for s, e, loss in
              ((50, 0.32, 0.33), (150, 0.95, 0.22), (318, 2.0, 0.15))]
    decision = decide(points, GATE, total_epochs=2.0)
    assert check(decision, "not memorising").passed is True
    assert "skipped" in check(decision, "not memorising").detail
    assert decision.should_continue


def test_terminal_slope_is_reported_but_never_gates():
    """It trends to zero under any annealed schedule, so it informs, not decides."""
    decision = decide(gapped(0.1450), GATE, total_epochs=2.0)
    slope = check(decision, "terminal slope (context, not a gate)")
    assert slope.passed is True
    assert "+0.20%" in slope.detail          # 0.1529 -> 0.1526 in the real run
    assert decision.should_continue          # ...and it did not stop it


def test_curve_reads_the_train_loss_beside_each_eval(tmp_path):
    ckpt = tmp_path / "checkpoint-300"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"log_history": [
            {"loss": 0.30, "step": 45, "epoch": 0.28},
            {"loss": 0.20, "step": 150, "epoch": 0.94},
            {"eval_loss": 0.22, "step": 150, "epoch": 0.94},
        ]}),
        encoding="utf-8",
    )
    point = read_curve(tmp_path)[0]
    assert point.train_loss == 0.20
    assert abs(point.gap - (0.22 - 0.20) / 0.22) < 1e-9


def test_final_point_does_not_take_the_trainers_mean_train_loss(tmp_path):
    """trainer_metrics['train_loss'] is the mean over the whole run.

    On a descending curve that sits far above where the model actually ended --
    using it would report a *negative* gap and wave through any amount of
    memorisation. The nearest logged step is the right pairing.
    """
    run = write_run(
        tmp_path,
        [(300, 1.89, 0.1529)],
        final={"epoch": 2.0, "eval_loss": 0.1526, "train_loss": 0.9},
    )
    ckpt = run / "checkpoint-300" / "trainer_state.json"
    state = json.loads(ckpt.read_text(encoding="utf-8"))
    state["log_history"].append({"loss": 0.1255, "step": 315, "epoch": 1.98})
    ckpt.write_text(json.dumps(state), encoding="utf-8")

    final = read_curve(run)[-1]
    assert final.train_loss == 0.1255          # not 0.9
    assert final.gap > 0.15

"""Deterministic decision: has the last epoch earned another one?

The temptation at the end of a run is to squint at the eval curve and decide by
feel. That is not reproducible and it is biased toward "one more epoch", because
another epoch always *looks* like it might help. So the rule is written down as
arithmetic over thresholds that can be argued with before the run starts, and
the answer is the same whether or not anyone is watching.

Three conditions, all required:

  1. *Still learning.* The best eval loss of the final epoch must beat the best
     reached before it by at least ``min_rel_improvement``.

  2. *Not memorising.* At the end of the run, eval must sit no further above
     train than ``max_generalisation_gap``.

  3. *Still at the floor.* The last measurement must sit within
     ``overfit_tolerance`` of the best one seen.

Any one failing means stop. The bias is deliberately toward stopping: the corpus
repeats each fact 3x typical and 7x worst case, so epochs multiply on top of an
already-high exposure count, and the failure mode of a closed-book model trained
too long is a record-reciter that has memorised phrasing instead of
relationships. Under-training shows up in the grades and can be fixed by running
more; over-training is only fixable by throwing the run away.

Check 2 was added after the first real 2-epoch run, which showed the other two
are both weak instruments under a schedule that anneals the learning rate to
zero -- and weak in the *same* direction, toward continuing:

* Check 1 compares an epoch's best against everything before it, so it is an
  average over the epoch, dominated by its early part. On that run it read
  +30.30% while the last two measurements differed by +0.20% -- below the gate's
  own 0.5% bar. A decaying LR makes almost any epoch improve on average.

* Check 3 barely binds. As the LR approaches zero the model stops moving, so the
  final measurement is very nearly always also the best; on that run the two
  were the same point and the check passed trivially.

The tempting fix -- judge the terminal slope instead of the epoch average -- just
inverts the bias, because flatness at the end is largely an artifact of the LR
having decayed rather than of the model being saturated. The train/eval gap is
the one quantity here the schedule cannot fake, since both numbers are read off
the same model at the same step. On that run it went from ~5% at the end of
epoch 1 to ~18% at the end of epoch 2.

A caveat worth stating plainly: check 2 was chosen *after* seeing the data it
now fires on, which is exactly what makes a pre-registered rule worth less. It
is defensible on its own terms -- the gap is the failure mode this project cares
about, and it is schedule-independent -- but a rule revised post hoc should not
be the sole basis for the decision it was revised to change. Treat a split
verdict as a prompt to look at the task metrics from 'ftlab grade', which
measure the thing eval loss is only a proxy for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import GateConfig


@dataclass(frozen=True)
class EvalPoint:
    """One eval measurement, with the train loss logged nearest to it."""

    step: int
    epoch: float
    loss: float
    source: str = "trainer_state"
    train_loss: float | None = None

    @property
    def gap(self) -> float | None:
        """How far eval sits above train, relative to eval.

        The only signal here a learning-rate schedule cannot fake: both numbers
        come off the same model at the same step.
        """
        if not self.train_loss or not self.loss:
            return None
        return (self.loss - self.train_loss) / self.loss


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateDecision:
    should_continue: bool
    reason: str
    checks: list[Check] = field(default_factory=list)
    curve: list[EvalPoint] = field(default_factory=list)
    epochs_trained: float = 0.0
    baseline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_continue": self.should_continue,
            "reason": self.reason,
            "epochs_trained": self.epochs_trained,
            "baseline": self.baseline,
            "checks": [c.__dict__ for c in self.checks],
            "curve": [p.__dict__ for p in self.curve],
        }


# ---------------------------------------------------------------------------
# reading the curve back out of a finished run
# ---------------------------------------------------------------------------


def _latest_trainer_state(run_dir: Path) -> Path | None:
    """The trainer_state.json from the highest-numbered checkpoint.

    Checkpoints are the only place transformers writes log_history, and
    save_total_limit means the early ones are deleted -- but each surviving
    state carries the *whole* history, so the newest is the complete one.
    """
    states = []
    for ckpt in run_dir.glob("checkpoint-*"):
        state = ckpt / "trainer_state.json"
        if not state.exists():
            continue
        try:
            number = int(ckpt.name.split("-")[-1])
        except ValueError:
            continue
        states.append((number, state))
    if not states:
        direct = run_dir / "trainer_state.json"
        return direct if direct.exists() else None
    return max(states)[1]


def read_curve(run_dir: str | Path) -> list[EvalPoint]:
    """Every eval measurement of a run, in step order.

    Two sources, because neither alone is complete. ``log_history`` in the last
    checkpoint holds the evals that happened *during* training, but the final
    checkpoint is written before the last step when total steps is not a
    multiple of save_steps. ``trainer_metrics.json`` holds the explicit
    end-of-training evaluate() that ftlab.train runs afterwards, which is the
    measurement that matters most and appears in no log_history at all.
    """
    run_dir = Path(run_dir)
    points: list[EvalPoint] = []

    train_log: list[tuple[int, float]] = []
    state_path = _latest_trainer_state(run_dir)
    if state_path is not None:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = state.get("log_history", [])
        train_log = [
            (int(e.get("step", 0)), float(e["loss"]))
            for e in history
            if "loss" in e and "eval_loss" not in e
        ]
        for entry in history:
            if "eval_loss" not in entry:
                continue
            step = int(entry.get("step", 0))
            points.append(
                EvalPoint(
                    step=step,
                    epoch=float(entry.get("epoch", 0.0)),
                    loss=float(entry["eval_loss"]),
                    train_loss=_nearest_train_loss(train_log, step),
                )
            )

    final_path = run_dir / "trainer_metrics.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if "eval_loss" in final:
            step = int(final.get("step") or _final_step(run_dir, points))
            epoch = float(final.get("epoch") or _final_epoch(run_dir, points))
            if not any(p.step == step for p in points):
                points.append(
                    EvalPoint(
                        step=step,
                        epoch=epoch,
                        loss=float(final["eval_loss"]),
                        source="trainer_metrics",
                        # Deliberately NOT final["train_loss"]: the Trainer
                        # reports that as the mean over the whole run, which at
                        # the end of a descending curve is far above the value
                        # the model is actually at. Pair with the nearest
                        # logged step instead.
                        train_loss=_nearest_train_loss(train_log, step),
                    )
                )

    return sorted(points, key=lambda p: p.step)


def _nearest_train_loss(train_log: list[tuple[int, float]], step: int) -> float | None:
    if not train_log:
        return None
    return min(train_log, key=lambda row: abs(row[0] - step))[1]


def _final_step(run_dir: Path, points: list[EvalPoint]) -> int:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        schedule = json.loads(meta.read_text(encoding="utf-8")).get("schedule") or {}
        if schedule.get("total_steps"):
            return int(schedule["total_steps"])
    return (points[-1].step + 1) if points else 0


def _final_epoch(run_dir: Path, points: list[EvalPoint]) -> float:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        config = json.loads(meta.read_text(encoding="utf-8")).get("config") or {}
        epochs = (config.get("train") or {}).get("epochs")
        if epochs:
            return float(epochs)
    return (points[-1].epoch) if points else 0.0


def epochs_trained(run_dir: str | Path, curve: list[EvalPoint]) -> float:
    """How many epochs the run actually completed."""
    return _final_epoch(Path(run_dir), curve)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def decide(
    curve: list[EvalPoint],
    gate: GateConfig,
    total_epochs: float,
    baseline: float | None = None,
) -> GateDecision:
    """Apply the rule. ``baseline`` stands in for history from a prior run.

    Without a baseline, a single-epoch run has nothing to compare its final
    epoch against and the gate refuses rather than guessing -- which is what
    makes the extra epoch itself gateable: pass the previous run's final eval
    loss and the same arithmetic answers "would a fourth epoch help".
    """
    if not gate.enabled:
        return GateDecision(False, "gate disabled in config", curve=curve,
                            epochs_trained=total_epochs, baseline=baseline)
    if len(curve) < 2:
        return GateDecision(
            False,
            f"only {len(curve)} eval point(s); not enough curve to judge",
            curve=curve, epochs_trained=total_epochs, baseline=baseline,
        )

    # The final epoch's window. Points are split by epoch, not by step count,
    # so this stays correct if eval_steps or dataset size changes.
    boundary = max(0.0, total_epochs - 1.0)
    last = [p for p in curve if p.epoch > boundary]
    prior = [p for p in curve if p.epoch <= boundary]

    if not last:
        last, prior = curve[-1:], curve[:-1]

    best_prior_loss = min((p.loss for p in prior), default=None)
    if best_prior_loss is None:
        best_prior_loss = baseline
    elif baseline is not None:
        best_prior_loss = min(best_prior_loss, baseline)

    if best_prior_loss is None:
        return GateDecision(
            False,
            "no eval history before the final epoch to compare against "
            "(pass --baseline with the previous run's final eval loss)",
            curve=curve, epochs_trained=total_epochs, baseline=baseline,
        )

    best_last = min(last, key=lambda p: p.loss)
    improvement = (best_prior_loss - best_last.loss) / best_prior_loss
    improving = improvement >= gate.min_rel_improvement

    best = min(curve, key=lambda p: p.loss)
    final = curve[-1]
    ceiling = best.loss * (1.0 + gate.overfit_tolerance)
    at_floor = final.loss <= ceiling

    gap = final.gap
    generalising = gap is None or gap <= gate.max_generalisation_gap

    if gap is None:
        gap_detail = "no train loss logged alongside the final eval; check skipped"
    else:
        gap_detail = (
            f"eval {final.loss:.4f} vs train {final.train_loss:.4f} = "
            f"{gap:.1%} above (limit {gate.max_generalisation_gap:.1%})"
        )

    checks = [
        Check(
            "still learning",
            improving,
            f"final epoch best {best_last.loss:.4f} vs prior best "
            f"{best_prior_loss:.4f} = {improvement:+.2%} "
            f"(need >= {gate.min_rel_improvement:.2%})",
        ),
        Check("not memorising", generalising, gap_detail),
        Check(
            "still at the floor",
            at_floor,
            f"last {final.loss:.4f} @ step {final.step} vs best {best.loss:.4f} "
            f"@ step {best.step}; ceiling {ceiling:.4f}",
        ),
    ]

    # The terminal slope is not a check -- under a decaying LR it always trends
    # to zero -- but it is the number that shows how much of check 1's verdict
    # came from the start of the epoch rather than the end.
    if len(curve) >= 2:
        prev_point = curve[-2]
        slope = (prev_point.loss - final.loss) / prev_point.loss if prev_point.loss else 0.0
        checks.append(
            Check(
                "terminal slope (context, not a gate)",
                True,
                f"{prev_point.loss:.4f} -> {final.loss:.4f} = {slope:+.2%} "
                f"between the last two measurements",
            )
        )

    if improving and at_floor and not generalising:
        reason = (
            f"eval loss is still falling ({improvement:+.2%}), but eval now sits "
            f"{gap:.1%} above train -- the model is fitting phrasing rather than "
            f"content, and another epoch would deepen that"
        )
    elif improving and at_floor:
        reason = (
            f"eval loss still falling ({improvement:+.2%} over the final epoch) "
            f"and the curve has not turned up -- another epoch is earned"
        )
    elif not improving and not at_floor:
        reason = (
            f"eval loss has flattened ({improvement:+.2%}) and turned up off its "
            f"minimum -- training is done"
        )
    elif not improving:
        reason = (
            f"the final epoch bought only {improvement:+.2%} eval loss, below the "
            f"{gate.min_rel_improvement:.2%} bar -- another epoch is not worth it"
        )
    else:
        reason = (
            f"eval loss improved {improvement:+.2%} on average but the last point "
            f"({final.loss:.4f}) is off the {best.loss:.4f} floor -- stop here"
        )

    return GateDecision(
        should_continue=improving and at_floor and generalising,
        reason=reason,
        checks=checks,
        curve=curve,
        epochs_trained=total_epochs,
        baseline=baseline,
    )


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------


def render(decision: GateDecision) -> str:
    lines = ["", "Eval curve", "-" * 58]
    if decision.curve:
        best = min(decision.curve, key=lambda p: p.loss)
        boundary = max(0.0, decision.epochs_trained - 1.0)
        for point in decision.curve:
            marks = " <- best" if point is best else ""
            window = "final epoch" if point.epoch > boundary else "earlier"
            gap = "" if point.gap is None else f"  gap {point.gap:>5.1%}"
            train = "" if point.train_loss is None else f"  train {point.train_loss:.4f}"
            lines.append(
                f"  step {point.step:>5}  epoch {point.epoch:>5.2f}  "
                f"eval {point.loss:.4f}{train}{gap}  {window}{marks}"
            )
    else:
        lines.append("  (no eval measurements found)")

    if decision.baseline is not None:
        lines.append(f"  baseline from previous run: {decision.baseline:.4f}")

    lines += ["", "Gate", "-" * 58]
    for check in decision.checks:
        lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}")
        lines.append(f"         {check.detail}")

    verdict = "CONTINUE" if decision.should_continue else "STOP"
    lines += ["", f"  verdict  {verdict}", f"  because  {decision.reason}", ""]
    return "\n".join(lines)


def run_gate(
    run_dir: str | Path,
    gate: GateConfig,
    baseline: float | None = None,
) -> GateDecision:
    """Read a finished run, decide, and write gate.json beside it."""
    run_dir = Path(run_dir)
    curve = read_curve(run_dir)
    decision = decide(curve, gate, epochs_trained(run_dir, curve), baseline)
    (run_dir / "gate.json").write_text(
        json.dumps(decision.to_dict(), indent=2), encoding="utf-8"
    )
    return decision

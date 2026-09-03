"""
Eval harness for DisputeCourt. Compares a set of predictions against the
rules-matrix ground truth (from scripts/labeler.py) and reports the four
numbers PLAN.md commits to: correctness, calibration, abstention rate, and
false-positive cost in both directions.

Works standalone -- feed it (true_verdict, pred_verdict, pred_confidence)
triples from either the prompted baseline or the RL-tuned model, and it
produces a report. Meant to be called on both and diffed side by side.
"""

from collections import Counter
from dataclasses import dataclass


VERDICTS = ("represent", "accept", "abstain")


@dataclass
class Prediction:
    case_id: str
    true_verdict: str
    pred_verdict: str
    pred_confidence: float  # model's own stated confidence, 0-1


def accuracy(preds: list[Prediction]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p in preds if p.pred_verdict == p.true_verdict)
    return correct / len(preds)


def confusion_matrix(preds: list[Prediction]) -> dict[tuple[str, str], int]:
    """(true, pred) -> count. Use to see *what kind* of mistakes are happening."""
    cm = Counter((p.true_verdict, p.pred_verdict) for p in preds)
    return dict(cm)


def abstention_rate(preds: list[Prediction]) -> float:
    if not preds:
        return 0.0
    return sum(1 for p in preds if p.pred_verdict == "abstain") / len(preds)


def brier_score(preds: list[Prediction]) -> float:
    """
    Binary Brier score on 'was the prediction correct', using the model's
    stated confidence as the probability. Lower is better; 0 is perfect
    calibration, 0.25 is what an uninformative always-0.5 predictor gets.
    This is the calibration term PLAN.md asks the reward function to target
    directly -- this function is the eval-side twin of that reward term.
    """
    if not preds:
        return 0.0
    total = 0.0
    for p in preds:
        correctness_indicator = 1.0 if p.pred_verdict == p.true_verdict else 0.0
        total += (p.pred_confidence - correctness_indicator) ** 2
    return total / len(preds)


def false_positive_costs(
    preds: list[Prediction],
    cost_wrong_represent: float = 1.0,
    cost_wrong_accept: float = 1.0,
) -> dict:
    """
    Two distinct failure modes, costed separately -- this is what the
    Track 02 bar means by 'honest metrics including false-positive cost':

    - wrong_represent: model said represent, true answer was accept.
      Cost = wasted dispute-response effort / representment fee on a case
      that was never winnable.
    - wrong_accept: model said accept, true answer was represent.
      Cost = recoverable revenue left on the table.

    Abstains are never counted as false positives in either direction --
    abstaining on an uncertain case is the correct behavior, not an error.
    """
    wrong_represent = sum(
        1 for p in preds if p.pred_verdict == "represent" and p.true_verdict == "accept"
    )
    wrong_accept = sum(
        1 for p in preds if p.pred_verdict == "accept" and p.true_verdict == "represent"
    )
    return {
        "wrong_represent_count": wrong_represent,
        "wrong_accept_count": wrong_accept,
        "wrong_represent_cost": wrong_represent * cost_wrong_represent,
        "wrong_accept_cost": wrong_accept * cost_wrong_accept,
        "total_cost": wrong_represent * cost_wrong_represent + wrong_accept * cost_wrong_accept,
    }


def full_report(preds: list[Prediction], label: str = "model") -> dict:
    report = {
        "label": label,
        "n": len(preds),
        "accuracy": accuracy(preds),
        "abstention_rate": abstention_rate(preds),
        "brier_score": brier_score(preds),
        "false_positive_costs": false_positive_costs(preds),
        "confusion_matrix": confusion_matrix(preds),
    }
    return report


def print_report(report: dict) -> None:
    print(f"\n=== {report['label']} (n={report['n']}) ===")
    print(f"  accuracy:         {report['accuracy']:.1%}")
    print(f"  abstention rate:  {report['abstention_rate']:.1%}")
    print(f"  brier score:      {report['brier_score']:.3f}  (0=perfect, 0.25=uninformative)")
    fp = report["false_positive_costs"]
    print(f"  wrong-represent:  {fp['wrong_represent_count']} cases  (cost {fp['wrong_represent_cost']:.1f})")
    print(f"  wrong-accept:     {fp['wrong_accept_count']} cases  (cost {fp['wrong_accept_cost']:.1f})")
    print(f"  total cost:       {fp['total_cost']:.1f}")


def compare_reports(baseline: dict, tuned: dict) -> None:
    """Side-by-side diff -- this is the table PLAN.md's Day 3 pitch needs."""
    print(f"\n{'metric':<20}{'baseline':>15}{'RL-tuned':>15}{'delta':>12}")
    rows = [
        ("accuracy", baseline["accuracy"], tuned["accuracy"], "{:.1%}"),
        ("abstention rate", baseline["abstention_rate"], tuned["abstention_rate"], "{:.1%}"),
        ("brier score", baseline["brier_score"], tuned["brier_score"], "{:.3f}"),
        (
            "total FP cost",
            baseline["false_positive_costs"]["total_cost"],
            tuned["false_positive_costs"]["total_cost"],
            "{:.1f}",
        ),
    ]
    for name, b, t, fmt in rows:
        delta = t - b
        print(f"{name:<20}{fmt.format(b):>15}{fmt.format(t):>15}{delta:>+12.3f}")

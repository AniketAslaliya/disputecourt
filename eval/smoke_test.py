"""
Smoke test: simulates a noisy 'baseline' predictor and a better-calibrated
'tuned' predictor against the real seed_cases_labeled.jsonl ground truth,
to prove eval/metrics.py actually works before any real model exists.
Delete or repurpose once real panel outputs are available -- this is
scaffolding, not a real result.
"""

import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from metrics import Prediction, full_report, print_report, compare_reports  # noqa: E402

random.seed(7)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_cases_labeled.jsonl"


def load_truth():
    return [json.loads(line) for line in DATA_PATH.open()]


def simulate_baseline(rows):
    """Deliberately mediocre + overconfident, to stand in for an untrained prompted baseline."""
    preds = []
    for r in rows:
        true = r["verdict"]
        if random.random() < 0.65:
            pred = true
        else:
            pred = random.choice([v for v in ("represent", "accept", "abstain") if v != true])
        conf = random.uniform(0.75, 0.95)  # overconfident regardless of correctness
        preds.append(Prediction(r["case_id"], true, pred, conf))
    return preds


def simulate_tuned(rows):
    """Better accuracy + confidence that tracks correctness, standing in for the RL-tuned model."""
    preds = []
    for r in rows:
        true = r["verdict"]
        if random.random() < 0.85:
            pred = true
        else:
            pred = random.choice([v for v in ("represent", "accept", "abstain") if v != true])
        conf = random.uniform(0.75, 0.9) if pred == true else random.uniform(0.4, 0.6)
        preds.append(Prediction(r["case_id"], true, pred, conf))
    return preds


if __name__ == "__main__":
    rows = load_truth()
    baseline_preds = simulate_baseline(rows)
    tuned_preds = simulate_tuned(rows)

    baseline_report = full_report(baseline_preds, label="baseline (simulated)")
    tuned_report = full_report(tuned_preds, label="RL-tuned (simulated)")

    print_report(baseline_report)
    print_report(tuned_report)
    compare_reports(baseline_report, tuned_report)

    print("\n[smoke test] eval/metrics.py runs end to end. Swap in real")
    print("predictions from panel/run_panel.py once the panel pipeline is live.")

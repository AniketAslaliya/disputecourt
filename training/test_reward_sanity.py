"""
Checks the reward function in reward.py against several strategies BEFORE
any real training run. The specific failure mode this guards against
(flagged in PLAN.md as the expected 'what broke' story): calibration reward
collapsing the policy to always-low-confidence, or to always-abstain,
because that looks safe under a naive reward shape. This script proves
whether that collapse is actually incentivized by the reward as written,
using the true verdict distribution from the real seed dataset -- not a
hypothetical one.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_reward  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
# Averaged over seeded trials. An earlier version sampled once over the 40
# seed cases with an unseeded RNG, which made the ordering of the two
# non-degenerate strategies a coin flip between runs -- a test that is only
# right on average is not evidence you can put in a README.
N_TRIALS = 12


def load_true_verdicts() -> list[str]:
    rows = [json.loads(line) for line in DATA_PATH.open()]
    return [r["verdict"] for r in rows]


def strategy_always_abstain_low_conf(true_verdict: str) -> str:
    return json.dumps({"verdict": "abstain", "confidence": 0.5})


def strategy_always_represent_overconfident(true_verdict: str) -> str:
    """The other obvious degenerate shortcut: always guess the most common
    class with high confidence, ignoring the case entirely."""
    return json.dumps({"verdict": "represent", "confidence": 0.9})


def strategy_calibrated_80pct_accurate(true_verdict: str) -> str:
    """Simulates a genuinely decent policy: usually right, confidence
    tracks correctness. This should beat both degenerate strategies."""
    import random

    is_correct = random.random() < 0.8
    if is_correct:
        pred = true_verdict
        conf = random.uniform(0.75, 0.9)
    else:
        other = [v for v in ("represent", "accept", "abstain") if v != true_verdict]
        pred = random.choice(other)
        conf = random.uniform(0.55, 0.7)
    return json.dumps({"verdict": pred, "confidence": conf})


def strategy_constant_half_confidence_but_accurate(true_verdict: str) -> str:
    """Guesses correctly at 80% but always states confidence=0.5 regardless
    -- checks whether the calibration term actually punishes this 'safe'
    dodge relative to real calibration."""
    import random

    if random.random() < 0.8:
        pred = true_verdict
    else:
        other = [v for v in ("represent", "accept", "abstain") if v != true_verdict]
        pred = random.choice(other)
    return json.dumps({"verdict": pred, "confidence": 0.5})


def evaluate_strategy(name: str, strategy_fn, true_verdicts: list[str]) -> tuple[float, float]:
    import random as _random
    import statistics

    trial_means = []
    for t in range(N_TRIALS):
        _random.seed(1000 * t)
        rewards = [compute_reward(strategy_fn(tv), tv) for tv in true_verdicts]
        trial_means.append(sum(rewards) / len(rewards))
    spread = statistics.stdev(trial_means) if len(trial_means) > 1 else 0.0
    return statistics.mean(trial_means), spread


if __name__ == "__main__":
    true_verdicts = load_true_verdicts()
    dist = Counter(true_verdicts)
    total = len(true_verdicts)
    print(f"Ground truth distribution (n={total}):")
    for v, c in dist.most_common():
        print(f"  {v:10s} {c:3d}  ({c/total:.0%})")

    strategies = [
        ("always-abstain, conf=0.5 (degenerate)", strategy_always_abstain_low_conf),
        ("always-represent, conf=0.9 (degenerate)", strategy_always_represent_overconfident),
        ("~80% accurate, poorly calibrated (conf=0.5 always)", strategy_constant_half_confidence_but_accurate),
        ("~80% accurate, well calibrated", strategy_calibrated_80pct_accurate),
    ]

    print("\nAverage reward per strategy:")
    results = {}
    for name, fn in strategies:
        avg, spread = evaluate_strategy(name, fn, true_verdicts)
        results[name] = avg
        print(f"  {avg:+.3f}  +/-{spread:.3f}   {name}")

    best = max(results, key=results.get)
    print(f"\nHighest-reward strategy: '{best}'")

    if "degenerate" in best:
        print("\nFAIL: a degenerate strategy scores highest -- the reward shape")
        print("needs fixing before training, or GRPO will find this shortcut.")
        raise SystemExit(1)
    else:
        print("\nPASS: genuine good behavior beats both degenerate shortcuts.")
        print("Reward shape looks safe to train against.")

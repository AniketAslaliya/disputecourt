"""
GRPO reward function for DisputeCourt. Pure logic, no model dependency --
verify this file with test_reward_sanity.py BEFORE wiring it into
train_grpo.py. A wrong reward shape silently produces a bad policy over
hours of training; a wrong reward function is a five-second bug to catch
here.
"""

import json
import re


CORRECTNESS_WEIGHT = 1.0
CALIBRATION_WEIGHT = 0.5
ABSTENTION_WEIGHT = 0.3

VALID_VERDICTS = {"represent", "accept", "abstain"}


def parse_completion(text: str) -> dict | None:
    """Returns None on anything unparseable -- that's a deliberate hard
    signal, not a thing to patch around with a lenient parser."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if parsed.get("verdict") not in VALID_VERDICTS:
        return None
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        return None
    return {"verdict": parsed["verdict"], "confidence": float(conf)}


def compute_reward(completion_text: str, true_verdict: str) -> float:
    """
    Single-case reward, roughly in [-1.65, 1.3].

    Three terms, matching PLAN.md's reward design exactly:
    - correctness: +/- 1.0 for right/wrong verdict
    - calibration: Brier-style penalty, confident+wrong hurts more than
      uncertain+wrong -- this is the eval-time brier_score's training-time twin
    - abstention credit: rewards abstaining on genuinely ambiguous
      (true_verdict == abstain) cases, small penalty for abstaining to
      dodge an easy one
    """
    parsed = parse_completion(completion_text)
    if parsed is None:
        return -1.5  # hard penalty: get to valid JSON before anything else matters

    pred_verdict = parsed["verdict"]
    pred_conf = parsed["confidence"]
    is_correct = pred_verdict == true_verdict

    correctness_term = CORRECTNESS_WEIGHT * (1.0 if is_correct else -1.0)

    correctness_indicator = 1.0 if is_correct else 0.0
    calibration_term = -CALIBRATION_WEIGHT * (pred_conf - correctness_indicator) ** 2

    abstention_term = 0.0
    if true_verdict == "abstain" and pred_verdict == "abstain":
        abstention_term = ABSTENTION_WEIGHT
    elif true_verdict != "abstain" and pred_verdict == "abstain":
        abstention_term = -ABSTENTION_WEIGHT * 0.5

    return correctness_term + calibration_term + abstention_term


def trl_reward_fn(completions: list[str], true_verdicts: list[str], **kwargs) -> list[float]:
    """
    Adapter matching TRL's GRPOTrainer reward_fn signature: takes a batch of
    completions plus the corresponding ground-truth column from the dataset,
    returns a list of floats. Wire this directly into GRPOConfig in
    train_grpo.py -- don't reimplement the reward math there.
    """
    return [compute_reward(c, v) for c, v in zip(completions, true_verdicts)]

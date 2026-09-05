"""
Non-AI control: the strongest keyword/regex extractor we could write by hand,
feeding the same deterministic rules matrix the model has to reconstruct.

Why this file exists
--------------------
The first version of this dataset used one fixed sentence per evidence item
("Carrier tracking confirms a successful delivery scan" == E1, all 122 times).
A 12-line keyword matcher scored 94% on that eval split, which meant the
reported "accuracy" measured nothing about the model -- the task was solvable
without reading. The dataset was regenerated with paraphrase pools and negative
distractors (scripts/generate_cases.py) to fix that.

This module is kept, and reported as a column in the README, so the claim
"the task is not trivially solvable" is something a reader can check rather
than take on trust. It is deliberately tuned to be as strong as possible --
handling negation, not just presence -- so the headroom it leaves is real
headroom and not a strawman.

Run:
    python eval/keyword_baseline.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from labeler import Case, label  # noqa: E402
from metrics import Prediction, full_report, print_report  # noqa: E402


# Positive cues per evidence item.
POSITIVE = {
    "E1": [r"delivery scan", r"completed handover", r"proof of delivery",
           r"delivered event", r"delivery receipt", r"delivered status"],
    "E2": [r"access logs", r"activated the entitlement", r"audit trail captures",
           r"usage telemetry", r"fulfilment logs", r"provisioned and used",
           r"subsequently redeemed"],
    "E3": [r"avs y-match", r"avs returned a full match", r"address verification came back y",
           r"avs m-response", r"identical to the billing address", r"matches billing"],
    "E4": [r"signature was captured", r"collected a signature", r"obtained a signature"],
    "E5": [r"prior undisputed transaction", r"never disputed", r"settled purchase",
           r"previous clean transaction"],
    "E6": [r"employment verification", r"hr confirmation", r"was staff at the delivery"],
    "E7": [r"support tickets show", r"email thread between", r"helpdesk log records"],
}

# Negative cues: phrases that mention the same vocabulary but establish absence.
# Without these the extractor is badly fooled by the distractors; with them it
# is about as good as hand-written rules get on this data.
NEGATIVE = {
    "E1": [r"no delivered scan", r"never progressed", r"never confirmed arrival",
           r"no delivery event"],
    "E2": [r"no access or download logs", r"no record of the account ever signing in",
           r"no usage or entitlement logs"],
    "E3": [r"former residence", r"avs returned an n-response", r"does not correspond to the billing",
           r"address verification was not run", r"never matched against it"],
    "E4": [r"no signature was collected", r"signature on delivery was waived", r"left blank"],
    "E5": [r"first order", r"device fingerprinting was not enabled", r"charged back as well"],
    "E6": [r"no proof of the cardholder's employment", r"no employment record",
           r"never verified the cardholder worked"],
    "E7": [r"no ticket or email thread was retained", r"no inbound customer contact"],
}

CONTRADICTION = [
    r"different city", r"returned to sender", r"wrong unit", r"several hundred miles",
    r"different account email", r"unrelated account", r"name unrelated to the cardholder",
    r"refund was already processed", r"cancelled and credited back",
]


def extract(narrative: str) -> tuple[set, bool]:
    text = narrative.lower()
    evidence = set()
    for item, cues in POSITIVE.items():
        if any(re.search(c, text) for c in cues):
            if not any(re.search(n, text) for n in NEGATIVE.get(item, [])):
                evidence.add(item)
    contradicted = any(re.search(c, text) for c in CONTRADICTION)
    return evidence, contradicted


def predict(narrative: str) -> tuple[str, float]:
    evidence, contradicted = extract(narrative)
    verdict, _ = label(Case("kw", narrative, evidence, contradicted))
    # A rules engine has no calibrated belief; 0.75 flat is the honest stand-in.
    return verdict.value, 0.75


def main():
    rows = [json.loads(l) for l in (ROOT / "data" / "eval.jsonl").open(encoding="utf-8")]
    preds = []
    for r in rows:
        v, c = predict(r["narrative"])
        preds.append(Prediction(r["case_id"], r["verdict"], v, c))
    report = full_report(preds, label="Keyword baseline (no AI)")
    print_report(report)
    out = ROOT / "data" / "results_keyword.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps({
                "case_id": p.case_id, "true_verdict": p.true_verdict,
                "pred_verdict": p.pred_verdict, "pred_confidence": p.pred_confidence,
            }) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""
Runs the real Court Panel (panel/personas.py) against a dataset using the
live Gemini API, then scores the results with eval/metrics.py. This is the
Day 2 entrypoint -- the "get an end-to-end prompted baseline running"
deliverable from PLAN.md.

Usage:
    export GEMINI_API_KEY=...
    python3 run_panel_live.py --data ../data/seed_cases_labeled.jsonl --limit 10

Start with --limit 10 given free-tier pacing (3 Gemini calls per case,
~8.5s apart -> roughly 4-5 minutes for 10 cases). Drop --limit once you've
confirmed output quality.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from personas import run_panel  # noqa: E402
from llm_client import call_gemini  # noqa: E402
from metrics import Prediction, full_report, print_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/seed_cases_labeled.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="../data/panel_baseline_results.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).open()]
    if args.limit:
        rows = rows[: args.limit]

    results = []
    predictions = []
    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {row['case_id']}...")
        try:
            result = run_panel(
                case_id=row["case_id"],
                narrative=row["narrative"],
                evidence_present=row["evidence_present"],
                llm_call_fn=call_gemini,
            )
        except ValueError as e:
            # The disqualification-risk safety check in personas.py raises here --
            # do not catch-and-continue past it silently, surface it.
            print(f"  SAFETY CHECK TRIPPED on {row['case_id']}: {e}")
            raise

        results.append(
            {
                "case_id": result.case_id,
                "true_verdict": row["verdict"],
                "pred_verdict": result.verdict,
                "pred_confidence": result.confidence,
                "rebuttal_draft": result.rebuttal_draft,
                "reasoning": result.reasoning,
            }
        )
        predictions.append(
            Prediction(result.case_id, row["verdict"], result.verdict, result.confidence)
        )
        print(f"  -> {result.verdict} (confidence {result.confidence:.2f}), true={row['verdict']}")

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}")

    report = full_report(predictions, label="prompted baseline (live Gemini)")
    print_report(report)


if __name__ == "__main__":
    main()

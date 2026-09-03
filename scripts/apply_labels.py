import json
from collections import Counter
from pathlib import Path

from labeler import Case, label, confidence_for

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    raw = json.loads((DATA_DIR / "seed_cases.json").read_text())
    labeled = []
    counts = Counter()

    for row in raw:
        case = Case(
            case_id=row["case_id"],
            narrative=row["narrative"],
            evidence_present=set(row["evidence_present"]),
            contradicted=row["contradicted"],
        )
        verdict, reasoning = label(case)
        conf = confidence_for(case, verdict)
        counts[verdict.value] += 1
        labeled.append(
            {
                **row,
                "verdict": verdict.value,
                "reference_confidence": conf,
                "labeling_reasoning": reasoning,
            }
        )

    out_path = DATA_DIR / "seed_cases_labeled.jsonl"
    with out_path.open("w") as f:
        for row in labeled:
            f.write(json.dumps(row) + "\n")

    total = len(labeled)
    print(f"Labeled {total} seed cases -> {out_path}")
    print("\nVerdict distribution:")
    for verdict, n in counts.most_common():
        print(f"  {verdict:10s} {n:3d}  ({n/total:.0%})")

    abstain_frac = counts["abstain"] / total
    print(f"\nAbstain fraction: {abstain_frac:.0%} (PLAN.md target: 20-30%)")
    if not (0.15 <= abstain_frac <= 0.35):
        print("WARNING: abstain fraction outside target band — adjust seed cases before expanding.")
    else:
        print("Within target band.")


if __name__ == "__main__":
    main()

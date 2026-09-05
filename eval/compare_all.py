"""
Builds the README's metrics table from whichever result files exist.

Reads data/results_*.jsonl (each a list of case_id / true_verdict /
pred_verdict / pred_confidence records), scores them all through the same
eval/metrics.py functions, and prints one table plus data/comparison.json.

Missing files are skipped with a note rather than faked, so the table always
reflects runs that actually happened.

    python eval/compare_all.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import Prediction, full_report  # noqa: E402

# (filename, column label). Order is the order they appear in the table.
SOURCES = [
    ("results_keyword.jsonl", "Keyword control (no AI)"),
    ("results_panel.jsonl", "Prompted panel (3-persona)"),
    ("results_base.jsonl", "Base Qwen2.5-0.5B"),
    ("results_grpo.jsonl", "GRPO-tuned"),
]


def load(path: Path) -> list[Prediction]:
    preds = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        preds.append(Prediction(
            r["case_id"], r["true_verdict"], r["pred_verdict"], float(r["pred_confidence"])
        ))
    return preds


def main():
    reports = []
    for fname, label in SOURCES:
        p = ROOT / "data" / fname
        if not p.exists() or p.stat().st_size == 0:
            print(f"  (skipping {fname} -- not present)")
            continue
        preds = load(p)
        if not preds:
            print(f"  (skipping {fname} -- empty)")
            continue
        reports.append(full_report(preds, label=label))

    if not reports:
        raise SystemExit("No result files found. Run eval/keyword_baseline.py first.")

    rows = [
        ("Accuracy", lambda r: f"{r['accuracy']:.1%}"),
        ("Abstention rate", lambda r: f"{r['abstention_rate']:.1%}"),
        ("Brier score", lambda r: f"{r['brier_score']:.3f}"),
        ("Wrong-represent (n)", lambda r: str(r["false_positive_costs"]["wrong_represent_count"])),
        ("Wrong-accept (n)", lambda r: str(r["false_positive_costs"]["wrong_accept_count"])),
        ("Total FP cost", lambda r: f"{r['false_positive_costs']['total_cost']:.1f}"),
    ]

    w0 = 22
    w = 26
    print()
    print("Metric".ljust(w0) + "".join(r["label"][:w - 2].ljust(w) for r in reports))
    print("-" * (w0 + w * len(reports)))
    for name, fn in rows:
        print(name.ljust(w0) + "".join(fn(r).ljust(w) for r in reports))
    print()
    print(f"n = {reports[0]['n']} held-out cases")

    # Markdown, ready to paste into the README.
    print("\n--- markdown ---\n")
    header = "| Metric | " + " | ".join(r["label"] for r in reports) + " |"
    sep = "|---|" + "---|" * len(reports)
    print(header)
    print(sep)
    for name, fn in rows:
        print(f"| {name} | " + " | ".join(fn(r) for r in reports) + " |")

    out = ROOT / "data" / "comparison.json"
    serialisable = []
    for r in reports:
        r = dict(r)
        r["confusion_matrix"] = {f"{t}->{p}": c for (t, p), c in r["confusion_matrix"].items()}
        serialisable.append(r)
    out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

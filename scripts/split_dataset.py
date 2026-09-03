"""
Splits the generated + seed data into train and eval JSONL files.

Uses stratified random sampling to keep the verdict distribution
proportional across both splits.

Usage:
    python scripts/split_dataset.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_SIZE = 100
SEED = 42


def main():
    all_rows = []

    seed_path = DATA_DIR / "seed_cases_labeled.jsonl"
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_rows.append(json.loads(line))

    gen_path = DATA_DIR / "generated_cases.jsonl"
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_rows.append(json.loads(line))

    # deduplicate by case_id
    seen = set()
    unique = []
    for r in all_rows:
        if r["case_id"] not in seen:
            seen.add(r["case_id"])
            unique.append(r)
    all_rows = unique

    # stratified split by verdict
    by_verdict = defaultdict(list)
    for r in all_rows:
        by_verdict[r["verdict"]].append(r)

    random.seed(SEED)
    eval_rows = []
    train_rows = []

    total = len(all_rows)
    for verdict, rows in by_verdict.items():
        random.shuffle(rows)
        n_eval = max(1, round(EVAL_SIZE * len(rows) / total))
        eval_rows.extend(rows[:n_eval])
        train_rows.extend(rows[n_eval:])

    random.shuffle(train_rows)
    random.shuffle(eval_rows)

    train_path = DATA_DIR / "train.jsonl"
    eval_path = DATA_DIR / "eval.jsonl"

    for path, rows in [(train_path, train_rows), (eval_path, eval_rows)]:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"Total unique cases: {len(all_rows)}")
    print(f"Train: {len(train_rows)} -> {train_path}")
    print(f"Eval:  {len(eval_rows)} -> {eval_path}")

    from collections import Counter
    for name, rows in [("train", train_rows), ("eval", eval_rows)]:
        dist = Counter(r["verdict"] for r in rows)
        print(f"\n{name} distribution:")
        for v, n in dist.most_common():
            print(f"  {v:10s} {n:4d}  ({n/len(rows):.0%})")


if __name__ == "__main__":
    main()

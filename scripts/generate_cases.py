"""
Expands seed_cases.json into a larger synthetic dataset for training/eval.

CRITICAL DESIGN CONSTRAINT (see CLAUDE.md): the LLM is only ever asked to
propose a case narrative + which evidence items (E1-E7) it contains. It is
NEVER asked "what's the verdict" -- the deterministic labeler in
scripts/labeler.py assigns the verdict from the rules matrix. This is the
whole point of the project's grounding story; do not change this.

Usage:
    export GEMINI_API_KEY=...
    python3 generate_cases.py --pilot          # 15 cases first, check quality
    python3 generate_cases.py --n 300          # full run

Requires: pip install google-genai --break-system-packages

Free-tier note: this is paced at ~7 requests/minute (see llm_client.py) to
stay under Gemini's free-tier rate limits. 300 cases will take roughly
45-50 minutes of wall-clock time because of that pacing -- this is expected,
not a bug. Run the --pilot batch first so you're not waiting 45 minutes to
discover a prompt problem.
"""

import argparse
import json
import random
import re
from pathlib import Path

from labeler import Case, label, confidence_for
from llm_client import call_gemini

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VALID_EVIDENCE = {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}

GENERATION_PROMPT = """You are generating a synthetic chargeback dispute case for Visa reason \
code 13.1 (Merchandise/Services Not Received), for an ML training dataset.

Evidence vocabulary (use ONLY these IDs, never invent new ones):
E1 - Delivery/tracking confirmation showing successful delivery
E2 - Proof of digital delivery or access logs
E3 - Delivery address matches billing address / AVS Y-or-M match
E4 - Signature confirmation
E5 - Same device/card used in a prior undisputed transaction
E6 - Proof cardholder was employed at delivery address at time of delivery
E7 - Customer communication log (support tickets, emails)

Generate ONE realistic case as JSON with this exact schema:
{{
  "narrative": "2-3 sentence realistic dispute scenario, varied dollar amount and product type",
  "evidence_present": ["E1", "E3"],
  "contradicted": false
}}

Target evidence pattern for this case: {pattern_hint}

Vary the product/service type, dollar amount, and phrasing across cases --
do not reuse the same scenario template. Output ONLY the JSON object, no
other text.
"""

# Pattern hints to force variety and keep the abstain band populated --
# without this, an LLM left to its own devices over-generates clean wins.
PATTERN_HINTS = [
    "clear delivery + address match (should be a clean represent case)",
    "no delivery evidence at all (should be a clean accept case)",
    "delivery evidence present but the only identity link is weak or missing (should land in the ambiguous/abstain zone)",
    "evidence that actively contradicts the merchant, e.g. delivered to wrong address or already refunded (should be accept)",
    "digital good/service with partial identity evidence (ambiguous)",
    "business-address delivery with employment evidence (represent)",
    "high-value item with signature but no other identity link (ambiguous)",
]


def call_llm(prompt: str) -> str:
    """Thin wrapper around the shared, rate-limited Gemini client."""
    return call_gemini(prompt)


def parse_generated_case(raw_text: str) -> dict | None:
    """Defensive JSON extraction -- models sometimes wrap output in prose or fences."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if "narrative" not in parsed or "evidence_present" not in parsed:
        return None
    evidence = set(parsed["evidence_present"])
    if not evidence.issubset(VALID_EVIDENCE):
        return None  # reject hallucinated evidence IDs outright, don't try to fix them
    return {
        "narrative": parsed["narrative"],
        "evidence_present": sorted(evidence),
        "contradicted": bool(parsed.get("contradicted", False)),
    }


def generate_n_cases(n: int, start_index: int = 100) -> list[dict]:
    generated = []
    attempts = 0
    max_attempts = n * 3  # allow for some parse failures without looping forever

    while len(generated) < n and attempts < max_attempts:
        attempts += 1
        pattern_hint = random.choice(PATTERN_HINTS)
        prompt = GENERATION_PROMPT.format(pattern_hint=pattern_hint)
        raw = call_llm(prompt)
        parsed = parse_generated_case(raw)
        if parsed is None:
            continue

        case_id = f"gen-{start_index + len(generated):04d}"
        case = Case(
            case_id=case_id,
            narrative=parsed["narrative"],
            evidence_present=set(parsed["evidence_present"]),
            contradicted=parsed["contradicted"],
        )
        verdict, reasoning = label(case)  # <-- ground truth from the rules matrix, not the LLM
        conf = confidence_for(case, verdict)

        generated.append(
            {
                "case_id": case_id,
                "narrative": parsed["narrative"],
                "evidence_present": parsed["evidence_present"],
                "contradicted": parsed["contradicted"],
                "verdict": verdict.value,
                "reference_confidence": conf,
                "labeling_reasoning": reasoning,
            }
        )

    if attempts >= max_attempts and len(generated) < n:
        print(f"WARNING: only generated {len(generated)}/{n} cases after {attempts} attempts "
              f"-- check parse failures before relying on this count.")

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--pilot", action="store_true",
                         help="Generate just 15 cases first to sanity-check quality/cost before the full run.")
    args = parser.parse_args()

    n = 15 if args.pilot else args.n
    cases = generate_n_cases(n)

    from collections import Counter
    dist = Counter(c["verdict"] for c in cases)
    total = len(cases)
    print(f"\nGenerated {total} cases.")
    for v, count in dist.most_common():
        print(f"  {v:10s} {count:4d}  ({count/total:.0%})" if total else "  (none generated)")

    abstain_frac = dist.get("abstain", 0) / total if total else 0
    if not (0.15 <= abstain_frac <= 0.35):
        print(f"\nWARNING: abstain fraction is {abstain_frac:.0%}, outside the 15-35% target band.")
        print("Re-run with more weight on the ambiguous PATTERN_HINTS, or filter/rebalance before training.")

    out_path = DATA_DIR / ("generated_cases_pilot.jsonl" if args.pilot else "generated_cases.jsonl")
    with out_path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"\nWrote {out_path}")
    if args.pilot:
        print("This was a --pilot run (15 cases). Check quality, then run without --pilot for the full set.")


if __name__ == "__main__":
    main()

"""
Expands seed_cases.json into a larger synthetic dataset for training/eval.

CRITICAL DESIGN CONSTRAINT (see CLAUDE.md): generation never assigns a
verdict. The deterministic labeler in scripts/labeler.py applies the rules
matrix. This is the whole point of the project's grounding story.

Default backend is --local (template narratives, no API key). --llm still
exists for Gemini expansion if a key is available.

Usage:
    python generate_cases.py --local --pilot
    python generate_cases.py --local --n 300
    python generate_cases.py --llm --pilot    # needs GEMINI_API_KEY
"""

import argparse
import json
import random
import re
from pathlib import Path

from labeler import Case, label, confidence_for

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

HARD RULES for evidence_present and contradicted:
- List an E-item ONLY if the narrative explicitly states that the merchant
  actually possesses that proof. Do not tag an ID just because a topic was
  mentioned (an address, a signature dispute, a company HQ, an email).
- E1 is successful PHYSICAL delivery (delivered scan / proof of delivery).
  "Shipped" / "order processed" / no scans is NOT E1. Never use E1 for a
  digital good, subscription, file, or login.
- E2 is digital access, downloads, or provisioning logs. Use E2 (not E1)
  for digital goods/services.
- E3 ONLY if the narrative says delivery/billing matched or AVS Y/M.
  Mentioning an address is not enough. If delivery went to a different
  address than billing, omit E3 and set contradicted to true.
- E4 ONLY if a signature was actually captured.
- E5 ONLY if a prior undisputed transaction on the same device or card
  is actually described.
- E6 ONLY if employment proof at the delivery address is actually described.
  Shipping to a company HQ is not E6 by itself.
- contradicted is true ONLY when the evidence undermines the merchant
  (wrong-address delivery, already refunded). A bare "I never received it"
  claim is not a contradiction. If contradicted is true, omit E3.
- If the target pattern is ambiguous/abstain, do NOT add E3, E5, or E6
  to strengthen the merchant's case. E1+E4 with no address/device/employment
  link, or E2 alone, is the intended abstain shape.

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
    "digital good/service with E2 only and no device/card history (ambiguous/abstain)",
    "business-address delivery with employment evidence (represent)",
    "high-value item with signature (E1+E4) but no AVS/device/employment link — omit E3/E5/E6 (ambiguous/abstain)",
    # extra weight so the abstain band is not starved by the model adding E3
    "delivery evidence present but the only identity link is weak or missing (should land in the ambiguous/abstain zone)",
    "digital good/service with E2 only and no device/card history (ambiguous/abstain)",
    "high-value item with signature (E1+E4) but no AVS/device/employment link — omit E3/E5/E6 (ambiguous/abstain)",
]


PHYSICAL_PRODUCTS = [
    "leather handbag", "espresso machine", "ergonomic office chair", "gaming laptop",
    "wireless headphones", "kitchen stand mixer", "running shoes", "winter coat",
    "mechanical keyboard", "cast-iron cookware set", "kids' bicycle", "yoga mat bundle",
    "external SSD", "desk lamp", "ceramic dinnerware set", "power drill",
    "luxury wristwatch", "camera lens", "robotic vacuum", "office supplies carton",
]
DIGITAL_PRODUCTS = [
    "online course subscription", "stock photo library membership", "ebook bundle",
    "cloud storage upgrade", "software license", "language-learning app year",
    "coding tutorial subscription", "digital textbook", "game download",
    "streaming add-on month", "webinar ticket", "photo-editing plugin",
]
AMOUNTS = [
    29, 38, 45, 55, 65, 79, 95, 110, 129, 149, 175, 199, 249, 289, 349,
    420, 489, 530, 610, 680, 750, 890, 1050, 1250, 1450, 1850, 2199,
]

# Evidence is chosen first. Narratives only mention those items.
# Verdict still comes from labeler.py, not from this table.
LOCAL_PATTERNS = {
    "represent": [
        (["E1", "E3"], False, "physical"),
        (["E1", "E3", "E4"], False, "physical"),
        (["E1", "E3", "E5"], False, "physical"),
        (["E1", "E3", "E7"], False, "physical"),
        (["E1", "E5"], False, "physical"),
        (["E1", "E6"], False, "physical"),
        (["E1", "E3", "E6"], False, "physical"),
        (["E2", "E5"], False, "digital"),
        (["E2", "E3"], False, "digital"),
        (["E2", "E5", "E7"], False, "digital"),
    ],
    "accept": [
        ([], False, "physical"),
        (["E7"], False, "physical"),
        (["E5"], False, "physical"),
        (["E4"], False, "physical"),
        ([], True, "physical"),
        (["E1"], True, "physical"),
        (["E1", "E7"], True, "physical"),
        (["E2"], True, "digital"),
        (["E4"], True, "physical"),
    ],
    "abstain": [
        (["E1"], False, "physical"),
        (["E1", "E4"], False, "physical"),
        (["E1", "E7"], False, "physical"),
        (["E1", "E4", "E7"], False, "physical"),
        (["E2"], False, "digital"),
        (["E2", "E7"], False, "digital"),
    ],
}

def _money() -> int:
    return random.choice(AMOUNTS)


def _opener(amount: int, product: str) -> str:
    return random.choice(
        [
            f"Customer disputes a ${amount} {product} purchase, claiming it was never received.",
            f"The cardholder filed a Visa 13.1 dispute on a ${amount} {product} order.",
            f"A ${amount} {product} charge was disputed as merchandise/services not received.",
        ]
    )


def render_local_narrative(evidence: list[str], contradicted: bool, kind: str) -> str:
    """Narrative mentions only the tagged evidence (and explicit gaps for abstain)."""
    ev = set(evidence)
    amount = _money()
    product = random.choice(DIGITAL_PRODUCTS if kind == "digital" else PHYSICAL_PRODUCTS)
    clauses: list[str] = []

    if contradicted:
        if "E1" in ev:
            clauses.append(
                random.choice(
                    [
                        "Merchant tracking shows delivery to a different city than the billing address on this transaction.",
                        "The carrier scan shows the package was returned to sender as undeliverable before the dispute was filed.",
                        "Tracking shows the package was left at the wrong unit, not confirmed as received by the cardholder.",
                    ]
                )
            )
        elif "E2" in ev:
            clauses.append(
                "Access logs show the digital good was provisioned to a different account email than the one on this transaction."
            )
        elif ev == {"E4"}:
            clauses.append(
                "A delivery signature is on file for a name unrelated to the cardholder, at an address that does not match billing."
            )
        else:
            clauses.append(
                "Merchant records show a full refund was already processed for this order before the dispute was filed."
            )
        if "E7" in ev:
            clauses.append("A support ticket notes the customer reported the failed or misdirected delivery.")
        return _opener(amount, product) + " " + " ".join(clauses)

    if not ev:
        clauses.append(
            random.choice(
                [
                    "Merchant has no shipping record, tracking number, or delivery confirmation on file.",
                    "The merchant response contains only the original invoice with no delivery or access evidence.",
                ]
            )
        )
    if "E1" in ev:
        clauses.append("Carrier tracking confirms a successful delivery scan.")
    if "E2" in ev:
        clauses.append("Merchant access logs show the digital item was provisioned and used.")
    if "E3" in ev:
        clauses.append("The delivery or account address matches billing with an AVS Y-match.")
    if "E4" in ev:
        clauses.append("A signature was captured at drop-off.")
    if "E5" in ev:
        clauses.append("The same device fingerprint was used on a prior undisputed transaction with this card.")
    if "E6" in ev:
        clauses.append("Employment verification shows the cardholder worked at the delivery address on the delivery date.")
    if "E7" in ev:
        clauses.append("Support tickets show the customer contacted the merchant after the order.")

    # Make abstain gaps explicit in the text so the panel is not guessing.
    if ev & {"E1", "E2"} and not (ev & {"E3", "E5", "E6"}):
        clauses.append(
            "Merchant has no AVS match, device history, or employment record linking this fulfillment to the cardholder."
        )

    return _opener(amount, product) + " " + " ".join(clauses)


def labeled_row(case_id: str, narrative: str, evidence: list[str], contradicted: bool) -> dict:
    case = Case(
        case_id=case_id,
        narrative=narrative,
        evidence_present=set(evidence),
        contradicted=contradicted,
    )
    verdict, reasoning = label(case)
    return {
        "case_id": case_id,
        "narrative": narrative,
        "evidence_present": sorted(set(evidence)),
        "contradicted": contradicted,
        "verdict": verdict.value,
        "reference_confidence": confidence_for(case, verdict),
        "labeling_reasoning": reasoning,
    }


def generate_n_cases_local(n: int, start_index: int = 100, out_path: Path | None = None) -> list[dict]:
    generated = []
    n_abs = max(1, round(n * 0.25)) if n >= 10 else max(0, round(n * 0.25))
    n_rep = round(n * 0.40)
    n_acc = n - n_abs - n_rep
    if n_acc < 0:
        n_rep += n_acc
        n_acc = 0
    buckets = ["represent"] * n_rep + ["accept"] * n_acc + ["abstain"] * n_abs
    random.shuffle(buckets)
    out_handle = out_path.open("a", encoding="utf-8") if out_path else None
    try:
        for i, bucket in enumerate(buckets):
            evidence, contradicted, kind = random.choice(LOCAL_PATTERNS[bucket])
            narrative = render_local_narrative(evidence, contradicted, kind)
            case_id = f"gen-{start_index + i:04d}"
            row = labeled_row(case_id, narrative, evidence, contradicted)
            retries = 0
            while row["verdict"] != bucket and retries < 8:
                evidence, contradicted, kind = random.choice(LOCAL_PATTERNS[bucket])
                narrative = render_local_narrative(evidence, contradicted, kind)
                row = labeled_row(case_id, narrative, evidence, contradicted)
                retries += 1
            generated.append(row)
            if out_handle:
                out_handle.write(json.dumps(row) + "\n")
                out_handle.flush()
        print(f"  local backend: wrote {len(generated)} cases", flush=True)
    finally:
        if out_handle:
            out_handle.close()
    return generated


def call_llm(prompt: str) -> str:
    """Thin wrapper around the shared, rate-limited Gemini client."""
    from llm_client import call_gemini

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


def generate_n_cases(n: int, start_index: int = 100, out_path: Path | None = None) -> list[dict]:
    generated = []
    attempts = 0
    max_attempts = n * 3  # allow for some parse failures without looping forever
    out_handle = out_path.open("a", encoding="utf-8") if out_path else None

    try:
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

            row = {
                "case_id": case_id,
                "narrative": parsed["narrative"],
                "evidence_present": parsed["evidence_present"],
                "contradicted": parsed["contradicted"],
                "verdict": verdict.value,
                "reference_confidence": conf,
                "labeling_reasoning": reasoning,
            }
            generated.append(row)
            if out_handle:
                out_handle.write(json.dumps(row) + "\n")
                out_handle.flush()
            print(f"  generated {len(generated)}/{n} ({case_id}, {verdict.value})", flush=True)
    finally:
        if out_handle:
            out_handle.close()

    if attempts >= max_attempts and len(generated) < n:
        print(f"WARNING: only generated {len(generated)}/{n} cases after {attempts} attempts "
              f"-- check parse failures before relying on this count.")

    return generated


def load_existing(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    rows = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--pilot", action="store_true",
                         help="Generate just 15 cases first to sanity-check quality/cost before the full run.")
    parser.add_argument("--local", action="store_true",
                         help="Template backend: no API key. Default if --llm is not set.")
    parser.add_argument("--llm", action="store_true",
                         help="Gemini backend. Requires GEMINI_API_KEY.")
    parser.add_argument("--fresh", action="store_true",
                         help="Overwrite the output file instead of resuming.")
    args = parser.parse_args()

    use_llm = bool(args.llm)
    n = 15 if args.pilot else args.n
    out_path = DATA_DIR / ("generated_cases_pilot.jsonl" if args.pilot else "generated_cases.jsonl")

    existing = []
    if use_llm and not args.pilot and not args.fresh:
        existing = load_existing(out_path)

    if existing:
        print(f"Resuming: {len(existing)} cases already in {out_path}")

    remaining = n - len(existing)
    new_cases = []
    if remaining > 0:
        if not existing:
            out_path.write_text("", encoding="utf-8")
        start_index = 100 + len(existing)
        if use_llm:
            new_cases = generate_n_cases(remaining, start_index=start_index, out_path=out_path)
        else:
            print("Using --local template backend (no Gemini). Verdicts still come from labeler.py.")
            new_cases = generate_n_cases_local(remaining, start_index=start_index, out_path=out_path)
    cases = existing + new_cases

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

    print(f"\nWrote {out_path}")
    if args.pilot:
        print("This was a --pilot run (15 cases). Check quality, then run without --pilot for the full set.")


if __name__ == "__main__":
    main()

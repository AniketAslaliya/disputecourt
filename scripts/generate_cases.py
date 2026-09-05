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
            f"Chargeback received under reason code 13.1 on a ${amount} {product} transaction.",
            f"The issuer raised a non-receipt dispute for ${amount} covering a {product}.",
            f"A ${amount} {product} order is being contested; the cardholder says nothing arrived.",
        ]
    )


# Paraphrase pools. Each E-item has several surface forms so no single phrase is
# a reliable tell -- the first version of this dataset used one fixed sentence
# per E-item, which let a 12-line keyword matcher score 94% on the eval split.
POSITIVE_CLAUSES = {
    "E1": [
        "Carrier tracking confirms a successful delivery scan.",
        "The courier recorded a completed handover at the destination on the expected date.",
        "Proof of delivery is on file: the parcel shows a delivered status with a timestamped scan.",
        "The shipping partner's API returned a final delivered event for this consignment.",
        "Merchant supplied the carrier's delivery receipt showing the package reached the address.",
    ],
    "E2": [
        "Merchant access logs show the digital item was provisioned and used.",
        "Server-side records show the account activated the entitlement and opened it several times.",
        "The platform's audit trail captures the download completing from the customer's session.",
        "Usage telemetry confirms the subscription was consumed after purchase.",
        "Fulfilment logs show the licence was issued and subsequently redeemed.",
    ],
    "E3": [
        "The delivery or account address matches billing with an AVS Y-match.",
        "The ship-to address is identical to the billing address on the card; AVS returned a full match.",
        "Address verification came back Y, and the parcel went to that same verified address.",
        "The processor recorded an AVS M-response, and delivery was made to the matched address.",
    ],
    "E4": [
        "A signature was captured at drop-off.",
        "The carrier collected a signature when the parcel was handed over.",
        "Delivery required and obtained a signature at the door.",
    ],
    "E5": [
        "The same device fingerprint was used on a prior undisputed transaction with this card.",
        "This card and browser fingerprint appear on an earlier order that was never disputed.",
        "The customer has an older, settled purchase placed from the identical device signature.",
        "Risk records tie this session to a previous clean transaction on the same card.",
    ],
    "E6": [
        "Employment verification shows the cardholder worked at the delivery address on the delivery date.",
        "HR confirmation establishes the cardholder was employed at the ship-to business on that date.",
        "The merchant obtained written confirmation that the cardholder was staff at the delivery site.",
    ],
    "E7": [
        "Support tickets show the customer contacted the merchant after the order.",
        "There is an email thread between the buyer and support following the purchase.",
        "The helpdesk log records inbound contact from the cardholder post-delivery.",
    ],
}

# Decoys. Each mentions the SAME vocabulary as the positive clause it shadows
# (AVS, tracking, signature, device, employment, logs) while stating the
# merchant does NOT hold that proof. These make the task require reading rather
# than keyword spotting. They never alter the label, which still comes from the
# structured evidence set via labeler.py.
DISTRACTOR_CLAUSES = {
    "E1": [
        "A shipping label was generated and the carrier accepted the parcel, but no delivered scan was ever recorded.",
        "Tracking last updated to out-for-delivery and never progressed to a completed status.",
        "The merchant can show the order was dispatched, though the courier never confirmed arrival.",
        "Carrier tracking exists but terminates at the sorting facility with no delivery event.",
    ],
    "E2": [
        "The fulfilment system issued a licence key, but there are no access or download logs for it.",
        "An activation email was queued, though the platform holds no record of the account ever signing in.",
        "The merchant can evidence provisioning intent but produced no usage or entitlement logs.",
    ],
    "E3": [
        "The cardholder states the billing address on file is a former residence.",
        "AVS returned an N-response on this authorisation.",
        "The shipping address was typed in manually at checkout and does not correspond to the billing address on record.",
        "Address verification was not run for this transaction, so no AVS result is available.",
        "The order shipped to an address the cardholder had used before, but billing details were never matched against it.",
    ],
    "E4": [
        "The carrier's proof-of-delivery photo shows the parcel left unattended; no signature was collected.",
        "Signature on delivery was waived for this order value.",
        "A signature line appears on the docket but was left blank.",
    ],
    "E5": [
        "The merchant notes this was the customer's first order, with no prior transaction history on the card.",
        "Device fingerprinting was not enabled at the time of this purchase.",
        "Earlier orders exist on this account, but all of them were charged back as well.",
    ],
    "E6": [
        "The parcel was addressed to a company mailroom, but the merchant holds no proof of the cardholder's employment there.",
        "The delivery site is a shared office building; no employment record ties the cardholder to it.",
        "The merchant assumed a workplace delivery but never verified the cardholder worked there.",
    ],
    "E7": [
        "The merchant believes it replied to the cardholder, but no ticket or email thread was retained.",
        "No inbound customer contact was logged before the chargeback was filed.",
    ],
}


def _distractors_for(ev: set, kind: str, limit: int = 2) -> list:
    """Pick decoys only for E-items the case does NOT have, so the narrative
    stays truthful against the structured label."""
    candidates = []
    for item, pool in DISTRACTOR_CLAUSES.items():
        if item in ev:
            continue
        # Don't offer a physical-delivery decoy on a digital case, or vice versa.
        if kind == "digital" and item in {"E1", "E4"}:
            continue
        if kind == "physical" and item == "E2":
            continue
        candidates.append(random.choice(pool))
    random.shuffle(candidates)
    if not candidates:
        return []
    return candidates[: random.randint(1, limit)]


def render_local_narrative(evidence: list[str], contradicted: bool, kind: str) -> str:
    """Narrative states the tagged evidence, plus decoys for evidence the
    merchant does not have. Verdict still comes from labeler.py, never here."""
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
                        "The delivered scan resolves to a postcode several hundred miles from the cardholder's billing address.",
                    ]
                )
            )
        elif "E2" in ev:
            clauses.append(
                random.choice(
                    [
                        "Access logs show the digital good was provisioned to a different account email than the one on this transaction.",
                        "The entitlement was activated under an unrelated account the cardholder has never controlled.",
                    ]
                )
            )
        elif ev == {"E4"}:
            clauses.append(
                "A delivery signature is on file for a name unrelated to the cardholder, at an address that does not match billing."
            )
        else:
            clauses.append(
                random.choice(
                    [
                        "Merchant records show a full refund was already processed for this order before the dispute was filed.",
                        "The merchant's own ledger shows this order was cancelled and credited back weeks earlier.",
                    ]
                )
            )
        if "E7" in ev:
            clauses.append(random.choice(POSITIVE_CLAUSES["E7"]))
        body = clauses + _distractors_for(ev | {"E3"}, kind, limit=1)
        random.shuffle(body)
        return _opener(amount, product) + " " + " ".join(body)

    if not ev:
        clauses.append(
            random.choice(
                [
                    "Merchant has no shipping record, tracking number, or delivery confirmation on file.",
                    "The merchant response contains only the original invoice with no delivery or access evidence.",
                    "Nothing in the merchant's submission speaks to fulfilment; the packet is an order confirmation alone.",
                ]
            )
        )

    for item in ("E1", "E2", "E3", "E4", "E5", "E6", "E7"):
        if item in ev:
            clauses.append(random.choice(POSITIVE_CLAUSES[item]))

    clauses.extend(_distractors_for(ev, kind, limit=2))
    random.shuffle(clauses)
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

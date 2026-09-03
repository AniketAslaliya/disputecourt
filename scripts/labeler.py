"""
Ground-truth labeler for Visa 13.1 (Merchandise/Services Not Received) cases.

Implements the labeling logic from SKILL.md mechanically. This is the single
source of truth for what a case's verdict "should" be — the panel's job
(panel/personas.py) is to reconstruct this judgment from unstructured case
text, not the other way around. Never replace this with an LLM call.
"""

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    REPRESENT = "represent"
    ACCEPT = "accept"
    ABSTAIN = "abstain"


# Evidence item IDs, matching SKILL.md's table exactly.
DELIVERY_ITEMS = {"E1", "E2"}          # something was delivered / completed
IDENTITY_LINK_ITEMS = {"E3", "E5", "E6"}  # links delivery to the cardholder
SUPPORTIVE_ONLY = {"E4", "E7"}         # never sufficient alone


@dataclass
class Case:
    case_id: str
    narrative: str
    evidence_present: set[str] = field(default_factory=set)
    contradicted: bool = False  # e.g. delivery went to a different address than this transaction's


def label(case: Case) -> tuple[Verdict, str]:
    """Returns (verdict, reasoning). Mirrors SKILL.md's labeling logic exactly."""
    has_delivery = bool(case.evidence_present & DELIVERY_ITEMS)
    identity_links = case.evidence_present & IDENTITY_LINK_ITEMS

    if case.contradicted:
        return Verdict.ACCEPT, "Evidence contradicts the merchant's position."

    if not has_delivery:
        return Verdict.ACCEPT, "No proof of delivery or completion (E1/E2 absent)."

    if has_delivery and identity_links:
        return (
            Verdict.REPRESENT,
            f"Delivery/completion evidence present with identity link(s): {sorted(identity_links)}.",
        )

    # Delivery present but no clean identity link — check for the weak/partial case.
    only_supportive = bool(case.evidence_present & SUPPORTIVE_ONLY) and not identity_links
    if has_delivery and (only_supportive or not case.evidence_present & (DELIVERY_ITEMS | IDENTITY_LINK_ITEMS | SUPPORTIVE_ONLY)):
        return (
            Verdict.ABSTAIN,
            "Delivery evidence present but identity link is weak, partial, or unverifiable.",
        )

    return Verdict.ABSTAIN, "Evidence does not resolve cleanly under the matrix."


def confidence_for(case: Case, verdict: Verdict) -> float:
    """
    Reference confidence for synthetic ground truth (NOT the model's confidence —
    this is only used to sanity-check that generated cases have a plausible
    confidence distribution, e.g. abstains should cluster near 0.5).
    """
    identity_links = case.evidence_present & IDENTITY_LINK_ITEMS
    if verdict == Verdict.ABSTAIN:
        return 0.5
    if verdict == Verdict.REPRESENT:
        return min(0.6 + 0.1 * len(identity_links), 0.95)
    if verdict == Verdict.ACCEPT:
        return 0.85 if case.contradicted else 0.75
    return 0.5


if __name__ == "__main__":
    # Sanity check against the labeled example in SKILL.md.
    example = Case(
        case_id="skill-example-1",
        narrative="Customer disputes non-receipt of a $340 order. Merchant has "
        "tracking showing delivery to the billing address (AVS: Y-match), no "
        "signature required for this order value.",
        evidence_present={"E1", "E3"},
    )
    v, reasoning = label(example)
    conf = confidence_for(example, v)
    print(f"verdict={v.value} confidence={conf:.2f}")
    print(f"reasoning={reasoning}")
    assert v == Verdict.REPRESENT, "Labeler disagrees with SKILL.md's worked example!"
    print("\nMatches SKILL.md's worked example. Labeler is consistent.")

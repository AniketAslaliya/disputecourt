"""
The Court Panel: three personas debate a case, the Referee produces the
final verdict. This is the prompted baseline (Day 2 of PLAN.md) -- keep it
working and callable even after GRPO training exists (training/), since
it's the comparison point eval/metrics.py needs for the RL-tuned-vs-baseline
table.

The LLM call is injected as `llm_call_fn` so this module is fully testable
without a real API key -- see panel/smoke_test.py.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

CARDHOLDER_ADVOCATE_PROMPT = """You are the Cardholder-side advocate on a chargeback \
evidence review panel. Your job is to find every evidence gap, weakness, or \
ambiguity in the merchant's case. Do not fabricate gaps that aren't there --  \
only point to evidence that is genuinely missing, weak, or contradicted.

Case: {narrative}
Evidence the merchant has on file: {evidence_present}

List the weaknesses in 2-3 sentences. Be specific about which evidence items \
(E1-E7) are missing or insufficient."""

MERCHANT_ADVOCATE_PROMPT = """You are the Merchant-side advocate on a chargeback \
evidence review panel. Your job is to state which evidence items are present \
and explain why they meet the bar. Do not overstate weak evidence as strong.

Case: {narrative}
Evidence the merchant has on file: {evidence_present}

List the strengths in 2-3 sentences. Be specific about which evidence items \
(E1-E7) are satisfied and why."""

REFEREE_PROMPT = """You are the Network-Rules Referee on a chargeback evidence \
review panel. Apply this rules matrix literally -- you do not get to have \
opinions the matrix doesn't support:

REPRESENT if: (E1 or E2 present) AND (E3 or E5 or E6 present) AND not contradicted
ACCEPT if: (E1 and E2 both absent) OR contradicted
ABSTAIN if: neither of the above resolves cleanly (e.g. delivery evidence \
present but the only identity link is weak/partial/unverifiable)

Case: {narrative}
Evidence on file: {evidence_present}
Cardholder-side argument: {cardholder_argument}
Merchant-side argument: {merchant_argument}

Output ONLY a JSON object matching this schema, no other text:
{{
  "verdict": "represent" | "accept" | "abstain",
  "confidence": <float 0-1>,
  "evidence_present": [...],
  "rebuttal_draft": "<string, ONLY if verdict is represent -- must restate only \
evidence actually present in the case, never invent or strengthen it. Empty \
string otherwise.>",
  "reasoning": "<short, references specific E-items, no invented facts>"
}}"""


@dataclass
class PanelResult:
    case_id: str
    verdict: str
    confidence: float
    evidence_present: list
    rebuttal_draft: str
    reasoning: str
    cardholder_argument: str
    merchant_argument: str


LlmCallFn = Callable[[str], str]


def run_panel(case_id: str, narrative: str, evidence_present: list, llm_call_fn: LlmCallFn) -> PanelResult:
    cardholder_arg = llm_call_fn(
        CARDHOLDER_ADVOCATE_PROMPT.format(narrative=narrative, evidence_present=evidence_present)
    )
    merchant_arg = llm_call_fn(
        MERCHANT_ADVOCATE_PROMPT.format(narrative=narrative, evidence_present=evidence_present)
    )
    referee_raw = llm_call_fn(
        REFEREE_PROMPT.format(
            narrative=narrative,
            evidence_present=evidence_present,
            cardholder_argument=cardholder_arg,
            merchant_argument=merchant_arg,
        )
    )

    parsed = _parse_referee_output(referee_raw)

    return PanelResult(
        case_id=case_id,
        verdict=parsed["verdict"],
        confidence=parsed["confidence"],
        evidence_present=parsed.get("evidence_present", evidence_present),
        rebuttal_draft=parsed.get("rebuttal_draft", ""),
        reasoning=parsed.get("reasoning", ""),
        cardholder_argument=cardholder_arg,
        merchant_argument=merchant_arg,
    )


def _parse_referee_output(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Referee output had no parseable JSON: {raw_text[:200]}")
    parsed = json.loads(match.group(0))

    if parsed.get("verdict") not in ("represent", "accept", "abstain"):
        raise ValueError(f"Referee returned an invalid verdict: {parsed.get('verdict')}")

    # Safety check from CLAUDE.md's hard constraints -- catch a rebuttal on a
    # non-represent verdict, which should never happen but must never ship if it does.
    if parsed["verdict"] != "represent" and parsed.get("rebuttal_draft"):
        raise ValueError(
            "Referee produced a rebuttal_draft on a non-represent verdict -- "
            "this is exactly the disqualification risk flagged in CLAUDE.md. "
            "Do not silently drop this; investigate the referee prompt."
        )

    return parsed

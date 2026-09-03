---
name: chargeback-evidence-rules
description: Ground-truth rules matrix and Court Panel persona definitions for adjudicating Visa reason code 13.1 (Merchandise/Services Not Received) chargeback evidence. Read this before writing any labeling, panel-prompt, or reward-function code in this repo — it is the single source of truth for what counts as sufficient evidence.
---

# Chargeback Evidence Rules — Visa 13.1 (Merchandise/Services Not Received)

**Source note:** built from public secondary-source summaries of Visa's
reason code 13.1 and Compelling Evidence 3.0 requirements. Visa's own
merchant-facing documentation is the primary source if there's spare time on
Day 1 to cross-check — but this is solid enough to build against now, and
using *some* explicit, cited rules matrix is the point, not perfection on
every clause.

## Scope

Only Visa 13.1. Do not extend this matrix to other reason codes (13.2, 13.3,
10.4, etc.) mid-build — different reason codes have different evidence
requirements and mixing them dilutes the dataset's ground truth.

## Evidence items (what a case may contain)

| ID | Item | What it establishes |
|---|---|---|
| E1 | Delivery/tracking confirmation showing successful delivery | Goods physically arrived |
| E2 | Proof of digital delivery or access logs (for digital goods/services) | Service/digital good was delivered or accessed |
| E3 | Delivery address matches billing address / AVS Y-or-M match | Delivery went to the cardholder, not a stranger |
| E4 | Signature confirmation | Delivery was accepted by a specific person (needed for higher-value goods) |
| E5 | Same device/card used in a prior undisputed transaction | Continuity signal linking this transaction to the genuine cardholder |
| E6 | Proof cardholder was employed at delivery address at time of delivery | Only relevant when E3 fails because delivery went to a business address |
| E7 | Customer communication log (support tickets, emails) | Supportive context, never sufficient alone |

## Labeling logic (ground truth — apply this mechanically, do not ask an LLM to judge)

```
REPRESENT (fight the dispute) if:
  (E1 present OR E2 present)                      # something was delivered
  AND (E3 present OR E5 present OR E6 present)     # and it's linked to this cardholder
  AND NOT contradicted                             # no evidence contradicts the above (e.g. delivery to a
                                                    # different address than the one on this transaction)

ACCEPT THE LOSS if:
  (E1 absent AND E2 absent)                        # nothing shows delivery/completion happened
  OR contradicted                                  # evidence actively undermines the merchant's position

ABSTAIN (escalate to human) if:
  neither of the above resolves cleanly —
  e.g. E1/E2 present but the only identity link is E4 alone without E3/E5/E6,
  or E3 fails and E6 is itself unverifiable,
  or the case has conflicting evidence that doesn't clearly contradict
```

This is deliberately conservative: the abstain band should be wider than
feels natural for a demo, because a system that never abstains is a system
that's guessing, and guessing is what the "honest metrics" bar exists to
catch. When generating synthetic cases, deliberately construct a meaningful
fraction (~20-30%) to land in the abstain band — don't only generate clean
wins and clean losses.

## Court Panel personas

Three roles debate each case before the verdict is declared. Keep prompts
short and role-bounded — the panel's value is structured disagreement, not
each persona writing an essay.

**Cardholder-side advocate** — argues from the evidence gaps. Job is to find
every E-item that's missing, weak, or ambiguous. Never fabricates a gap that
isn't there.

**Merchant-side advocate** — argues from the evidence present. Job is to
state which E-items are satisfied and why they meet the bar. Never overstates
weak evidence as strong.

**Network-Rules Referee** — applies the labeling logic above literally,
mechanically, against what the other two surfaced. This persona's output
*is* the verdict + confidence + abstain decision — it does not get to have
opinions the matrix doesn't support.

## Output schema

```json
{
  "verdict": "represent | accept | abstain",
  "confidence": 0.0-1.0,
  "evidence_present": ["E1", "E3", ...],
  "rebuttal_draft": "string, only populated if verdict == represent",
  "reasoning": "short, references specific E-items, no invented facts"
}
```

`rebuttal_draft` must only ever restate evidence the case actually contains.
If the panel is tempted to strengthen a weak case in the rebuttal text,
that's the disqualification risk from `CLAUDE.md` — catch it here, not
after training.

## One labeled example

```json
{
  "case": "Customer disputes non-receipt of a $340 order. Merchant has tracking
    showing delivery to the billing address (AVS: Y-match), no signature
    required for this order value.",
  "evidence_present": ["E1", "E3"],
  "verdict": "represent",
  "confidence": 0.82,
  "reasoning": "E1 (delivery confirmed) + E3 (AVS Y-match to billing address)
    satisfies the matrix. No signature requirement at this order value, so E4
    absence doesn't block. No contradicting evidence."
}
```

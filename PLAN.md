# DisputeCourt — Razorpay AI Buildathon (Track 02: AI Risk Manager)

**Deadline:** Sept 5, 2026. **Today:** Sept 2. **Working days left: 3.**

## What we're building

A calibrated, RL-trained chargeback-evidence responder for one dispute reason
code: **Visa "Merchandise/Services Not Received" (compelling evidence
category)**. Given a disputed transaction + whatever evidence the merchant
has, the system outputs:

1. **Verdict** — represent (fight the dispute) / accept the loss / abstain (escalate to human)
2. **Confidence** — calibrated, not just a softmax number
3. **Rebuttal draft** — if representing, a drafted evidence packet a merchant could actually submit

Port of the DebateFloor / ClaimCourt architecture (Court Panel + confidence
declaration + GRPO reward shaping), retargeted from insurance claims to
chargeback evidence. Reuse the mechanic, not the domain code.

## Why this reason code specifically

It has a clean, public, checklist-style evidence requirement (proof of
delivery, tracking ID matched to ship-to address, customer communication,
IP/device match) — which means we can build **real ground truth from rules**,
not from an LLM's opinion. That grounding is the whole differentiator. Do not
scope-creep into other reason codes.

## No Adaption Labs — data pipeline is scripted

Seed set (~30 hand-built cases against the rules matrix) → expand via a
plain LLM-generation script (not a platform dependency) → label every case
by running the rules matrix against the evidence present, not by asking a
model to judge. ~300–500 cases total, ~100 held out for eval. See
`SKILL.md` for the actual rules matrix and labeling logic.

## Reward shaping (the "AI judgment" story)

Three terms, all reused from DebateFloor's calibration mechanic:
- **Correctness** — predicted verdict vs. rules-matrix ground truth
- **Calibration** — penalize confident-and-wrong harder than uncertain-and-wrong
- **Abstention credit** — reward "insufficient evidence, escalate" on genuinely ambiguous cases instead of forcing a guess

Report all three as numbers in the README — this is what makes the "honest
metrics including false-positive cost" bar a literal yes, not a gesture.

## Safety framing (avoid disqualification)

Track 02 explicitly disqualifies anything offense-capable. One line, stated
plainly in the README and the pitch: **low-confidence/no-evidence cases route
to "accept the loss," never to a coached or fabricated rebuttal.** The system
helps a merchant truthfully assemble evidence it already has — it does not
help win disputes it should legitimately lose.

## 3-day timeline

**Day 1 — Sept 2 (today)**
- [ ] Confirm/lock the rules matrix for the reason code (`SKILL.md`)
- [ ] Hand-write 20–30 seed cases against the matrix
- [ ] Fork DebateFloor/ClaimCourt scaffolding, strip insurance-specific parts
- [ ] Repo skeleton + this plan committed

**Day 2 — Sept 3**
- [ ] Scripted expansion to 300–500 cases, labeled via the rules matrix
- [ ] Wire the 3-persona panel (Cardholder-side / Merchant-side / Network-Rules Referee) to the new domain
- [ ] Get a plain-prompted (no RL yet) end-to-end pipeline running: case in → verdict + confidence + rebuttal draft out
- [ ] Keep this baseline around — it's the comparison point for Day 3

**Day 3 — Sept 4**
- [ ] GRPO fine-tune with the 3-term reward
- [ ] Eval harness: correctness / calibration / abstention rate / false-positive cost, RL-tuned vs. baseline
- [ ] Pick 3 pitch examples: one clear win, one clear loss, one abstain (not cherry-picked wins only)
- [ ] README with metrics table + safety framing
- [ ] Record 5-min pitch video

**Sept 5 — buffer**
- [ ] Polish repo, fill the 12-field application form, submit early — not at the wire

## Application form fields to have ready

Track (02 — AI Risk Manager), project name, what it solves, GitHub URL,
pitch video link, "what broke and how you fixed it" (use a real failure,
e.g. reward collapsing to always-low-confidence if the calibration term
isn't correctness-constrained — a known GRPO failure mode worth watching for
on Day 3).

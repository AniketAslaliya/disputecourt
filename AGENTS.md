# CLAUDE.md — DisputeCourt

Project context for Claude Code sessions in this repo. Read `PLAN.md` for
the timeline and `SKILL.md` for the domain rules before writing adjudication
logic — don't improvise evidence-sufficiency rules from general knowledge.

## What this is

A 3-day hackathon build for Razorpay's AI Buildathon (Track 02: AI Risk
Manager). A calibrated RL system that decides whether a merchant should
represent (fight) a chargeback, accept the loss, or escalate to a human —
for one Visa dispute reason code (Merchandise/Services Not Received).

It's a port of two earlier projects by the same author: **DebateFloor**
(3rd place, Meta PyTorch x Scaler Grand Finale — multi-agent Court Panel +
confidence-declared verdicts, GRPO on Qwen 0.5B) and **NyayaClaim**
(calibrated legal-insurance reasoning, GRPO + QLoRA on Qwen2.5-7B). Reuse
their architecture patterns; do not reuse their domain content.

## Hard constraints — do not violate

- **No Adaption Labs / Adaptive Data dependency.** Time-boxed build — the
  data pipeline is a plain generation script, no external platform in the
  critical path.
- **Ground truth comes from the rules matrix in `SKILL.md`, never from an
  LLM's judgment call.** If you're labeling data by asking a model "would
  this win," stop — that's the thing this project is explicitly not doing.
- **Low-confidence or evidence-thin cases must route to "accept the loss,"
  never to a fabricated or coached rebuttal.** This is a disqualification
  risk for the track (defense-only requirement), not a style preference.
- **Scope is one reason code.** Do not generalize the rules matrix to other
  dispute types mid-build — that's Day-4-that-doesn't-exist scope creep.

## Structure (expected)

```
/data/          seed cases + expanded generated dataset + eval split
/scripts/       data generation, labeling (rules matrix), training, eval
/panel/         the 3-persona debate logic (Cardholder-side, Merchant-side, Network-Rules Referee)
/training/      GRPO reward functions, fine-tuning entrypoint
SKILL.md        the evidence rules matrix + persona definitions — read this first
PLAN.md         timeline and scope
README.md       final deliverable doc — metrics table + safety framing, written Day 3
```

## What "done" looks like for each piece

- **Data:** 300–500 labeled cases, ~100 held out, every label traceable to
  the rules matrix in `SKILL.md`.
- **Panel pipeline:** runs end-to-end on a single case, prompted baseline
  first (no RL) — keep this baseline callable even after RL training exists,
  it's the comparison point for the pitch.
- **Reward:** three terms — correctness, calibration, abstention credit.
  Watch for reward collapsing to always-low-confidence if calibration isn't
  correctness-constrained; if that happens, it's a legitimate "what broke"
  story for the submission form, not just a bug to silently fix.
- **Eval:** report correctness, calibration error, abstention rate, and
  false-positive cost (both directions — wrongly represented vs. wrongly
  accepted) — RL-tuned vs. baseline, side by side.

## Working style for this build

Time-boxed to 3 days. Prefer the smallest thing that produces a real,
reportable number over a more elegant design that risks not finishing.
When in doubt about scope, check `PLAN.md`'s day-by-day list before adding
anything not on it.

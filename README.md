# DisputeCourt

**Razorpay AI Buildathon — Track 02: AI Risk Manager**

A calibrated chargeback-evidence responder for **Visa reason code 13.1
(Merchandise/Services Not Received)**. Given a disputed transaction and the
merchant's available evidence, the system outputs a verdict (represent /
accept / abstain), a calibrated confidence score, and — only when
representing — a rebuttal draft that restates evidence the merchant already
has.

## Why this reason code

Visa 13.1 has a clean, public, checklist-style evidence requirement (proof of
delivery, AVS match, device continuity, employment verification). That means
ground truth can come from a **deterministic rules matrix**, not from an LLM's
judgment call. The grounding is the differentiator.

## Architecture

```
Case (narrative + evidence IDs)
        │
        ▼
┌─────────────────────────────────┐
│   3-Persona Court Panel         │
│                                 │
│  Cardholder-side advocate       │  ← finds evidence gaps
│  Merchant-side advocate         │  ← states which E-items are met
│  Network-Rules Referee          │  ← applies matrix literally
└─────────────────────────────────┘
        │
        ▼
  Verdict + Confidence + Rebuttal Draft
```

**Prompted baseline** (panel/personas.py): three Gemini calls per case,
structured disagreement, referee applies the rules matrix.

**RL-tuned policy** (training/train_grpo.py): Qwen2.5-0.5B-Instruct fine-tuned
with GRPO using a 3-term reward (correctness, calibration, abstention credit).
Single forward pass at inference — no multi-agent calls needed.

## Evidence vocabulary (SKILL.md)

| ID | Item | What it establishes |
|---|---|---|
| E1 | Delivery/tracking confirmation | Goods physically arrived |
| E2 | Digital delivery / access logs | Service/digital good delivered |
| E3 | Address matches billing / AVS Y-or-M | Delivery went to the cardholder |
| E4 | Signature confirmation | Delivery was accepted (supportive only) |
| E5 | Same device/card in prior undisputed txn | Continuity signal |
| E6 | Employment proof at delivery address | Business-address identity link |
| E7 | Customer communication log | Supportive context, never sufficient alone |

## Labeling logic (ground truth)

```
REPRESENT if: (E1 or E2) AND (E3 or E5 or E6) AND NOT contradicted
ACCEPT    if: (E1 absent AND E2 absent) OR contradicted
ABSTAIN   if: neither resolves cleanly
```

Labels are assigned by `scripts/labeler.py`, mechanically, from the rules
matrix. The LLM is never asked "what's the verdict" during data generation.

## Dataset

| Split | Cases | Represent | Accept | Abstain |
|---|---|---|---|---|
| Train | 240 | 96 (40%) | 84 (35%) | 60 (25%) |
| Eval  | 100 | 40 (40%) | 35 (35%) | 25 (25%) |

~20–30% abstain fraction by design: a system that never abstains is guessing.

## Results

<!-- Fill after Day 3 GRPO run -->

| Metric | Prompted Baseline | RL-Tuned | Delta |
|---|---|---|---|
| Accuracy | — | — | — |
| Abstention rate | — | — | — |
| Brier score | — | — | — |
| Wrong-represent (FP cost) | — | — | — |
| Wrong-accept (FP cost) | — | — | — |

## Safety framing

**Low-confidence or evidence-thin cases route to "accept the loss," never to
a fabricated or coached rebuttal.** The system helps a merchant truthfully
assemble evidence it already has — it does not help win disputes it should
legitimately lose.

- The rebuttal draft is only populated when the verdict is "represent."
- It only ever restates evidence actually present in the case.
- The safety check in `panel/personas.py` raises a hard error if a rebuttal
  appears on a non-represent verdict.

## Repo structure

```
data/           seed cases, generated dataset, train/eval splits
scripts/        data generation, labeling (rules matrix), train/eval split
panel/          3-persona Court Panel (prompted baseline)
training/       GRPO reward function, fine-tuning entrypoint, Colab notebook
eval/           metrics (accuracy, Brier, abstention, FP cost)
SKILL.md        evidence rules matrix + persona definitions
PLAN.md         3-day timeline
```

## Running

```bash
# Label seed cases
python scripts/apply_labels.py

# Generate training data (no API key needed)
python scripts/generate_cases.py --local --n 300

# Split into train/eval
python scripts/split_dataset.py

# Run prompted baseline (needs GEMINI_API_KEY)
python panel/run_panel_live.py --data data/eval.jsonl --limit 10

# GRPO training (GPU required — use Colab notebook)
# training/train_grpo_colab.ipynb
```

## Prior work

Architecture ported from **DebateFloor** (3rd place, Meta PyTorch x Scaler
Grand Finale — Court Panel + GRPO on Qwen 0.5B) and **NyayaClaim**
(calibrated legal-insurance reasoning, GRPO + QLoRA on Qwen2.5-7B).
Domain content is original to this project.

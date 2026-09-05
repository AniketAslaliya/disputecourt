# DisputeCourt

**Razorpay AI Buildathon — Track 02: AI Risk Manager**

A calibrated chargeback-evidence adjudicator for **Visa reason code 13.1
(Merchandise/Services Not Received)** that is willing to say "I don't know."

Given a disputed transaction and the merchant's evidence file, it returns a
verdict (**represent** / **accept the loss** / **abstain and escalate**), a
calibrated confidence, and — only when representing — a rebuttal draft that
restates evidence the merchant already holds.

---

## The problem

Merchants lose money on chargebacks in two directions at once:

- **Fighting disputes they can't win** — representment fees spent on cases the
  evidence never supported.
- **Conceding disputes they could have won** — recoverable revenue written off.

A single win-rate number cannot tell you which direction you are bleeding,
because the two errors partially cancel in the average. DisputeCourt reports
them **separately**, and its training reward penalises them **differently**.

---

## What makes this more than a prompt

**1. Ground truth is deterministic.** Every label comes from a rules matrix
derived from Visa's published 13.1 compelling-evidence requirements
(`SKILL.md`, implemented in `scripts/labeler.py`). An LLM is *never* asked
"would this dispute win?" during labeling. Training a model to imitate another
model's guess and calling the result accuracy is the specific thing this
project refuses to do.

**2. Abstention is a real output.** A system that always answers is guessing on
hard cases and hiding it in the average. ~25% of the dataset is constructed to
be genuinely ambiguous, and the reward pays for escalating those rather than
forcing a call.

**3. Both error directions are costed.** `wrong-represent` and `wrong-accept`
carry different penalties in the GRPO reward, scaled by how confidently the
mistake was made.

---

## Architecture

```
                    Case file (narrative prose only)
                                 │
          ┌──────────────────────┴──────────────────────┐
          │                                             │
   PROMPTED BASELINE                            RL-TUNED POLICY
   3-persona Court Panel                        Qwen2.5-0.5B + LoRA
                                                trained with GRPO
   ┌─────────────────────────┐                         │
   │ Cardholder advocate     │ finds evidence gaps      │
   │ Merchant advocate       │ states what's satisfied  │
   │ Network-Rules Referee   │ applies the matrix       │
   └─────────────────────────┘                         │
          │  3 LLM calls                                │  1 forward pass
          └──────────────────────┬──────────────────────┘
                                 ▼
              verdict + calibrated confidence + rebuttal draft
                                 │
                                 ▼
                    scored against the rules matrix
             accuracy · Brier · abstention · FP cost (both ways)
```

The model sees **narrative prose only**. It has to recover the evidence set
itself; the structured evidence field exists solely to produce the ground-truth
label, and is never shown to the model. (It used to be — see *What broke*, #1.)

---

## Evidence vocabulary

| ID | Item | What it establishes |
|---|---|---|
| E1 | Delivery/tracking confirmation | Goods physically arrived |
| E2 | Digital delivery / access logs | Service/digital good delivered |
| E3 | Address matches billing / AVS Y-or-M | Delivery went to the cardholder |
| E4 | Signature confirmation | Supportive only — never an identity link alone |
| E5 | Same device/card in prior undisputed txn | Continuity signal |
| E6 | Employment proof at delivery address | Business-address identity link |
| E7 | Customer communication log | Supportive context, never sufficient alone |

## Labeling logic (ground truth)

```
REPRESENT if: (E1 or E2) AND (E3 or E5 or E6) AND NOT contradicted
ACCEPT    if: (E1 absent AND E2 absent) OR contradicted
ABSTAIN   if: neither resolves cleanly
```

## Reward (three terms + cost asymmetry)

| Term | Purpose |
|---|---|
| Correctness | ±1.0 against the rules-matrix verdict |
| Calibration | Brier-style; confident-and-wrong hurts more than uncertain-and-wrong |
| Abstention credit | Pays for escalating genuinely ambiguous cases; small penalty for dodging easy ones |
| FP asymmetry | `wrong-represent` penalised 0.6, `wrong-accept` 0.25, scaled by stated confidence |

`training/test_reward_sanity.py` verifies — over 12 seeded trials on the full
train split — that genuine calibrated behaviour out-scores both degenerate
shortcuts (always-abstain, always-represent) *and* the flat-0.5 hedging dodge,
before any compute is spent on training.

---

## Dataset

| Split | Cases | Represent | Accept | Abstain |
|---|---|---|---|---|
| Train | 340 | 136 (40%) | 119 (35%) | 85 (25%) |
| Eval  | 100 | 40 (40%)  | 35 (35%)  | 25 (25%) |

Narratives are generated from paraphrase pools with **negative distractors** —
sentences that use the same vocabulary as a piece of evidence in order to state
that it is *absent* ("AVS returned an N-response", "tracking never progressed
past out-for-delivery"). Labels come from the structured evidence set via
`labeler.py`, never from the generator and never from an LLM.

---

## Results

<!-- Generated by: python eval/compare_all.py -->

`[FILL FROM data/comparison.json — do not ship this file with this line in it]`

**How to read the keyword control.** The first column is a hand-written
keyword/regex extractor feeding the same rules matrix — no AI at all. It scores
high, and the honest reason is that it was written *after* the data generator,
so it matches templates its author had already read. **Any finite template
generator is invertible by someone who has read the templates.** So this column
is an *informed oracle*, not a floor: it marks the ceiling that perfect evidence
extraction reaches on synthetic data. It is reported rather than deleted because
deleting the number that complicates the story is how metrics stop meaning
anything.

The honest limitation this implies: **these numbers establish that the pipeline,
reward, and calibration machinery work end-to-end on a task with verifiable
ground truth. They do not establish real-world win rates.** That would need real
merchant dispute files, which this dataset is a stand-in for.

---

## Safety framing (defense-only)

**Low-confidence or evidence-thin cases route to "accept the loss," never to a
fabricated or coached rebuttal.**

- The rebuttal draft is populated **only** on a `represent` verdict.
- It may only restate evidence the case actually contains.
- `panel/personas.py` raises a hard error if a rebuttal ever appears on a
  non-represent verdict — it fails loudly rather than shipping quietly.

The system helps a merchant truthfully assemble evidence it already has. It
does not help win disputes it should legitimately lose, and it will not
manufacture a persuasive case out of thin evidence.

---

## What broke

**1. The model was handed the answer key.** The prompt included
`evidence_present` — the same structured field the labeler reads to produce
ground truth. The model wasn't reading a case file and deciding anything; it was
re-executing a lookup it had been given the inputs for. Every accuracy number
before this fix measured nothing. The prompt is now narrative-only.

**2. The eval set was solvable without AI.** A 12-line keyword matcher scored
**94%**, because the generator used one fixed sentence per evidence item —
`"Carrier tracking confirms a successful delivery scan"` *was* E1, all 122 times
it appeared. Regenerated with paraphrase pools and negative distractors.

**3. The fix didn't work as expected, which was more informative than if it had.**
The rewritten keyword control still scored ~95%, for the reason described above.
Rather than bury it, it became a reported column and a stated limitation.

**4. GRPO wouldn't run on free Colab, so the dependency was removed.** Three
attempts died in the TRL/peft/torchao version matrix. GRPO's update rule is
short enough not to need a framework, so it is implemented directly in
`training/grpo_minimal.py` against torch + transformers + peft: sample G
completions, score them, compute the advantage *within the group* (that
group-relative baseline is the whole idea — it removes the need for a value
network), backprop the advantage-weighted token log-probs, with a k3 KL penalty
against the LoRA-disabled base model as a zero-cost reference. Trains in fp32,
because fp16 GRPO on a T4 goes NaN once advantages get small.

**5. The reward sanity test was a coin flip.** It sampled once over 40 cases
with an unseeded RNG, and on one run reported that flat-0.5 hedging beat genuine
calibration — which would have meant the policy collapses to always-uncertain.
Averaged over 12 seeded trials on the full 340-case split, the correct ordering
holds (calibrated **+0.611** vs hedging **+0.539**, both far above the
degenerate strategies at ≈ **−0.65**). The reward was fine; the test wasn't.

---

## Repo structure

```
data/           seed cases, generated dataset, train/eval splits, results
scripts/        data generation, labeling (rules matrix), train/eval split
panel/          3-persona Court Panel (prompted baseline)
training/       reward, self-contained GRPO, Colab notebook
eval/           metrics, keyword control, model eval, comparison table
app.py          Gradio demo
SKILL.md        evidence rules matrix + persona definitions
SUBMISSION.md   application form answers
VIDEO_SCRIPT.md pitch video shot list
```

## Running

```bash
# 1. Regenerate the dataset (no API key needed)
python scripts/generate_cases.py --local --n 400 --fresh
python scripts/split_dataset.py

# 2. Verify the reward before spending compute on it
python training/test_reward_sanity.py

# 3. Non-AI control
python eval/keyword_baseline.py

# 4. Base model, no RL
python eval/run_model_eval.py --out data/results_base.jsonl

# 5. GRPO  (free Colab T4: training/train_grpo_colab.ipynb)
python training/grpo_minimal.py --steps 250 --group-size 6

# 6. Tuned model + the comparison table
python eval/run_model_eval.py --adapter training/checkpoints --out data/results_grpo.jsonl
python eval/compare_all.py

# Demo
python app.py
```

## Prior work

Architecture ported from **DebateFloor** (3rd place, Meta PyTorch × Scaler Grand
Finale — Court Panel + GRPO on Qwen 0.5B) and **NyayaClaim** (calibrated
legal-insurance reasoning, GRPO + QLoRA on Qwen2.5-7B). Domain content, rules
matrix, dataset, and reward design are original to this project.

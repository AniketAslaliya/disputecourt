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

100 held-out cases, same prompt and same parser for every model column.

| Metric | Keyword control (no AI) | Base Qwen2.5-0.5B | GRPO-tuned |
|---|---|---|---|
| Accuracy | 95.0% | 39.0% | 40.0% |
| Abstention rate | 24.0% | 2.0% | 0.0% |
| Brier score | 0.087 | 0.504 | **0.407** |
| Wrong-represent (n) | 0 | 34 | 35 |
| Wrong-accept (n) | 4 | 0 | 0 |
| Total FP cost | 4.0 | 34.0 | 35.0 |
| Unparseable output | — | 2 | **0** |

GRPO: 250 steps, group size 6 (1,500 sampled completions), LoRA r=16 on
Qwen2.5-0.5B, free Colab T4. Training reward rose from **−0.924** to **−0.650**
(`data/training_curve.png`).

### The headline result is negative, and the mechanism is legible

**Training reward improved. Task performance did not.** Accuracy moved 39% → 40%
— one case, indistinguishable from noise. Wrong-represents went 34 → 35.
Abstention went 2% → **0%**, which is worse.

What genuinely improved is narrower and identifiable:

- **Format compliance:** 2 unparseable outputs → 0.
- **Calibration:** Brier 0.504 → 0.407. At near-identical accuracy that is
  purely a confidence drop — solving the Brier decomposition gives a mean
  stated confidence of **~0.91 → ~0.81**.

So the reward went up because the policy learned to **emit valid JSON and hedge
its confidence**, not to adjudicate. The reward has four terms; a 0.5B model
given 1,500 samples can learn two of them and not the third:

| Reward term | Learnable in 250 steps? |
|---|---|
| Valid JSON (−1.5 floor) | **Yes** — pure format |
| Calibration (Brier) | **Yes** — just lower confidence |
| Abstention credit | No — requires knowing *which* cases are ambiguous |
| Verdict correctness | No — requires actually reading the case file |

The policy took the available points. Two design details made that the
path of least resistance, and both are ours:

1. **The FP penalty is scaled by stated confidence**
   (`−0.6 × conf` for wrong-represent). Lowering confidence reduces the penalty
   *without getting a single additional case right*. We intended this to
   discourage confident errors; it also created a gradient toward hedging that
   is far easier to follow than learning the task.
2. **Abstention collapsed rather than grew.** Correctly abstaining is the
   highest-reward action available (+1.0 correctness +0.3 credit = +1.3), but
   only if you can tell which cases qualify. A model that can't will find that
   always-`represent` (40% prevalence) beats always-`abstain` (25%) — so it
   went to 0% abstention, which is rational under the reward and useless in
   production.

**What we would change with more time**, in priority order: (a) start from a
supervised warm-start on evidence extraction so the correctness term has a
non-zero gradient before RL begins — GRPO cannot bootstrap a skill the base
policy never exhibits; (b) decouple the FP penalty from stated confidence so
hedging is not a cheap substitute for accuracy; (c) a 1.5B–7B base — 0.5B may
simply lack the capacity to track seven evidence items with distractors; (d)
far more than 1,500 samples.

**Why this is reported rather than buried.** The pipeline, reward, calibration
measurement, and training loop all work — verifiably, end to end, on a task with
deterministic ground truth. What the run shows is that *this reward at this
scale* optimises the wrong subset of its own objective. That is the failure mode
`CLAUDE.md` flagged before training started, it is diagnosable from the numbers
above, and reporting an RL result that did not work is the entire point of a
bar that says "honest metrics."

**The base model is a constant predictor, and its accuracy number hides that.**
Its confusion matrix (`data/results_base.summary.json`):

```
accept    -> represent : 34
represent -> represent : 39
abstain   -> represent : 25     98 / 100 predictions are "represent"
accept    -> abstain   :  1
represent -> abstain   :  1
```

Untuned Qwen2.5-0.5B answers `represent` to essentially every case. It never
once says "accept the loss." Its 39% accuracy is not competence — it is simply
the prevalence of `represent` in the split (40%), which is exactly what a
constant predictor scores. This is the clearest possible illustration of why
accuracy alone is a bad metric for this problem, and why the abstention rate and
the two FP directions are reported next to it: a single accuracy figure makes a
model that has learned nothing look like it is nearly 40% of the way there.

It is also the *unsafe* failure direction — a system that tells merchants to
contest every dispute, including the 35 it should concede. The reward's
asymmetric false-positive penalty and its abstention credit were aimed precisely
at this, and **they did not move it**: the tuned column still shows 35
wrong-represents and 0% abstention. See the analysis under the results table.

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
SKILL.md         evidence rules matrix + persona definitions
ARCHITECTURE.md  system design, data flow, training loop, safety gates
SUBMISSION.md    application form answers
VIDEO_SCRIPT.md  pitch video script
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

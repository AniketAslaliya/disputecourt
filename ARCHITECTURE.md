# Architecture — DisputeCourt

Track 02 (AI Risk Manager). One Visa reason code: **13.1, Merchandise/Services
Not Received.**

---

## 1. System shape

```
                       Merchant dispute case file
                            (narrative prose)
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        ADJUDICATOR            │
                    │  (one of two interchangeable  │
                    │   implementations)            │
                    └───────────────────────────────┘
                       │                        │
        ┌──────────────┘                        └──────────────┐
        ▼                                                      ▼
  PROMPTED BASELINE                                     RL-TUNED POLICY
  panel/personas.py                                     training/grpo_minimal.py
                                                        Qwen2.5-0.5B + LoRA
  ┌────────────────────────┐
  │ Cardholder advocate    │  finds evidence gaps
  │ Merchant advocate      │  states what is satisfied
  │ Network-Rules Referee  │  applies the matrix, declares verdict
  └────────────────────────┘
        3 LLM calls / case                              1 forward pass / case
        │                                                      │
        └──────────────────────────┬───────────────────────────┘
                                   ▼
              { verdict, confidence, evidence_present,
                rebuttal_draft, reasoning }
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            SAFETY GATE                    EVAL HARNESS
      rebuttal only on "represent"    eval/metrics.py
      hard-fails otherwise            accuracy · Brier · abstention
                                      · FP cost, both directions
```

## 2. The critical design constraint

**The adjudicator sees narrative prose only.**

Every case has two representations:

| Field | Who sees it | Purpose |
|---|---|---|
| `narrative` | the model | prose the model must read and interpret |
| `evidence_present`, `contradicted` | **only `labeler.py`** | produces the ground-truth verdict |

An earlier version passed `evidence_present` into the model prompt. That made
the task circular — the model was handed the labeler's exact input and was
re-executing a deterministic lookup, so its accuracy measured nothing. The
separation above is the fix, and it is enforced in `train_grpo.build_prompt()`,
which accepts `evidence_present` for call compatibility and deliberately ignores
it.

## 3. Data pipeline

```
seed_cases.json  (hand-written against the matrix)
        │
        ▼
scripts/generate_cases.py --local        no LLM, no external platform
        │   paraphrase pools (4-5 surface forms per evidence item)
        │   negative distractors (same vocabulary, states absence)
        ▼
scripts/labeler.py                       DETERMINISTIC. The only labeller.
        │   REPRESENT if (E1|E2) & (E3|E5|E6) & !contradicted
        │   ACCEPT    if (!E1 & !E2) | contradicted
        │   ABSTAIN   otherwise
        ▼
scripts/split_dataset.py                 stratified 340 train / 100 eval
```

No LLM is ever asked "would this dispute win?". Ground truth is a pure function
of the structured evidence set. This is what makes the reported accuracy a
measurement rather than an agreement rate between two models.

**Known limitation.** Narratives are generated from a finite template set, and
any finite template generator is invertible by someone who has read it — a
hand-written keyword extractor scores 95% on the eval split. That number is
reported as a column (`eval/keyword_baseline.py`) and framed as an informed
oracle rather than a floor. See README § Results.

## 4. Reward design

`training/reward.py`, four components:

| Component | Weight | Rationale |
|---|---|---|
| Correctness | ±1.0 | verdict vs. rules-matrix ground truth |
| Calibration | −0.5·(conf − correct)² | Brier-style; the training-time twin of the eval metric |
| Abstention credit | +0.3 / −0.15 | pays for escalating true-ambiguous; penalises dodging easy cases |
| FP asymmetry | −0.6·conf / −0.25·conf | wrong-represent costs more than wrong-accept |

Unparseable output returns −1.5 — a hard floor, because valid JSON has to come
before anything else can be optimised.

`training/test_reward_sanity.py` verifies over 12 seeded trials on the full
train split that a genuinely calibrated policy out-scores (a) always-abstain,
(b) always-represent, and (c) flat-0.5 hedging, *before* compute is spent.

## 5. Training loop

`training/grpo_minimal.py` — GRPO implemented directly on torch + transformers +
peft. No TRL: its API churn against Colab's torchao/peft versions cost three
build attempts, and the update rule is short enough not to need a framework.

Per prompt:
1. Sample **G** completions from the current policy.
2. Score each with `compute_reward`.
3. **Advantage = (r − mean r) / (std r + ε), computed within the group.** This
   group-relative baseline is the core of GRPO — it removes the need for a
   separate value network.
4. Loss = −(advantage × token log-prob) over completion tokens, plus a k3 KL
   penalty against the reference policy.
5. Zero-variance groups are skipped rather than divided by ~0.

Implementation notes that matter:

- **The reference policy is this model with the LoRA adapter disabled**
  (`model.disable_adapter()`), so KL costs no extra memory.
- **No full-vocab `log_softmax`.** Qwen's vocab is 151,936; materialising
  `[batch, seq, vocab]` in fp32 OOMs a T4. It uses
  `logit[target] − logsumexp(logits)` instead, allocating `[batch, seq]`.
- **Both passes chunk over the group** (`--micro-batch`). Each chunk divides by a
  normaliser computed over the whole group beforehand, so accumulated gradients
  equal the full-group gradient exactly; the flag trades peak memory for step
  count without changing the update.
- **fp32 on GPU**, deliberately — fp16 GRPO on a T4 produces NaN once advantages
  get small, and 0.5B in fp32 fits 16GB comfortably.

## 6. Evaluation

`eval/metrics.py` reports four things on the same 100 held-out cases, for every
adjudicator, through the same parser:

- **Accuracy** — against rules-matrix ground truth
- **Abstention rate** — vs. the true 25%
- **Brier score** — calibration, using the model's own stated confidence
- **False-positive cost, both directions** — `wrong-represent` (fought an
  unwinnable dispute) and `wrong-accept` (conceded a winnable one), costed
  separately

Abstentions are never counted as false positives in either direction —
abstaining on a genuinely uncertain case is correct behaviour, not an error.

`eval/compare_all.py` assembles the columns that exist and skips the ones that
don't, so the table always reflects runs that actually happened.

**Measured outcome (see README § Results for the full table and analysis).** The
GRPO run improved training reward (−0.924 → −0.650) without improving the task:
accuracy 39% → 40%, abstention 2% → **0%**, wrong-represents 34 → 35. What did
improve was format compliance (2 unparseable → 0) and calibration (Brier 0.504 →
0.407, a pure confidence drop at flat accuracy).

Architecturally, the lesson sits in §4: the reward has four terms, and only two
of them — JSON validity and calibration — are reachable by a 0.5B policy in 1,500
samples. Verdict correctness needs the model to actually read the case file, and
abstention credit needs it to identify ambiguity; neither had a usable gradient
because the base policy never emitted `accept` or `abstain` at all. **GRPO cannot
bootstrap a behaviour the base policy never exhibits**, so a supervised
warm-start on evidence extraction has to precede RL. Compounding it, the FP
penalty is scaled by stated confidence (§4), which made hedging a cheaper way to
raise reward than getting cases right.

## 7. Safety architecture (defense-only)

Three enforcement points, not a policy statement:

1. `rebuttal_draft` is populated **only** when `verdict == "represent"`.
2. It may only restate evidence the case actually contains.
3. `panel/personas.py::_parse_referee_output` **raises** if a rebuttal appears on
   a non-represent verdict. It fails loudly rather than shipping quietly.

Low-confidence and evidence-thin cases route to *accept the loss*. The system
helps a merchant truthfully assemble evidence they already hold; it does not
help win disputes that should legitimately be lost, and it will not manufacture
a case out of thin evidence.

## 8. Module map

| Path | Responsibility |
|---|---|
| `SKILL.md` | Rules matrix + persona definitions — the domain source of truth |
| `scripts/labeler.py` | Deterministic ground-truth labeller |
| `scripts/generate_cases.py` | Synthetic case generation (paraphrases + distractors) |
| `scripts/split_dataset.py` | Stratified train/eval split |
| `panel/personas.py` | 3-persona prompted baseline + safety gate |
| `training/reward.py` | GRPO reward (4 components) |
| `training/grpo_minimal.py` | Self-contained GRPO training loop |
| `eval/metrics.py` | Accuracy, Brier, abstention, bidirectional FP cost |
| `eval/keyword_baseline.py` | Non-AI control column |
| `eval/run_model_eval.py` | Runs any HF model/adapter over the eval split |
| `eval/compare_all.py` | Assembles the comparison table |
| `app.py` | Gradio demo (rules / RL policy / panel) |

# Submission — Razorpay AI Builder Internship 2026

Draft answers for the application form. Numbers marked `[FILL]` come from
`data/comparison.json` once the Colab run finishes — paste them in before
submitting, and do not submit with a placeholder still in place.

---

## Selected Track

**Track 02 — AI Risk Manager**

---

## Project Name / Title

**DisputeCourt — a calibrated chargeback-evidence adjudicator that knows when to abstain**

---

## Project Objectives / What does it solve?

Merchants lose money on chargebacks twice: once by fighting disputes they were
never going to win (representment fees on unwinnable cases), and once by
conceding disputes they had the evidence to win. Both errors are invisible in
aggregate win-rate reporting, because a single "win rate" number cannot tell
you which direction you are bleeding.

DisputeCourt takes a disputed transaction and the merchant's evidence file for
one Visa reason code — 13.1, Merchandise/Services Not Received — and returns
three things: a **verdict** (represent / accept the loss / escalate to a human),
a **calibrated confidence**, and, only when representing, a **rebuttal draft
that restates evidence the merchant already holds**.

Three things make it different from wrapping an LLM in a prompt:

1. **Ground truth is deterministic, not model opinion.** Every label comes from
   a rules matrix (`SKILL.md`, implemented in `scripts/labeler.py`) derived from
   Visa's published 13.1 compelling-evidence requirements. No LLM is ever asked
   "would this win?" during labeling. That is what makes the accuracy number
   mean something.

2. **Abstention is a first-class output.** A system that always produces an
   answer is guessing on the hard cases and hiding it in the average. ~25% of
   the dataset is deliberately constructed to be genuinely ambiguous, and the
   RL reward pays for correctly escalating those instead of forcing a call.

3. **Both error directions are costed separately.** Wrongly representing
   (fighting an unwinnable dispute) and wrongly accepting (conceding a winnable
   one) have different costs, and the GRPO reward encodes that asymmetry rather
   than treating "wrong" as one undifferentiated bucket.

**Safety / defense-only.** Low-confidence or evidence-thin cases route to
"accept the loss." The rebuttal draft is populated only on a `represent`
verdict and may only restate evidence the case actually contains; a hard check
in `panel/personas.py` raises rather than emitting a rebuttal on any other
verdict. The system helps a merchant truthfully assemble evidence it already
has. It does not help win disputes it should legitimately lose, and it will not
draft a persuasive case out of thin evidence.

**Architecture.** A 3-persona Court Panel (Cardholder-side advocate finds
evidence gaps, Merchant-side advocate states what is satisfied, Network-Rules
Referee applies the matrix and declares the verdict) as the prompted baseline,
plus a Qwen2.5-0.5B policy fine-tuned with GRPO against a three-term reward
(correctness + calibration + abstention credit, with asymmetric false-positive
cost). The RL policy collapses the 3-call debate into a single forward pass at
inference.

---

## GitHub Repository URL

https://github.com/AniketAslaliya/disputecourt

---

## 5-min Pitch Video Link

`[FILL — see VIDEO_SCRIPT.md]`

---

## Build Challenges & Technical Obstacles

*(Everything below actually happened during the build. Kept in the order it was
found, because the second one was only visible after fixing the first.)*

**1. The model was being handed the answer key.**
The prompt included `evidence_present` — the same structured field the
deterministic labeler reads to produce the ground-truth verdict. So the model
was not reading a case file and deciding anything; it was re-executing a lookup
it had been given the inputs for. Any accuracy it scored measured nothing.
Fixed by making the prompt narrative-only: the label still derives from the
structured field, but the model only ever sees prose and has to recover the
evidence set itself.

**2. The eval set was solvable without AI, and I could prove it.**
After fixing (1) I wrote a 12-line keyword matcher as a control. It scored
**94%** on the held-out split. The cause was the data generator: it used one
fixed sentence per evidence item, so `"Carrier tracking confirms a successful
delivery scan"` *was* E1, all 122 times it appeared. I regenerated the dataset
with paraphrase pools (4–5 surface forms per item) and **negative distractors** —
sentences using the same vocabulary (AVS, tracking, signature, employment) to
state that the proof is *absent*, e.g. *"AVS returned an N-response"* or
*"tracking never progressed past out-for-delivery."*

**3. The fix did not work the way I expected, which was the more useful result.**
The rewritten keyword control still scored ~95%, because I had written it *after*
writing the paraphrase pools — it was matching my own templates. The real lesson
is that **any finite template generator is invertible by someone who has read the
templates**, so a keyword control on synthetic data is an informed oracle, not a
floor. I kept it in the repo and report it as exactly that (`eval/keyword_baseline.py`),
rather than quietly deleting the number that undercut my story. Treating it as a
"we beat the dumb baseline" trophy would have been the dishonest move.

**4. GRPO would not run on free Colab, so I removed the dependency.**
Three separate attempts died in the TRL / peft / torchao version matrix
(`GRPOTrainer` API drift, then a torchao import clash, then fp16/bf16). GRPO's
update rule is short enough not to need a framework, so I implemented it
directly against torch + transformers + peft (`training/grpo_minimal.py`, ~60
lines of actual algorithm): sample G completions per prompt, score each with the
reward, compute the advantage *within the group* — that group-relative baseline
is the entire point of GRPO and removes the need for a value network — and
backprop the token log-probs weighted by advantage, with a k3 KL penalty against
the LoRA-disabled base model as a free reference. The notebook now installs one
package. Related: I train in **fp32 on the T4**, because fp16 GRPO goes NaN once
advantages get small, and a 0.5B model in fp32 fits in 16GB comfortably.

**5. My reward sanity test was a coin flip.**
The test that certifies the reward doesn't incentivise degenerate policies
sampled *once* over 40 cases with an unseeded RNG. On one run it reported that
flat-0.5 hedging beat genuine calibration — which, if true, would mean GRPO
collapses to always-uncertain and the whole calibration story dies. Re-running
over the full 340-case split averaged across 12 seeded trials showed the correct
ordering (calibrated +0.611 vs hedging +0.539, both far above the degenerate
always-abstain/always-represent strategies at ≈ −0.65). The reward was fine; the
test wasn't. A test that is only right on average is not evidence you can put in
a README.

---

## Final Submission Confirmation

Tick only once the video link is in and every `[FILL]` above is replaced.

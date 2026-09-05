# 5-Minute Pitch Video — Word-for-Word Script

Read the quoted lines aloud while screen-recording. Narration is ~4:15 at a
normal pace (~145 wpm), leaving ~40s for clicking through the demo and letting
the table render. Lands around 4:50.

**Setup before you hit record**
- `python app.py` running in a browser tab, examples loaded
- A terminal at the repo root, ready to run `python eval/compare_all.py`
- `SKILL.md` and `training/reward.py` open in tabs
- OBS or Loom at 1080p, mic tested

**The one rule:** never say a number that isn't in `data/comparison.json`. If
GRPO didn't finish, use **Branch B** at 3:00 — it's written, and it's honest.

---

## 0:00 – 0:30 — Problem  *(webcam or title card)*

> "Merchants lose money on chargebacks in two opposite directions at once. They
> fight disputes they were never going to win, paying a representment fee for the
> privilege. And they concede disputes they had the evidence to win.
>
> A single win-rate number can't tell you which way you're bleeding — the two
> errors cancel in the average. DisputeCourt measures them separately.
>
> Track 2, AI Risk Manager, on one Visa reason code: 13.1, merchandise or
> services not received."

---

## 0:30 – 1:20 — Demo  *(screen: the Gradio app — run all three examples)*

> "Three outputs: a verdict, a calibrated confidence, and a rebuttal draft.
>
> Tracking confirms delivery, AVS matches billing — represent, with a rebuttal
> restating only evidence the merchant actually holds.
>
> No delivery record — accept the loss, and the rebuttal field is empty. That's
> enforced in code.
>
> And the one I care about: delivery confirmed, but the only identity link is an
> illegible signature. Nothing ties that parcel to this cardholder. It doesn't
> guess — it escalates. A quarter of my dataset lands there by design."

---

## 1:20 – 2:05 — Ground truth  *(screen: `SKILL.md`, then `scripts/labeler.py`)*

> "Here's what makes the numbers mean anything.
>
> Every label comes from a rules matrix built from Visa's published evidence
> requirements for 13.1. Proof of delivery, plus something linking it to this
> cardholder, nothing contradicting it — represent. No proof of delivery, or
> evidence that undermines the merchant — accept. Anything unresolved — abstain.
>
> Ninety lines of Python, and it's the only thing here that assigns a label. I
> never ask a model 'would this win?' — that's training a model to imitate
> another model's guess and reporting the agreement as accuracy."

---

## 2:05 – 2:50 — Reward and GRPO  *(screen: `training/reward.py`, then `grpo_minimal.py`)*

> "The policy is a half-billion-parameter Qwen fine-tuned with GRPO.
>
> Three reward terms: correctness against the matrix; a Brier-style calibration
> term, so confident-and-wrong hurts more than uncertain-and-wrong; and abstention
> credit, so escalating an ambiguous case pays better than guessing.
>
> Plus what the brief names explicitly — false-positive cost. Wrongly representing
> and wrongly accepting aren't equally bad, so they carry different penalties.
>
> And this is GRPO written directly, not a framework call — the advantage is
> computed within the group, which is what removes the value network."

---

## 2:50 – 4:00 — The numbers  *(run `python eval/compare_all.py` live)*

> "Same hundred held-out cases, same prompt, same parser for every column.
>
> Base model, before any RL: thirty-nine percent. That number is a trap, and the
> confusion matrix shows why — it answers 'represent' to ninety-eight cases out of
> a hundred. It never once says 'accept the loss.' Thirty-nine percent is just the
> base rate of represent in my split. A constant predictor that learned nothing,
> and plain accuracy makes it look forty percent competent.
>
> That's why abstention rate and both error directions sit next to accuracy here.
> Thirty-four wrong-represents, zero wrong-accepts — it tells merchants to fight
> everything. The expensive direction, and the unsafe one."

### → Branch A — GRPO finished

> "After GRPO: [accuracy]. But the number I care about is wrong-represents
> dropping from thirty-four to [N], and abstention moving from two percent toward
> the true twenty-five. The reward's cost asymmetry did that."

### → Branch B — GRPO didn't finish

> "The GRPO run didn't finish in my window — three attempts lost to Colab
> environment failures, one to a real bug of mine, a full-vocabulary softmax that
> OOM'd a T4. The loop runs and is verified end to end on CPU. What I don't have
> is a tuned column, and I'd rather show an empty cell than a number I can't
> reproduce."

### Either branch, then:

> "The first column is a keyword matcher — no AI at all — and I want to be
> straight about it. Ninety-five percent, and it scores that high because I wrote
> it after I wrote the data generator, so it matches my own templates. Any finite
> template generator is invertible by whoever read the templates. That's a real
> limitation of synthetic data. It's in the README, and I kept the column rather
> than delete the number that complicates my story."

---

## 4:00 – 4:35 — What broke

> "The one that mattered most: my prompt was handing the model the structured
> evidence field — the exact input my labeler uses to build the ground truth. It
> wasn't reading a case file, it was re-running a lookup it already had the
> answers to. Every accuracy number before I caught that was measuring nothing.
>
> That's why the model now only ever sees prose, and has to recover the evidence
> set itself."

*(Two more are written up in the form answer and the README — the unseeded
reward sanity test, and the full-vocab softmax that OOM'd the T4. Only mention
them if you're comfortably under time.)*

---

## 4:35 – 4:50 — Safety and close

> "One rule throughout: low-confidence or evidence-thin cases route to accept the
> loss — never to a fabricated or coached rebuttal. The drafter is only reachable
> on a represent verdict, it can only restate evidence the case contains, and a
> hard check raises if a rebuttal appears anywhere else.
>
> This helps a merchant truthfully assemble evidence they already have. It does
> not help them win a dispute they should lose. Repo's linked — thanks."

---

## Before you upload

- [ ] Under 5:00
- [ ] Every number spoken matches `data/comparison.json`
- [ ] Uploaded **unlisted**, not private — open the link in a logged-out window
      and confirm it plays. A private link scores zero.
- [ ] Link pasted into `SUBMISSION.md`

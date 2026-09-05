# 5-Minute Pitch Video — Word-for-Word Script

Read the quoted lines aloud while screen-recording. ~640 spoken words ≈ **4:05 at
a normal pace**, leaving ~40s for clicking through the demo and letting the table
render. Lands around **4:45**.

**Setup before you hit record**
- `python app.py` running in a browser tab, examples loaded
- A terminal at the repo root with `python eval/compare_all.py` ready to run
- `SKILL.md` and `data/training_curve.png` open in tabs
- OBS or Loom at 1080p, mic tested

**The one rule:** every number below is from `data/comparison.json`. Don't add any.

---

## 0:00 – 0:25 — Problem

> "Merchants lose money on chargebacks in two opposite directions at once. They
> fight disputes they were never going to win, paying a representment fee for the
> privilege. And they concede disputes they had the evidence to win.
>
> A single win-rate number can't tell you which way you're bleeding — the two
> errors cancel in the average. DisputeCourt measures them separately. Track 2,
> on one Visa reason code: 13.1, merchandise not received."

---

## 0:25 – 1:10 — Demo  *(run all three examples)*

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

## 1:10 – 1:50 — Ground truth  *(screen: `SKILL.md`, then `scripts/labeler.py`)*

> "Here's what makes the numbers mean anything.
>
> Every label comes from a rules matrix built from Visa's published evidence
> requirements. Proof of delivery, plus something linking it to this cardholder,
> nothing contradicting it — represent. No proof of delivery — accept. Anything
> unresolved — abstain.
>
> Ninety lines of Python, and it's the only thing here that assigns a label. I
> never ask a model 'would this win?' — that's training a model to imitate another
> model's guess and reporting the agreement as accuracy."

---

## 1:50 – 2:20 — Reward  *(screen: `training/reward.py`)*

> "The policy is a half-billion-parameter Qwen tuned with GRPO, written directly
> rather than through a framework.
>
> Four reward terms: correctness against the matrix, Brier-style calibration,
> abstention credit for escalating ambiguous cases, and — the thing the brief names
> explicitly — false-positive cost, with wrongly representing penalised harder than
> wrongly accepting."

---

## 2:20 – 3:10 — The base model  *(run `python eval/compare_all.py`)*

> "Same hundred held-out cases, same prompt, same parser for every column.
>
> Base model, before RL: thirty-nine percent. That number is a trap, and the
> confusion matrix shows why — it answers 'represent' to ninety-eight cases out of
> a hundred. It never once says 'accept the loss.' Thirty-nine percent is just the
> prevalence of represent in my split. A constant predictor that learned nothing,
> and plain accuracy makes it look forty percent competent.
>
> That's why abstention rate and both error directions sit next to accuracy here.
> Thirty-four wrong-represents, zero wrong-accepts — it tells merchants to fight
> everything. The expensive direction, and the unsafe one."

---

## 3:10 – 4:05 — The RL result  *(show `data/training_curve.png`)*

> "Now the RL column — and this is a negative result. I'd rather walk you through
> it than dress it up.
>
> Training reward improved, minus point nine to minus point six-five. But accuracy
> went thirty-nine to forty percent — one case, noise. Abstention went two percent
> to zero, which is worse.
>
> What did improve: unparseable outputs, two to zero. And Brier, point five-oh to
> point four-one — at flat accuracy that's purely a confidence drop.
>
> So it learned to emit valid JSON and hedge. It did not learn to adjudicate. Of
> my four reward terms, a half-billion-parameter model on fifteen hundred samples
> can learn two — format and calibration — but not verdict correctness, which
> needs reading, or abstention credit, which needs knowing which cases are
> ambiguous. It took the points that were available.
>
> Two of those were my design errors. My false-positive penalty scales with stated
> confidence, so hedging reduces it without getting anything right. And
> always-represent at forty percent prevalence beats always-abstain at twenty-five,
> so abstention collapsed. Rational under my reward. Useless in production. The fix
> is a supervised warm-start on evidence extraction — GRPO can't bootstrap a skill
> the base policy never shows."

---

## 4:05 – 4:45 — Keyword control, safety, close

*If you're already past 4:15 when you reach this, **skip the keyword-control
paragraph** and go straight to the safety lines. It's fully covered in the
README; the safety close is not, and matters more on camera.*

> "One more column: a keyword matcher, no AI. Ninety-five percent — and it scores
> that high because I wrote it *after* the data generator, so it matches my own
> templates. Any finite template generator is invertible by whoever read it. That's
> a real limit of synthetic data, it's in the README, and I kept the column rather
> than delete the number that complicates my story.
>
> And one rule throughout: evidence-thin cases route to accept the loss, never to
> a fabricated rebuttal. The drafter is only reachable on a represent verdict, and
> a hard check raises if a rebuttal appears anywhere else.
>
> This helps a merchant truthfully assemble evidence they already have. It doesn't
> help them win a dispute they should lose. Repo's linked — thanks."

---

## Before you upload

- [ ] Under 5:00
- [ ] Every number spoken matches `data/comparison.json`
- [ ] Uploaded **unlisted**, not private — open the link in a logged-out window
      and confirm it plays. A private link scores zero.
- [ ] Link pasted into `SUBMISSION.md`

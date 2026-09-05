# 5-Minute Pitch Video — Space-Only Script

Everything happens on one URL: **https://huggingface.co/spaces/AniketAsla/disputecourt**

No terminal, no editor, no repo. Screen-record that page for the whole video.
~690 spoken words ≈ **4:15–4:50** including click pauses.

**Before you record**
- Open the Space, wait for it to finish building, and click one example once so
  the model is warm and the page is scrolled to the top
- Browser zoom **125%**
- Collapse the "Results & method" accordion — you'll open it on camera at 2:45
- Have `data/comparison.json` numbers nowhere else; every figure you say is
  already on screen inside that accordion

**The one rule:** in **Rules-matrix mode the verdict comes from the checkboxes,
not the narrative text.** Never say "it reads the case and decides" in that mode
— that's the RL policy's job, and the mismatch is exactly what a technical judge
catches.

---

## 0:00 – 0:25 — Problem  *(Space landing page, top)*

> "Merchants lose money on chargebacks in two opposite directions at once. They
> fight disputes they were never going to win, paying a representment fee for the
> privilege. And they concede disputes they had the evidence to win.
>
> A single win-rate number can't tell you which way you're bleeding — the two
> errors cancel in the average. DisputeCourt measures them separately. Track 2,
> on one Visa reason code: 13.1, merchandise not received."

---

## 0:25 – 1:15 — The three verdicts  *(scroll to Examples, click rows 1, 2, 3)*

Click **Example 1** → Adjudicate.

> "Three outputs: a verdict, a calibrated confidence, and a rebuttal draft. Here
> tracking confirms delivery and the address matches billing — represent, with a
> rebuttal that restates only evidence the merchant actually holds."

Click **Example 2** → Adjudicate. *Cursor on the output where no rebuttal appears.*

> "No delivery record at all — accept the loss. And notice there's no rebuttal.
> That's enforced in code, not a style choice: the drafter is unreachable on any
> verdict except represent."

Click **Example 3** → Adjudicate.

> "And the one I care about. Delivery is confirmed, but the only identity link is
> an illegible signature — nothing ties that parcel to this cardholder. It
> doesn't guess. It escalates to a human. A quarter of my dataset is built to
> land exactly there."

---

## 1:15 – 1:50 — Ground truth  *(cursor over the E1–E7 checkboxes)*

> "These seven items are the whole evidence vocabulary — proof of delivery,
> digital access logs, address match, signature, device continuity, employment
> proof, support logs.
>
> And here's what makes the numbers mean anything. I tell the system which of
> these the merchant actually holds, and a rules matrix decides — built from
> Visa's published requirements. Delivery proven, plus something linking it to
> this cardholder, nothing contradicting it: represent. No proof of delivery:
> accept. Anything unresolved: abstain.
>
> That matrix is ninety lines of Python and it's the only thing that assigns a
> label. I never ask a model 'would this win?' — that's training a model to
> imitate another model's guess and reporting the agreement as accuracy."

---

## 1:50 – 2:20 — The three modes  *(cursor over the mode radio)*

> "Three ways to adjudicate. The rules matrix, deterministic, what you just saw —
> that's the ground truth.
>
> The Court Panel: three personas — a cardholder-side advocate finding evidence
> gaps, a merchant-side advocate arguing what's satisfied, and a network-rules
> referee that applies the matrix and declares the verdict.
>
> And the RL policy: a half-billion-parameter Qwen trained with GRPO that gets the
> case as **prose only**. It never sees these checkboxes — it has to recover the
> evidence set by reading. That's the real research question here, and it's where
> my honest result is."

---

## 2:20 – 2:45 — Show the model failing  *(select RL policy, re-run Example 2)*

Click **Example 2** again, switch mode to **RL policy**, Adjudicate. Wait for it.

> "Same case — no delivery evidence whatsoever. The rules matrix said accept the
> loss. Watch what the trained policy says.
>
> Represent. It tells the merchant to fight a dispute with nothing behind it.
> I'm showing you that deliberately, because I'm about to quantify it."

---

## 2:45 – 4:00 — The numbers  *(open "Results & method")*

> "Same hundred held-out cases for every column.
>
> The base model, before RL: thirty-nine percent — and that's a trap. It answers
> represent to ninety-eight cases out of a hundred, and never once says accept.
> Thirty-nine percent is just the prevalence of represent in my split. A constant
> predictor that learned nothing, and plain accuracy makes it look forty percent
> competent. That's why abstention rate and both error directions sit right next
> to accuracy.
>
> Now the RL column, and this is a negative result. Training reward improved,
> minus point nine to minus point six-five. But accuracy went thirty-nine to
> forty — one case, noise. Abstention went two percent to zero, which is worse.
>
> What did improve: unparseable outputs, two to zero. And calibration — Brier from
> point five-oh to point four-one, which at flat accuracy is a pure confidence
> drop. So it learned to emit valid JSON and hedge. It did not learn to
> adjudicate.
>
> Of my four reward terms, a half-billion model on fifteen hundred samples can
> learn two — format and calibration — but not verdict correctness, which needs
> reading, or abstention credit, which needs knowing which cases are ambiguous. It
> took the points that were available. And the base policy never emitted accept or
> abstain at all, so there was nothing for the correctness gradient to reinforce.
> GRPO can't bootstrap a behaviour the base policy never shows. The fix is a
> supervised warm-start on evidence extraction, first."

---

## 4:00 – 4:45 — Keyword control, safety, close  *(scroll within the accordion)*

> "One more column: a keyword matcher, no AI, ninety-five percent. And it scores
> that high because I wrote it *after* the data generator, so it matches my own
> templates. Any finite template generator is invertible by whoever read it.
> That's a real limit of synthetic data — I kept the column rather than delete the
> number that complicates my story.
>
> And one rule throughout: evidence-thin cases route to accept the loss, never to
> a fabricated rebuttal — which is why example two came back empty.
>
> This helps a merchant truthfully assemble evidence they already have. It doesn't
> help them win a dispute they should lose. Code and full analysis are linked at
> the bottom. Thanks."

---

## Before you upload

- [ ] Under 5:00
- [ ] Every number spoken is visible on screen in the accordion
- [ ] Uploaded **unlisted**, not private — open the link logged-out and confirm
- [ ] Link pasted into `SUBMISSION.md`

**If you're running long:** cut the keyword-control paragraph at 4:00 (it's in the
accordion anyway) and the mode walkthrough at 1:50. Never cut the safety close or
the 2:20 failure demo — those are the two that carry the most credibility.

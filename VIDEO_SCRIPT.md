# 5-Minute Pitch Video — Shot List and Script

Record with OBS or Loom, 1080p, screen + mic. No slides deck needed beyond two
title cards — screen-recording the actual repo and the actual metrics table is
more convincing than slides, and it is the thing the brief asks to see.

**Rule for this recording: never show a number that isn't in `data/comparison.json`.**
If the GRPO run didn't finish, say so on camera and show the columns you do
have. A missing column you name is survivable; a number you can't reproduce
when asked is not.

---

### 0:00 – 0:30 — The problem (webcam or voice over a title card)

> "Merchants lose money on chargebacks in two directions at once. They fight
> disputes they were never going to win, and pay a representment fee for the
> privilege. And they concede disputes they actually had the evidence to win.
>
> A single win-rate number can't tell you which way you're bleeding — the two
> errors cancel out in the average. That's the problem DisputeCourt is built
> around."

---

### 0:30 – 1:20 — What it does (screen: the Gradio app)

Run the app. Walk through **three** cases from the Examples row — deliberately
one of each, and say out loud that you're showing all three kinds:

1. **Represent** — tracking + AVS match. Show the verdict, the confidence, and
   the rebuttal draft.
2. **Accept** — no delivery evidence at all. Point out the rebuttal field is
   *empty*, and that this is enforced, not stylistic.
3. **Abstain** — delivery confirmed but the only identity link is an illegible
   signature.

> "Three outputs: a verdict — represent, accept the loss, or escalate to a human
> — a calibrated confidence, and a rebuttal draft that only ever restates
> evidence the merchant already has.
>
> Note the third case. Delivery is confirmed, but nothing links it to this
> cardholder. The system doesn't guess. It escalates. Roughly a quarter of the
> dataset is built to land there on purpose."

---

### 1:20 – 2:10 — Why the ground truth is real (screen: `SKILL.md`, then `scripts/labeler.py`)

> "Here's what makes the numbers mean something. Every label in this dataset
> comes from a rules matrix built from Visa's published requirements for reason
> code 13.1 — merchandise or services not received.
>
> Represent if there's proof of delivery *and* something linking that delivery
> to this cardholder, and nothing contradicts it. Accept if there's no proof of
> delivery at all, or the evidence undermines the merchant. Anything that
> doesn't resolve cleanly, abstain.
>
> That's ninety lines of Python, and it is the only thing that assigns a label.
> I never ask a model 'would this win?' — because then I'd be training a model
> to imitate another model's guess and calling the result accuracy."

---

### 2:10 – 3:00 — The RL (screen: `training/reward.py`, then the training curve)

> "The baseline is a three-persona panel — one advocate arguing the evidence
> gaps, one arguing the evidence present, and a referee that applies the matrix.
> Then I fine-tune a half-billion-parameter Qwen with GRPO to do the same job in
> a single forward pass.
>
> The reward has three terms. Correctness against the matrix. A Brier-style
> calibration term, so being confident and wrong hurts more than being uncertain
> and wrong. And abstention credit, so escalating a genuinely ambiguous case
> pays better than guessing it.
>
> Plus one thing the track brief asked for by name: false-positive cost. Wrongly
> representing and wrongly accepting are not equally bad, so they carry
> different penalties, scaled by how confidently the model made the mistake."

Show the training curve (`data/training_curve.png`).

---

### 3:00 – 4:00 — The numbers (screen: `python eval/compare_all.py`)

Run it live. Let the table render on camera.

> "Four columns, same hundred held-out cases, same parser.
>
> Accuracy. Abstention rate. Brier score — that's calibration, lower is better.
> And the two error directions costed separately, because that's the number that
> tells a merchant which way they're losing money."

Then — the part that earns trust:

> "The first column is a keyword matcher. No AI at all. And I want to be
> straight about what it means: it scores high, and it scores high because I
> wrote it after I wrote the data generator, so it's matching my own templates.
>
> Any finite template generator is invertible by whoever read the templates.
> That's a real limitation of synthetic data, it's in the README, and I kept the
> column in rather than delete the number that complicates my story."

---

### 4:00 – 4:40 — What broke (screen: the diff or `SUBMISSION.md`)

Pick **two**, don't rush all five:

> "Two things that broke. First, my prompt was handing the model
> `evidence_present` — the exact structured field the labeler uses to make the
> ground truth. The model wasn't reading anything, it was re-running a lookup
> it had been given the inputs for. Every accuracy number before that fix was
> measuring nothing.
>
> Second, GRPO wouldn't run on free Colab — three attempts died in TRL and
> torchao version conflicts. So I removed the framework and implemented GRPO
> directly: sample a group of completions, score them, take the advantage
> *within* the group — that group-relative baseline is the whole idea, it's what
> removes the need for a value network — and backprop. Sixty lines. The notebook
> now installs one package and runs start to finish on a free T4."

---

### 4:40 – 5:00 — Safety and close

> "One design rule throughout: low-confidence or evidence-thin cases route to
> accept the loss. Never to a fabricated or coached rebuttal. The rebuttal
> drafter is only reachable on a represent verdict, and it can only restate
> evidence the case already contains — there's a hard check that raises if a
> rebuttal ever appears on any other verdict.
>
> This helps a merchant truthfully assemble evidence they already have. It does
> not help them win a dispute they should lose. Repo's linked. Thanks."

---

## Recording checklist

- [ ] `python app.py` works and all three examples render before you hit record
- [ ] `python eval/compare_all.py` prints a full table (no "skipping" lines you
      didn't intend)
- [ ] `data/training_curve.png` exists, or you've cut the curve beat
- [ ] Under 5:00 — the limit is a filter, don't fail it
- [ ] Uploaded **unlisted, not private**, and the link opens in a logged-out
      browser window. A private link is a zero.

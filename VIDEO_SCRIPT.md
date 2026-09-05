# 5-Minute Pitch Video, Simple Spoken Script

One screen the whole time: **https://huggingface.co/spaces/AniketAsla/disputecourt**

Short sentences. Say them like you're explaining to a friend. Pause at **[pause]**.
About 4:15 total.

**Before recording:** open the Space, click any example once so the page is warm,
zoom to 125%, and keep "Results & method" closed at the bottom.

**Fastest way to demo:** scroll down to the **Examples** table and click the rows.
Each row fills the text box, ticks the right boxes, and sets the mode for you.
The manual setup is written out below in case a row doesn't fill.

---

## 0:00 to 0:30 · The story

*Nothing to click. Space homepage on screen.*

> "Imagine you run an online store.
>
> Someone buys headphones from you. Two weeks later they call their bank and say,
> I never got this. The bank pulls the money out of your account. That's a
> chargeback.
>
> Now you choose. Fight it? Or accept the loss and move on?
>
> **[pause]**
>
> Fight and lose, you pay a fee on top of losing the money. Accept when you could
> have won, you gave away money that was yours.
>
> Most merchants track one number, how many disputes they win. That hides which
> mistake is costing them. So I built something that tracks both."

---

## 0:30 to 1:15 · Three answers

### CASE 1 : click **Examples row 1**

| Setting | Value |
|---|---|
| **Paste in Case Narrative** | `Customer disputes a $210 physical order. Tracking confirms delivery to the billing address (AVS Y-match). Standard order value, no signature required.` |
| **Tick** | E1, E3 |
| **Evidence contradicted** | leave unticked |
| **Adjudication mode** | Rules matrix (deterministic, no AI) |
| **You will get** | REPRESENT, confidence 0.70 |

> "You give it a dispute. It gives one of three answers.
>
> First case. Courier confirmed delivery, and the address matches the card's
> billing address. So, fight it. And it drafts your reply using only the proof you
> actually have."

### CASE 2 : click **Examples row 2**

| Setting | Value |
|---|---|
| **Paste in Case Narrative** | `Customer disputes a $95 charge for merchandise. Merchant has no shipping record, no tracking number, and no delivery confirmation on file.` |
| **Tick** | nothing, leave all seven empty |
| **Evidence contradicted** | leave unticked |
| **Adjudication mode** | Rules matrix (deterministic, no AI) |
| **You will get** | ACCEPT, confidence 0.75, no rebuttal |

*Point your cursor at the output where no draft reply appears.*

> "Second case. No tracking, no delivery record, nothing. Answer, accept the loss.
>
> And look. There's no draft reply. That's deliberate. No proof, no argument. I
> blocked it in the code."

### CASE 3 : click **Examples row 3**

| Setting | Value |
|---|---|
| **Paste in Case Narrative** | `Customer disputes a $890 jewelry order. Delivery confirmed with signature capture, but the merchant has no AVS match, device match, or employment record on file, the signature name is illegible.` |
| **Tick** | E1, E4 |
| **Evidence contradicted** | leave unticked |
| **Adjudication mode** | Rules matrix (deterministic, no AI) |
| **You will get** | ABSTAIN, confidence 0.50 |

> "Third case, my favourite. The package was delivered and signed for. But the
> signature is unreadable, and nothing connects that delivery to this customer.
>
> So it says, I don't know. Send this to a human.
>
> **[pause]**
>
> That's the whole point. A system that always has an answer is guessing on the
> hard ones and hiding it."

---

## 1:15 to 1:50 · How it decides

*Move your cursor slowly over the seven checkboxes. Don't click anything.*

> "These seven boxes are every kind of proof a merchant can have. Delivery
> confirmation. Address match. Signature. Whether they've bought from you before.
>
> I tick what the merchant has, and a fixed rulebook decides. That rulebook comes
> straight from Visa's published rules. Did something get delivered? Is it linked
> to this customer? Both yes, fight it. Nothing delivered, accept it. Anything in
> between, ask a human.
>
> **[pause]**
>
> And this part matters. That rulebook is the only thing deciding the right answer.
> I never asked ChatGPT, would this dispute win? Then I'd just be checking whether
> my AI agrees with another AI. That's not a real score."

---

## 1:50 to 2:30 · The AI, and it fails

### CASE 4 : same case 2, but change the mode

| Setting | Value |
|---|---|
| **Paste in Case Narrative** | `Customer disputes a $95 charge for merchandise. Merchant has no shipping record, no tracking number, and no delivery confirmation on file.` |
| **Tick** | nothing (the AI never reads the boxes anyway) |
| **Evidence contradicted** | leave unticked |
| **Adjudication mode** | **RL policy, Qwen2.5-0.5B + GRPO (narrative only)** |
| **You will get** | REPRESENT, which is the wrong answer. That's the point. |

*Takes 20 to 40 seconds. Keep talking while it runs.*

> "Now the AI. I trained a small model with reinforcement learning. It only gets
> the story in plain English, it never sees those checkboxes. It has to work out
> the proof by reading.
>
> Let me run it on that second case. No tracking, no delivery, no proof at all.
>
> **[pause while it runs]**
>
> It says fight it.
>
> That's wrong. That's telling a merchant to spend money fighting a case with
> nothing behind it. I'm showing you on purpose. Now let me show you how badly it
> did, and why."

---

## 2:30 to 3:40 · The honest numbers

*Scroll down and click **"Results & method"** to open it.*

> "Same hundred cases for every column.
>
> Look at the model before training. Thirty-nine percent. Sounds like it half
> works. It doesn't. It said fight it to ninety-eight cases out of a hundred. It
> never once said accept the loss. It wasn't thinking. About forty percent of my
> cases happen to be fight cases, so it scored forty percent by saying the same
> thing every single time.
>
> **[pause]**
>
> Then I trained it. Honest result, it barely moved. Thirty-nine to forty percent.
> That's one case. And how often it says I don't know dropped to zero. Worse than
> before.
>
> Two things did improve. It stopped producing broken output, and it got less
> overconfident.
>
> So it's like a student who learned neat handwriting, and learned to say I think
> maybe instead of shouting a wrong answer. But never learned the subject.
>
> **[pause]**
>
> Why? The model I started with never once said accept the loss. Not a single
> time. And this kind of training can only reward things the model already does
> sometimes. It can't reward an answer it has never given. There was nothing to
> build on.
>
> The fix is to teach it the basics normally first, then train it this way. I ran
> out of time. I'd rather tell you that than show a number I can't defend."

---

## 3:40 to 4:15 · Close

*Stay on the results panel.*

> "One last thing on this table. That first column isn't AI at all, just keyword
> matching. Ninety-five percent. I'm keeping it because it's embarrassing and it's
> true. I wrote it after I wrote my data generator, so it already knows my
> patterns. Real merchant cases wouldn't look like that.
>
> **[pause]**
>
> And the rule I care about most. No proof means accept the loss. This tool will
> never write you a clever argument out of nothing.
>
> It helps an honest merchant use the proof they already have. It doesn't help
> anyone win a case they should lose.
>
> It's all on GitHub. Thank you."

---

## Quick reference, every click in order

| # | Mode to select | Tick these | Expected answer |
|---|---|---|---|
| 1 | Rules matrix | E1, E3 | REPRESENT 0.70 |
| 2 | Rules matrix | none | ACCEPT 0.75, no rebuttal |
| 3 | Rules matrix | E1, E4 | ABSTAIN 0.50 |
| 4 | **RL policy** | none | REPRESENT (wrong, on purpose) |

Then open "Results & method" and talk over the table.

---

## Checklist

- [ ] Under 5:00
- [ ] Upload **unlisted**, not private. Open it in a private window and check it
      plays.
- [ ] Paste the link into `SUBMISSION.md`

**Running long?** Cut the keyword matching paragraph at 3:40. Never cut case 3
(the I don't know answer) or case 4 (the AI failing). Those two are what make
people trust everything else.

# 5-Minute Pitch Video — Simple Spoken Script

One screen the whole time: **https://huggingface.co/spaces/AniketAsla/disputecourt**

Short sentences. Say them like you're explaining to a friend. Pause at **[pause]**.
About **4:15**.

**Before recording:** open the Space, click one example so it's warm, zoom 125%,
and keep "Results & method" closed at the bottom.

---

## 0:00 – 0:30 · The story

> "Imagine you run an online store.
>
> Someone buys headphones from you. Two weeks later they call their bank and say —
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
> Most merchants track one number — how many disputes they win. That hides which
> mistake is costing them. So I built something that tracks both."

---

## 0:30 – 1:15 · Three answers

*(Click Example 1 → Adjudicate)*

> "You give it a dispute. It gives one of three answers.
>
> First case. Courier confirmed delivery, and the address matches the card's
> billing address. So — fight it. And it drafts your reply using only the proof
> you actually have."

*(Click Example 2 → Adjudicate. Point at the output.)*

> "Second case. No tracking, no delivery record, nothing. Answer — accept the loss.
>
> And look. There's no draft reply. That's deliberate. No proof, no argument. I
> blocked it in the code."

*(Click Example 3 → Adjudicate)*

> "Third case, my favourite. The package was delivered and signed for. But the
> signature is unreadable, and nothing connects that delivery to this customer.
>
> So it says — I don't know. Send this to a human.
>
> **[pause]**
>
> That's the whole point. A system that always has an answer is guessing on the
> hard ones and hiding it."

---

## 1:15 – 1:50 · How it decides

*(Cursor over the E1–E7 checkboxes)*

> "These seven boxes are every kind of proof a merchant can have. Delivery
> confirmation. Address match. Signature. Whether they've bought from you before.
>
> I tick what the merchant has, and a fixed rulebook decides — taken straight from
> Visa's published rules. Did something get delivered? Is it linked to this
> customer? Both yes, fight it. Nothing delivered, accept it. Anything in between,
> ask a human.
>
> **[pause]**
>
> And this part matters. That rulebook is the only thing deciding the right answer.
> I never asked ChatGPT 'would this dispute win?' Then I'd just be checking whether
> my AI agrees with another AI. That's not a real score."

---

## 1:50 – 2:30 · The AI — and it fails

*(Switch mode to RL policy. Click Example 2 again. Adjudicate.)*

> "Now the AI. I trained a small model with reinforcement learning. It only gets
> the story in plain English — it never sees those checkboxes. It has to work out
> the proof by reading.
>
> Let me run it on that second case. No tracking, no delivery, no proof at all.
>
> **[pause while it runs]**
>
> It says fight it.
>
> That's wrong. That's telling a merchant to spend money fighting a case with
> nothing behind it. I'm showing you on purpose — now let me show you how badly it
> did, and why."

---

## 2:30 – 3:40 · The honest numbers

*(Open "Results & method")*

> "Same hundred cases for every column.
>
> Look at the model before training. Thirty-nine percent. Sounds like it half
> works. It doesn't. It said 'fight it' to ninety-eight cases out of a hundred. It
> never once said 'accept the loss.' It wasn't thinking — about forty percent of my
> cases happen to be fight cases, so it scored forty percent by saying the same
> thing every single time.
>
> **[pause]**
>
> Then I trained it. Honest result — it barely moved. Thirty-nine to forty percent.
> That's one case. And how often it says 'I don't know' dropped to zero. Worse than
> before.
>
> Two things did improve. It stopped producing broken output, and it got less
> overconfident.
>
> So it's like a student who learned neat handwriting, and learned to say 'I think
> maybe' instead of shouting a wrong answer. But never learned the subject.
>
> **[pause]**
>
> Why? The model I started with never once said 'accept the loss.' Not a single
> time. And this kind of training can only reward things the model already does
> sometimes. It can't reward an answer it has never given. There was nothing to
> build on.
>
> The fix is to teach it the basics normally first, then train it this way. I ran
> out of time. I'd rather tell you that than show a number I can't defend."

---

## 3:40 – 4:15 · Close

> "One last thing on this table. That first column isn't AI at all — just keyword
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

## Checklist

- [ ] Under 5:00
- [ ] Upload **unlisted** (not private) — open it in a private window and check
      it plays
- [ ] Paste the link into `SUBMISSION.md`

**Running long?** Cut the keyword-matching paragraph at 3:40. Never cut the
"I don't know" case at 0:30, or the model failing at 1:50 — those two are what
make people trust everything else.

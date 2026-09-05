"""
Self-contained GRPO for DisputeCourt. No TRL.

Why this exists instead of train_grpo.py's GRPOTrainer
------------------------------------------------------
TRL's GRPOTrainer API has moved fast, and on free Colab it collides with
whatever torchao/peft/transformers versions the runtime ships that week --
three commits in this repo's history are version-clash fixes, not modelling
work. GRPO's actual update is short enough to write directly, so this file
depends only on torch + transformers + peft, all of which Colab already has.

The algorithm, in full:
  1. For each prompt, sample G completions from the current policy.
  2. Score each with reward.py's compute_reward.
  3. Advantage = (r - mean(r)) / (std(r) + eps), computed WITHIN the group --
     that group-relative baseline is the whole idea of GRPO; it removes the
     need for a separate value network.
  4. Loss = -(advantage * token_logprob), averaged over completion tokens.
     Updates are strictly on-policy (we backprop the same samples we just
     drew), so the PPO importance ratio is identically 1 and is omitted.
  5. Optional KL-to-reference penalty. The reference is this same model with
     the LoRA adapter disabled, so it costs no extra memory.

Precision note: fp32 on GPU, deliberately. A 0.5B model trains fine in fp32
inside a T4's 16GB, and fp16 GRPO on a T4 tends to produce NaN losses once
advantages get small. Trading a little speed for not debugging NaNs at 2am
is the right call under a deadline.

Run:
    python training/grpo_minimal.py --steps 120 --group-size 6
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward import compute_reward  # noqa: E402
from train_grpo import build_prompt  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def completion_logprobs(model, input_ids, attention_mask, prompt_len):
    """Per-token logprobs for the completion portion only.

    logits[:, t] predicts token t+1, so we align by dropping the last logit
    and the first token id.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Mask: completion tokens only, and only up to (and including) the first EOS.
    mask = attention_mask[:, 1:].clone().float()
    mask[:, : prompt_len - 1] = 0.0
    return tok_logp, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data" / "train.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "training" / "checkpoints"))
    ap.add_argument("--steps", type=int, default=120, help="Number of prompts (one group each).")
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-beta", type=float, default=0.02, help="0 disables the KL penalty.")
    ap.add_argument("--accum", type=int, default=2, help="Groups per optimizer step.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.config.use_cache = True

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    rows = load_rows(Path(args.data))
    random.shuffle(rows)
    rows = rows[: args.steps]
    print(f"{len(rows)} prompts, group size {args.group_size}")

    history = []
    t0 = time.time()
    opt.zero_grad()

    for step, row in enumerate(rows, 1):
        prompt = build_prompt(row["narrative"])
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        enc = tok(chat, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]

        model.eval()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                num_return_sequences=args.group_size,
                pad_token_id=tok.pad_token_id,
            )

        texts = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        rewards = torch.tensor(
            [compute_reward(t, row["verdict"]) for t in texts], dtype=torch.float32, device=device
        )

        # Group-relative advantage. If every sample in the group scored the
        # same, the advantage is undefined and there is nothing to learn from
        # this prompt -- skip it rather than dividing by ~0 and injecting noise.
        if rewards.std() < 1e-6:
            history.append({"step": step, "reward_mean": rewards.mean().item(),
                            "reward_std": 0.0, "skipped": True})
            continue
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        attn = (gen != tok.pad_token_id).long()
        attn[:, :prompt_len] = enc["attention_mask"].repeat(args.group_size, 1)

        model.train()
        tok_logp, mask = completion_logprobs(model, gen, attn, prompt_len)
        pg = -(adv.unsqueeze(1) * tok_logp * mask).sum() / mask.sum().clamp(min=1)

        kl = torch.tensor(0.0, device=device)
        if args.kl_beta > 0:
            with torch.no_grad(), model.disable_adapter():
                ref_logp, _ = completion_logprobs(model, gen, attn, prompt_len)
            # k3 estimator: always non-negative, lower variance than (logp_ref - logp).
            diff = ref_logp - tok_logp
            kl = ((diff.exp() - diff - 1.0) * mask).sum() / mask.sum().clamp(min=1)

        loss = (pg + args.kl_beta * kl) / args.accum
        loss.backward()

        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad()

        history.append({
            "step": step, "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(), "kl": kl.item(),
            "loss": pg.item(), "skipped": False,
        })

        if step % 5 == 0:
            recent = [h["reward_mean"] for h in history[-20:]]
            el = time.time() - t0
            print(f"step {step}/{len(rows)}  reward(last20)={sum(recent)/len(recent):+.3f}  "
                  f"kl={kl.item():.4f}  {el/step:.1f}s/step  "
                  f"~{el/step*(len(rows)-step)/60:.1f} min left", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    scored = [h["reward_mean"] for h in history if not h["skipped"]]
    if scored:
        n = max(1, len(scored) // 4)
        print(f"\nmean reward, first quarter: {sum(scored[:n])/n:+.3f}")
        print(f"mean reward, last quarter:  {sum(scored[-n:])/n:+.3f}")
    print(f"skipped (zero-variance groups): {sum(1 for h in history if h['skipped'])}/{len(history)}")
    print(f"\nadapter saved to {out}")


if __name__ == "__main__":
    main()

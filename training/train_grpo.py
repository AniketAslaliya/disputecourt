"""
GRPO fine-tuning for DisputeCourt's single-model policy (Day 3 of PLAN.md).

This is the RL-tuned counterpart to panel/personas.py's prompted baseline --
same task, but a small model trained to directly output a calibrated
verdict instead of a 3-agent debate at inference time. Compare the two with
eval/metrics.py's compare_reports() once both have run.

NOT RUNNABLE IN A SANDBOX WITHOUT A GPU. This is a correct scaffold to hand
to Cursor with real compute (local GPU or Colab) -- verify reward.py's
sanity check passes (it does, see test_reward_sanity.py) before spending
compute time here.

Setup:
    pip install trl peft transformers bitsandbytes accelerate datasets --break-system-packages

Usage:
    python3 train_grpo.py --data ../data/generated_cases.jsonl --output ./checkpoints

Model choice: Qwen2.5-0.5B-Instruct by default -- matches DebateFloor's
original choice and is realistic to train on a single consumer GPU or a
free Colab T4 within the remaining timeline. Swap --model if you have
access to more compute and want a stronger base.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import trl_reward_fn  # noqa: E402

SKILL_MD_RULES_COMPACT = """Apply this rules matrix to decide the verdict:
REPRESENT if: (E1 or E2 present) AND (E3 or E5 or E6 present) AND not contradicted
ACCEPT if: (E1 and E2 both absent) OR contradicted
ABSTAIN if: neither resolves cleanly (e.g. delivery present but identity link is weak/partial)

Evidence codes: E1=delivery confirmed, E2=digital delivery/access proof,
E3=AVS address match, E4=signature confirmation, E5=device/card continuity,
E6=employment-at-business-address proof, E7=support communication log."""

PROMPT_TEMPLATE = """You are adjudicating a Visa 13.1 (Merchandise/Services Not \
Received) chargeback dispute.

{rules}

Read the merchant's case file below and decide which evidence items the merchant
actually HOLDS. A case file may mention an evidence type only to say it is
missing, unverified, or inconclusive -- that does not count as holding it.

Case file: {narrative}

Output ONLY a JSON object: {{"verdict": "represent"|"accept"|"abstain", \
"confidence": <float 0-1>, "reasoning": "<short, references specific E-items>"}}"""


def build_prompt(narrative: str, evidence_present: list | None = None) -> str:
    """Narrative-only by design.

    An earlier version passed `evidence_present` into the prompt alongside the
    rules matrix. That made the task circular: the model was handed the exact
    structured input the deterministic labeler uses, so it was re-executing a
    lookup rather than reading a case file, and any accuracy it scored measured
    nothing. `evidence_present` is accepted and ignored so existing callers do
    not break; the ground-truth label still comes from it via labeler.py, but
    the model never sees it.
    """
    return PROMPT_TEMPLATE.format(rules=SKILL_MD_RULES_COMPACT, narrative=narrative)


def load_dataset(path: str):
    """
    Returns a HF Datasets object with columns: prompt, true_verdict.
    Kept as a plain function (not baked into main()) so Cursor/you can unit
    test it against a small fixture file without loading a model.
    """
    from datasets import Dataset

    rows = [json.loads(line) for line in Path(path).open()]
    records = [
        {
            "prompt": build_prompt(r["narrative"], r["evidence_present"]),
            "true_verdict": r["verdict"],
        }
        for r in rows
    ]
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4,
                         help="Completions sampled per prompt for GRPO's group-relative advantage.")
    args = parser.parse_args()

    # Deliberately deferred imports -- keeps --help and load_dataset()
    # usable without heavy ML deps installed, and gives a clear error
    # message instead of an import traceback if deps are missing.
    try:
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise SystemExit(
            "Missing training deps. Run:\n"
            "  pip install trl peft transformers bitsandbytes accelerate datasets --break-system-packages"
        ) from e

    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset)} training examples from {args.data}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        num_generations=args.group_size,
        per_device_train_batch_size=args.group_size,  # one prompt's group per step to start
        learning_rate=1e-5,
        logging_steps=5,
        save_steps=50,
        fp16=True,   # T4 (Colab free GPU) does not support bf16
        bf16=False,
        # Watch this run closely for the collapse mode flagged in
        # CLAUDE.md/PLAN.md: if reward variance within a group flatlines
        # early, the policy has likely collapsed onto one verdict regardless
        # of input. test_reward_sanity.py rules out a *reward-shape* cause
        # of collapse, but a training-dynamics collapse can still happen --
        # that's a legitimate "what broke" story either way.
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        reward_funcs=trl_reward_fn,
    )

    trainer.train()
    trainer.save_model(args.output)
    print(f"Training complete. Checkpoint saved to {args.output}")
    print("\nNext: run this model's outputs through eval/metrics.py and compare")
    print("against panel/run_panel_live.py's baseline results with compare_reports().")


if __name__ == "__main__":
    main()

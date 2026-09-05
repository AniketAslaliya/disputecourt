"""
Runs a HuggingFace causal LM over the eval split and reports the four metrics
PLAN.md commits to. Used for both columns of the comparison table:

  base model (no RL):
      python eval/run_model_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \\
          --out data/results_base.jsonl

  GRPO-tuned (LoRA adapter on top of the same base):
      python eval/run_model_eval.py --model Qwen/Qwen2.5-0.5B-Instruct \\
          --adapter training/checkpoints --out data/results_grpo.jsonl

Same prompt, same eval split, same parser both times -- the only thing that
differs between the two runs is the adapter, which is what makes the delta
attributable to GRPO rather than to prompt or harness changes.

Runs on CPU (slow but workable for 0.5B) or CUDA if available.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_grpo import build_prompt  # noqa: E402  (heavy deps are deferred inside main())
from reward import parse_completion  # noqa: E402
from metrics import Prediction, full_report, print_report  # noqa: E402


def load_model(model_id: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).to(device)
        print(f"loaded LoRA adapter from {adapter}")
    model.eval()
    return tok, model, device


def generate(tok, model, device, prompt: str, max_new_tokens: int = 160) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="Path to a LoRA adapter directory.")
    ap.add_argument("--data", default=str(ROOT / "data" / "eval.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "results_model.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all cases.")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tok, model, device = load_model(args.model, args.adapter)
    label_name = args.label or ("GRPO-tuned" if args.adapter else "Base model (no RL)")
    print(f"{label_name}: {len(rows)} cases on {device}")

    preds, unparseable = [], 0
    t0 = time.time()
    with Path(args.out).open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            raw = generate(tok, model, device, build_prompt(r["narrative"]))
            parsed = parse_completion(raw)
            if parsed is None:
                unparseable += 1
                # An unparseable answer is not silently dropped and not counted
                # as correct -- it is recorded as a maximally-unhelpful abstain
                # so the accuracy number stays honest about format failures.
                pred_v, pred_c = "abstain", 0.5
            else:
                pred_v, pred_c = parsed["verdict"], parsed["confidence"]
            preds.append(Prediction(r["case_id"], r["verdict"], pred_v, pred_c))
            f.write(json.dumps({
                "case_id": r["case_id"], "true_verdict": r["verdict"],
                "pred_verdict": pred_v, "pred_confidence": pred_c,
                "parsed_ok": parsed is not None, "raw": raw[:400],
            }) + "\n")
            if i % 10 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(rows)}  ({rate:.1f}s/case, ~{rate*(len(rows)-i)/60:.1f} min left)",
                      flush=True)

    report = full_report(preds, label=label_name)
    print_report(report)
    print(f"  unparseable JSON: {unparseable}/{len(rows)}")
    print(f"\nwrote {args.out}")

    summary = Path(args.out).with_suffix(".summary.json")
    report["unparseable"] = unparseable
    # confusion_matrix is keyed by (true, pred) tuples, which JSON cannot encode.
    report["confusion_matrix"] = {
        f"{t}->{p}": c for (t, p), c in report["confusion_matrix"].items()
    }
    summary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()

"""
DisputeCourt — Gradio demo for Razorpay AI Buildathon (Track 02).

Runs the Court Panel (3-persona debate) on a single case and returns
the verdict, confidence, reasoning, and rebuttal draft.
"""

import json
import os
import sys
from pathlib import Path

import gradio as gr

try:
    import spaces

    _gpu_decorator = spaces.GPU
except ImportError:
    # Local dev — no HF ZeroGPU runtime.
    def _gpu_decorator(fn):
        return fn

sys.path.insert(0, str(Path(__file__).resolve().parent / "panel"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))

from personas import run_panel  # noqa: E402
from labeler import Case, label, confidence_for, Verdict  # noqa: E402
from train_grpo import build_prompt  # noqa: E402
from reward import parse_completion  # noqa: E402

MODE_RULES = "Rules matrix (deterministic, no AI)"
MODE_POLICY = "RL policy — Qwen2.5-0.5B + GRPO (narrative only)"
MODE_PANEL = "Court Panel — 3 personas (needs GEMINI_API_KEY)"

EVIDENCE_LABELS = {
    "E1": "E1 — Delivery/tracking confirmation",
    "E2": "E2 — Digital delivery / access logs",
    "E3": "E3 — Address matches billing (AVS Y/M)",
    "E4": "E4 — Signature confirmation",
    "E5": "E5 — Same device/card in prior undisputed txn",
    "E6": "E6 — Employment proof at delivery address",
    "E7": "E7 — Customer communication log",
}

EVIDENCE_IDS = list(EVIDENCE_LABELS.keys())


def get_llm_fn():
    """Returns a Gemini call function if keys are available, else None."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS"):
        from llm_client import call_gemini
        return call_gemini
    return None


_POLICY = {}


def run_rl_policy(narrative):
    """The actual product: narrative in, verdict out, single forward pass.

    Loads the GRPO adapter if one has been trained, otherwise the base model,
    and says which it used -- an untuned base model labelled 'RL policy' in a
    demo would be a lie by omission.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not _POLICY:
        base = "Qwen/Qwen2.5-0.5B-Instruct"
        adapter = Path(__file__).resolve().parent / "training" / "checkpoints"
        tok = AutoTokenizer.from_pretrained(base)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        tuned = False
        if (adapter / "adapter_config.json").exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter)).to(device)
            tuned = True
        model.eval()
        _POLICY.update(tok=tok, model=model, device=device, tuned=tuned)

    prompt = build_prompt(narrative)
    text = _POLICY["tok"].apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = _POLICY["tok"](text, return_tensors="pt").to(_POLICY["device"])
    with torch.no_grad():
        out = _POLICY["model"].generate(
            **inputs, max_new_tokens=160, do_sample=False,
            pad_token_id=_POLICY["tok"].eos_token_id,
        )
    raw = _POLICY["tok"].decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    parsed = parse_completion(raw)
    mode = ("GRPO-tuned policy" if _POLICY["tuned"]
            else "Base Qwen2.5-0.5B (no adapter found — NOT RL-tuned)")
    if parsed is None:
        return {
            "verdict": "abstain", "confidence": 0.5,
            "reasoning": "Model did not return parseable JSON; abstaining rather than guessing.",
            "rebuttal_draft": "", "mode": mode + " — unparseable output",
        }
    full = {}
    try:
        import json as _json
        import re as _re

        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        full = _json.loads(m.group(0)) if m else {}
    except Exception:
        full = {}
    rebuttal = full.get("rebuttal_draft", "") if parsed["verdict"] == "represent" else ""
    return {
        "verdict": parsed["verdict"],
        "confidence": round(parsed["confidence"], 2),
        "reasoning": full.get("reasoning", raw[:300]),
        "rebuttal_draft": rebuttal,
        "mode": mode + " (narrative-only, 1 forward pass)",
    }


def run_rules_only(narrative, evidence_ids, contradicted):
    """Deterministic rules-matrix verdict (no LLM needed)."""
    case = Case(
        case_id="demo",
        narrative=narrative,
        evidence_present=set(evidence_ids),
        contradicted=contradicted,
    )
    verdict, reasoning = label(case)
    conf = confidence_for(case, verdict)
    return {
        "verdict": verdict.value,
        "confidence": round(conf, 2),
        "reasoning": reasoning,
        "rebuttal_draft": "",
        "mode": "rules-matrix (deterministic)",
    }


def run_full_panel(narrative, evidence_ids, contradicted):
    """Full 3-persona Court Panel via Gemini."""
    llm_fn = get_llm_fn()
    if llm_fn is None:
        return None

    result = run_panel(
        case_id="demo",
        narrative=narrative,
        evidence_present=list(evidence_ids),
        llm_call_fn=llm_fn,
        contradicted=contradicted,
    )
    return {
        "verdict": result.verdict,
        "confidence": round(result.confidence, 2),
        "reasoning": result.reasoning,
        "rebuttal_draft": result.rebuttal_draft,
        "cardholder_argument": result.cardholder_argument,
        "merchant_argument": result.merchant_argument,
        "mode": "Court Panel (3-persona, Gemini)",
    }


@_gpu_decorator
def predict(narrative, evidence_checkboxes, contradicted, mode):
    if not narrative.strip():
        return "Please enter a case narrative."

    evidence_ids = [eid for eid in EVIDENCE_IDS if EVIDENCE_LABELS[eid] in evidence_checkboxes]

    if mode == MODE_PANEL:
        result = run_full_panel(narrative, evidence_ids, contradicted)
        if result is None:
            result = run_rules_only(narrative, evidence_ids, contradicted)
            result["mode"] += " (GEMINI_API_KEY not set — fell back to rules-only)"
    elif mode == MODE_POLICY:
        # Narrative only. The evidence checkboxes are deliberately NOT passed:
        # handing the model the structured evidence set would be handing it the
        # labeler's input, which is the circularity this project had to fix.
        result = run_rl_policy(narrative)
    else:
        result = run_rules_only(narrative, evidence_ids, contradicted)

    lines = [
        f"**Mode:** {result['mode']}",
        f"**Verdict:** {result['verdict'].upper()}",
        f"**Confidence:** {result['confidence']}",
        f"**Reasoning:** {result['reasoning']}",
    ]
    if result.get("rebuttal_draft"):
        lines.append(f"**Rebuttal Draft:** {result['rebuttal_draft']}")
    if result.get("cardholder_argument"):
        lines.append(f"\n---\n**Cardholder Advocate:** {result['cardholder_argument']}")
    if result.get("merchant_argument"):
        lines.append(f"**Merchant Advocate:** {result['merchant_argument']}")

    return "\n\n".join(lines)


EXAMPLES = [
    [
        "Customer disputes a $210 physical order. Tracking confirms delivery to the billing address (AVS Y-match). Standard order value, no signature required.",
        [EVIDENCE_LABELS["E1"], EVIDENCE_LABELS["E3"]],
        False,
        MODE_RULES,
    ],
    [
        "Customer disputes a $95 charge for merchandise. Merchant has no shipping record, no tracking number, and no delivery confirmation on file.",
        [],
        False,
        MODE_RULES,
    ],
    [
        "Customer disputes a $890 jewelry order. Delivery confirmed with signature capture, but the merchant has no AVS match, device match, or employment record on file — the signature name is illegible.",
        [EVIDENCE_LABELS["E1"], EVIDENCE_LABELS["E4"]],
        False,
        MODE_RULES,
    ],
    [
        "Customer disputes a $150 order. Merchant's tracking shows the package was delivered, but to a different city than the cardholder's billing address.",
        [EVIDENCE_LABELS["E1"]],
        True,
        MODE_RULES,
    ],
]

with gr.Blocks(title="DisputeCourt — Visa 13.1 Chargeback Adjudicator") as demo:
    gr.Markdown(
        """
        # DisputeCourt
        ### Visa 13.1 Chargeback Evidence Adjudicator
        *Razorpay AI Buildathon — Track 02: AI Risk Manager*

        Enter a dispute case below. The system applies the **rules matrix** from
        SKILL.md to determine whether the merchant should **represent** (fight),
        **accept** the loss, or **abstain** (escalate to a human).

        Low-confidence or evidence-thin cases route to accept — never to a
        fabricated rebuttal.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            narrative = gr.Textbox(
                label="Case Narrative",
                placeholder="Describe the dispute scenario...",
                lines=4,
            )
            evidence = gr.CheckboxGroup(
                choices=list(EVIDENCE_LABELS.values()),
                label="Evidence Present",
            )
            contradicted = gr.Checkbox(label="Evidence contradicted (e.g. wrong address)")
            mode = gr.Radio(
                choices=[MODE_RULES, MODE_POLICY, MODE_PANEL],
                value=MODE_RULES,
                label="Adjudication mode",
                info="The RL policy reads the narrative only — it never sees the "
                     "checkboxes above. First run downloads the model (~1GB) and "
                     "takes ~20s on CPU.",
            )
            submit = gr.Button("Adjudicate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Result")

    submit.click(predict, inputs=[narrative, evidence, contradicted, mode], outputs=output)
    gr.Examples(examples=EXAMPLES, inputs=[narrative, evidence, contradicted, mode])

demo.queue()

if __name__ == "__main__":
    demo.launch()

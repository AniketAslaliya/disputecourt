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

sys.path.insert(0, str(Path(__file__).resolve().parent / "panel"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from personas import run_panel  # noqa: E402
from labeler import Case, label, confidence_for, Verdict  # noqa: E402

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


def predict(narrative, evidence_checkboxes, contradicted, use_panel):
    if not narrative.strip():
        return "Please enter a case narrative."

    evidence_ids = [eid for eid in EVIDENCE_IDS if EVIDENCE_LABELS[eid] in evidence_checkboxes]

    if use_panel:
        result = run_full_panel(narrative, evidence_ids, contradicted)
        if result is None:
            result = run_rules_only(narrative, evidence_ids, contradicted)
            result["mode"] += " (GEMINI_API_KEY not set — fell back to rules-only)"
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
        False,
    ],
    [
        "Customer disputes a $95 charge for merchandise. Merchant has no shipping record, no tracking number, and no delivery confirmation on file.",
        [],
        False,
        False,
    ],
    [
        "Customer disputes a $890 jewelry order. Delivery confirmed with signature capture, but the merchant has no AVS match, device match, or employment record on file — the signature name is illegible.",
        [EVIDENCE_LABELS["E1"], EVIDENCE_LABELS["E4"]],
        False,
        False,
    ],
    [
        "Customer disputes a $150 order. Merchant's tracking shows the package was delivered, but to a different city than the cardholder's billing address.",
        [EVIDENCE_LABELS["E1"]],
        True,
        False,
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
            with gr.Row():
                contradicted = gr.Checkbox(label="Evidence contradicted (e.g. wrong address)")
                use_panel = gr.Checkbox(
                    label="Use Court Panel (needs GEMINI_API_KEY)",
                    value=False,
                )
            submit = gr.Button("Adjudicate", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Result")

    submit.click(predict, inputs=[narrative, evidence, contradicted, use_panel], outputs=output)
    gr.Examples(examples=EXAMPLES, inputs=[narrative, evidence, contradicted, use_panel])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
else:
    # HF Spaces imports app.py and launches `demo` itself.
    demo.queue()

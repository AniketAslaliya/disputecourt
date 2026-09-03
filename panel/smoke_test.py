"""
Validates panel/personas.py's plumbing with a mocked llm_call_fn -- no real
API key needed. Three scenarios: a normal represent case, a normal accept
case, and a deliberately broken referee response that should trip the
rebuttal-on-non-represent safety check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from personas import run_panel  # noqa: E402


def make_mock_llm(referee_response: str):
    call_log = []

    def mock_llm(prompt: str) -> str:
        call_log.append(prompt[:40])
        if "Network-Rules Referee" in prompt:
            return referee_response
        if "Cardholder-side advocate" in prompt:
            return "Weakness: no clean identity link to the cardholder (E3/E5/E6 all absent)."
        return "Strength: delivery is confirmed via tracking (E1 present)."

    return mock_llm, call_log


def test_represent_case():
    referee_output = """{"verdict": "represent", "confidence": 0.82, "evidence_present": ["E1", "E3"], "rebuttal_draft": "Delivery confirmed to billing address via tracking, AVS Y-match on file.", "reasoning": "E1+E3 satisfies the matrix, no contradiction."}"""
    mock_llm, log = make_mock_llm(referee_output)
    result = run_panel("test-1", "Customer disputes a $300 order, tracking confirms delivery to billing address.", ["E1", "E3"], mock_llm)
    assert result.verdict == "represent"
    assert result.rebuttal_draft != ""
    assert len(log) == 3, "Should call the LLM exactly 3 times: cardholder, merchant, referee"
    print("test_represent_case: PASS")


def test_accept_case_no_rebuttal():
    referee_output = """{"verdict": "accept", "confidence": 0.9, "evidence_present": [], "rebuttal_draft": "", "reasoning": "No delivery evidence on file."}"""
    mock_llm, _ = make_mock_llm(referee_output)
    result = run_panel("test-2", "Customer disputes a $95 charge, merchant has no delivery record.", [], mock_llm)
    assert result.verdict == "accept"
    assert result.rebuttal_draft == ""
    print("test_accept_case_no_rebuttal: PASS")


def test_safety_check_catches_bad_rebuttal():
    """A referee that outputs 'accept' but still writes a rebuttal should be caught, not shipped."""
    broken_referee_output = """{"verdict": "accept", "confidence": 0.7, "evidence_present": [], "rebuttal_draft": "We should still fight this because...", "reasoning": "test"}"""
    mock_llm, _ = make_mock_llm(broken_referee_output)
    try:
        run_panel("test-3", "Some case.", [], mock_llm)
        print("test_safety_check_catches_bad_rebuttal: FAIL -- should have raised ValueError")
        raise SystemExit(1)
    except ValueError as e:
        assert "disqualification risk" in str(e)
        print("test_safety_check_catches_bad_rebuttal: PASS (correctly rejected)")


if __name__ == "__main__":
    test_represent_case()
    test_accept_case_no_rebuttal()
    test_safety_check_catches_bad_rebuttal()
    print("\nAll panel plumbing tests pass. Swap the mock llm_call_fn for a real")
    print("API call (e.g. anthropic.Anthropic().messages.create) to go live.")

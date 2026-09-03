"""
Shared Gemini call wrapper for DisputeCourt, built for the free tier.

Free-tier limits are tight and change model-to-model (roughly 10-15
requests/minute depending on which Flash variant you're on) -- check
ai.google.dev for current numbers before a long run. This wrapper paces
calls to stay under a conservative default and retries with backoff on
rate-limit errors instead of crashing your generation run halfway through.

Usage:
    export GEMINI_API_KEY=...
    pip install google-genai --break-system-packages
"""

import os
import time
import threading

# Flash models are what the free tier actually gives you -- Pro models are
# capped very low (as little as 50 requests/day) or behind billing entirely.
# Check ai.google.dev/gemini-api/docs/models for the current lineup; this is
# a reasonable default as of the time this was written, not a guarantee.
MODEL = "gemini-2.5-flash"

# Conservative pacing: 7 requests/minute (~8.5s between calls) stays under
# every published free-tier RPM ceiling for Flash models with margin.
# Tighten this only after confirming your actual quota in AI Studio.
MIN_SECONDS_BETWEEN_CALLS = 8.5

_client = None
_last_call_lock = threading.Lock()
_last_call_time = [0.0]


def get_client():
    global _client
    if _client is None:
        from google import genai  # local import so this module is importable without the dep installed

        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("Set GEMINI_API_KEY before making calls.")
        _client = genai.Client()  # reads GEMINI_API_KEY from env
    return _client


def _throttle():
    """Blocks just long enough to keep calls under MIN_SECONDS_BETWEEN_CALLS apart."""
    with _last_call_lock:
        elapsed = time.time() - _last_call_time[0]
        wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()


def call_gemini(prompt: str, max_retries: int = 5, base_delay: float = 5.0) -> str:
    """
    Rate-limited, retrying call to Gemini. Returns the response text.
    Raises RuntimeError if it exhausts retries -- treat that as "stop and
    check your quota in AI Studio", not a bug to silently swallow.
    """
    client = get_client()

    for attempt in range(max_retries):
        _throttle()
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return resp.text
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  [rate limited, attempt {attempt + 1}/{max_retries}] backing off {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini call failed after {attempt + 1} attempt(s): {msg}") from e

    raise RuntimeError("Gemini call failed after max retries.")

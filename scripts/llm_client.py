"""
Shared Gemini call wrapper for DisputeCourt, built for the free tier.

Free-tier limits are tight and change model-to-model. This wrapper paces
calls and, on 429/quota, rotates to the next API key instead of dying on
the first bucket.

Usage:
    $env:GEMINI_API_KEY = "..."
    $env:GEMINI_API_KEY_2 = "..."          # optional second (or third) key
    # or: $env:GEMINI_API_KEYS = "key1,key2"

    pip install google-genai
"""

import os
import time
import threading

# gemini-2.5-flash free tier is 20 RPD. Flash-Lite variants have a higher
# daily cap and a separate bucket per API project/key.
MODEL = "gemini-3.1-flash-lite"

# Conservative pacing: 7 requests/minute (~8.5s between calls).
MIN_SECONDS_BETWEEN_CALLS = 8.5

_last_call_lock = threading.Lock()
_last_call_time = [0.0]
_key_lock = threading.Lock()
_clients = {}  # api_key -> genai.Client
_key_index = [0]
_exhausted = set()  # keys that hit a daily-quota 429


def _collect_keys() -> list[str]:
    keys = []
    bundled = os.environ.get("GEMINI_API_KEYS", "")
    for part in bundled.split(","):
        part = part.strip()
        if part:
            keys.append(part)
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        val = os.environ.get(name, "").strip()
        if val:
            keys.append(val)
    # preserve order, drop duplicates
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def _client_for(api_key: str):
    if api_key not in _clients:
        from google import genai

        _clients[api_key] = genai.Client(api_key=api_key)
    return _clients[api_key]


def _active_keys() -> list[str]:
    keys = [k for k in _collect_keys() if k not in _exhausted]
    if not keys:
        # all marked exhausted — try them again in case the quota window moved
        _exhausted.clear()
        keys = _collect_keys()
    return keys


def get_client():
    keys = _active_keys()
    if not keys:
        raise SystemExit(
            "Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_2) before making calls."
        )
    with _key_lock:
        idx = _key_index[0] % len(keys)
        return _client_for(keys[idx]), keys[idx]


def _rotate_key(current: str) -> str | None:
    """Mark current key exhausted if needed and return the next usable key."""
    keys = _collect_keys()
    with _key_lock:
        remaining = [k for k in keys if k not in _exhausted and k != current]
        if remaining:
            _key_index[0] = keys.index(remaining[0])
            print(f"  [key rotate] switching API key ({len(remaining)} remaining)")
            return remaining[0]
        # last key: keep using it so backoff can still retry
        return current if current not in _exhausted else (keys[0] if keys else None)


def _throttle():
    """Blocks just long enough to keep calls under MIN_SECONDS_BETWEEN_CALLS apart."""
    with _last_call_lock:
        elapsed = time.time() - _last_call_time[0]
        wait = MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()


def _is_daily_quota(msg: str) -> bool:
    return "PerDay" in msg or "per day" in msg.lower() or "quotaValue" in msg


def call_gemini(prompt: str, max_retries: int = 8, base_delay: float = 5.0) -> str:
    """
    Rate-limited, retrying call to Gemini. Rotates keys on daily-quota 429s.
    Raises RuntimeError if it exhausts retries.
    """
    keys = _active_keys()
    if not keys:
        raise SystemExit(
            "Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_2) before making calls."
        )

    last_error = None
    for attempt in range(max_retries):
        _throttle()
        client, key = get_client()
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return resp.text
        except Exception as e:
            msg = str(e)
            last_error = msg
            is_retryable = (
                "429" in msg
                or "RESOURCE_EXHAUSTED" in msg
                or "quota" in msg.lower()
                or "503" in msg
                or "UNAVAILABLE" in msg
                or "high demand" in msg.lower()
            )
            if not is_retryable or attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Gemini call failed after {attempt + 1} attempt(s): {msg}"
                ) from e

            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if _is_daily_quota(msg):
                    _exhausted.add(key)
                    nxt = _rotate_key(key)
                    if nxt and nxt != key:
                        continue  # try the other key immediately
                delay = base_delay * (2 ** attempt)
                print(f"  [retryable, attempt {attempt + 1}/{max_retries}] backing off {delay:.0f}s...")
                time.sleep(delay)
                continue

            delay = base_delay * (2 ** attempt)
            print(f"  [retryable, attempt {attempt + 1}/{max_retries}] backing off {delay:.0f}s...")
            time.sleep(delay)

    raise RuntimeError(f"Gemini call failed after max retries: {last_error}")

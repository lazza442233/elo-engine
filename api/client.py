"""
API client for fetching match data from the Dribl API.
"""

import gzip
import json
import time
import urllib.error
import urllib.request

from config.constants import fixtures_url

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds; retry waits: 2, 4, 8 …


def fetch_dribl_data(url: str) -> list[dict]:
    """
    Fetch match data from the Dribl API.

    Uses urllib (avoids requests' Cloudflare fingerprinting issues).
    Spoofs a standard Chrome User-Agent as required by the endpoint.
    Retries up to 3 times with exponential backoff on transient failures.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, */*",
        "Accept-Language": "en-AU,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.dribl.com/",
        "Origin": "https://www.dribl.com",
    }

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "") == "gzip":
                    raw = gzip.decompress(raw)

            payload = json.loads(raw)

            # Dribl wraps everything in a top-level "data" key
            if isinstance(payload, dict) and "data" in payload:
                return payload["data"]
            if isinstance(payload, list):
                return payload

            raise ValueError(
                f"Unexpected API response shape. Top-level keys: {list(payload.keys())}"
            )

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
            continue

    raise ConnectionError(
        f"Dribl API unreachable after {_MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def detect_next_round(max_round: int = 30, league_key: str | None = None) -> tuple[int, list[dict]]:
    """
    Auto-detect the next upcoming round by iterating roundrobin_{N}
    until a non-empty fixture list is returned.

    Returns (round_number, fixtures_data).
    Raises RuntimeError if no fixtures found up to max_round.
    """
    kwargs = {"league_key": league_key} if league_key else {}
    for n in range(1, max_round + 1):
        try:
            data = fetch_dribl_data(fixtures_url(n, **kwargs))
            if data:
                return n, data
        except Exception:
            continue
    raise RuntimeError(f"No upcoming fixtures found (checked rounds 1–{max_round})")

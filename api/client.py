"""
API client for fetching match data from the Dribl API.
"""

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from config.constants import fixtures_url

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds; retry waits: 2, 4, 8 …
_MAX_PAGES = 10  # safety cap to avoid infinite pagination loops


def _fetch_single_page(url: str, headers: dict) -> dict:
    """Fetch a single page from the Dribl API with retries.

    Returns the full JSON payload (including meta/links for pagination).
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "") == "gzip":
                    raw = gzip.decompress(raw)

            return json.loads(raw)

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
            continue

    raise ConnectionError(
        f"Dribl API unreachable after {_MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def fetch_dribl_data(url: str) -> list[dict]:
    """
    Fetch match data from the Dribl API, following cursor pagination.

    Uses urllib (avoids requests' Cloudflare fingerprinting issues).
    Spoofs a standard Chrome User-Agent as required by the endpoint.
    Retries up to 3 times with exponential backoff on transient failures.
    Automatically follows cursor-based pagination (API caps at 30/page).
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

    all_data: list[dict] = []
    current_url = url

    for _ in range(_MAX_PAGES):
        payload = _fetch_single_page(current_url, headers)

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict) and "data" in payload:
            all_data.extend(payload["data"])

            # Follow cursor pagination if present
            next_cursor = (
                payload.get("meta", {}).get("next_cursor")
            )
            if not next_cursor:
                break

            # Append cursor to the original URL's query params
            sep = "&" if "?" in url else "?"
            current_url = f"{url}{sep}cursor={urllib.parse.quote(next_cursor, safe='')}"
            continue

        raise ValueError(
            f"Unexpected API response shape. Top-level keys: {list(payload.keys())}"
        )

    return all_data


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

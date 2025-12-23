from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


def request_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 10,
    max_retries: int = 5,
) -> Any:
    headers = {"User-Agent": "100k_strategy/0.1"}
    backoff = 1.0

    for attempt in range(max_retries):
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in {418, 429, 500, 502, 503, 504}:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = backoff
            else:
                sleep_s = backoff
            time.sleep(sleep_s)
            backoff *= 2.0
            continue

        resp.raise_for_status()

    resp.raise_for_status()
    return None

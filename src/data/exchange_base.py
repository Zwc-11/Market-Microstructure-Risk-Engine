from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from src.utils.http import request_json


class ExchangeBase:
    def __init__(
        self,
        base_url: str,
        name: str,
        rate_limit_per_sec: Optional[float] = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.rate_limit_per_sec = rate_limit_per_sec
        self.timeout = timeout
        self._last_request_ts = 0.0
        self._session = requests.Session()

    def _throttle(self) -> None:
        if not self.rate_limit_per_sec:
            return
        min_interval = 1.0 / float(self.rate_limit_per_sec)
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_ts = time.time()

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._throttle()
        url = f"{self.base_url}{path}"
        return request_json(self._session, url, params=params, timeout=self.timeout)

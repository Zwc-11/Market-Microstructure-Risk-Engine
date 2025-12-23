from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.data.exchange_base import ExchangeBase
from src.data.schemas import SCHEMAS, enforce_schema


class DeepcoinRest(ExchangeBase):
    def __init__(self, rate_limit_per_sec: float = 1.0) -> None:
        super().__init__(base_url="https://api.deepcoin.com", name="deepcoin", rate_limit_per_sec=rate_limit_per_sec)

    def fetch_klines_1m(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows: List[Dict] = []
        params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "endTime": end_ms}
        data = self._get_json("/public/market/kline", params=params)
        if not data:
            return pd.DataFrame()

        for k in data:
            rows.append(
                {
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_ts": int(k[0]) + 60_000,
                    "quote_volume": 0.0,
                    "trade_count": 0,
                    "taker_buy_base": 0.0,
                    "taker_buy_quote": 0.0,
                    "symbol": symbol,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("ts").drop_duplicates(subset=["symbol", "ts"], keep="last")
        return enforce_schema(df, SCHEMAS["klines_1m"])

    def fetch_trades(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows: List[Dict] = []
        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms}
        data = self._get_json("/public/market/trades", params=params)
        if not data:
            return pd.DataFrame()

        for t in data:
            side = t.get("side", "").lower()
            rows.append(
                {
                    "agg_id": int(t.get("id", t.get("tradeId", 0))),
                    "ts": int(t["timestamp"]),
                    "price": float(t["price"]),
                    "qty": float(t["size"]),
                    "is_buyer_maker": side == "sell",
                    "symbol": symbol,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values(["ts", "agg_id"]).drop_duplicates(subset=["symbol", "agg_id"], keep="last")
        return enforce_schema(df, SCHEMAS["agg_trades"])

    def fetch_depth_snapshot(self, symbol: str, limit: int = 50) -> Dict:
        params = {"symbol": symbol, "limit": limit}
        return self._get_json("/public/market/depth", params=params)

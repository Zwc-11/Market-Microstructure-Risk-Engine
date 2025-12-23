from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.binance import BinanceFuturesRest


def test_binance_klines_pagination_and_schema(monkeypatch):
    calls = []

    def fake_get_json(path, params=None):
        calls.append(params)
        start = params["startTime"]
        if start == 0:
            return [
                [0, "1", "2", "0.5", "1.5", "10", 60000, "0", "1", "0", "0", "0"],
                [60000, "1.1", "2.1", "0.6", "1.6", "11", 120000, "0", "1", "0", "0", "0"],
            ]
        if start == 120000:
            return [[120000, "1.2", "2.2", "0.7", "1.7", "12", 180000, "0", "1", "0", "0", "0"]]
        return []

    client = BinanceFuturesRest()
    monkeypatch.setattr(client, "_get_json", fake_get_json)

    df = client.fetch_klines_1m("BTCUSDT", 0, 180000)
    assert list(df["ts"]) == [0, 60000, 120000]
    assert df["symbol"].unique().tolist() == ["BTCUSDT"]
    assert len(calls) >= 2


def test_binance_agg_trades_window(monkeypatch):
    params_seen = []

    def fake_get_json(path, params=None):
        params_seen.append(params)
        return [
            {"a": 1, "p": "100.0", "q": "0.1", "T": params["startTime"] + 1, "m": False},
            {"a": 2, "p": "100.1", "q": "0.2", "T": params["startTime"] + 2, "m": True},
        ]

    client = BinanceFuturesRest()
    monkeypatch.setattr(client, "_get_json", fake_get_json)
    df = client.fetch_agg_trades("BTCUSDT", 0, 3600_000 * 2)

    assert not df.empty
    for params in params_seen:
        assert params["endTime"] - params["startTime"] <= 55 * 60 * 1000

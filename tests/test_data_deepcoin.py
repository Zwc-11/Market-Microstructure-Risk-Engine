from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.deepcoin import DeepcoinRest


def test_deepcoin_klines_and_trades_schema(monkeypatch):
    def fake_get_json(path, params=None):
        if "kline" in path:
            return [[0, "1", "2", "0.5", "1.5", "10"]]
        return [{"id": 1, "timestamp": 0, "price": "100", "size": "0.1", "side": "sell"}]

    client = DeepcoinRest()
    monkeypatch.setattr(client, "_get_json", fake_get_json)

    klines = client.fetch_klines_1m("BTCUSDT", 0, 60_000)
    trades = client.fetch_trades("BTCUSDT", 0, 60_000)

    assert not klines.empty
    assert not trades.empty
    assert klines["symbol"].iloc[0] == "BTCUSDT"
    assert trades["symbol"].iloc[0] == "BTCUSDT"

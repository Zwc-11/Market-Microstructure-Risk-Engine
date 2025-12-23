from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import build_dataset as build_mod


class DummyBinance:
    def fetch_klines_1m(self, symbol, start_ms, end_ms):
        return pd.DataFrame(
            {
                "ts": [start_ms, start_ms + 60_000],
                "open": [1.0, 1.1],
                "high": [1.5, 1.6],
                "low": [0.9, 1.0],
                "close": [1.2, 1.3],
                "volume": [10.0, 11.0],
                "close_ts": [start_ms + 60_000, start_ms + 120_000],
                "quote_volume": [0.0, 0.0],
                "trade_count": [1, 1],
                "taker_buy_base": [0.0, 0.0],
                "taker_buy_quote": [0.0, 0.0],
                "symbol": [symbol, symbol],
            }
        )

    def fetch_agg_trades(self, symbol, start_ms, end_ms):
        return pd.DataFrame(
            {
                "agg_id": [1, 2],
                "ts": [start_ms + 1, start_ms + 2],
                "price": [100.0, 100.1],
                "qty": [0.1, 0.2],
                "is_buyer_maker": [False, True],
                "symbol": [symbol, symbol],
            }
        )


def _config(tmp_path):
    return {
        "paths": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "processed")},
        "data": {"quality": {"max_gap_seconds": 120}},
    }


def test_build_dataset_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(build_mod, "BinanceFuturesRest", lambda: DummyBinance())
    cfg = _config(tmp_path)

    build_mod.build_dataset(
        cfg,
        exchange="binance",
        symbol="BTCUSDT",
        start="2025-01-01",
        end="2025-01-02",
        build_bars=True,
    )

    raw_dir = Path(cfg["paths"]["raw_dir"])
    manifest = raw_dir / "binance" / "usdm" / "BTCUSDT" / "klines_1m" / "MANIFEST.json"
    assert manifest.exists()
    assert (Path(cfg["paths"]["processed_dir"]) / "bars_1m.parquet").exists()
    assert (Path(cfg["paths"]["processed_dir"]) / "bars_5m.parquet").exists()

    build_mod.build_dataset(
        cfg,
        exchange="binance",
        symbol="BTCUSDT",
        start="2025-01-01",
        end="2025-01-02",
        build_bars=True,
    )
    assert manifest.exists()

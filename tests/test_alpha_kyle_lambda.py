from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alpha.kyle_lambda import kyle_lambda_features, signed_volume_by_bucket


def test_signed_volume_tick_rule():
    times = pd.date_range("2025-01-01 00:00", periods=4, freq="30s", tz="UTC")
    ts = (times.view("int64") // 1_000_000).astype("int64")
    trades = pd.DataFrame(
        {
            "ts": ts,
            "price": [100.0, 100.0, 101.0, 100.0],
            "qty": [1.0, 1.0, 1.0, 1.0],
        }
    )

    flow = signed_volume_by_bucket(trades, bucket_seconds=60)
    assert flow.iloc[0] == 2.0


def test_kyle_lambda_no_lookahead():
    bars_time = pd.date_range("2025-01-01 00:00", periods=6, freq="1min")
    bars = pd.DataFrame({"timestamp": bars_time, "mid_close": [100, 101, 103, 106, 110, 115]})

    trade_time = pd.date_range("2025-01-01 00:00", periods=6, freq="1min", tz="UTC")
    ts = (trade_time.view("int64") // 1_000_000).astype("int64")
    trades = pd.DataFrame(
        {
            "ts": ts,
            "price": [100, 101, 102, 103, 104, 105],
            "qty": [1, 2, 3, 4, 5, 6],
            "is_buyer_maker": [False] * 6,
        }
    )

    cfg = {"window_minutes": 3, "zscore_window_minutes": 3, "bucket_seconds": 60}

    full = kyle_lambda_features(bars, trades, cfg)
    trunc = kyle_lambda_features(bars.iloc[:4], trades.iloc[:4], cfg)

    assert full["lambda_raw"].iloc[-1] > 0
    pd.testing.assert_series_equal(full["lambda_raw"].iloc[:4], trunc["lambda_raw"])

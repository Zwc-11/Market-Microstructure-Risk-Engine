from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alpha.vpin import vpin_features


def _bars(periods=2):
    times = pd.date_range("2025-01-01 00:00", periods=periods, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100.0] * periods})


def test_vpin_bucket_basic():
    bars = _bars()
    trade_times = pd.to_datetime(
        ["2025-01-01 00:00:10", "2025-01-01 00:00:20", "2025-01-01 00:00:40"],
        utc=True,
    )
    ts = (trade_times.view("int64") // 1_000_000).astype("int64")
    trades = pd.DataFrame(
        {
            "ts": ts,
            "price": [100.0, 100.0, 100.0],
            "qty": [5.0, 5.0, 2.0],
            "is_buyer_maker": [False, False, True],
        }
    )

    cfg = {
        "bucket_mult": 1.0,
        "volume_window_minutes": 1,
        "window_buckets": 1,
        "cdf_window_buckets": 1,
    }

    feats = vpin_features(bars, trades, cfg)
    expected = abs(10.0 - 2.0) / 12.0

    assert np.isclose(feats["vpin"].iloc[0], expected)
    assert np.isclose(feats["vpin_cdf"].iloc[0], 1.0)


def test_vpin_no_lookahead():
    bars = _bars(periods=3)
    trade_times = pd.to_datetime(
        ["2025-01-01 00:00:10", "2025-01-01 00:01:10", "2025-01-01 00:01:40"],
        utc=True,
    )
    ts = (trade_times.view("int64") // 1_000_000).astype("int64")
    trades = pd.DataFrame(
        {
            "ts": ts,
            "price": [100.0, 100.0, 100.0],
            "qty": [5.0, 5.0, 2.0],
            "is_buyer_maker": [False, False, True],
        }
    )

    cfg = {
        "bucket_mult": 1.0,
        "volume_window_minutes": 1,
        "window_buckets": 1,
        "cdf_window_buckets": 1,
    }

    full = vpin_features(bars, trades, cfg)
    trunc = vpin_features(bars.iloc[:2], trades.iloc[:1], cfg)

    pd.testing.assert_series_equal(full["vpin"].iloc[:1], trunc["vpin"].iloc[:1])

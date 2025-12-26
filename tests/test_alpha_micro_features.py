from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alpha.micro_features import build_micro_features


def _bars():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100.0, 101.0, 103.0]})


def _l2():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "bid_price_1": [100.0, 100.0, 100.0],
            "ask_price_1": [101.0, 101.0, 101.0],
            "bid_size_1": [10.0, 12.0, 11.0],
            "ask_size_1": [10.0, 9.0, 11.0],
        }
    )


def _trades():
    times = pd.to_datetime(
        ["2025-01-01 00:00:10", "2025-01-01 00:00:20", "2025-01-01 00:01:10"],
        utc=True,
    )
    ts = (times.view("int64") // 1_000_000).astype("int64")
    return pd.DataFrame(
        {
            "ts": ts,
            "price": [100.0, 100.0, 101.0],
            "qty": [5.0, 2.0, 6.0],
            "is_buyer_maker": [False, True, False],
        }
    )


def test_micro_features_columns():
    bars = _bars()
    l2 = _l2()
    trades = _trades()
    cfg = {
        "ofi": {"enabled": True, "levels": [1], "decay": 0.7, "zscore_window": 2, "include_raw": False},
        "obi": {"enabled": True, "levels": [1], "decay": 0.7},
        "kyle_lambda": {"enabled": True, "window_minutes": 2, "zscore_window_minutes": 2, "bucket_seconds": 60, "outputs": ["lambda_z"]},
        "vpin": {"enabled": True, "bucket_mult": 1.0, "volume_window_minutes": 1, "window_buckets": 1, "cdf_window_buckets": 1},
        "extras": {"spread": True, "mid_change": True, "depth_levels": [1]},
    }

    feats = build_micro_features(bars, l2, trades, cfg)
    expected = {"ofi_z_L1", "w_obi_L1", "lambda_z", "vpin", "vpin_cdf", "spread", "mid_change", "depth_L1"}
    assert expected.issubset(set(feats.columns))
    expected_index = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True))
    assert feats.index.equals(expected_index)


def test_micro_features_no_lookahead():
    bars = _bars()
    l2 = _l2()
    trades = _trades()
    cfg = {
        "ofi": {"enabled": True, "levels": [1], "decay": 0.7, "zscore_window": 2, "include_raw": False},
        "obi": {"enabled": True, "levels": [1], "decay": 0.7},
        "kyle_lambda": {"enabled": True, "window_minutes": 2, "zscore_window_minutes": 2, "bucket_seconds": 60, "outputs": ["lambda_z"]},
        "vpin": {"enabled": True, "bucket_mult": 1.0, "volume_window_minutes": 1, "window_buckets": 1, "cdf_window_buckets": 1},
    }

    full = build_micro_features(bars, l2, trades, cfg)
    trunc = build_micro_features(bars.iloc[:2], l2.iloc[:2], trades.iloc[:2], cfg)

    pd.testing.assert_series_equal(full["w_obi_L1"].iloc[:2], trunc["w_obi_L1"])

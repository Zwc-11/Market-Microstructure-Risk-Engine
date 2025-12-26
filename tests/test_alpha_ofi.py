from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alpha.ofi import ofi_features


def _bars(periods=3):
    times = pd.date_range("2025-01-01 00:00", periods=periods, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100.0] * periods})


def _l2():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "bid_price_1": [100.0, 100.0, 100.0],
            "ask_price_1": [101.0, 101.0, 101.0],
            "bid_size_1": [10.0, 12.0, 12.0],
            "ask_size_1": [10.0, 9.0, 11.0],
            "bid_price_2": [99.0, 99.0, 99.0],
            "ask_price_2": [102.0, 102.0, 102.0],
            "bid_size_2": [5.0, 4.0, 6.0],
            "ask_size_2": [5.0, 6.0, 5.0],
        }
    )


def test_weighted_ofi_levels():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": [1, 2], "decay": 0.5, "include_raw": True, "include_zscore": False}

    feats = ofi_features(bars, l2, cfg)

    assert np.isclose(feats["ofi_L1"].iloc[1], 1.5)
    assert np.isclose(feats["ofi_L2"].iloc[1], 1.0)


def test_ofi_zscore_no_lookahead():
    bars = _bars(periods=5)
    l2 = _l2()
    l2 = pd.concat([l2, l2.iloc[[1, 2]]], ignore_index=True)
    l2["timestamp"] = pd.date_range("2025-01-01 00:00", periods=5, freq="1min")

    cfg = {"levels": [1], "decay": 0.7, "zscore_window": 3, "include_raw": False, "include_zscore": True}

    full = ofi_features(bars, l2, cfg)
    trunc = ofi_features(bars.iloc[:4], l2.iloc[:4], cfg)

    pd.testing.assert_series_equal(full["ofi_L1_z"].iloc[:4], trunc["ofi_L1_z"])

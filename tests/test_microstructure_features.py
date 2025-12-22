from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.microstructure import ofi_features


def _bars():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100.0, 100.0, 100.0]})


def _l2():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "bid_price_1": [100.0, 100.0, 100.0],
            "ask_price_1": [101.0, 101.0, 100.5],
            "bid_size_1": [10.0, 12.0, 12.0],
            "ask_size_1": [10.0, 10.0, 11.0],
        }
    )


def test_ofi_sign_and_normalization():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": [1], "normalize_by_depth": True, "eps": 1.0e-9}

    feats = ofi_features(bars, l2, cfg)
    ofi = feats["ofi_L1"]

    expected_t1 = 2.0 / 22.0
    expected_t2 = -11.0 / 23.0

    assert np.isclose(ofi.iloc[1], expected_t1)
    assert np.isclose(ofi.iloc[2], expected_t2)


def test_ofi_no_lookahead_and_deterministic():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": [1], "normalize_by_depth": False, "eps": 1.0e-9}

    full = ofi_features(bars, l2, cfg)
    trunc = ofi_features(bars.iloc[:2], l2.iloc[:2], cfg)

    pd.testing.assert_series_equal(full["ofi_L1"].iloc[:2], trunc["ofi_L1"])
    pd.testing.assert_frame_equal(full, ofi_features(bars, l2, cfg))

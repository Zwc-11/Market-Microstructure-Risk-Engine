from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alpha.imbalance import obi_features


def _bars():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100.0, 100.0, 100.0]})


def _l2():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "bid_size_1": [10.0, 12.0, 11.0],
            "ask_size_1": [10.0, 8.0, 11.0],
            "bid_size_2": [5.0, 6.0, 6.0],
            "ask_size_2": [5.0, 10.0, 6.0],
        }
    )


def test_weighted_obi_formula():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": [1, 2], "decay": 0.7, "eps": 1.0e-9}

    feats = obi_features(bars, l2, cfg)

    num = 0.7 * (12.0 - 8.0) + (0.7**2) * (6.0 - 10.0)
    den = 0.7 * (12.0 + 8.0) + (0.7**2) * (6.0 + 10.0)
    expected = num / den

    assert np.isclose(feats["w_obi_L2"].iloc[1], expected)


def test_obi_no_lookahead():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": [1], "decay": 0.7, "eps": 1.0e-9}

    full = obi_features(bars, l2, cfg)
    trunc = obi_features(bars.iloc[:2], l2.iloc[:2], cfg)

    pd.testing.assert_series_equal(full["w_obi_L1"].iloc[:2], trunc["w_obi_L1"])

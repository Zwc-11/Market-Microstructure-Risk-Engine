from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.replenishment import replenishment_features


def _bars():
    times = pd.date_range("2025-01-01 00:00", periods=4, freq="1min")
    return pd.DataFrame({"timestamp": times, "mid_close": [100.0, 100.0, 100.0, 100.0]})


def _l2():
    times = pd.date_range("2025-01-01 00:00", periods=4, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "bid_price_1": [100.0, 100.0, 100.0, 100.0],
            "ask_price_1": [101.0, 101.0, 101.0, 101.0],
            "bid_size_1": [5.0, 4.0, 6.0, 6.0],
            "ask_size_1": [5.0, 4.0, 5.0, 5.0],
        }
    )


def test_replenishment_depth_and_ratio():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": 1, "eps": 1.0, "outputs": ["depth", "depletion", "replenish", "repl_ratio", "spread_mean", "microprice"]}

    feats = replenishment_features(bars, l2, cfg, repl_window=2)
    depth = feats["depth_L1"]
    depletion = feats["depletion_L1"]
    replenish = feats["replenish_L1"]
    ratio = feats["repl_ratio_L1"]

    assert np.isclose(depth.iloc[0], 10.0)
    assert np.isclose(depletion.iloc[1], 2.0)
    assert np.isclose(replenish.iloc[2], 3.0)
    assert np.isclose(ratio.iloc[2], 1.5)
    assert np.isclose(feats["spread_mean"].iloc[0], 1.0)
    assert np.isclose(feats["microprice"].iloc[2], (101 * 6 + 100 * 5) / 11)


def test_replenishment_no_lookahead_and_deterministic():
    bars = _bars()
    l2 = _l2()
    cfg = {"levels": 1, "eps": 1.0, "outputs": ["depth", "depletion", "replenish", "repl_ratio", "spread_mean", "microprice"]}

    full = replenishment_features(bars, l2, cfg, repl_window=2)
    trunc = replenishment_features(bars.iloc[:3], l2.iloc[:3], cfg, repl_window=2)

    pd.testing.assert_frame_equal(full.iloc[:3], trunc)
    pd.testing.assert_frame_equal(full, replenishment_features(bars, l2, cfg, repl_window=2))

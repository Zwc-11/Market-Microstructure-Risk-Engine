from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regime.regime import classify_regime


def _make_bars(prices):
    times = pd.date_range("2025-01-01 00:00", periods=len(prices), freq="5min")
    return pd.DataFrame({"timestamp": times, "mid_close": prices})


def _regime_cfg(hazard_rv_percentile=0.9, trend_strength_threshold=1.25):
    return {
        "windows": {"rv_window_bars": 6, "ema_fast_bars": 6, "ema_slow_bars": 18},
        "thresholds": {
            "hazard_rv_percentile": hazard_rv_percentile,
            "trend_strength_threshold": trend_strength_threshold,
        },
    }


def test_regime_trend_case():
    n = 120
    base = 1.0 + 0.001 * np.arange(n)
    prices = base * (1.0 + 0.0005 * np.sin(np.arange(n) / 5.0))
    bars = _make_bars(prices)

    out = classify_regime(bars, _regime_cfg())
    valid = out["rv"].notna()

    trend_share = (out.loc[valid, "regime"] == "TREND").mean()
    hazard_share = out.loc[valid, "hazard_flag"].mean()

    assert trend_share > 0.6
    assert hazard_share < 0.3


def test_regime_range_case():
    n = 140
    t = np.arange(n)
    prices = 1.0 + 0.01 * np.sin(t / 1.5) + 0.003 * np.sin(t / 0.3)
    bars = _make_bars(prices)

    out = classify_regime(bars, _regime_cfg())
    valid = out["rv"].notna()

    range_share = (out.loc[valid, "regime"] == "RANGE").mean()
    hazard_share = out.loc[valid, "hazard_flag"].mean()

    assert range_share > 0.6
    assert hazard_share < 0.3


def test_regime_hazard_case():
    n = 150
    prices = [1.0]
    for i in range(1, n):
        step = 0.02 if i % 2 == 0 else -0.02
        prices.append(prices[-1] * (1.0 + step))
    bars = _make_bars(prices)

    out = classify_regime(bars, _regime_cfg(hazard_rv_percentile=0.7))
    valid = out["rv"].notna()

    hazard_share = (out.loc[valid, "regime"] == "HAZARD").mean()
    assert hazard_share > 0.4


def test_regime_no_lookahead_percentile():
    n = 140
    low_vol = 1.0 + 0.001 * np.arange(n)
    prices = low_vol.copy()
    for i in range(110, n):
        prices[i] = prices[i - 1] * (1.0 + (0.05 if i % 2 == 0 else -0.05))

    bars_full = _make_bars(prices)
    bars_trunc = bars_full.iloc[:100].copy()

    out_full = classify_regime(bars_full, _regime_cfg())
    out_trunc = classify_regime(bars_trunc, _regime_cfg())

    cutoff = bars_trunc["timestamp"].iloc[-1]
    full_slice = out_full.loc[out_full.index <= cutoff, ["regime", "hazard_flag", "trend_flag"]]
    trunc_slice = out_trunc[["regime", "hazard_flag", "trend_flag"]]

    pd.testing.assert_frame_equal(full_slice, trunc_slice)


def test_regime_deterministic():
    prices = 1.0 + 0.001 * np.arange(120)
    bars = _make_bars(prices)

    out_a = classify_regime(bars, _regime_cfg())
    out_b = classify_regime(bars, _regime_cfg())

    pd.testing.assert_frame_equal(out_a, out_b)

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.entries_5m import generate_entries_5m


def _make_bars(prices):
    times = pd.date_range("2025-01-01 00:00", periods=len(prices), freq="5min")
    bars = pd.DataFrame(
        {
            "timestamp": times,
            "mid_close": prices,
            "mid_high": np.array(prices) + 0.1,
            "mid_low": np.array(prices) - 0.1,
            "volume": np.ones(len(prices)),
        }
    )
    return bars


def _range_cfg(window=3, k=1.0, cooldown=1):
    return {
        "range": {
            "enabled": True,
            "center": {"kind": "rolling_vwap", "window_5m_bars": window},
            "band": {"kind": "rolling_std_mid", "window_5m_bars": window, "k": k},
            "cooldown_5m_bars": cooldown,
        },
        "trend": {"enabled": False},
    }


def _trend_cfg():
    return {
        "range": {"enabled": False},
        "trend": {
            "enabled": True,
            "ema_fast_bars": 3,
            "ema_slow_bars": 5,
            "pullback": {"kind": "atr", "atr_window_5m_bars": 2, "tolerance_atr": 2.0},
        },
    }


def test_range_entries_and_cooldown():
    prices = [100.0, 100.0, 100.0, 110.0, 111.0, 100.0, 100.0, 90.0, 90.0, 100.0]
    bars = _make_bars(prices)
    regime_vals = ["HAZARD", "HAZARD", "HAZARD", "RANGE", "RANGE", "HAZARD", "HAZARD", "RANGE", "RANGE", "HAZARD"]
    regime = pd.Series(regime_vals, index=pd.to_datetime(bars["timestamp"]))

    events = generate_entries_5m(bars, regime, _range_cfg(window=3, k=1.0, cooldown=1))

    assert len(events) == 2
    assert events.iloc[0]["reason"] == "range_upper_touch"
    assert events.iloc[0]["side"] == -1
    assert events.iloc[1]["reason"] == "range_lower_touch"
    assert events.iloc[1]["side"] == 1
    assert pd.Timestamp("2025-01-01 00:20") not in set(events["entry_ts"])
    assert pd.Timestamp("2025-01-01 00:40") not in set(events["entry_ts"])


def test_trend_pullback_long_entry():
    prices = [1.0, 1.05, 1.1, 1.15, 1.12, 1.16, 1.2, 1.25]
    bars = _make_bars(prices)
    regime = pd.Series(["TREND"] * len(prices), index=pd.to_datetime(bars["timestamp"]))

    events = generate_entries_5m(bars, regime, _trend_cfg())
    assert (events["reason"] == "trend_pullback_long").any()
    assert (events["side"] == 1).any()


def test_trend_pullback_short_entry():
    prices = [1.3, 1.25, 1.2, 1.15, 1.18, 1.12, 1.08, 1.05]
    bars = _make_bars(prices)
    regime = pd.Series(["TREND"] * len(prices), index=pd.to_datetime(bars["timestamp"]))

    events = generate_entries_5m(bars, regime, _trend_cfg())
    assert (events["reason"] == "trend_pullback_short").any()
    assert (events["side"] == -1).any()


def test_hazard_regime_no_entries():
    prices = [100.0, 101.0, 102.0, 103.0]
    bars = _make_bars(prices)
    regime = pd.Series(["HAZARD"] * len(prices), index=pd.to_datetime(bars["timestamp"]))

    events = generate_entries_5m(bars, regime, _range_cfg())
    assert events.empty


def test_entries_no_lookahead_truncation():
    prices = [100.0, 100.0, 100.0, 110.0, 111.0, 100.0, 100.0, 90.0, 90.0, 100.0]
    bars = _make_bars(prices)
    regime = pd.Series(["RANGE"] * len(prices), index=pd.to_datetime(bars["timestamp"]))

    events_full = generate_entries_5m(bars, regime, _range_cfg(window=3, k=1.0, cooldown=1))
    cutoff = bars["timestamp"].iloc[6]

    bars_trunc = bars.iloc[:7].copy()
    regime_trunc = regime.loc[regime.index <= cutoff]
    events_trunc = generate_entries_5m(bars_trunc, regime_trunc, _range_cfg(window=3, k=1.0, cooldown=1))

    full_early = events_full[events_full["entry_ts"] <= cutoff].reset_index(drop=True)
    trunc = events_trunc.reset_index(drop=True)
    pd.testing.assert_frame_equal(full_early, trunc)

import pandas as pd

from src.strategy.entries_5m import generate_entries_5m


def _bars_from_prices(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "mid_close": prices,
            "mid_high": [p + 0.5 for p in prices],
            "mid_low": [p - 0.5 for p in prices],
            "volume": [1.0 for _ in prices],
        }
    )


def _trend_cfg():
    return {
        "enabled": True,
        "ema_fast_bars": 2,
        "ema_slow_bars": 3,
        "cooldown_5m_bars": 0,
        "pullback": {
            "atr_window_5m_bars": 2,
            "tolerance_atr": 10.0,
        },
    }


def test_disable_trend_pullback_long():
    bars = _bars_from_prices([100, 101, 102, 103, 104, 105])
    regime = pd.Series(["TREND"] * len(bars), index=pd.to_datetime(bars["timestamp"]))
    entries_cfg = {"range": {"enabled": False}, "trend": _trend_cfg(), "subtypes": {}}

    events = generate_entries_5m(bars, regime, entries_cfg)
    assert (events["reason"] == "trend_pullback_long").any()

    entries_cfg["subtypes"] = {"trend_pullback_long": False}
    events_disabled = generate_entries_5m(bars, regime, entries_cfg)
    assert not (events_disabled["reason"] == "trend_pullback_long").any()


def test_disable_range_vwap_band():
    bars = _bars_from_prices([100, 101, 102])
    regime = pd.Series(["RANGE"] * len(bars), index=pd.to_datetime(bars["timestamp"]))
    entries_cfg = {
        "range": {
            "enabled": True,
            "center": {"kind": "rolling_vwap", "window_5m_bars": 2},
            "band": {"kind": "rolling_std_mid", "window_5m_bars": 2, "k": 0.0},
            "cooldown_5m_bars": 0,
        },
        "trend": {"enabled": False},
        "subtypes": {},
    }

    events = generate_entries_5m(bars, regime, entries_cfg)
    assert not events.empty

    entries_cfg["subtypes"] = {"range_vwap_band": False}
    events_disabled = generate_entries_5m(bars, regime, entries_cfg)
    assert events_disabled.empty


def test_trend_pullback_long_filter_requires_confirmation():
    entries_cfg = {"range": {"enabled": False}, "trend": _trend_cfg(), "subtypes": {}}
    entries_cfg["trend"]["pullback"]["long_filter"] = {
        "enabled": True,
        "require_ema_fast_slope_positive": True,
        "confirm_bars": 2,
    }

    bars_fail = _bars_from_prices([100, 101, 100, 101])
    regime_fail = pd.Series(["TREND"] * len(bars_fail), index=pd.to_datetime(bars_fail["timestamp"]))
    events_fail = generate_entries_5m(bars_fail, regime_fail, entries_cfg)
    assert not (events_fail["reason"] == "trend_pullback_long").any()

    bars_pass = _bars_from_prices([100, 101, 102, 103])
    regime_pass = pd.Series(["TREND"] * len(bars_pass), index=pd.to_datetime(bars_pass["timestamp"]))
    events_pass = generate_entries_5m(bars_pass, regime_pass, entries_cfg)
    assert (events_pass["reason"] == "trend_pullback_long").any()

import pandas as pd

from src.features.intrabar import build_intrabar_features


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "mid_open": prices,
            "mid_high": [p + 0.5 for p in prices],
            "mid_low": [p - 0.5 for p in prices],
            "mid_close": prices,
            "volume": [1.0 for _ in prices],
        }
    )


def test_intrabar_no_lookahead():
    bars = _bars([100, 101, 102, 103, 104, 105])
    entry_ts = pd.Series([bars["timestamp"].iloc[-1]])
    feat = build_intrabar_features(bars, entry_ts, lookback_min=5)
    ret_expected = (104 / 103) - 1.0
    assert abs(float(feat["ret_1m_last"].iloc[0]) - ret_expected) < 1.0e-9

    bars2 = bars.copy()
    bars2.loc[bars2.index[-1], "mid_close"] = 1000.0
    feat2 = build_intrabar_features(bars2, entry_ts, lookback_min=5)
    assert abs(float(feat2["ret_1m_last"].iloc[0]) - ret_expected) < 1.0e-9


def test_intrabar_missing_columns_raises():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="1min", tz="UTC"),
            "close": [1.0, 1.1, 1.2],
        }
    )
    entry_ts = pd.Series([bars["timestamp"].iloc[-1]])
    try:
        build_intrabar_features(bars, entry_ts, lookback_min=2)
    except ValueError as exc:
        assert "OHLC" in str(exc) or "columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing OHLC columns")

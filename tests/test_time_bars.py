from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bars.time_bars import build_time_bars, resample_time_bars


def _make_trades():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:10",
                    "2025-01-01 00:00:50",
                    "2025-01-01 00:01:10",
                    "2025-01-01 00:01:40",
                ]
            ),
            "price": [100.0, 102.0, 101.0, 103.0],
            "qty": [1.0, 2.0, 1.0, 1.0],
        }
    )


def _make_l2():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:05",
                    "2025-01-01 00:00:55",
                    "2025-01-01 00:01:20",
                    "2025-01-01 00:01:50",
                ]
            ),
            "bid": [99.0, 100.0, 100.0, 101.0],
            "ask": [101.0, 102.0, 104.0, 105.0],
            "bid_size": [1.0, 3.0, 3.0, 2.0],
            "ask_size": [1.0, 1.0, 1.0, 2.0],
        }
    )


def test_time_bars_ohlc_and_vwap_with_l2():
    trades = _make_trades()
    l2 = _make_l2()

    bars = build_time_bars(
        trades,
        l2=l2,
        freq="1min",
        time_col="timestamp",
        price_col="price",
        qty_col="qty",
        l2_time_col="timestamp",
    )

    bar1_end = pd.Timestamp("2025-01-01 00:01:00")
    bar2_end = pd.Timestamp("2025-01-01 00:02:00")

    bar1 = bars.loc[bar1_end]
    bar2 = bars.loc[bar2_end]

    assert bar1["mid_open"] == 100.0
    assert bar1["mid_high"] == 101.0
    assert bar1["mid_low"] == 100.0
    assert bar1["mid_close"] == 101.0

    assert bar2["mid_open"] == 102.0
    assert bar2["mid_high"] == 103.0
    assert bar2["mid_low"] == 102.0
    assert bar2["mid_close"] == 103.0

    assert bar1["volume"] == 3.0
    assert bar2["volume"] == 2.0

    assert np.isclose(bar1["vwap"], (100.0 * 1.0 + 102.0 * 2.0) / 3.0)
    assert np.isclose(bar2["vwap"], (101.0 * 1.0 + 103.0 * 1.0) / 2.0)

    assert "spread_mean" in bars.columns
    assert "microprice_close" in bars.columns
    assert np.isclose(bar1["spread_mean"], 2.0)
    assert np.isclose(bar2["spread_mean"], 4.0)
    assert np.isclose(bar1["microprice_close"], 101.5)
    assert np.isclose(bar2["microprice_close"], 103.0)


def test_time_bar_timestamps_aligned_to_minute():
    trades = _make_trades()
    l2 = _make_l2()

    bars = build_time_bars(
        trades,
        l2=l2,
        freq="1min",
        time_col="timestamp",
        price_col="price",
        qty_col="qty",
        l2_time_col="timestamp",
    )

    assert all(ts.second == 0 and ts.microsecond == 0 for ts in bars.index)
    assert (bars.index == bars.index.floor("min")).all()


def test_resample_1m_to_5m_exact_values():
    times = pd.date_range("2025-01-01 00:01", periods=10, freq="1min")
    bars_1m = pd.DataFrame(
        {
            "mid_open": [float(i) for i in range(1, 11)],
            "mid_high": [float(i) + 0.5 for i in range(1, 11)],
            "mid_low": [float(i) - 0.5 for i in range(1, 11)],
            "mid_close": [float(i) + 0.1 for i in range(1, 11)],
            "volume": [float(i) for i in range(1, 11)],
            "vwap": [float(i) + 0.2 for i in range(1, 11)],
            "spread_mean": [0.1 * float(i) for i in range(1, 11)],
            "microprice_close": [float(i) + 0.3 for i in range(1, 11)],
        },
        index=times,
    )

    bars_5m = resample_time_bars(bars_1m, freq="5min")
    bar1 = bars_5m.loc[pd.Timestamp("2025-01-01 00:05")]
    bar2 = bars_5m.loc[pd.Timestamp("2025-01-01 00:10")]

    assert bar1["mid_open"] == 1.0
    assert bar1["mid_high"] == 5.5
    assert bar1["mid_low"] == 0.5
    assert bar1["mid_close"] == 5.1
    assert bar1["volume"] == 15.0

    vwap_1 = sum((i + 0.2) * i for i in range(1, 6)) / sum(range(1, 6))
    assert np.isclose(bar1["vwap"], vwap_1)
    assert np.isclose(bar1["spread_mean"], sum(0.1 * i for i in range(1, 6)) / 5.0)
    assert bar1["microprice_close"] == 5.3

    assert bar2["mid_open"] == 6.0
    assert bar2["mid_high"] == 10.5
    assert bar2["mid_low"] == 5.5
    assert bar2["mid_close"] == 10.1
    assert bar2["volume"] == 40.0

    vwap_2 = sum((i + 0.2) * i for i in range(6, 11)) / sum(range(6, 11))
    assert np.isclose(bar2["vwap"], vwap_2)
    assert np.isclose(bar2["spread_mean"], sum(0.1 * i for i in range(6, 11)) / 5.0)
    assert bar2["microprice_close"] == 10.3

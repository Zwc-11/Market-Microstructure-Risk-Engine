from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.simulator import run_backtest


def _config():
    return {
        "labeling": {
            "triple_barrier": {
                "horizon_minutes": 10,
                "vol": {"kind": "rolling", "window_1m_bars": 2, "min_sigma": 0.01},
                "barriers": {"pt_mult": 1.0, "sl_mult": 1.0},
                "price_source": "mid_close",
                "tie_break": {"mode": "worst_case"},
            }
        },
        "backtest": {
            "initial_capital": 10000,
            "leverage": 1,
            "fees_bps": {"taker": 10},
            "slippage_bps": 5,
            "latency_ms": 0,
            "sizing": {"max_position_notional_pct": 0.1},
            "reporting": {},
        },
    }


def _bars_for_long_pt():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "mid_open": [100.0, 100.0, 100.0],
            "mid_high": [100.0, 101.5, 100.5],
            "mid_low": [100.0, 99.5, 99.8],
            "mid_close": [100.0, 101.0, 100.2],
        }
    )


def _bars_for_long_sl():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "mid_open": [100.0, 100.0, 100.0],
            "mid_high": [100.0, 100.4, 100.2],
            "mid_low": [100.0, 98.5, 99.0],
            "mid_close": [100.0, 99.0, 99.5],
        }
    )


def _event(ts, price, side=1):
    return pd.DataFrame(
        {
            "event_id": ["evt1"],
            "symbol": ["TEST"],
            "entry_ts": [ts],
            "entry_price": [price],
            "side": [side],
            "regime": ["RANGE"],
            "reason": ["range_lower_touch"],
        }
    )


def test_backtest_long_tp_and_costs():
    bars = _bars_for_long_pt()
    events = _event(bars["timestamp"].iloc[0], 100.0, side=1)
    trades, equity, summary = run_backtest(bars, events, _config())

    assert len(trades) == 1
    trade = trades.iloc[0]

    expected_exit = 100.0 * (1.0 + 0.01)
    assert np.isclose(trade["exit_price"], expected_exit)
    assert trade["event_type"] == "pt"
    assert trade["net_pnl"] < trade["gross_pnl"]
    assert summary["trade_count"] == 1
    assert equity["equity"].iloc[-1] > 0


def test_backtest_long_sl():
    bars = _bars_for_long_sl()
    events = _event(bars["timestamp"].iloc[0], 100.0, side=1)
    trades, _, _ = run_backtest(bars, events, _config())

    assert len(trades) == 1
    trade = trades.iloc[0]
    expected_exit = 100.0 * (1.0 - 0.01)
    assert np.isclose(trade["exit_price"], expected_exit)
    assert trade["event_type"] == "sl"


def test_backtest_deterministic():
    bars = _bars_for_long_pt()
    events = _event(bars["timestamp"].iloc[0], 100.0, side=1)
    trades_a, equity_a, summary_a = run_backtest(bars, events, _config())
    trades_b, equity_b, summary_b = run_backtest(bars, events, _config())

    pd.testing.assert_frame_equal(trades_a, trades_b)
    pd.testing.assert_frame_equal(equity_a, equity_b)
    assert summary_a == summary_b

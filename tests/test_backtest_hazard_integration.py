from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.simulator import run_backtest, run_backtest_enhanced


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
            "fees_bps": {"taker": 1},
            "slippage_bps": 0,
            "latency_ms": 0,
            "sizing": {"max_position_notional_pct": 0.1},
            "reporting": {},
        },
        "policy": {
            "exit": {"enabled": True, "hazard_threshold": 0.7, "consecutive_minutes": 1, "require_rising": False},
            "fail_fast": {"enabled": False, "hazard_threshold": 0.9, "confirm_signals": {}},
            "add_risk": {"enabled": True, "hazard_max_to_add": 0.2},
        },
        "hazard": {"feature_window_minutes": 3},
    }


def _bars_5m():
    times = pd.date_range("2025-01-01 00:00", periods=3, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "mid_open": [100.0, 100.0, 99.0],
            "mid_high": [100.0, 100.2, 99.2],
            "mid_low": [100.0, 98.5, 98.8],
            "mid_close": [100.0, 99.0, 98.9],
        }
    )


def _bars_1m():
    times = pd.date_range("2025-01-01 00:00", periods=6, freq="1min")
    prices = [100.0, 99.7, 99.6, 99.4, 99.2, 98.8]
    return pd.DataFrame(
        {
            "timestamp": times,
            "mid_close": prices,
            "mid_high": np.array(prices) + 0.1,
            "mid_low": np.array(prices) - 0.1,
        }
    )


def _events():
    return pd.DataFrame(
        {
            "event_id": ["evt1"],
            "symbol": ["TEST"],
            "entry_ts": [pd.Timestamp("2025-01-01 00:00")],
            "entry_price": [100.0],
            "side": [1],
            "regime": ["RANGE"],
            "reason": ["range_lower_touch"],
        }
    )


def test_enhanced_exits_before_sl_and_reduces_loss():
    bars_5m = _bars_5m()
    bars_1m = _bars_1m()
    events = _events()
    cfg = _config()

    base_trades, _, _ = run_backtest(bars_5m, events, cfg)
    hazard_prob = pd.Series(
        [0.0, 0.0, 0.9, 0.0, 0.0, 0.0], index=pd.to_datetime(bars_1m["timestamp"])
    )
    enh_trades, _, _, _ = run_backtest_enhanced(
        bars_5m, events, cfg, bars_1m, hazard_prob=hazard_prob, policy_mode="hazard_exit_only"
    )

    assert len(base_trades) == 1
    assert len(enh_trades) == 1
    assert enh_trades.iloc[0]["exit_ts"] < base_trades.iloc[0]["exit_ts"]
    assert enh_trades.iloc[0]["net_pnl"] > base_trades.iloc[0]["net_pnl"]


def test_constant_zero_matches_baseline():
    bars_5m = _bars_5m()
    bars_1m = _bars_1m()
    events = _events()
    cfg = _config()

    base_trades, _, _ = run_backtest(bars_5m, events, cfg)
    hazard_prob = pd.Series(0.0, index=pd.to_datetime(bars_1m["timestamp"]))
    enh_trades, _, _, _ = run_backtest_enhanced(
        bars_5m, events, cfg, bars_1m, hazard_prob=hazard_prob, policy_mode="full_policy"
    )

    pd.testing.assert_frame_equal(
        base_trades[["exit_ts", "exit_price", "net_pnl"]],
        enh_trades[["exit_ts", "exit_price", "net_pnl"]],
    )

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.walkforward import generate_walkforward_folds, run_walkforward


def _make_bars(start: str, days: int) -> pd.DataFrame:
    periods = days * 24 * 12
    times = pd.date_range(start, periods=periods, freq="5min")
    pattern = np.array([100.0, 100.0, 101.0, 99.0, 100.0, 102.0])
    prices = np.tile(pattern, int(np.ceil(periods / len(pattern))))[:periods]
    return pd.DataFrame(
        {
            "timestamp": times,
            "mid_close": prices,
            "mid_high": prices + 0.5,
            "mid_low": prices - 0.5,
            "volume": np.ones(periods),
        }
    )


def _config():
    return {
        "regime": {
            "windows": {"rv_window_bars": 6, "ema_fast_bars": 3, "ema_slow_bars": 9},
            "thresholds": {"hazard_rv_percentile": 0.99, "trend_strength_threshold": 5.0},
        },
        "strategy": {
            "entries_5m": {
                "range": {
                    "enabled": True,
                    "center": {"kind": "rolling_vwap", "window_5m_bars": 3},
                    "band": {"kind": "rolling_std_mid", "window_5m_bars": 3, "k": 0.5},
                    "cooldown_5m_bars": 1,
                },
                "trend": {"enabled": False},
            }
        },
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
        "model": {
            "training": {
                "validation": {
                    "embargo_minutes": 0,
                    "splits": {"train_days": 1, "test_days": 1, "step_days": 1},
                }
            }
        },
    }


def test_generate_walkforward_folds():
    bars = _make_bars("2025-01-01", days=4)
    folds = generate_walkforward_folds(
        pd.to_datetime(bars["timestamp"]),
        train_days=1,
        test_days=1,
        step_days=1,
        embargo_minutes=0,
    )
    assert len(folds) == 2
    assert folds[0]["test_start"] == pd.Timestamp("2025-01-02 00:00:00")
    assert folds[0]["test_end"] == pd.Timestamp("2025-01-03 00:00:00")
    assert folds[1]["test_start"] == pd.Timestamp("2025-01-03 00:00:00")
    assert folds[1]["test_end"] == pd.Timestamp("2025-01-04 00:00:00")


def test_walkforward_summary_and_determinism():
    bars = _make_bars("2025-01-01", days=4)
    cfg = _config()

    summary_a = run_walkforward(
        bars,
        cfg,
        modes=["range_only", "trend_only"],
        fee_mults=[1.0],
        slippage_mults=[1.0, 2.0],
    )
    summary_b = run_walkforward(
        bars,
        cfg,
        modes=["range_only", "trend_only"],
        fee_mults=[1.0],
        slippage_mults=[1.0, 2.0],
    )

    assert not summary_a.empty
    expected_rows = 2 * 2 * 1 * 2
    assert len(summary_a) == expected_rows
    assert set(
        [
            "fold_id",
            "test_start",
            "test_end",
            "mode",
            "fee_mult",
            "slippage_mult",
            "pnl_net",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "trade_count",
            "total_fees",
            "total_slippage",
        ]
    ).issubset(summary_a.columns)

    pd.testing.assert_frame_equal(summary_a, summary_b)

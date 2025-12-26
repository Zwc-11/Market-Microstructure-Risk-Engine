import pandas as pd

from src.modeling.train_meta import apply_cost_label


def _bars(prices):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "mid_open": prices,
            "mid_high": [p + 0.5 for p in prices],
            "mid_low": [p - 0.5 for p in prices],
            "mid_close": prices,
        }
    )


def test_cost_label_positive_net():
    bars = _bars([100, 101, 102])
    events = pd.DataFrame(
        {
            "event_id": ["evt1"],
            "entry_ts": [bars["timestamp"].iloc[0]],
            "side": [1],
            "entry_price": [100.0],
        }
    )
    cfg = {
        "backtest": {
            "initial_capital": 10000,
            "leverage": 1.0,
            "fees_bps": {"taker": 0.0},
            "slippage_bps": 0.0,
            "sizing": {"max_position_notional_pct": 0.1},
        },
        "labeling": {
            "triple_barrier": {
                "horizon_minutes": 10,
                "vol": {"kind": "rolling", "window_1m_bars": 2, "min_sigma": 1.0e-6},
                "barriers": {"pt_mult": 0.5, "sl_mult": 0.5},
                "tie_break": {"mode": "worst_case"},
                "by_regime": {},
            }
        },
    }
    labeled = apply_cost_label(events, bars, cfg)
    assert labeled["net_pnl_est"].iloc[0] > 0
    assert labeled["label"].iloc[0] == 1

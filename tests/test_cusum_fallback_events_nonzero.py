from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import run_all as run_mod


def test_cusum_fallback_events_nonzero():
    ts = pd.date_range("2025-01-01", periods=60, freq="1min", tz="UTC")
    prices = pd.Series(100.0, index=ts)
    prices.iloc[10:] += 2.0
    prices.iloc[30:] -= 3.0

    bars_1m = pd.DataFrame(
        {
            "mid_open": prices.values,
            "mid_high": prices.values + 0.5,
            "mid_low": prices.values - 0.5,
            "mid_close": prices.values,
        },
        index=ts,
    )

    cfg = {
        "labeling": {
            "cusum": {"threshold_k": 0.5, "vol_window_1m_bars": 5, "min_sigma": 1.0e-6},
            "triple_barrier": {
                "horizon_minutes": 5,
                "barriers": {"pt_mult": 1.0, "sl_mult": 1.0},
                "price_source": "mid_close",
                "tie_break": {"mode": "worst_case"},
                "vol": {"kind": "ewma", "window_1m_bars": 5, "min_sigma": 1.0e-6},
            },
        }
    }

    trades = run_mod._cusum_fallback_trades(bars_1m, cfg, "BTCUSDT")
    assert not trades.empty

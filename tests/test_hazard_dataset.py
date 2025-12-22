from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.hazard_dataset import build_hazard_dataset
from src.labeling.triple_barrier import compute_volatility


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
        "hazard": {"horizon_minutes": 5, "label_mode": "adverse_barrier", "adverse_barrier": {"which": "SL"}},
    }


def _bars():
    times = pd.date_range("2025-01-01 00:00", periods=12, freq="1min")
    prices = [100, 100, 100, 100, 100, 99.0, 98.5, 100, 101.0, 101.5, 101.0, 100.5]
    return pd.DataFrame(
        {
            "timestamp": times,
            "mid_close": prices,
            "mid_high": np.array(prices) + 0.3,
            "mid_low": np.array(prices) - 0.3,
        }
    )


def _trades():
    return pd.DataFrame(
        [
            {
                "event_id": "evt_long",
                "symbol": "TEST",
                "entry_ts": pd.Timestamp("2025-01-01 00:01"),
                "exit_ts": pd.Timestamp("2025-01-01 00:07"),
                "side": 1,
                "entry_price": 100.0,
            },
            {
                "event_id": "evt_short",
                "symbol": "TEST",
                "entry_ts": pd.Timestamp("2025-01-01 00:06"),
                "exit_ts": pd.Timestamp("2025-01-01 00:10"),
                "side": -1,
                "entry_price": 100.0,
            },
        ]
    )


def test_hazard_labels_long_short_and_caps():
    bars = _bars()
    trades = _trades()
    cfg = _config()

    hazard = build_hazard_dataset(trades, bars, cfg)

    assert not hazard.empty
    assert (hazard["t"] <= hazard["exit_ts"]).all()
    assert (hazard["horizon_end_ts"] <= hazard["exit_ts"]).all()

    long_rows = hazard[hazard["event_id"] == "evt_long"].reset_index(drop=True)
    assert (long_rows["t"] > long_rows["entry_ts"]).all()

    long_y = long_rows.set_index("t")["y"]
    assert long_y.loc[pd.Timestamp("2025-01-01 00:02")] == 1
    assert long_y.loc[pd.Timestamp("2025-01-01 00:05")] == 1

    short_rows = hazard[hazard["event_id"] == "evt_short"].reset_index(drop=True)
    short_y = short_rows.set_index("t")["y"]
    assert short_y.loc[pd.Timestamp("2025-01-01 00:07")] == 1


def test_hazard_sl_matches_triple_barrier():
    bars = _bars()
    trades = _trades().iloc[[0]].copy()
    cfg = _config()

    hazard = build_hazard_dataset(trades, bars, cfg)
    entry_ts = trades.iloc[0]["entry_ts"]
    entry_price = trades.iloc[0]["entry_price"]

    price_col = cfg["labeling"]["triple_barrier"]["price_source"]
    sigma = compute_volatility(
        bars.set_index("timestamp"),
        price_col="mid_close" if price_col == "mid_close" else "close",
        kind=cfg["labeling"]["triple_barrier"]["vol"]["kind"],
        window=int(cfg["labeling"]["triple_barrier"]["vol"]["window_1m_bars"]),
        min_sigma=float(cfg["labeling"]["triple_barrier"]["vol"]["min_sigma"]),
    )
    sl_mult = float(cfg["labeling"]["triple_barrier"]["barriers"]["sl_mult"])
    sigma_t0 = float(sigma.loc[entry_ts])
    expected_sl = entry_price * (1.0 - sl_mult * sigma_t0)

    t_check = pd.Timestamp("2025-01-01 00:02")
    row = hazard[hazard["t"] == t_check].iloc[0]
    assert row["future_min_low"] <= expected_sl


def test_hazard_deterministic():
    bars = _bars()
    trades = _trades()
    cfg = _config()

    a = build_hazard_dataset(trades, bars, cfg)
    b = build_hazard_dataset(trades, bars, cfg)
    pd.testing.assert_frame_equal(a, b)

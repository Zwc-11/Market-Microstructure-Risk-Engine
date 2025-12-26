import pandas as pd

from src.labeling.triple_barrier import triple_barrier_labels_by_regime


def test_regime_specific_barriers():
    start = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    index = pd.date_range(start, periods=20, freq="min")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        },
        index=index,
    )

    events = pd.DataFrame(
        {
            "event_id": ["evt_trend", "evt_range"],
            "t0": [start, start],
            "side": [1, 1],
            "entry_price": [100.0, 100.0],
            "regime": ["TREND", "RANGE"],
        }
    )

    cfg = {
        "labeling": {
            "triple_barrier": {
                "horizon_minutes": 10,
                "vol": {"kind": "rolling", "window_1m_bars": 2, "min_sigma": 1.0e-6},
                "barriers": {"pt_mult": 1.0, "sl_mult": 1.0},
                "tie_break": {"mode": "worst_case"},
                "by_regime": {
                    "TREND": {"horizon": 10, "pt_mult": 1.0, "sl_mult": 1.0},
                    "RANGE": {"horizon": 5, "pt_mult": 1.0, "sl_mult": 1.0},
                },
            }
        }
    }

    labeled = triple_barrier_labels_by_regime(events, bars, cfg, price_col="close", time_col=None)
    t1_trend = labeled.loc[0, "t1"]
    t1_range = labeled.loc[1, "t1"]

    assert t1_trend > t1_range
    assert t1_range == start + pd.Timedelta(minutes=5)

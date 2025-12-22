from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.cusum import cusum_events


def _price_series(values):
    times = pd.date_range("2025-01-01 00:00", periods=len(values), freq="1min")
    return pd.DataFrame({"close": values}, index=times)


def test_no_events_when_returns_small():
    prices = _price_series([100, 100.001, 100.002, 100.003, 100.004, 100.005])

    events = cusum_events(
        prices,
        threshold_k=3.0,
        vol_window=3,
        min_sigma=0.01,
        price_col="close",
    )

    assert events.empty


def test_events_trigger_on_cumulative_drift():
    up_moves = [100]
    for _ in range(4):
        up_moves.append(up_moves[-1] * (1.0 + 0.0015))
    down_moves = [up_moves[-1]]
    for _ in range(3):
        down_moves.append(down_moves[-1] * (1.0 - 0.0015))

    prices = _price_series(up_moves + down_moves[1:])

    events = cusum_events(
        prices,
        threshold_k=2.0,
        vol_window=2,
        min_sigma=0.001,
        price_col="close",
    )

    assert len(events) >= 2
    assert events.iloc[0]["side"] == 1
    assert (events["side"] == -1).any()


def test_deterministic_output():
    prices = _price_series([100, 100.2, 100.4, 100.6, 100.8, 101.0])

    events_a = cusum_events(
        prices,
        threshold_k=1.5,
        vol_window=2,
        min_sigma=0.0005,
        price_col="close",
    )
    events_b = cusum_events(
        prices,
        threshold_k=1.5,
        vol_window=2,
        min_sigma=0.0005,
        price_col="close",
    )

    pd.testing.assert_frame_equal(events_a, events_b)

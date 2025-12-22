from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.triple_barrier import triple_barrier_labels


def _bars_from_rows(rows):
    times = pd.date_range("2025-01-01 00:00", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=times)


def test_long_profit_hit():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 100, "low": 100, "close": 100},
            {"open": 100, "high": 101.2, "low": 99.5, "close": 101},
            {"open": 101, "high": 101.1, "low": 100.2, "close": 100.8},
        ]
    )
    events = pd.DataFrame({"t0": [bars.index[0]], "side": [1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=2,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="worst_case",
    )

    row = out.iloc[0]
    assert row["label"] == 1
    assert row["t1"] == bars.index[1]


def test_short_stop_hit():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 100, "low": 100, "close": 100},
            {"open": 100, "high": 101.2, "low": 99.8, "close": 101},
            {"open": 101, "high": 101.1, "low": 100.2, "close": 100.8},
        ]
    )
    events = pd.DataFrame({"t0": [bars.index[0]], "side": [-1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=2,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="worst_case",
    )

    row = out.iloc[0]
    assert row["label"] == -1
    assert row["t1"] == bars.index[1]


def test_tie_break_worst_case():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 100, "low": 100, "close": 100},
            {"open": 100, "high": 101.5, "low": 98.5, "close": 100.5},
            {"open": 100.5, "high": 100.8, "low": 99.8, "close": 100.2},
        ]
    )
    events = pd.DataFrame({"t0": [bars.index[0]], "side": [1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=2,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="worst_case",
    )

    row = out.iloc[0]
    assert row["label"] == -1
    assert row["event_type"] == "sl"


def test_tie_break_first_touch_bullish():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 100, "low": 100, "close": 100},
            {"open": 100, "high": 101.5, "low": 98.5, "close": 101},
            {"open": 101, "high": 101.2, "low": 100.6, "close": 100.9},
        ]
    )
    events = pd.DataFrame({"t0": [bars.index[0]], "side": [1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=2,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="first_touch",
    )

    row = out.iloc[0]
    assert row["label"] == 1
    assert row["event_type"] == "pt"


def test_timeout_vertical_barrier():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 100, "low": 100, "close": 100},
            {"open": 100, "high": 100.5, "low": 99.5, "close": 100.1},
            {"open": 100.1, "high": 100.4, "low": 99.6, "close": 100.0},
        ]
    )
    events = pd.DataFrame({"t0": [bars.index[0]], "side": [1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=2,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="worst_case",
    )

    row = out.iloc[0]
    assert row["label"] == 0
    assert row["t1"] == bars.index[2]
    assert row["event_type"] == "timeout"


def test_no_lookahead_entry_bar_ignored():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 105, "low": 95, "close": 100},
            {"open": 100, "high": 100.5, "low": 99.5, "close": 100.1},
        ]
    )
    events = pd.DataFrame({"t0": [bars.index[0]], "side": [1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=1,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="worst_case",
    )

    row = out.iloc[0]
    assert row["label"] == 0
    assert row["t1"] == bars.index[1]


def test_asof_entry_price():
    bars = _bars_from_rows(
        [
            {"open": 100, "high": 100.2, "low": 99.8, "close": 100},
            {"open": 100, "high": 100.3, "low": 99.7, "close": 100.1},
        ]
    )
    t0 = bars.index[0] + pd.Timedelta(seconds=30)
    events = pd.DataFrame({"t0": [t0], "side": [1]})

    out = triple_barrier_labels(
        events,
        bars,
        horizon_minutes=1,
        pt_mult=1.0,
        sl_mult=1.0,
        vol_kind="rolling",
        vol_window=2,
        min_sigma=0.01,
        tie_break="worst_case",
    )

    row = out.iloc[0]
    assert row["entry_price"] == 100
    assert row["entry_time"] == bars.index[0]

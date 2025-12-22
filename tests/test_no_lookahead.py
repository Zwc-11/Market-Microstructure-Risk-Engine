from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.cusum import cusum_events
from src.labeling.triple_barrier import compute_volatility


def _bars_from_prices(prices):
    times = pd.date_range("2025-01-01 00:00", periods=len(prices), freq="1min")
    return pd.DataFrame({"close": prices}, index=times)


def _assert_no_future_merge(merged, row_time_col, source_time_col):
    if (merged[source_time_col] > merged[row_time_col]).any():
        raise AssertionError("Found future timestamp usage in merged data")


def test_forward_merge_leak_detected():
    left = pd.DataFrame(
        {"t": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01"])}
    )
    right = pd.DataFrame({"t_feature": pd.to_datetime(["2025-01-01 00:02"]), "x": [1.0]})

    merged = pd.merge_asof(
        left,
        right,
        left_on="t",
        right_on="t_feature",
        direction="forward",
    )

    with pytest.raises(AssertionError):
        _assert_no_future_merge(merged, "t", "t_feature")


def test_backward_merge_passes():
    left = pd.DataFrame(
        {"t": pd.to_datetime(["2025-01-01 00:01", "2025-01-01 00:02"])}
    )
    right = pd.DataFrame(
        {"t_feature": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01"]), "x": [1.0, 2.0]}
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="t",
        right_on="t_feature",
        direction="backward",
    )

    _assert_no_future_merge(merged, "t", "t_feature")


def test_volatility_no_lookahead_future_perturbation():
    prices = [100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6]
    bars = _bars_from_prices(prices)
    bars_future = bars.copy()
    bars_future.loc[bars_future.index[5:], "close"] = [1000, 10]

    sigma_base = compute_volatility(
        bars, price_col="close", kind="rolling", window=3, min_sigma=1.0e-6
    )
    sigma_future = compute_volatility(
        bars_future, price_col="close", kind="rolling", window=3, min_sigma=1.0e-6
    )

    cutoff = bars.index[4]
    pd.testing.assert_series_equal(sigma_base.loc[:cutoff], sigma_future.loc[:cutoff])


def test_cusum_no_lookahead_future_perturbation():
    prices = [100, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2]
    bars = _bars_from_prices(prices)
    bars_future = bars.copy()
    bars_future.loc[bars_future.index[5:], "close"] = [150, 50]

    events_base = cusum_events(
        bars, threshold_k=2.0, vol_window=3, min_sigma=1.0e-4, price_col="close"
    )
    events_future = cusum_events(
        bars_future, threshold_k=2.0, vol_window=3, min_sigma=1.0e-4, price_col="close"
    )

    cutoff = bars.index[4]
    base_cut = events_base[events_base["t0"] <= cutoff].reset_index(drop=True)
    future_cut = events_future[events_future["t0"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_cut, future_cut)

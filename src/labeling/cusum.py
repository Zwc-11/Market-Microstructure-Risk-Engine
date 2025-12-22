from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def cusum_events(
    data: pd.Series | pd.DataFrame,
    threshold_k: float,
    vol_window: int,
    min_sigma: float = 1.0e-6,
    time_col: Optional[str] = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Symmetric CUSUM filter with a rolling volatility threshold.

    Formulas:
      r_t = log(p_t / p_{t-1})
      sigma_t = std(r) over rolling window at time t (no lookahead)
      h_t = threshold_k * max(sigma_t, min_sigma)

      g_pos = max(0, g_pos + r_t)
      g_neg = min(0, g_neg + r_t)

    Event at time t when g_pos > h_t (side=+1) or g_neg < -h_t (side=-1),
    then reset both cumulative sums to 0.
    """
    if threshold_k <= 0:
        raise ValueError("threshold_k must be positive")
    if vol_window <= 0:
        raise ValueError("vol_window must be positive")
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive")

    if isinstance(data, pd.DataFrame):
        if time_col is not None:
            if time_col not in data.columns:
                raise ValueError(f"time_col '{time_col}' not found in data")
            data = data.copy()
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.set_index(time_col)
        if price_col not in data.columns:
            raise ValueError(f"price_col '{price_col}' not found in data")
        prices = data[price_col].astype(float)
    else:
        prices = data.astype(float)

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("data must have a DatetimeIndex or a valid time_col")

    prices = prices.sort_index()
    returns = np.log(prices).diff()
    sigma = returns.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
    sigma = sigma.fillna(min_sigma).clip(lower=min_sigma)
    threshold = threshold_k * sigma

    g_pos = 0.0
    g_neg = 0.0
    events = []

    for t, r_t in returns.items():
        if not np.isfinite(r_t):
            continue

        g_pos = max(0.0, g_pos + r_t)
        g_neg = min(0.0, g_neg + r_t)

        h_t = float(threshold.loc[t])

        if g_pos > h_t:
            events.append({"t0": t, "side": 1, "threshold": h_t, "sigma": float(sigma.loc[t])})
            g_pos = 0.0
            g_neg = 0.0
        elif g_neg < -h_t:
            events.append({"t0": t, "side": -1, "threshold": h_t, "sigma": float(sigma.loc[t])})
            g_pos = 0.0
            g_neg = 0.0

    return pd.DataFrame(events)

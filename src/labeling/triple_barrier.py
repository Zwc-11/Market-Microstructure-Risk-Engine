from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


_TIE_BREAK_MODES = {"worst_case", "best_case", "first_touch"}


def compute_volatility(
    bars: pd.DataFrame,
    price_col: str,
    kind: str,
    window: int,
    min_sigma: float,
) -> pd.Series:
    """
    Compute per-bar volatility using only information <= t.

    Formula:
      r_t = (p_t / p_{t-1}) - 1
      sigma_t = std(r) over the rolling window or EWMA std at time t
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive")
    if price_col not in bars.columns:
        raise ValueError(f"price_col '{price_col}' not found in bars")

    prices = bars[price_col].astype(float)
    returns = prices.pct_change()

    if kind == "ewma":
        sigma = returns.ewm(span=window, adjust=False).std(bias=False)
    elif kind == "rolling":
        sigma = returns.rolling(window=window, min_periods=window).std(ddof=0)
    else:
        raise ValueError(f"Unknown vol kind '{kind}'")

    sigma = sigma.fillna(min_sigma).clip(lower=min_sigma)
    return sigma


def _ensure_datetime_index(bars: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in bars.columns:
            raise ValueError(f"time_col '{time_col}' not found in bars")
        bars = bars.copy()
        bars[time_col] = pd.to_datetime(bars[time_col])
        bars = bars.set_index(time_col)

    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex or a valid time_col")

    return bars.sort_index()


def _resolve_tie(open_px: float, close_px: float, side: int, tie_break: str) -> str:
    if tie_break == "worst_case":
        return "sl"
    if tie_break == "best_case":
        return "pt"

    if close_px >= open_px:
        first_leg = "high"
    else:
        first_leg = "low"

    if side > 0:
        return "pt" if first_leg == "high" else "sl"
    return "pt" if first_leg == "low" else "sl"


def triple_barrier_labels(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    horizon_minutes: int,
    pt_mult: float,
    sl_mult: float,
    price_col: str = "close",
    tie_break: str = "worst_case",
    vol_kind: str = "ewma",
    vol_window: int = 60,
    min_sigma: float = 1.0e-6,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Triple-barrier labeling with OHLC barrier detection and a vertical time stop.

    Formulas:
      r_t = (p_t / p_{t-1}) - 1
      sigma_t = std(r) over window (rolling) or EWMA std at t
      For long:
        pt = p0 * (1 + pt_mult * sigma_t0)
        sl = p0 * (1 - sl_mult * sigma_t0)
      For short:
        pt = p0 * (1 - pt_mult * sigma_t0)
        sl = p0 * (1 + sl_mult * sigma_t0)

    Barrier checks use bar high/low and consider bars strictly after the entry bar
    up to t0 + horizon_minutes. Ties are resolved via tie_break:
      - worst_case: choose SL
      - best_case: choose PT
      - first_touch: infer order using open/close (open->high->low if close>=open).
    """
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if tie_break not in _TIE_BREAK_MODES:
        raise ValueError(f"tie_break must be one of {sorted(_TIE_BREAK_MODES)}")

    bars = _ensure_datetime_index(bars, time_col)

    required_cols = {"open", "high", "low", price_col}
    missing_cols = required_cols - set(bars.columns)
    if missing_cols:
        raise ValueError(f"bars missing required columns: {sorted(missing_cols)}")

    if "t0" not in events.columns:
        raise ValueError("events must include a 't0' column")

    side_col = "side" if "side" in events.columns else "direction" if "direction" in events.columns else None
    entry_col = (
        "entry_price"
        if "entry_price" in events.columns
        else "price"
        if "price" in events.columns
        else None
    )

    sigma = compute_volatility(bars, price_col=price_col, kind=vol_kind, window=vol_window, min_sigma=min_sigma)
    times = bars.index

    results = []
    for _, row in events.iterrows():
        t0 = pd.Timestamp(row["t0"])
        side_val = row[side_col] if side_col else 1
        if not np.isfinite(side_val) or side_val == 0:
            raise ValueError("side/direction must be a non-zero numeric value")
        side = 1 if side_val > 0 else -1

        pos = int(times.searchsorted(t0, side="right") - 1)
        if pos < 0:
            raise ValueError(f"event t0 {t0} is before the first bar")
        entry_time = times[pos]

        if entry_col:
            entry_price = float(row[entry_col])
        else:
            entry_price = float(bars.loc[entry_time, price_col])

        sigma_t0 = float(sigma.loc[entry_time])
        if not np.isfinite(sigma_t0) or sigma_t0 <= 0:
            sigma_t0 = min_sigma

        if side > 0:
            pt_price = entry_price * (1.0 + pt_mult * sigma_t0)
            sl_price = entry_price * (1.0 - sl_mult * sigma_t0)
        else:
            pt_price = entry_price * (1.0 - pt_mult * sigma_t0)
            sl_price = entry_price * (1.0 + sl_mult * sigma_t0)

        end_time = t0 + pd.Timedelta(minutes=horizon_minutes)
        window = bars.loc[(times > entry_time) & (times <= end_time)]

        label = 0
        t1 = None
        event_type = "timeout"

        for bar_time, bar in window.iterrows():
            if side > 0:
                pt_hit = bar["high"] >= pt_price
                sl_hit = bar["low"] <= sl_price
            else:
                pt_hit = bar["low"] <= pt_price
                sl_hit = bar["high"] >= sl_price

            if pt_hit or sl_hit:
                if pt_hit and sl_hit:
                    hit = _resolve_tie(bar["open"], bar["close"], side, tie_break)
                else:
                    hit = "pt" if pt_hit else "sl"

                label = 1 if hit == "pt" else -1
                t1 = bar_time
                event_type = hit
                break

        if label == 0:
            if window.empty:
                t1 = end_time
            else:
                t1 = window.index[-1]

        results.append(
            {
                "t0": t0,
                "t1": t1,
                "side": side,
                "label": label,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "pt_price": pt_price,
                "sl_price": sl_price,
                "sigma": sigma_t0,
                "event_type": event_type,
            }
        )

    return pd.DataFrame(results, index=events.index)

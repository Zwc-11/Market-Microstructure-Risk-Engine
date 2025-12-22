from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    rv_window_bars: int
    ema_fast_bars: int
    ema_slow_bars: int
    hazard_rv_percentile: float
    trend_strength_threshold: float


def _ensure_datetime_index(bars: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in bars.columns:
            raise ValueError(f"time_col '{time_col}' not found in bars")
        bars = bars.copy()
        bars[time_col] = pd.to_datetime(bars[time_col])
        bars = bars.set_index(time_col)

    if not isinstance(bars.index, pd.DatetimeIndex):
        if "timestamp" in bars.columns:
            bars = bars.copy()
            bars["timestamp"] = pd.to_datetime(bars["timestamp"])
            bars = bars.set_index("timestamp")
        elif "ts" in bars.columns:
            bars = bars.copy()
            bars["ts"] = pd.to_datetime(bars["ts"])
            bars = bars.set_index("ts")
        else:
            raise ValueError("bars must have a DatetimeIndex or a timestamp/ts column")

    return bars.sort_index()


def compute_returns(price: pd.Series) -> pd.Series:
    """r_t = log(p_t / p_{t-1})."""
    return pd.Series(np.log(price).diff(), index=price.index)


def compute_rv(returns: pd.Series, window: int) -> pd.Series:
    """rv_t = rolling_std(r_t, window, ddof=0)."""
    if window <= 0:
        raise ValueError("rv window must be positive")
    return returns.rolling(window=window, min_periods=window).std(ddof=0)


def compute_hazard_threshold(rv: pd.Series, percentile: float, window: int) -> pd.Series:
    """hazard_thr_t = rolling_quantile(rv_t, q, window), backward-only."""
    if not 0.0 < percentile < 1.0:
        raise ValueError("hazard_rv_percentile must be between 0 and 1")
    if window <= 0:
        raise ValueError("hazard threshold window must be positive")
    return rv.rolling(window=window, min_periods=window).quantile(percentile)


def compute_trend_strength(
    ema_fast: pd.Series, ema_slow: pd.Series, rv: pd.Series, eps: float
) -> pd.Series:
    """trend_strength_t = |ema_fast - ema_slow| / (rv + eps)."""
    return (ema_fast - ema_slow).abs() / (rv + eps)


def apply_hysteresis(flags: pd.Series, enter: int, exit: int) -> pd.Series:
    """
    Debounce a boolean series by requiring consecutive confirmations.

    enter: consecutive True bars required to switch on
    exit: consecutive False bars required to switch off
    """
    if enter <= 0 or exit <= 0:
        raise ValueError("enter/exit must be positive")

    clean = flags.fillna(False).astype(bool)
    state = False
    on_count = 0
    off_count = 0
    out = []

    for value in clean:
        if value:
            on_count += 1
            off_count = 0
        else:
            off_count += 1
            on_count = 0

        if not state and on_count >= enter:
            state = True
        elif state and off_count >= exit:
            state = False

        out.append(state)

    return pd.Series(out, index=clean.index)


def classify_regime(
    bars: pd.DataFrame,
    regime_cfg: dict,
    time_col: Optional[str] = None,
    price_col: Optional[str] = None,
    trend_enter_bars: int = 2,
    trend_exit_bars: int = 2,
    hazard_enter_bars: int = 2,
    hazard_exit_bars: int = 2,
    eps: float = 1.0e-12,
) -> pd.DataFrame:
    """
    Regime classifier with backward-only hazard detection and trend/range labeling.

    Formulas:
      r_t = log(p_t / p_{t-1})
      rv_t = rolling_std(r_t, window=rv_window, ddof=0)
      hazard_thr_t = rolling_quantile(rv_t, q, window=max(5*rv_window, 60))
      hazard_flag_t = rv_t >= hazard_thr_t (False when rv_t is NaN)

      ema_fast_t = EMA(price, span=ema_fast_bars)
      ema_slow_t = EMA(price, span=ema_slow_bars)
      trend_strength_t = |ema_fast_t - ema_slow_t| / (rv_t + eps)
      trend_flag_t = trend_strength_t > trend_strength_threshold

    Regime assignment:
      - HAZARD if hazard_flag
      - TREND if trend_flag and not hazard
      - RANGE otherwise
    """
    if price_col is None:
        price_col = "mid_close" if "mid_close" in bars.columns else "close"
    if price_col not in bars.columns:
        raise ValueError("bars must include mid_close or close price")

    bars = _ensure_datetime_index(bars, time_col)
    price = bars[price_col].astype(float)

    cfg = RegimeConfig(
        rv_window_bars=int(regime_cfg["windows"]["rv_window_bars"]),
        ema_fast_bars=int(regime_cfg["windows"]["ema_fast_bars"]),
        ema_slow_bars=int(regime_cfg["windows"]["ema_slow_bars"]),
        hazard_rv_percentile=float(regime_cfg["thresholds"]["hazard_rv_percentile"]),
        trend_strength_threshold=float(regime_cfg["thresholds"]["trend_strength_threshold"]),
    )

    returns = compute_returns(price)
    rv = compute_rv(returns, cfg.rv_window_bars)

    hazard_window = max(5 * cfg.rv_window_bars, 60)
    hazard_thr = compute_hazard_threshold(rv, cfg.hazard_rv_percentile, hazard_window)
    hazard_flag = (rv >= hazard_thr) & rv.notna()

    ema_fast = price.ewm(span=cfg.ema_fast_bars, adjust=False).mean()
    ema_slow = price.ewm(span=cfg.ema_slow_bars, adjust=False).mean()
    trend_strength = compute_trend_strength(ema_fast, ema_slow, rv, eps=eps)
    trend_flag = trend_strength > cfg.trend_strength_threshold

    hazard_state = apply_hysteresis(hazard_flag, enter=hazard_enter_bars, exit=hazard_exit_bars)
    trend_state = apply_hysteresis(trend_flag, enter=trend_enter_bars, exit=trend_exit_bars)

    regime = pd.Series("RANGE", index=bars.index, dtype="object")
    regime[trend_state] = "TREND"
    regime[hazard_state] = "HAZARD"

    out = bars.copy()
    out["rv"] = rv
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["trend_strength"] = trend_strength
    out["hazard_flag"] = hazard_state
    out["trend_flag"] = trend_state
    out["regime"] = regime
    return out

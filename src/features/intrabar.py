from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in bars")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "ts" in df.columns:
            df = df.copy()
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.set_index("ts")
        else:
            raise ValueError("bars must have a DatetimeIndex or timestamp/ts column")
    return df.sort_index()


def _pick_ohlc_cols(bars: pd.DataFrame) -> Dict[str, str]:
    if {"mid_open", "mid_high", "mid_low", "mid_close"}.issubset(bars.columns):
        return {
            "open": "mid_open",
            "high": "mid_high",
            "low": "mid_low",
            "close": "mid_close",
        }
    if {"open", "high", "low", "close"}.issubset(bars.columns):
        return {"open": "open", "high": "high", "low": "low", "close": "close"}
    missing = {"open", "high", "low", "close"} - set(bars.columns)
    raise ValueError(f"bars missing required OHLC columns: {sorted(missing)}")


def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    x = np.arange(len(values), dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def build_intrabar_features(
    bars_1m: pd.DataFrame,
    entry_ts_series: pd.Series,
    lookback_min: int = 5,
) -> pd.DataFrame:
    """
    Build intrabar candle-structure features using 1m bars strictly BEFORE entry time.

    Window: [T - lookback_min, T) (no leakage).
    """
    if lookback_min <= 0:
        raise ValueError("lookback_min must be positive")

    bars = _ensure_datetime_index(bars_1m, None)
    ohlc_cols = _pick_ohlc_cols(bars)
    required = [ohlc_cols["open"], ohlc_cols["high"], ohlc_cols["low"], ohlc_cols["close"]]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    entries = pd.to_datetime(entry_ts_series)
    eps = 1.0e-12
    features: List[Dict[str, float]] = []

    vol_series = bars["volume"].astype(float) if "volume" in bars.columns else None
    trade_series = None
    for col in ("trade_count", "count", "num_trades"):
        if col in bars.columns:
            trade_series = bars[col].astype(float)
            break

    close = bars[ohlc_cols["close"]].astype(float)
    open_px = bars[ohlc_cols["open"]].astype(float)
    high = bars[ohlc_cols["high"]].astype(float)
    low = bars[ohlc_cols["low"]].astype(float)
    returns = close.pct_change()

    for ts in entries:
        window_start = ts - pd.Timedelta(minutes=lookback_min)
        window = bars.loc[(bars.index >= window_start) & (bars.index < ts)]
        feat: Dict[str, float] = {
            "entry_ts": ts,
            "ret_1m_last": float("nan"),
            "ret_2m_sum": float("nan"),
            "ret_5m_sum": float("nan"),
            "ret_slope_5": float("nan"),
            "realized_vol_5": float("nan"),
            "body_ratio_mean": float("nan"),
            "upper_wick_ratio_max": float("nan"),
            "lower_wick_ratio_max": float("nan"),
            "close_location_last": float("nan"),
            "close_location_mean": float("nan"),
            "vol_zscore_5": float("nan"),
            "trade_count_zscore_5": float("nan"),
        }

        if window.empty:
            features.append(feat)
            continue

        window_idx = window.index
        win_close = close.loc[window_idx]
        win_open = open_px.loc[window_idx]
        win_high = high.loc[window_idx]
        win_low = low.loc[window_idx]

        win_returns = returns.loc[window_idx].dropna()
        if not win_returns.empty:
            feat["ret_1m_last"] = float(win_returns.iloc[-1])
            feat["ret_2m_sum"] = float(win_returns.iloc[-2:].sum())
            feat["ret_5m_sum"] = float(win_returns.iloc[-5:].sum())
            feat["realized_vol_5"] = float(win_returns.std(ddof=0))

        if len(win_close) >= 5:
            feat["ret_slope_5"] = _linear_slope(win_close.tail(5).to_numpy())

        rng = (win_high - win_low).astype(float) + eps
        body = (win_close - win_open).abs()
        body_ratio = body / rng
        upper_wick = win_high - np.maximum(win_open, win_close)
        lower_wick = np.minimum(win_open, win_close) - win_low
        upper_ratio = upper_wick / rng
        lower_ratio = lower_wick / rng

        feat["body_ratio_mean"] = float(body_ratio.mean())
        feat["upper_wick_ratio_max"] = float(upper_ratio.max())
        feat["lower_wick_ratio_max"] = float(lower_ratio.max())

        last_close = float(win_close.iloc[-1])
        last_low = float(win_low.iloc[-1])
        last_high = float(win_high.iloc[-1])
        feat["close_location_last"] = float((last_close - last_low) / ((last_high - last_low) + eps))
        feat["close_location_mean"] = float(((win_close - win_low) / rng).mean())

        if vol_series is not None:
            vol_window = vol_series.loc[window_idx]
            baseline = vol_series.loc[(bars.index >= ts - pd.Timedelta(minutes=60)) & (bars.index < ts)]
            base_mean = float(baseline.mean()) if not baseline.empty else float("nan")
            base_std = float(baseline.std(ddof=0)) if not baseline.empty else float("nan")
            if np.isfinite(base_std) and base_std > 0:
                feat["vol_zscore_5"] = float((vol_window.mean() - base_mean) / base_std)
            else:
                feat["vol_zscore_5"] = 0.0

        if trade_series is not None:
            trade_window = trade_series.loc[window_idx]
            baseline = trade_series.loc[(bars.index >= ts - pd.Timedelta(minutes=60)) & (bars.index < ts)]
            base_mean = float(baseline.mean()) if not baseline.empty else float("nan")
            base_std = float(baseline.std(ddof=0)) if not baseline.empty else float("nan")
            if np.isfinite(base_std) and base_std > 0:
                feat["trade_count_zscore_5"] = float((trade_window.mean() - base_mean) / base_std)
            else:
                feat["trade_count_zscore_5"] = 0.0

        features.append(feat)

    out = pd.DataFrame(features)
    return out

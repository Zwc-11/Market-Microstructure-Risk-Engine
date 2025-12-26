from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from src.features.impact import compute_kyle_lambda


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    if is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        median = float(values.dropna().median()) if values.notna().any() else 0.0
        unit = "ms" if median >= 1_000_000_000_000 else "s"
        return pd.to_datetime(values, unit=unit, utc=True)
    return pd.to_datetime(series, utc=True)


def _ensure_datetime_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in data")
        df = df.copy()
        df[time_col] = _coerce_timestamp(df[time_col])
        df = df.set_index(time_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = _coerce_timestamp(df["timestamp"])
            df = df.set_index("timestamp")
        elif "ts" in df.columns:
            df = df.copy()
            df["ts"] = _coerce_timestamp(df["ts"])
            df = df.set_index("ts")
        else:
            raise ValueError("data must have a DatetimeIndex or timestamp/ts column")
    else:
        df = df.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    return df.sort_index()


def _rolling_zscore(series: pd.Series, window: int, eps: float) -> pd.Series:
    if window <= 0:
        raise ValueError("zscore window must be positive")
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    denom = std.where(std > 0, np.nan) + eps
    return (series - mean) / denom


def _tick_rule_sign(prices: pd.Series) -> pd.Series:
    delta = prices.diff()
    sign = np.sign(delta)
    sign = sign.replace(0, np.nan).ffill().fillna(1.0)
    return sign


def signed_volume_by_bucket(
    trades: pd.DataFrame,
    bucket_seconds: int = 60,
    price_col: str = "price",
    qty_col: str = "qty",
    side_col: Optional[str] = None,
    is_buyer_maker_col: str = "is_buyer_maker",
) -> pd.Series:
    """
    Signed volume per time bucket using exchange side when available; otherwise tick rule.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    if "ts" not in trades.columns:
        raise ValueError("trades must include 'ts' column")
    if price_col not in trades.columns or qty_col not in trades.columns:
        raise ValueError("trades missing required price/qty columns")

    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["ts"], unit="ms", utc=True, errors="coerce")
    trades = trades.dropna(subset=["timestamp"])

    sort_cols = ["timestamp"]
    if "agg_id" in trades.columns:
        sort_cols.append("agg_id")
    trades = trades.sort_values(sort_cols)

    if is_buyer_maker_col in trades.columns:
        sign = trades[is_buyer_maker_col].map(lambda x: -1.0 if bool(x) else 1.0)
    elif side_col is not None and side_col in trades.columns:
        side = trades[side_col].astype(str).str.lower()
        sign = side.map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
        sign = sign.replace(0.0, np.nan).ffill().fillna(1.0)
    else:
        sign = _tick_rule_sign(trades[price_col].astype(float))

    signed_qty = trades[qty_col].astype(float) * sign
    bucket = trades["timestamp"].dt.floor(pd.Timedelta(seconds=bucket_seconds))
    return signed_qty.groupby(bucket).sum().sort_index()


def kyle_lambda_features(
    bars_1m: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: Dict,
    time_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Kyle lambda via rolling regression: delta_price ~ signed volume per bucket.

    lambda_t = cov(x, y) / var(x)
    lambda_z = (lambda - mean) / std over rolling window.
    illiquidity_flag = lambda_z > z_thr.
    """
    bars = _ensure_datetime_index(bars_1m, time_col)
    window = int(cfg["window_minutes"])
    z_window = int(cfg.get("zscore_window_minutes", window))
    z_thr = float(cfg.get("illiquidity_z", 2.0))
    eps = float(cfg.get("zscore_eps", 1.0e-9))
    bucket_seconds = int(cfg.get("bucket_seconds", 60))

    signed_flow = signed_volume_by_bucket(
        trades,
        bucket_seconds=bucket_seconds,
        price_col=str(cfg.get("trade_price_col", "price")),
        qty_col=str(cfg.get("trade_qty_col", "qty")),
        side_col=cfg.get("trade_side_col"),
        is_buyer_maker_col=str(cfg.get("trade_is_buyer_maker_col", "is_buyer_maker")),
    )
    signed_flow = signed_flow.reindex(bars.index, fill_value=0.0)

    lambda_df = compute_kyle_lambda(
        bars,
        signed_flow,
        window=window,
        separate_up_down=False,
        price_col=price_col,
    )
    lambda_raw = lambda_df["lambda"].astype(float)
    lambda_z = _rolling_zscore(lambda_raw, z_window, eps) if z_window > 0 else pd.Series(index=bars.index)

    out = pd.DataFrame(
        {
            "lambda_raw": lambda_raw,
            "lambda_z": lambda_z,
            "illiquidity_flag": lambda_z > z_thr,
        },
        index=bars.index,
    )
    return out

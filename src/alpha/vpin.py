from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


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


def _align_to_bars(bars: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    bars_idx = bars.index
    bars_df = pd.DataFrame({"ts": bars_idx}).sort_values("ts")
    feat_df = features.reset_index().rename(columns={features.index.name or "index": "ts"}).sort_values("ts")
    merged = pd.merge_asof(bars_df, feat_df, on="ts", direction="backward")
    merged = merged.set_index("ts")
    return merged.reindex(bars_idx)


def _tick_rule_sign(prices: pd.Series) -> pd.Series:
    delta = prices.diff()
    sign = np.sign(delta)
    sign = sign.replace(0, np.nan).ffill().fillna(1.0)
    return sign


def _rolling_cdf(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("cdf window must be positive")
    return series.rolling(window=window, min_periods=window).apply(lambda x: float(np.mean(x <= x[-1])), raw=True)


def _minute_trade_volumes(
    trades: pd.DataFrame,
    price_col: str,
    qty_col: str,
    side_col: Optional[str],
    is_buyer_maker_col: str,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["buy_vol", "sell_vol", "total_vol"])
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

    qty = trades[qty_col].astype(float)
    trades["buy_vol"] = qty.where(sign > 0, 0.0)
    trades["sell_vol"] = qty.where(sign < 0, 0.0).abs()
    trades["minute"] = trades["timestamp"].dt.floor("min")

    grouped = trades.groupby("minute")[["buy_vol", "sell_vol"]].sum()
    grouped["total_vol"] = grouped["buy_vol"] + grouped["sell_vol"]
    return grouped.sort_index()


def compute_vpin(
    trades: pd.DataFrame,
    bucket_mult: float,
    volume_window_minutes: int,
    window_buckets: int,
    bucket_seconds: int = 60,
    min_bucket_volume: float = 1.0,
    trade_price_col: str = "price",
    trade_qty_col: str = "qty",
    trade_side_col: Optional[str] = None,
    trade_is_buyer_maker_col: str = "is_buyer_maker",
) -> pd.DataFrame:
    """
    VPIN via volume buckets:

      Bucket size V_t = mean(volume_1m, window) * bucket_mult.
      For each bucket b: imbalance_b = |Vb - Vs| / V.
      VPIN_t = mean(imbalance_b) over last N buckets.
    """
    if bucket_mult <= 0:
        raise ValueError("bucket_mult must be positive")
    if volume_window_minutes <= 0:
        raise ValueError("volume_window_minutes must be positive")
    if window_buckets <= 0:
        raise ValueError("window_buckets must be positive")
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    if min_bucket_volume <= 0:
        raise ValueError("min_bucket_volume must be positive")

    minute_vol = _minute_trade_volumes(
        trades,
        price_col=trade_price_col,
        qty_col=trade_qty_col,
        side_col=trade_side_col,
        is_buyer_maker_col=trade_is_buyer_maker_col,
    )
    if minute_vol.empty:
        return pd.DataFrame(columns=["vpin"])

    vol_mean = minute_vol["total_vol"].rolling(window=volume_window_minutes, min_periods=volume_window_minutes).mean()
    vol_mean = vol_mean.fillna(minute_vol["total_vol"])
    bucket_size_series = (vol_mean * bucket_mult).clip(lower=min_bucket_volume)

    bucket_imbalances = []
    bucket_end_times = []
    bucket_size = None
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_filled = 0.0

    for minute, row in minute_vol.iterrows():
        total = float(row["total_vol"])
        if total <= 0:
            continue
        buy_remaining = float(row["buy_vol"])
        sell_remaining = float(row["sell_vol"])
        remaining = total

        while remaining > 0:
            if bucket_size is None:
                bucket_size = float(bucket_size_series.loc[minute])
                if not np.isfinite(bucket_size) or bucket_size <= 0:
                    bucket_size = float(min_bucket_volume)

            fill = min(bucket_size - bucket_filled, remaining)
            if remaining > 0:
                buy_fraction = buy_remaining / remaining
            else:
                buy_fraction = 0.0

            buy_fill = fill * buy_fraction
            sell_fill = fill - buy_fill

            bucket_buy += buy_fill
            bucket_sell += sell_fill
            bucket_filled += fill

            buy_remaining -= buy_fill
            sell_remaining -= sell_fill
            remaining -= fill

            if bucket_filled >= bucket_size - 1.0e-12:
                imbalance = abs(bucket_buy - bucket_sell) / bucket_size
                bucket_imbalances.append(imbalance)
                bucket_end_times.append(minute)
                bucket_size = None
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_filled = 0.0

    if not bucket_imbalances:
        return pd.DataFrame(columns=["vpin"])

    bucket_series = pd.Series(bucket_imbalances, index=pd.DatetimeIndex(bucket_end_times)).sort_index()
    vpin = bucket_series.rolling(window=window_buckets, min_periods=window_buckets).mean()
    return pd.DataFrame({"vpin": vpin}, index=bucket_series.index)


def vpin_features(
    bars_1m: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: Dict,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute VPIN + rolling CDF aligned to 1m bars using backward-only asof alignment.
    """
    bars = _ensure_datetime_index(bars_1m, time_col)
    vpin_df = compute_vpin(
        trades,
        bucket_mult=float(cfg.get("bucket_mult", 1.0)),
        volume_window_minutes=int(cfg.get("volume_window_minutes", 60)),
        window_buckets=int(cfg.get("window_buckets", 20)),
        bucket_seconds=int(cfg.get("bucket_seconds", 60)),
        min_bucket_volume=float(cfg.get("min_bucket_volume", 1.0)),
        trade_price_col=str(cfg.get("trade_price_col", "price")),
        trade_qty_col=str(cfg.get("trade_qty_col", "qty")),
        trade_side_col=cfg.get("trade_side_col"),
        trade_is_buyer_maker_col=str(cfg.get("trade_is_buyer_maker_col", "is_buyer_maker")),
    )

    if vpin_df.empty:
        out = pd.DataFrame(index=bars.index, columns=["vpin", "vpin_cdf"], dtype=float)
        return out

    aligned = _align_to_bars(bars, vpin_df)
    cdf_window = int(cfg.get("cdf_window_buckets", cfg.get("window_buckets", 20)))
    aligned["vpin_cdf"] = _rolling_cdf(aligned["vpin"], cdf_window)
    return aligned

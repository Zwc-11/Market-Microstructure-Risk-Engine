from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    if is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().any():
            median = float(values.dropna().median())
        else:
            median = 0.0
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


def compute_replenishment(
    l2: pd.DataFrame,
    levels: Iterable[int],
    eps: float = 1.0e-9,
    repl_window: int = 3,
    time_col: Optional[str] = None,
    bid_price_prefix: str = "bid_price_",
    ask_price_prefix: str = "ask_price_",
    bid_size_prefix: str = "bid_size_",
    ask_size_prefix: str = "ask_size_",
) -> pd.DataFrame:
    """
    Compute depth, depletion, replenish, repl_ratio and spread/microprice.

    depth_L = sum_{i=1..L}(bid_size_i + ask_size_i)
    depletion = max(0, depth_{t-1} - depth_t)
    replenish = max(0, depth_t - depth_{t-1})
    repl_ratio = rolling_mean(replenish / (depletion + eps), window=repl_window)
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if repl_window <= 0:
        raise ValueError("repl_window must be positive")

    l2 = _ensure_datetime_index(l2, time_col)
    levels = sorted(set(int(x) for x in levels))
    if not levels:
        raise ValueError("levels must be non-empty")

    for lvl in levels:
        for prefix in (bid_price_prefix, ask_price_prefix, bid_size_prefix, ask_size_prefix):
            col = f"{prefix}{lvl}"
            if col not in l2.columns:
                raise ValueError(f"l2 missing required column '{col}'")

    results = pd.DataFrame(index=l2.index)

    for lvl in levels:
        depth = pd.Series(0.0, index=l2.index)
        for i in levels:
            if i <= lvl:
                depth = depth + l2[f"{bid_size_prefix}{i}"].astype(float) + l2[f"{ask_size_prefix}{i}"].astype(float)

        depletion = (depth.shift(1) - depth).clip(lower=0)
        replenish = (depth - depth.shift(1)).clip(lower=0)
        repl_ratio = (replenish / (depletion + eps)).rolling(window=repl_window, min_periods=repl_window).mean()

        suffix = f"_L{lvl}"
        results[f"depth{suffix}"] = depth
        results[f"depletion{suffix}"] = depletion
        results[f"replenish{suffix}"] = replenish
        results[f"repl_ratio{suffix}"] = repl_ratio

    bid = l2[f"{bid_price_prefix}1"].astype(float)
    ask = l2[f"{ask_price_prefix}1"].astype(float)
    bid_size = l2[f"{bid_size_prefix}1"].astype(float)
    ask_size = l2[f"{ask_size_prefix}1"].astype(float)

    results["spread_mean"] = ask - bid
    results["microprice"] = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

    return results


def replenishment_features(
    bars_1m: pd.DataFrame,
    l2: pd.DataFrame,
    cfg: Dict,
    time_col: Optional[str] = None,
    l2_time_col: Optional[str] = None,
    repl_window: int = 3,
) -> pd.DataFrame:
    """
    Compute replenishment features aligned to 1m bars using backward-only asof alignment.
    """
    bars = _ensure_datetime_index(bars_1m, time_col)
    features = compute_replenishment(
        l2,
        levels=[int(cfg.get("levels", 1))] if not isinstance(cfg.get("levels"), list) else cfg["levels"],
        eps=float(cfg.get("eps", 1.0e-9)),
        repl_window=repl_window,
        time_col=l2_time_col,
    )

    outputs = cfg.get("outputs")
    if outputs:
        keep_cols = []
        for col in features.columns:
            base = col.split("_L")[0]
            if base in outputs or col in outputs:
                keep_cols.append(col)
        if keep_cols:
            features = features[keep_cols]

    return _align_to_bars(bars, features)

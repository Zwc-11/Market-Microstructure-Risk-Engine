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


def compute_weighted_obi(
    l2: pd.DataFrame,
    levels: Iterable[int],
    decay: float = 0.7,
    eps: float = 1.0e-9,
    time_col: Optional[str] = None,
    bid_size_prefix: str = "bid_size_",
    ask_size_prefix: str = "ask_size_",
) -> pd.DataFrame:
    """
    Weighted order book imbalance for depth L:

      W_OBI_L = sum_{i=1..L}(w_i * (Vbid_i - Vask_i))
                / (sum_{i=1..L}(w_i * (Vbid_i + Vask_i)) + eps)

    with w_i = decay^i.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if decay <= 0:
        raise ValueError("decay must be positive")

    l2 = _ensure_datetime_index(l2, time_col)
    levels = sorted(set(int(x) for x in levels))
    if not levels:
        raise ValueError("levels must be non-empty")

    for lvl in levels:
        for prefix in (bid_size_prefix, ask_size_prefix):
            col = f"{prefix}{lvl}"
            if col not in l2.columns:
                raise ValueError(f"l2 missing required column '{col}'")

    weights = {lvl: float(decay) ** int(lvl) for lvl in levels}
    results = pd.DataFrame(index=l2.index)

    for lvl in levels:
        num = pd.Series(0.0, index=l2.index)
        den = pd.Series(0.0, index=l2.index)
        for i in levels:
            if i <= lvl:
                weight = weights[i]
                bid = l2[f"{bid_size_prefix}{i}"].astype(float)
                ask = l2[f"{ask_size_prefix}{i}"].astype(float)
                num = num + weight * (bid - ask)
                den = den + weight * (bid + ask)
        results[f"w_obi_L{lvl}"] = num / (den + eps)

    return results


def obi_features(
    bars_1m: pd.DataFrame,
    l2: pd.DataFrame,
    cfg: Dict,
    time_col: Optional[str] = None,
    l2_time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute weighted OBI features aligned to 1m bars using backward-only asof alignment.
    """
    bars = _ensure_datetime_index(bars_1m, time_col)
    obi = compute_weighted_obi(
        l2,
        levels=cfg["levels"],
        decay=float(cfg.get("decay", 0.7)),
        eps=float(cfg.get("eps", 1.0e-9)),
        time_col=l2_time_col,
    )
    return _align_to_bars(bars, obi)

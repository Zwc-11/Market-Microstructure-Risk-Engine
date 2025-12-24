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


def _ofi_level(
    l2: pd.DataFrame,
    level: int,
    bid_price_prefix: str,
    ask_price_prefix: str,
    bid_size_prefix: str,
    ask_size_prefix: str,
) -> pd.Series:
    bp = l2[f"{bid_price_prefix}{level}"].astype(float)
    ap = l2[f"{ask_price_prefix}{level}"].astype(float)
    bs = l2[f"{bid_size_prefix}{level}"].astype(float)
    aS = l2[f"{ask_size_prefix}{level}"].astype(float)

    bp_prev = bp.shift(1)
    ap_prev = ap.shift(1)
    bs_prev = bs.shift(1)
    as_prev = aS.shift(1)

    bid_comp = np.where(
        bp > bp_prev,
        bs,
        np.where(bp == bp_prev, bs - bs_prev, -bs_prev),
    )
    ask_comp = np.where(
        ap < ap_prev,
        aS,
        np.where(ap == ap_prev, aS - as_prev, -as_prev),
    )

    ofi = bid_comp - ask_comp
    return pd.Series(ofi, index=l2.index).fillna(0.0)


def compute_ofi(
    l2: pd.DataFrame,
    levels: Iterable[int],
    normalize_by_depth: bool,
    eps: float = 1.0e-9,
    time_col: Optional[str] = None,
    bid_price_prefix: str = "bid_price_",
    ask_price_prefix: str = "ask_price_",
    bid_size_prefix: str = "bid_size_",
    ask_size_prefix: str = "ask_size_",
) -> pd.DataFrame:
    """
    Compute OFI for multiple depth levels using only past data.

    Formula per level i:
      ofi_i = bid_component_i - ask_component_i
    with bid/ask components following Cont et al. (2014).
    Aggregate:
      OFI_L = sum_{i=1..L} ofi_i
      if normalize_by_depth: OFI_L /= (sum_depth_L + eps)
    """
    if eps <= 0:
        raise ValueError("eps must be positive")

    l2 = _ensure_datetime_index(l2, time_col)
    levels = sorted(set(int(x) for x in levels))
    if not levels:
        raise ValueError("levels must be non-empty")

    for lvl in levels:
        for prefix in (bid_price_prefix, ask_price_prefix, bid_size_prefix, ask_size_prefix):
            col = f"{prefix}{lvl}"
            if col not in l2.columns:
                raise ValueError(f"l2 missing required column '{col}'")

    ofi_by_level = {}
    for lvl in levels:
        ofi_by_level[lvl] = _ofi_level(l2, lvl, bid_price_prefix, ask_price_prefix, bid_size_prefix, ask_size_prefix)

    results = pd.DataFrame(index=l2.index)
    for lvl in levels:
        ofi_sum = pd.Series(0.0, index=l2.index)
        depth_sum = pd.Series(0.0, index=l2.index)
        for i in levels:
            if i <= lvl:
                ofi_sum = ofi_sum + ofi_by_level[i]
                depth_sum = depth_sum + l2[f"{bid_size_prefix}{i}"].astype(float) + l2[
                    f"{ask_size_prefix}{i}"
                ].astype(float)

        if normalize_by_depth:
            ofi_sum = ofi_sum / (depth_sum + eps)

        results[f"ofi_L{lvl}"] = ofi_sum

    return results


def ofi_features(
    bars_1m: pd.DataFrame,
    l2: pd.DataFrame,
    cfg: Dict,
    time_col: Optional[str] = None,
    l2_time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute OFI features aligned to 1m bars using backward-only asof alignment.
    """
    bars = _ensure_datetime_index(bars_1m, time_col)
    ofi = compute_ofi(
        l2,
        levels=cfg["levels"],
        normalize_by_depth=bool(cfg.get("normalize_by_depth", True)),
        eps=float(cfg.get("eps", 1.0e-9)),
        time_col=l2_time_col,
    )
    return _align_to_bars(bars, ofi)

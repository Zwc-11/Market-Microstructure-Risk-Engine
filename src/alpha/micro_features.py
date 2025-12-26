from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from src.alpha.imbalance import obi_features
from src.alpha.kyle_lambda import kyle_lambda_features
from src.alpha.ofi import ofi_features
from src.alpha.vpin import vpin_features


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


def _depth_features(
    bars: pd.DataFrame,
    l2: pd.DataFrame,
    levels: Iterable[int],
    l2_time_col: Optional[str],
    bid_size_prefix: str = "bid_size_",
    ask_size_prefix: str = "ask_size_",
) -> pd.DataFrame:
    l2 = _ensure_datetime_index(l2, l2_time_col)
    levels = sorted(set(int(x) for x in levels))
    if not levels:
        return pd.DataFrame(index=bars.index)

    for lvl in levels:
        for prefix in (bid_size_prefix, ask_size_prefix):
            col = f"{prefix}{lvl}"
            if col not in l2.columns:
                raise ValueError(f"l2 missing required column '{col}'")

    data = {}
    for lvl in levels:
        depth = pd.Series(0.0, index=l2.index)
        for i in levels:
            if i <= lvl:
                depth = depth + l2[f"{bid_size_prefix}{i}"].astype(float) + l2[f"{ask_size_prefix}{i}"].astype(float)
        data[f"depth_L{lvl}"] = depth
    depth_df = pd.DataFrame(data, index=l2.index)
    return _align_to_bars(bars, depth_df)


def _spread_feature(
    bars: pd.DataFrame,
    l2: pd.DataFrame,
    l2_time_col: Optional[str],
    bid_price_prefix: str = "bid_price_",
    ask_price_prefix: str = "ask_price_",
) -> pd.DataFrame:
    l2 = _ensure_datetime_index(l2, l2_time_col)
    bid_col = f"{bid_price_prefix}1"
    ask_col = f"{ask_price_prefix}1"
    if bid_col not in l2.columns or ask_col not in l2.columns:
        raise ValueError("l2 missing bid/ask price columns for spread feature")
    spread = (l2[ask_col].astype(float) - l2[bid_col].astype(float)).to_frame("spread")
    return _align_to_bars(bars, spread)


def _mid_change_feature(bars: pd.DataFrame) -> pd.DataFrame:
    if "mid_close" in bars.columns:
        series = bars["mid_close"].astype(float)
    elif "close" in bars.columns:
        series = bars["close"].astype(float)
    else:
        raise ValueError("bars must include mid_close or close for mid_change feature")
    return pd.DataFrame({"mid_change": series.diff()}, index=bars.index)


def build_micro_features(
    bars_1m: pd.DataFrame,
    l2: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: Dict,
    time_col: Optional[str] = None,
    l2_time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build 1m microstructure feature frame (OFI/OBI/Kyle/VPIN) aligned to bars.
    """
    bars = _ensure_datetime_index(bars_1m, time_col)
    features = []

    ofi_cfg = cfg.get("ofi", {})
    if ofi_cfg.get("enabled", True):
        ofi_out = ofi_features(bars, l2, ofi_cfg, time_col=None, l2_time_col=l2_time_col)
        rename = {}
        for col in ofi_out.columns:
            if col.startswith("ofi_L") and col.endswith("_z"):
                level = col[len("ofi_L") : -2]
                rename[col] = f"ofi_z_L{level}"
        if rename:
            ofi_out = ofi_out.rename(columns=rename)
        features.append(ofi_out)

    obi_cfg = cfg.get("obi", {})
    if obi_cfg.get("enabled", True):
        obi_out = obi_features(bars, l2, obi_cfg, time_col=None, l2_time_col=l2_time_col)
        features.append(obi_out)

    kyle_cfg = cfg.get("kyle_lambda", {})
    if kyle_cfg.get("enabled", True):
        kyle_out = kyle_lambda_features(bars, trades, kyle_cfg, time_col=None)
        keep = kyle_cfg.get("outputs")
        if keep:
            kyle_out = kyle_out[[col for col in keep if col in kyle_out.columns]]
        features.append(kyle_out)

    vpin_cfg = cfg.get("vpin", {})
    if vpin_cfg.get("enabled", True):
        vpin_out = vpin_features(bars, trades, vpin_cfg, time_col=None)
        features.append(vpin_out)

    extras_cfg = cfg.get("extras", {})
    if extras_cfg.get("spread", False):
        features.append(_spread_feature(bars, l2, l2_time_col))
    if extras_cfg.get("mid_change", False):
        features.append(_mid_change_feature(bars))
    if extras_cfg.get("depth_levels"):
        features.append(_depth_features(bars, l2, extras_cfg.get("depth_levels", []), l2_time_col))

    if not features:
        raise ValueError("No microstructure features enabled")

    out = pd.concat(features, axis=1)
    return out.reindex(bars.index)

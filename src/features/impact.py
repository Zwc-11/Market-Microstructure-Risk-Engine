from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in data")
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
            raise ValueError("data must have a DatetimeIndex or timestamp/ts column")

    return df.sort_index()


def _get_price_col(bars: pd.DataFrame, price_col: Optional[str]) -> str:
    if price_col is not None:
        if price_col not in bars.columns:
            raise ValueError(f"price_col '{price_col}' not found in bars")
        return price_col
    if "mid_close" in bars.columns:
        return "mid_close"
    if "close" in bars.columns:
        return "close"
    raise ValueError("bars must include mid_close or close")


def compute_kyle_lambda(
    bars_1m: pd.DataFrame,
    signed_flow: pd.Series,
    window: int,
    separate_up_down: bool = False,
    time_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Kyle lambda via rolling regression: delta_mid ~ signed_flow.

    lambda_t = cov(x, y) / var(x)
    r2_t = cov(x, y)^2 / (var(x) * var(y))
    resid_std_t = std(y - lambda*x)
    """
    if window <= 0:
        raise ValueError("window must be positive")

    bars = _ensure_datetime_index(bars_1m, time_col)
    price_col = _get_price_col(bars, price_col)
    price = bars[price_col].astype(float)

    y = price.diff()
    x = signed_flow.reindex(bars.index).astype(float)

    cov_xy = x.rolling(window=window, min_periods=window).cov(y)
    var_x = x.rolling(window=window, min_periods=window).var(ddof=0)
    var_y = y.rolling(window=window, min_periods=window).var(ddof=0)

    lambda_all = cov_xy / var_x
    r2 = (cov_xy * cov_xy) / (var_x * var_y)
    resid = y - (lambda_all * x)
    resid_std = resid.rolling(window=window, min_periods=window).std(ddof=0)

    out = pd.DataFrame(
        {
            "lambda": lambda_all,
            "r2": r2,
            "resid_std": resid_std,
        },
        index=bars.index,
    )

    if separate_up_down:
        y_up = y.where(y > 0)
        x_up = x.where(y > 0)
        y_dn = y.where(y < 0)
        x_dn = x.where(y < 0)

        cov_up = x_up.rolling(window=window, min_periods=window).cov(y_up)
        var_up = x_up.rolling(window=window, min_periods=window).var(ddof=0)
        cov_dn = x_dn.rolling(window=window, min_periods=window).cov(y_dn)
        var_dn = x_dn.rolling(window=window, min_periods=window).var(ddof=0)

        out["lambda_up"] = cov_up / var_up
        out["lambda_down"] = cov_dn / var_dn

    return out


def kyle_lambda_features(
    bars_1m: pd.DataFrame,
    signed_flow: pd.Series,
    cfg: Dict,
    time_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Wrapper for Kyle lambda features using config parameters.
    """
    window = int(cfg["window_minutes"])
    separate = bool(cfg.get("separate_up_down", False))

    features = compute_kyle_lambda(
        bars_1m,
        signed_flow,
        window=window,
        separate_up_down=separate,
        time_col=time_col,
        price_col=price_col,
    )

    outputs = cfg.get("outputs")
    if outputs:
        keep = [col for col in outputs if col in features.columns]
        return features[keep]
    return features

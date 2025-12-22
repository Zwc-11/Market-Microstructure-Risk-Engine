from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np
import pandas as pd


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


def _coerce_regime_series(regime: pd.DataFrame | pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if isinstance(regime, pd.Series):
        regime_series = regime
    else:
        if "regime" not in regime.columns:
            raise ValueError("regime DataFrame must include 'regime' column")
        regime_series = regime["regime"]

    if not isinstance(regime_series.index, pd.DatetimeIndex):
        if isinstance(regime, pd.DataFrame):
            if "timestamp" in regime.columns:
                regime_series = regime.set_index(pd.to_datetime(regime["timestamp"]))["regime"]
            elif "ts" in regime.columns:
                regime_series = regime.set_index(pd.to_datetime(regime["ts"]))["regime"]
            else:
                raise ValueError("regime must have DatetimeIndex or timestamp/ts column")
        else:
            raise ValueError("regime must have DatetimeIndex or timestamp/ts column")

    return regime_series.sort_index().reindex(index)


def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    pv = (price * volume).rolling(window=window, min_periods=window).sum()
    vol = volume.rolling(window=window, min_periods=window).sum()
    return pv / vol


def _rolling_std(price: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    return price.rolling(window=window, min_periods=window).std(ddof=0)


def _compute_atr(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _event_id(symbol: Optional[str], entry_ts: pd.Timestamp, side: int, reason: str) -> str:
    symbol_val = "" if symbol is None else str(symbol)
    payload = f"{symbol_val}|{entry_ts.isoformat()}|{side}|{reason}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def generate_entries_5m(
    bars: pd.DataFrame,
    regime: pd.DataFrame | pd.Series,
    entries_cfg: Dict,
    symbol: Optional[str] = None,
    time_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate 5m entry events from regime labels with no lookahead.

    RANGE:
      center_t = rolling_vwap(price, volume, window)
      band_t = k * rolling_std(price, window)
      upper = center + band, lower = center - band
      if price >= upper => short, if price <= lower => long

    TREND:
      ema_fast/ema_slow on price
      bias long if price > ema_slow, short if price < ema_slow
      pullback when |price - ema_fast| <= tolerance_atr * ATR
      bounce long if price_t > price_{t-1}, short if price_t < price_{t-1}
    """
    bars = _ensure_datetime_index(bars, time_col)
    price_col = _get_price_col(bars, price_col)
    price = bars[price_col].astype(float)
    regime_series = _coerce_regime_series(regime, bars.index)

    range_cfg = entries_cfg.get("range", {})
    trend_cfg = entries_cfg.get("trend", {})

    range_enabled = bool(range_cfg.get("enabled", True))
    trend_enabled = bool(trend_cfg.get("enabled", True))

    range_cooldown = int(range_cfg.get("cooldown_5m_bars", 1))
    trend_cooldown = int(trend_cfg.get("cooldown_5m_bars", range_cooldown))

    center = pd.Series(index=bars.index, dtype="float64")
    band = pd.Series(index=bars.index, dtype="float64")
    upper = pd.Series(index=bars.index, dtype="float64")
    lower = pd.Series(index=bars.index, dtype="float64")

    if range_enabled:
        center_cfg = range_cfg.get("center", {})
        band_cfg = range_cfg.get("band", {})

        center_kind = center_cfg.get("kind", "rolling_vwap")
        center_window = int(center_cfg.get("window_5m_bars", 12))
        band_kind = band_cfg.get("kind", "rolling_std_mid")
        band_window = int(band_cfg.get("window_5m_bars", 12))
        band_k = float(band_cfg.get("k", 1.0))

        if center_kind == "rolling_vwap":
            if "volume" not in bars.columns:
                raise ValueError("bars must include volume for rolling_vwap center")
            center = _rolling_vwap(price, bars["volume"].astype(float), center_window)
        elif center_kind == "ema":
            center = price.ewm(span=center_window, adjust=False).mean()
        else:
            raise ValueError(f"Unknown center kind '{center_kind}'")

        if band_kind == "rolling_std_mid":
            band = band_k * _rolling_std(price, band_window)
        else:
            raise ValueError(f"Unknown band kind '{band_kind}'")

        upper = center + band
        lower = center - band

    ema_fast = pd.Series(index=bars.index, dtype="float64")
    ema_slow = pd.Series(index=bars.index, dtype="float64")
    atr = pd.Series(index=bars.index, dtype="float64")

    if trend_enabled:
        ema_fast_bars = int(trend_cfg.get("ema_fast_bars", 12))
        ema_slow_bars = int(trend_cfg.get("ema_slow_bars", 48))
        pullback_cfg = trend_cfg.get("pullback", {})
        atr_window = int(pullback_cfg.get("atr_window_5m_bars", 14))
        tolerance_atr = float(pullback_cfg.get("tolerance_atr", 0.3))

        ema_fast = price.ewm(span=ema_fast_bars, adjust=False).mean()
        ema_slow = price.ewm(span=ema_slow_bars, adjust=False).mean()

        if "mid_high" in bars.columns:
            high = bars["mid_high"].astype(float)
            low = bars["mid_low"].astype(float)
        else:
            high = price
            low = price

        atr = _compute_atr(price, high, low, atr_window)

    events = []
    last_range_idx = None
    last_trend_idx = None

    for idx, ts in enumerate(bars.index):
        regime_val = regime_series.iloc[idx]
        if pd.isna(regime_val) or regime_val == "HAZARD":
            continue

        if regime_val == "RANGE" and range_enabled:
            if pd.isna(upper.iloc[idx]) or pd.isna(lower.iloc[idx]):
                continue

            if last_range_idx is not None and idx - last_range_idx <= range_cooldown:
                continue

            entry_price = float(price.iloc[idx])
            reason = None
            side = None

            if entry_price >= float(upper.iloc[idx]):
                side = -1
                reason = "range_upper_touch"
            elif entry_price <= float(lower.iloc[idx]):
                side = 1
                reason = "range_lower_touch"

            if reason is None:
                continue

            event = {
                "event_id": _event_id(symbol, ts, side, reason),
                "symbol": symbol,
                "entry_ts": ts,
                "side": side,
                "entry_price": entry_price,
                "regime": "RANGE",
                "reason": reason,
                "center": float(center.iloc[idx]),
                "band": float(band.iloc[idx]),
                "upper": float(upper.iloc[idx]),
                "lower": float(lower.iloc[idx]),
                "ema_fast": float(ema_fast.iloc[idx]) if trend_enabled else np.nan,
                "ema_slow": float(ema_slow.iloc[idx]) if trend_enabled else np.nan,
                "atr": float(atr.iloc[idx]) if trend_enabled else np.nan,
            }
            events.append(event)
            last_range_idx = idx

        elif regime_val == "TREND" and trend_enabled:
            if pd.isna(ema_fast.iloc[idx]) or pd.isna(ema_slow.iloc[idx]) or pd.isna(atr.iloc[idx]):
                continue

            if last_trend_idx is not None and idx - last_trend_idx <= trend_cooldown:
                continue

            if idx == 0:
                continue

            entry_price = float(price.iloc[idx])
            prev_price = float(price.iloc[idx - 1])
            atr_val = float(atr.iloc[idx])
            if atr_val <= 0 or np.isnan(atr_val):
                continue

            tolerance = float(tolerance_atr) * atr_val
            near_fast = abs(entry_price - float(ema_fast.iloc[idx])) <= tolerance

            reason = None
            side = None

            if entry_price > float(ema_slow.iloc[idx]):
                if near_fast and entry_price > prev_price:
                    side = 1
                    reason = "trend_pullback_long"
            elif entry_price < float(ema_slow.iloc[idx]):
                if near_fast and entry_price < prev_price:
                    side = -1
                    reason = "trend_pullback_short"

            if reason is None:
                continue

            event = {
                "event_id": _event_id(symbol, ts, side, reason),
                "symbol": symbol,
                "entry_ts": ts,
                "side": side,
                "entry_price": entry_price,
                "regime": "TREND",
                "reason": reason,
                "center": float(center.iloc[idx]) if range_enabled else np.nan,
                "band": float(band.iloc[idx]) if range_enabled else np.nan,
                "upper": float(upper.iloc[idx]) if range_enabled else np.nan,
                "lower": float(lower.iloc[idx]) if range_enabled else np.nan,
                "ema_fast": float(ema_fast.iloc[idx]),
                "ema_slow": float(ema_slow.iloc[idx]),
                "atr": atr_val,
            }
            events.append(event)
            last_trend_idx = idx

    columns = [
        "event_id",
        "symbol",
        "entry_ts",
        "side",
        "entry_price",
        "regime",
        "reason",
        "center",
        "band",
        "upper",
        "lower",
        "ema_fast",
        "ema_slow",
        "atr",
    ]
    return pd.DataFrame(events, columns=columns)

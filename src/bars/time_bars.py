from __future__ import annotations

from typing import Optional

import pandas as pd


def _ensure_time_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in data")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("data must have a DatetimeIndex or a valid time_col")

    return df.sort_index()


def build_time_bars(
    trades: pd.DataFrame,
    l2: Optional[pd.DataFrame] = None,
    freq: str = "1min",
    time_col: str = "timestamp",
    price_col: str = "price",
    qty_col: str = "qty",
    l2_time_col: str = "timestamp",
    bid_col: str = "bid",
    ask_col: str = "ask",
    bid_size_col: str = "bid_size",
    ask_size_col: str = "ask_size",
) -> pd.DataFrame:
    """
    Build time bars using only data <= bar end (right-closed windows).

    Trades:
      volume_t = sum(qty)
      vwap_t = sum(price * qty) / sum(qty)

    L2 snapshots (if present):
      mid = (bid + ask) / 2
      spread = ask - bid
      microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

    Mid OHLC is computed from mid (or trade price if L2 is absent).
    """
    trades = _ensure_time_index(trades, time_col)
    if price_col not in trades.columns or qty_col not in trades.columns:
        raise ValueError("trades must include price_col and qty_col")

    price = trades[price_col].astype(float)
    qty = trades[qty_col].astype(float)

    resample_kwargs = {"label": "right", "closed": "right"}
    volume = qty.resample(freq, **resample_kwargs).sum()
    vwap = (price * qty).resample(freq, **resample_kwargs).sum() / volume

    if l2 is not None and not l2.empty:
        l2 = _ensure_time_index(l2, l2_time_col)
        required = {bid_col, ask_col, bid_size_col, ask_size_col}
        missing = required - set(l2.columns)
        if missing:
            raise ValueError(f"l2 missing required columns: {sorted(missing)}")

        bid = l2[bid_col].astype(float)
        ask = l2[ask_col].astype(float)
        bid_size = l2[bid_size_col].astype(float)
        ask_size = l2[ask_size_col].astype(float)

        mid = (bid + ask) / 2.0
        spread = ask - bid
        microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

        mid_ohlc = mid.resample(freq, **resample_kwargs).ohlc()
        spread_mean = spread.resample(freq, **resample_kwargs).mean()
        microprice_close = microprice.resample(freq, **resample_kwargs).last()
    else:
        mid_ohlc = price.resample(freq, **resample_kwargs).ohlc()
        spread_mean = None
        microprice_close = None

    bars = mid_ohlc.rename(
        columns={
            "open": "mid_open",
            "high": "mid_high",
            "low": "mid_low",
            "close": "mid_close",
        }
    )

    bars = pd.concat([bars, volume.rename("volume"), vwap.rename("vwap")], axis=1)

    if spread_mean is not None and microprice_close is not None:
        bars = pd.concat(
            [
                bars,
                spread_mean.rename("spread_mean"),
                microprice_close.rename("microprice_close"),
            ],
            axis=1,
        )

    return bars


def resample_time_bars(bars: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """
    Resample time bars using right-closed windows (no lookahead).

    Aggregations:
      - mid_open: first
      - mid_high: max
      - mid_low: min
      - mid_close: last
      - volume: sum
      - vwap: sum(vwap * volume) / sum(volume)
      - spread_mean: mean (if present)
      - microprice_close: last (if present)
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex")

    required = {"mid_open", "mid_high", "mid_low", "mid_close", "volume", "vwap"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    bars = bars.sort_index()
    resample_kwargs = {"label": "right", "closed": "right"}

    mid_open = bars["mid_open"].resample(freq, **resample_kwargs).first()
    mid_high = bars["mid_high"].resample(freq, **resample_kwargs).max()
    mid_low = bars["mid_low"].resample(freq, **resample_kwargs).min()
    mid_close = bars["mid_close"].resample(freq, **resample_kwargs).last()

    volume = bars["volume"].resample(freq, **resample_kwargs).sum()
    vwap = (bars["vwap"] * bars["volume"]).resample(freq, **resample_kwargs).sum() / volume

    out = pd.concat(
        [
            mid_open.rename("mid_open"),
            mid_high.rename("mid_high"),
            mid_low.rename("mid_low"),
            mid_close.rename("mid_close"),
            volume.rename("volume"),
            vwap.rename("vwap"),
        ],
        axis=1,
    )

    if "spread_mean" in bars.columns:
        spread_mean = bars["spread_mean"].resample(freq, **resample_kwargs).mean()
        out = pd.concat([out, spread_mean.rename("spread_mean")], axis=1)

    if "microprice_close" in bars.columns:
        microprice_close = bars["microprice_close"].resample(freq, **resample_kwargs).last()
        out = pd.concat([out, microprice_close.rename("microprice_close")], axis=1)

    return out

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.labeling.triple_barrier import compute_volatility


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


def _price_col(bars: pd.DataFrame, price_col: Optional[str]) -> str:
    if price_col is not None:
        if price_col not in bars.columns:
            raise ValueError(f"price_col '{price_col}' not found in bars")
        return price_col
    if "mid_close" in bars.columns:
        return "mid_close"
    if "close" in bars.columns:
        return "close"
    raise ValueError("bars must include mid_close or close")


def _compute_sl_price(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    side: int,
    cfg: Dict,
) -> float:
    price_col = cfg["labeling"]["triple_barrier"]["price_source"]
    price_col = "mid_close" if price_col == "mid_close" else "close"

    vol_cfg = cfg["labeling"]["triple_barrier"]["vol"]
    sigma = compute_volatility(
        bars,
        price_col=price_col,
        kind=vol_cfg["kind"],
        window=int(vol_cfg["window_1m_bars"]),
        min_sigma=float(vol_cfg["min_sigma"]),
    )

    idx = bars.index.searchsorted(entry_ts, side="right") - 1
    if idx < 0:
        raise ValueError("entry_ts precedes available bars")
    entry_time = bars.index[idx]
    sigma_t0 = float(sigma.loc[entry_time])
    sl_mult = float(cfg["labeling"]["triple_barrier"]["barriers"]["sl_mult"])

    if side > 0:
        return entry_price * (1.0 - sl_mult * sigma_t0)
    return entry_price * (1.0 + sl_mult * sigma_t0)


def build_hazard_dataset(
    trades: pd.DataFrame,
    bars_1m: pd.DataFrame,
    config: Dict,
    time_col: Optional[str] = None,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build per-minute hazard labels for active trades.

    For each trade and minute t in (entry_ts, exit_ts], label:
      y_t = 1 if the adverse barrier (SL) is touched within (t, t+horizon] else 0.

    If future bars needed for labeling are missing, y is NaN.
    """
    required_cols = {"event_id", "entry_ts", "exit_ts", "side", "entry_price", "symbol"}
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing required columns: {sorted(missing)}")

    bars = _ensure_datetime_index(bars_1m, time_col)
    price_col = _price_col(bars, price_col)

    horizon = int(config["hazard"]["horizon_minutes"])

    rows = []
    for _, trade in trades.iterrows():
        event_id = trade["event_id"]
        symbol = trade.get("symbol")
        entry_ts = pd.Timestamp(trade["entry_ts"])
        exit_ts = pd.Timestamp(trade["exit_ts"])
        side = int(trade["side"])
        entry_price = float(trade["entry_price"])

        if "sl_price" in trade:
            sl_price = float(trade["sl_price"])
        else:
            sl_price = _compute_sl_price(bars, entry_ts, entry_price, side, config)

        t = entry_ts + pd.Timedelta(minutes=1)
        while t <= exit_ts:
            horizon_end = min(t + pd.Timedelta(minutes=horizon), exit_ts)
            window = bars.loc[(bars.index > t) & (bars.index <= horizon_end)]

            if window.empty:
                y = np.nan
                future_min_low = np.nan
                future_max_high = np.nan
            else:
                if "mid_low" in bars.columns:
                    future_min_low = float(window["mid_low"].min())
                    future_max_high = float(window["mid_high"].max())
                else:
                    future_min_low = float(window[price_col].min())
                    future_max_high = float(window[price_col].max())

                if side > 0:
                    y = 1 if future_min_low <= sl_price else 0
                else:
                    y = 1 if future_max_high >= sl_price else 0

            mid_close_t = float(bars.loc[t, price_col]) if t in bars.index else np.nan

            rows.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "t": t,
                    "side": side,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "entry_price": entry_price,
                    "mid_close_t": mid_close_t,
                    "y": y,
                    "horizon_end_ts": horizon_end,
                    "future_min_low": future_min_low,
                    "future_max_high": future_max_high,
                }
            )

            t += pd.Timedelta(minutes=1)

    return pd.DataFrame(rows)

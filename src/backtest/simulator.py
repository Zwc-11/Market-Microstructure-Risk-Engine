from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_summary
from src.labeling.triple_barrier import triple_barrier_labels


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


def _prepare_tb_bars(bars: pd.DataFrame, price_col: str) -> pd.DataFrame:
    price = bars[price_col].astype(float)
    if "mid_open" in bars.columns:
        open_px = bars["mid_open"].astype(float)
        high_px = bars["mid_high"].astype(float)
        low_px = bars["mid_low"].astype(float)
    else:
        open_px = price
        high_px = price
        low_px = price

    out = pd.DataFrame(
        {
            "open": open_px,
            "high": high_px,
            "low": low_px,
            price_col: price,
        },
        index=bars.index,
    )
    return out


def _asof_price(bars: pd.DataFrame, ts: pd.Timestamp, price_col: str) -> float:
    idx = bars.index
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        raise ValueError("timestamp precedes available bars")
    return float(bars.iloc[pos][price_col])


def _apply_latency(
    entry_ts: pd.Timestamp, bars: pd.DataFrame, price_col: str, latency_ms: int
) -> Tuple[pd.Timestamp, float, int]:
    if latency_ms <= 0:
        return entry_ts, float(bars.loc[entry_ts, price_col]), 0

    if len(bars.index) < 2:
        return entry_ts, float(bars.loc[entry_ts, price_col]), 0

    bar_delta_ms = float(bars.index.to_series().diff().median().total_seconds() * 1000)
    if np.isnan(bar_delta_ms) or bar_delta_ms <= 0:
        return entry_ts, float(bars.loc[entry_ts, price_col]), 0

    if latency_ms < bar_delta_ms:
        return entry_ts, float(bars.loc[entry_ts, price_col]), 0

    pos = bars.index.searchsorted(entry_ts, side="right")
    if pos >= len(bars.index):
        raise ValueError("latency shifts entry beyond available bars")

    delayed_ts = bars.index[pos]
    return delayed_ts, float(bars.loc[delayed_ts, price_col]), 1


def run_backtest(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: Dict,
    price_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    bars = _ensure_datetime_index(bars, None)
    price_col = _get_price_col(bars, price_col)

    if events.empty:
        equity = pd.DataFrame({"equity": [config["backtest"]["initial_capital"]]}, index=[bars.index[0]])
        summary = compute_summary(pd.DataFrame(), equity, config["backtest"]["initial_capital"])
        return pd.DataFrame(), equity, summary

    events = events.copy()
    if "entry_ts" not in events.columns:
        raise ValueError("events must include entry_ts")
    if "side" not in events.columns:
        raise ValueError("events must include side")

    events["t0"] = pd.to_datetime(events["entry_ts"])
    events = events.sort_values("t0").reset_index(drop=True)

    tb_cfg = config["labeling"]["triple_barrier"]
    horizon_minutes = int(tb_cfg["horizon_minutes"])
    vol_cfg = tb_cfg["vol"]
    barrier_cfg = tb_cfg["barriers"]

    bars_tb = _prepare_tb_bars(bars, price_col)

    latency_ms = int(config["backtest"].get("latency_ms", 0))
    latency_bars = 0
    for idx in range(len(events)):
        entry_ts = events.at[idx, "t0"]
        delayed_ts, delayed_price, delayed_bars = _apply_latency(entry_ts, bars_tb, price_col, latency_ms)
        events.at[idx, "t0"] = delayed_ts
        events.at[idx, "entry_ts"] = delayed_ts
        events.at[idx, "entry_price"] = delayed_price
        latency_bars = max(latency_bars, delayed_bars)

    labels = triple_barrier_labels(
        events,
        bars_tb,
        horizon_minutes=horizon_minutes,
        pt_mult=float(barrier_cfg["pt_mult"]),
        sl_mult=float(barrier_cfg["sl_mult"]),
        price_col=price_col,
        tie_break=tb_cfg["tie_break"]["mode"],
        vol_kind=vol_cfg["kind"],
        vol_window=int(vol_cfg["window_1m_bars"]),
        min_sigma=float(vol_cfg["min_sigma"]),
    )

    merged = events.join(labels[["t1", "label", "pt_price", "sl_price", "event_type"]])

    backtest_cfg = config["backtest"]
    initial_capital = float(backtest_cfg["initial_capital"])
    leverage = float(backtest_cfg.get("leverage", 1.0))
    max_notional_pct = float(backtest_cfg["sizing"]["max_position_notional_pct"])
    fee_rate = float(backtest_cfg["fees_bps"]["taker"]) / 10000.0
    slippage_rate = float(backtest_cfg.get("slippage_bps", 0.0)) / 10000.0

    trades = []
    last_exit = None

    for _, row in merged.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        if last_exit is not None and entry_ts < last_exit:
            continue

        side = int(row["side"])
        entry_price = float(row["entry_price"])
        if not np.isfinite(entry_price):
            entry_price = _asof_price(bars_tb, entry_ts, price_col)

        event_type = row["event_type"]
        if event_type == "pt":
            exit_price = float(row["pt_price"])
        elif event_type == "sl":
            exit_price = float(row["sl_price"])
        else:
            exit_price = _asof_price(bars_tb, pd.Timestamp(row["t1"]), price_col)

        exit_ts = pd.Timestamp(row["t1"])

        notional = initial_capital * leverage * max_notional_pct
        qty = notional / entry_price
        gross_pnl = side * (exit_price - entry_price) * qty

        fees = 2.0 * notional * fee_rate
        slippage = 2.0 * notional * slippage_rate
        net_pnl = gross_pnl - fees - slippage

        trades.append(
            {
                "event_id": row.get("event_id"),
                "symbol": row.get("symbol"),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fees": fees,
                "slippage": slippage,
                "notional": notional,
                "label": int(row["label"]),
                "event_type": event_type,
                "regime": row.get("regime"),
                "reason": row.get("reason"),
                "latency_bars": latency_bars,
            }
        )

        last_exit = exit_ts

    trades_df = pd.DataFrame(trades)

    pnl_series = pd.Series(0.0, index=bars.index)
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            exit_idx = pnl_series.index.searchsorted(trade["exit_ts"], side="right") - 1
            if exit_idx >= 0:
                pnl_series.iloc[exit_idx] += trade["net_pnl"]

    equity = initial_capital + pnl_series.cumsum()
    equity_df = pd.DataFrame({"equity": equity}, index=bars.index)

    summary = compute_summary(trades_df, equity_df, initial_capital)
    return trades_df, equity_df, summary

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_summary
from src.labeling.triple_barrier import triple_barrier_labels
from src.strategy.policy_1m import evaluate_hazard_policy


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
                "exit_reason": f"barrier_{event_type}",
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


def _align_features_to_bars(bars_1m: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    bars_idx = bars_1m.index
    bars_df = pd.DataFrame({"ts": bars_idx}).sort_values("ts")
    if "t" in features.columns:
        feat_df = features.copy()
        feat_df["ts"] = pd.to_datetime(feat_df["t"])
    elif "timestamp" in features.columns:
        feat_df = features.copy()
        feat_df["ts"] = pd.to_datetime(feat_df["timestamp"])
    elif "ts" in features.columns:
        feat_df = features.copy()
        feat_df["ts"] = pd.to_datetime(feat_df["ts"])
    else:
        feat_df = features.reset_index().rename(columns={features.index.name or "index": "ts"})
    feat_df = feat_df.sort_values("ts")
    merged = pd.merge_asof(bars_df, feat_df, on="ts", direction="backward")
    merged = merged.set_index("ts")
    return merged.reindex(bars_idx)


def _predict_hazard_proba(model_pack: Dict, features: pd.DataFrame) -> pd.Series:
    feature_names = model_pack.get("features", [])
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    x = features[feature_names].astype(float).to_numpy()
    mean = model_pack.get("mean")
    std = model_pack.get("std")
    clip = model_pack.get("clip_zscore")

    if mean is not None and std is not None:
        mean = np.asarray(mean)
        std = np.asarray(std)
        std = np.where(std == 0, 1.0, std)
        x = (x - mean) / std
        if clip is not None:
            x = np.clip(x, -float(clip), float(clip))

    w = model_pack.get("model")
    if isinstance(w, np.ndarray):
        xb = np.hstack([np.ones((x.shape[0], 1)), x])
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
    elif hasattr(w, "predict_proba"):
        p = w.predict_proba(x)[:, 1]
    else:
        raise ValueError("Unsupported model type in hazard model pack")

    calibrator = model_pack.get("calibrator")
    if calibrator is not None:
        p = calibrator.predict(p)

    return pd.Series(p, index=features.index)


def run_backtest_enhanced(
    bars_5m: pd.DataFrame,
    events: pd.DataFrame,
    config: Dict,
    bars_1m: pd.DataFrame,
    hazard_features: Optional[pd.DataFrame] = None,
    hazard_model: Optional[Dict] = None,
    hazard_prob: Optional[pd.Series] = None,
    policy_mode: str = "full_policy",
    price_col_5m: Optional[str] = None,
    price_col_1m: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float]]:
    bars_5m = _ensure_datetime_index(bars_5m, None)
    bars_1m = _ensure_datetime_index(bars_1m, None)

    price_col_5m = _get_price_col(bars_5m, price_col_5m)
    price_col_1m = _get_price_col(bars_1m, price_col_1m)

    if hazard_prob is None:
        if hazard_model is None or hazard_features is None:
            print("WARNING: hazard model or features missing; falling back to baseline.")
            trades_df, equity_df, summary = run_backtest(bars_5m, events, config, price_col_5m)
            return trades_df, equity_df, summary, {"hazard_enabled": 0}

        features_aligned = _align_features_to_bars(bars_1m, hazard_features)
        try:
            hazard_prob = _predict_hazard_proba(hazard_model, features_aligned)
        except ValueError as exc:
            print(f"WARNING: {exc}; falling back to baseline.")
            trades_df, equity_df, summary = run_backtest(bars_5m, events, config, price_col_5m)
            return trades_df, equity_df, summary, {"hazard_enabled": 0}

    hazard_prob = hazard_prob.reindex(bars_1m.index).ffill()

    if events.empty:
        equity = pd.DataFrame({"equity": [config["backtest"]["initial_capital"]]}, index=[bars_5m.index[0]])
        summary = compute_summary(pd.DataFrame(), equity, config["backtest"]["initial_capital"])
        return pd.DataFrame(), equity, summary, {"hazard_enabled": 1}

    events = events.copy()
    events["t0"] = pd.to_datetime(events["entry_ts"])
    events = events.sort_values("t0").reset_index(drop=True)

    tb_cfg = config["labeling"]["triple_barrier"]
    horizon_minutes = int(tb_cfg["horizon_minutes"])
    vol_cfg = tb_cfg["vol"]
    barrier_cfg = tb_cfg["barriers"]

    bars_tb = _prepare_tb_bars(bars_5m, price_col_5m)
    labels = triple_barrier_labels(
        events,
        bars_tb,
        horizon_minutes=horizon_minutes,
        pt_mult=float(barrier_cfg["pt_mult"]),
        sl_mult=float(barrier_cfg["sl_mult"]),
        price_col=price_col_5m,
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
    hazard_exit_count = 0
    fail_fast_count = 0

    for _, row in merged.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        if last_exit is not None and entry_ts < last_exit:
            continue

        side = int(row["side"])
        entry_price = float(row["entry_price"])
        if not np.isfinite(entry_price):
            entry_price = _asof_price(bars_5m, entry_ts, price_col_5m)

        base_exit_ts = pd.Timestamp(row["t1"])
        base_event_type = row["event_type"]

        if base_event_type == "pt":
            base_exit_price = float(row["pt_price"])
        elif base_event_type == "sl":
            base_exit_price = float(row["sl_price"])
        else:
            base_exit_price = _asof_price(bars_5m, base_exit_ts, price_col_5m)

        start_idx = bars_1m.index.searchsorted(entry_ts, side="left")
        end_idx = bars_1m.index.searchsorted(base_exit_ts, side="right")
        hazard_index = bars_1m.index[start_idx:end_idx]

        if hazard_index.empty:
            exit_ts = base_exit_ts
            exit_price = base_exit_price
            exit_reason = f"barrier_{base_event_type}"
        else:
            p_series = hazard_prob.loc[hazard_index]
            features_slice = None
            if hazard_features is not None:
                features_slice = _align_features_to_bars(bars_1m.loc[hazard_index], hazard_features)

            hazard_ts, hazard_reason, diag = evaluate_hazard_policy(
                p_series, features_slice, config, mode=policy_mode
            )

            if hazard_ts is not None and hazard_ts < base_exit_ts:
                exit_ts = hazard_ts
                exit_price = _asof_price(bars_1m, hazard_ts, price_col_1m)
                exit_reason = hazard_reason
                if hazard_reason == "hazard_exit":
                    hazard_exit_count += 1
                elif hazard_reason == "hazard_fail_fast":
                    fail_fast_count += 1
            else:
                exit_ts = base_exit_ts
                exit_price = base_exit_price
                exit_reason = f"barrier_{base_event_type}"

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
                "event_type": base_event_type,
                "exit_reason": exit_reason,
                "regime": row.get("regime"),
                "reason": row.get("reason"),
                "latency_bars": 0,
            }
        )

        last_exit = exit_ts

    trades_df = pd.DataFrame(trades)

    pnl_series = pd.Series(0.0, index=bars_1m.index)
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            exit_idx = pnl_series.index.searchsorted(trade["exit_ts"], side="right") - 1
            if exit_idx >= 0:
                pnl_series.iloc[exit_idx] += trade["net_pnl"]

    equity = initial_capital + pnl_series.cumsum()
    equity_df = pd.DataFrame({"equity": equity}, index=bars_1m.index)

    summary = compute_summary(trades_df, equity_df, initial_capital)
    diagnostics = {"hazard_enabled": 1, "hazard_exits": hazard_exit_count, "fail_fast_exits": fail_fast_count}
    return trades_df, equity_df, summary, diagnostics

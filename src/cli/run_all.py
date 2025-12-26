from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    compute_hazard_exit_counts,
    compute_summary,
    compute_tail_metrics,
    compute_time_in_trade,
)
from src.backtest.simulator import run_backtest, run_backtest_enhanced
from src.backtest.walkforward import generate_walkforward_folds
from src.bars.time_bars import build_time_bars, resample_time_bars
from src.cli.build_dataset import build_dataset
from src.features.impact import kyle_lambda_features
from src.features.intrabar import build_intrabar_features
from src.features.microstructure import ofi_features
from src.features.replenishment import replenishment_features
from src.labeling.hazard_dataset import build_hazard_dataset
from src.labeling.cusum import cusum_events
from src.labeling.triple_barrier import triple_barrier_labels, triple_barrier_labels_by_regime
from src.modeling.load_model import load_hazard_model, predict_hazard_proba
from src.modeling.train_hazard import train_hazard_model
from src.modeling.train_meta import predict_meta, train_meta_model
from src.regime.regime import classify_regime
from src.strategy.entries_5m import generate_entries_5m


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_date_to_ms(date_str: str) -> int:
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 1_000_000)


def _parse_end_to_ms(date_str: str, end_inclusive_date: bool) -> int:
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if end_inclusive_date:
        ts = ts + pd.Timedelta(days=1)
    return int(ts.value // 1_000_000)


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for col in ("timestamp", "ts"):
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col).sort_index()
    raise ValueError("DataFrame must include a DatetimeIndex or timestamp/ts column.")


def _load_bars(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing bars file: {path}")
    df = pd.read_parquet(path)
    return _ensure_dt_index(df)


def _load_raw_dataset(
    raw_dir: Path,
    exchange: str,
    market: str,
    symbol: str,
    dataset: str,
    start_ms: int,
    end_ms: int,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    base = raw_dir / exchange / market / symbol / dataset
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.glob("date=*/part-000.parquet"))
    if not files:
        return pd.DataFrame()
    try:
        start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC").normalize()
        end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC").normalize()
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        start_dt = None
        end_dt = None

    if start_dt is not None and end_dt is not None:
        if end_dt <= start_dt:
            dates = {start_dt.strftime("%Y-%m-%d")}
        else:
            dates = {
                d.strftime("%Y-%m-%d")
                for d in pd.date_range(start_dt, end_dt - pd.Timedelta(days=1), freq="D")
            }
        filtered = []
        for path in files:
            date_part = path.parent.name.replace("date=", "")
            if date_part in dates:
                filtered.append(path)
        if filtered:
            files = filtered

    if columns is not None:
        cols = list(dict.fromkeys(columns))
        if "ts" not in cols:
            cols = ["ts"] + cols
    else:
        cols = None

    frames = [pd.read_parquet(path, columns=cols) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns and not df.empty:
        ts_min = df["ts"].min()
        ts_max = df["ts"].max()
        if pd.notna(ts_min) and pd.notna(ts_max):
            if not (int(ts_min) >= start_ms and int(ts_max) < end_ms):
                df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.reset_index(drop=True)


def _required_l2_columns(cfg: Dict) -> List[str]:
    levels: set[int] = set()
    ofi_cfg = cfg.get("features", {}).get("ofi", {})
    if ofi_cfg.get("enabled", False):
        levels.update(int(x) for x in ofi_cfg.get("levels", []))
    repl_cfg = cfg.get("features", {}).get("replenishment", {})
    if repl_cfg.get("enabled", False):
        repl_levels = repl_cfg.get("levels", 1)
        if isinstance(repl_levels, list):
            levels.update(int(x) for x in repl_levels)
        else:
            levels.add(int(repl_levels))
    levels.add(1)

    cols = ["ts"]
    for lvl in sorted(levels):
        cols.extend(
            [
                f"bid_price_{lvl}",
                f"ask_price_{lvl}",
                f"bid_size_{lvl}",
                f"ask_size_{lvl}",
            ]
        )
    return cols


def _load_bybit_trades_processed(
    processed_dir: Path,
    exchange: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    base = processed_dir / "agg_trades" / f"exchange={exchange}" / f"symbol={symbol}"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.glob("date=*/part-*.parquet"))
    if files:
        dates = set(_date_range_from_ms(start_ms, end_ms))
        filtered = []
        for path in files:
            date_part = path.parent.name.replace("date=", "")
            if date_part in dates:
                filtered.append(path)
        files = filtered
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns:
        df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
    return df.reset_index(drop=True)


def _bybit_trade_partitions(
    processed_dir: Path,
    exchange: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[Path]:
    base = processed_dir / "agg_trades" / f"exchange={exchange}" / f"symbol={symbol}"
    if not base.exists():
        return []
    files = sorted(base.glob("date=*/part-*.parquet"))
    if not files:
        return []
    dates = set(_date_range_from_ms(start_ms, end_ms))
    filtered = []
    for path in files:
        date_part = path.parent.name.replace("date=", "")
        if date_part in dates:
            filtered.append(path)
    return filtered


def _scan_bybit_trade_metrics(
    trade_paths: List[Path],
    start_ms: int,
    end_ms: int,
) -> Dict[str, object]:
    rows_total = 0
    rows_per_day: Dict[str, int] = {}
    min_ts = None
    max_ts = None
    minute_buckets: set[int] = set()

    for path in trade_paths:
        date_str = path.parent.name.replace("date=", "")
        df = pd.read_parquet(path, columns=["ts"])
        if df.empty or "ts" not in df.columns:
            continue
        ts = pd.to_numeric(df["ts"], errors="coerce").dropna()
        if ts.empty:
            continue
        rows = int(len(ts))
        rows_total += rows
        rows_per_day[date_str] = rows_per_day.get(date_str, 0) + rows
        ts_min = int(ts.min())
        ts_max = int(ts.max())
        min_ts = ts_min if min_ts is None else min(min_ts, ts_min)
        max_ts = ts_max if max_ts is None else max(max_ts, ts_max)
        minutes = ts.to_numpy(dtype="int64", copy=False) // 60_000
        minute_buckets.update(np.unique(minutes).tolist())

    expected = max(1, int((end_ms - start_ms) / 60_000))
    minute_coverage = float(len(minute_buckets) / expected)
    return {
        "rows_total": rows_total,
        "rows_per_day": rows_per_day,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "minute_coverage": minute_coverage,
    }


def _rows_per_day(df: pd.DataFrame, time_col: str) -> Dict[str, int]:
    if df.empty or time_col not in df.columns:
        return {}
    ts = pd.to_numeric(df[time_col], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return {}
    min_ts = int(ts.min())
    max_ts = int(ts.max())
    min_day = pd.to_datetime(min_ts, unit="ms", utc=True).strftime("%Y-%m-%d")
    max_day = pd.to_datetime(max_ts, unit="ms", utc=True).strftime("%Y-%m-%d")
    if min_day == max_day:
        return {min_day: int(len(ts))}

    day_index = (ts.to_numpy(dtype="int64", copy=False) // 86_400_000)
    days, counts = np.unique(day_index, return_counts=True)
    dates = pd.to_datetime(days * 86_400_000, unit="ms", utc=True).strftime("%Y-%m-%d")
    return {str(date): int(count) for date, count in zip(dates, counts)}


def _ts_range_utc(df: pd.DataFrame, time_col: str) -> Tuple[Optional[str], Optional[str]]:
    if df.empty or time_col not in df.columns:
        return None, None
    ts = pd.to_datetime(df[time_col], unit="ms", utc=True, errors="coerce")
    if ts.dropna().empty:
        return None, None
    return ts.min().isoformat(), ts.max().isoformat()


def _sampling_rate_sec(df: pd.DataFrame, time_col: str) -> Optional[float]:
    if df.empty or time_col not in df.columns or len(df) < 2:
        return None
    ts = pd.to_datetime(df[time_col], unit="ms", utc=True, errors="coerce")
    ts = ts.dropna().sort_values()
    if len(ts) < 2:
        return None
    diffs = ts.diff().dt.total_seconds().dropna()
    if diffs.empty:
        return None
    return float(diffs.median())


def _missing_minutes_ratio(bars_1m: pd.DataFrame, start_ms: int, end_ms: int) -> float:
    if bars_1m.empty:
        return 1.0
    expected = max(1, int((end_ms - start_ms) / 60_000))
    if "mid_close" in bars_1m.columns:
        actual = int(bars_1m["mid_close"].notna().sum())
    elif "close" in bars_1m.columns:
        actual = int(bars_1m["close"].notna().sum())
    else:
        idx = bars_1m.index
        if not isinstance(idx, pd.DatetimeIndex):
            return 1.0
        actual = idx.floor("min").nunique()
    return float(1.0 - min(actual / expected, 1.0))


def _write_pipeline_health(health: Dict, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / "pipeline_health.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2)


def _expected_dataset_paths(
    raw_dir: Path,
    processed_dir: Path,
    exchange: str,
    market: str,
    symbol: str,
    dataset: str,
    start: str,
    end: str,
) -> List[str]:
    dates = _date_range(start, end)
    paths = []
    if exchange == "bybit" and dataset == "agg_trades":
        base = processed_dir / "agg_trades" / f"exchange={exchange}" / f"symbol={symbol}"
        for date_str in dates:
            paths.append(str(base / f"date={date_str}" / "part-*.parquet"))
    else:
        base = raw_dir / exchange / market / symbol / dataset
        for date_str in dates:
            paths.append(str(base / f"date={date_str}" / "part-000.parquet"))
    return paths


def _signed_flow_from_trades(trades: pd.DataFrame, bars_1m: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(0.0, index=bars_1m.index)
    if "ts" not in trades.columns or "qty" not in trades.columns or "is_buyer_maker" not in trades.columns:
        raise ValueError("agg_trades missing required columns for signed flow")
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["ts"], unit="ms", utc=True)
    trades["signed_qty"] = trades["qty"].astype(float) * trades["is_buyer_maker"].map(lambda x: -1.0 if x else 1.0)
    trades["minute"] = trades["timestamp"].dt.floor("min")
    flow = trades.groupby("minute")["signed_qty"].sum()
    return flow.reindex(bars_1m.index, fill_value=0.0)


def _cusum_fallback_trades(bars_1m: pd.DataFrame, cfg: Dict, symbol: str) -> pd.DataFrame:
    if bars_1m.empty:
        return pd.DataFrame()
    price_col = "mid_close" if "mid_close" in bars_1m.columns else "close"
    cusum_cfg = cfg["labeling"]["cusum"]
    base_k = float(cusum_cfg["threshold_k"])
    vol_window = int(cusum_cfg["vol_window_1m_bars"])
    min_sigma = float(cusum_cfg.get("min_sigma", 1.0e-6))
    events = pd.DataFrame()
    for factor in (1.0, 0.5, 0.25, 0.1):
        events = cusum_events(
            bars_1m,
            threshold_k=base_k * factor,
            vol_window=vol_window,
            min_sigma=min_sigma,
            time_col=None,
            price_col=price_col,
        )
        if not events.empty:
            events["threshold_k_used"] = base_k * factor
            break
    if events.empty:
        return pd.DataFrame()

    bars_tb = bars_1m.copy()
    if "mid_open" in bars_tb.columns:
        bars_tb = bars_tb.rename(
            columns={
                "mid_open": "open",
                "mid_high": "high",
                "mid_low": "low",
                "mid_close": "close",
            }
        )

    tb_cfg = cfg["labeling"]["triple_barrier"]
    vol_cfg = tb_cfg["vol"]
    labeled = triple_barrier_labels(
        events=events,
        bars=bars_tb,
        horizon_minutes=int(tb_cfg["horizon_minutes"]),
        pt_mult=float(tb_cfg["barriers"]["pt_mult"]),
        sl_mult=float(tb_cfg["barriers"]["sl_mult"]),
        price_col="close",
        tie_break=str(tb_cfg["tie_break"]["mode"]),
        vol_kind=str(vol_cfg["kind"]),
        vol_window=int(vol_cfg["window_1m_bars"]),
        min_sigma=float(vol_cfg["min_sigma"]),
        time_col=None,
    )

    if labeled.empty:
        return pd.DataFrame()

    trades = pd.DataFrame(
        {
            "event_id": [f"cusum_{i}" for i in labeled.index],
            "entry_ts": labeled["entry_time"],
            "exit_ts": labeled["t1"],
            "side": labeled["side"].astype(int),
            "entry_price": labeled["entry_price"].astype(float),
            "symbol": symbol,
        }
    )
    return trades


def _write_signal_1m(
    hazard_features: pd.DataFrame,
    model_path: Path,
    cfg: Dict,
    output_path: Path,
) -> None:
    model_pack = load_hazard_model(model_path)
    features = hazard_features.copy()
    if "t" in features.columns:
        features["t"] = pd.to_datetime(features["t"])
        features = features.sort_values("t")
        timestamp = features["t"]
    elif "timestamp" in features.columns:
        features["timestamp"] = pd.to_datetime(features["timestamp"])
        features = features.sort_values("timestamp")
        timestamp = features["timestamp"]
    else:
        raise ValueError("hazard_features must include a timestamp column")

    p_hat = predict_hazard_proba(model_pack, features)

    policy_cfg = cfg.get("policy", {})
    exit_thr = float(policy_cfg.get("exit", {}).get("hazard_threshold", 0.7))
    ff_thr = float(policy_cfg.get("fail_fast", {}).get("hazard_threshold", 0.85))
    add_thr = float(policy_cfg.get("add_risk", {}).get("hazard_max_to_add", 0.2))

    def _recommend(p_val: float) -> str:
        if p_val >= ff_thr:
            return "exit"
        if p_val >= exit_thr:
            return "reduce"
        if p_val > add_thr:
            return "no_add"
        return "hold"

    hazard_state = ["high" if val >= exit_thr else "normal" for val in p_hat]
    recommended = [_recommend(float(val)) for val in p_hat]

    out = pd.DataFrame(
        {
            "timestamp": timestamp.values,
            "P_end": p_hat.values,
            "hazard_state": hazard_state,
            "recommended_action": recommended,
        }
    )
    out.to_parquet(output_path, index=False)


def _build_hazard_features(
    bars_1m: pd.DataFrame,
    l2: pd.DataFrame,
    agg_trades: pd.DataFrame,
    cfg: Dict,
) -> pd.DataFrame:
    features = []
    ofi = None

    if cfg["features"]["ofi"]["enabled"]:
        ofi = ofi_features(bars_1m, l2, cfg["features"]["ofi"], l2_time_col="ts")
        features.append(ofi)

    kyle_cfg = cfg["features"]["kyle_lambda"]
    if kyle_cfg["enabled"]:
        source = kyle_cfg.get("signed_flow_source", "ofi")
        if source == "ofi":
            if ofi is None:
                ofi = ofi_features(bars_1m, l2, cfg["features"]["ofi"], l2_time_col="ts")
                features.append(ofi)
            levels = sorted(int(x) for x in cfg["features"]["ofi"]["levels"])
            ofi_col = f"ofi_L{levels[-1]}"
            if ofi_col not in ofi.columns:
                raise ValueError(f"Missing OFI column '{ofi_col}' for Kyle lambda")
            signed_flow = ofi[ofi_col]
        else:
            signed_flow = _signed_flow_from_trades(agg_trades, bars_1m)

        kyle = kyle_lambda_features(bars_1m, signed_flow, kyle_cfg)
        features.append(kyle)

    repl_cfg = cfg["features"]["replenishment"]
    if repl_cfg["enabled"]:
        repl_window = int(cfg["hazard"].get("feature_window_minutes", 3))
        repl = replenishment_features(bars_1m, l2, repl_cfg, l2_time_col="ts", repl_window=repl_window)
        features.append(repl)

    if not features:
        raise ValueError("No hazard features enabled in config")

    out = pd.concat(features, axis=1)
    out["t"] = out.index
    return out.reset_index(drop=True)


def _run_enhanced_walkforward(
    bars_5m: pd.DataFrame,
    bars_1m: pd.DataFrame,
    hazard_df: pd.DataFrame,
    hazard_features: pd.DataFrame,
    cfg: Dict,
    artifacts_dir: Path,
    policy_variants: Iterable[str],
    events_override: Optional[pd.DataFrame] = None,
    entry_variant: str = "baseline",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_cfg = cfg["model"]["training"]["validation"]
    splits = val_cfg["splits"]
    folds = generate_walkforward_folds(
        pd.to_datetime(bars_5m.index),
        train_days=int(splits["train_days"]),
        test_days=int(splits["test_days"]),
        step_days=int(splits["step_days"]),
        embargo_minutes=int(val_cfg.get("embargo_minutes", 0)),
    )
    if not folds:
        idx = pd.to_datetime(bars_5m.index).sort_values()
        if len(idx) < 2:
            raise ValueError("No walk-forward folds produced; check data range and validation splits.")
        split = max(1, int(len(idx) * 0.7))
        if split >= len(idx):
            split = len(idx) - 1
        folds = [
            {
                "fold_id": 0,
                "train_start": idx[0],
                "train_end": idx[split - 1],
                "test_start": idx[split],
                "test_end": idx[-1],
            }
        ]

    rows = []
    trades_baseline_all = []
    trades_enhanced_all = []

    for fold in folds:
        test_start = fold["test_start"]
        test_end = fold["test_end"]

        bars_5m_slice = bars_5m.loc[bars_5m.index <= test_end]
        if events_override is None:
            regime = classify_regime(bars_5m_slice, cfg["regime"])
            events_all = generate_entries_5m(bars_5m_slice, regime, cfg["strategy"]["entries_5m"])
        else:
            events_all = events_override.copy()

        if not events_all.empty:
            events = events_all.loc[
                (events_all["entry_ts"] >= test_start) & (events_all["entry_ts"] <= test_end)
            ].reset_index(drop=True)
        else:
            events = events_all

        base_trades, base_equity, _ = run_backtest(bars_5m_slice, events, cfg)
        base_equity_test = base_equity.loc[(base_equity.index >= test_start) & (base_equity.index <= test_end)]
        base_summary = compute_summary(base_trades, base_equity_test, cfg["backtest"]["initial_capital"])
        base_tail = compute_tail_metrics(base_trades)
        base_time = compute_time_in_trade(base_trades)

        hazard_df["t"] = pd.to_datetime(hazard_df["t"])
        train_df = hazard_df[hazard_df["t"] <= fold["train_end"]]

        features_df = hazard_features.copy()
        if "t" in features_df.columns:
            features_df["t"] = pd.to_datetime(features_df["t"])
        elif "timestamp" in features_df.columns:
            features_df["t"] = pd.to_datetime(features_df["timestamp"])
        elif "ts" in features_df.columns:
            features_df["t"] = pd.to_datetime(features_df["ts"])
        else:
            raise ValueError("hazard_features must include a timestamp column")

        features_train = features_df[features_df["t"] <= fold["train_end"]]
        fold_dir = artifacts_dir / "models" / f"fold_{fold['fold_id']}"
        train_hazard_model(train_df, features_train, cfg, output_dir=fold_dir)

        model_path = fold_dir / "models" / "hazard_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing fold hazard model: {model_path}")
        import pickle

        with open(model_path, "rb") as pf:
            hazard_model = pickle.load(pf)

        bars_1m_slice = bars_1m.loc[bars_1m.index <= test_end]

        for variant in policy_variants:
            enh_trades, enh_equity, enh_summary, diagnostics = run_backtest_enhanced(
                bars_5m_slice,
                events,
                cfg,
                bars_1m_slice,
                hazard_features=hazard_features,
                hazard_model=hazard_model,
                hazard_prob=None,
                policy_mode=variant,
            )
            enh_equity_test = enh_equity.loc[(enh_equity.index >= test_start) & (enh_equity.index <= test_end)]
            enh_summary = compute_summary(enh_trades, enh_equity_test, cfg["backtest"]["initial_capital"])
            enh_tail = compute_tail_metrics(enh_trades)
            enh_time = compute_time_in_trade(enh_trades)
            hazard_counts = compute_hazard_exit_counts(enh_trades)

            baseline_sl = pd.DataFrame()
            if "exit_reason" in base_trades.columns:
                baseline_sl = base_trades[base_trades["exit_reason"] == "barrier_sl"]
            sl_reduction = 0.0
            if (
                not baseline_sl.empty
                and "event_id" in baseline_sl.columns
                and "exit_reason" in enh_trades.columns
                and "exit_ts" in enh_trades.columns
            ):
                merged = baseline_sl.merge(
                    enh_trades[["event_id", "exit_ts", "exit_reason"]],
                    on="event_id",
                    how="left",
                    suffixes=("_base", "_enh"),
                )
                improved = merged[
                    (merged["exit_reason_enh"] != "barrier_sl")
                    & (pd.to_datetime(merged["exit_ts_enh"]) < pd.to_datetime(merged["exit_ts_base"]))
                ]
                sl_reduction = float(len(improved) / len(baseline_sl))

            rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "test_start": test_start,
                    "test_end": test_end,
                    "policy_variant": variant,
                    "entry_variant": entry_variant,
                    "baseline_pnl_net": base_summary["pnl_net"],
                    "enhanced_pnl_net": enh_summary["pnl_net"],
                    "delta_pnl_net": enh_summary["pnl_net"] - base_summary["pnl_net"],
                    "baseline_sharpe": base_summary["sharpe"],
                    "enhanced_sharpe": enh_summary["sharpe"],
                    "delta_sharpe": enh_summary["sharpe"] - base_summary["sharpe"],
                    "baseline_max_drawdown": base_summary["max_drawdown"],
                    "enhanced_max_drawdown": enh_summary["max_drawdown"],
                    "delta_max_drawdown": enh_summary["max_drawdown"] - base_summary["max_drawdown"],
                    "baseline_trade_count": base_summary["trade_count"],
                    "enhanced_trade_count": enh_summary["trade_count"],
                    "baseline_win_rate": base_summary["win_rate"],
                    "enhanced_win_rate": enh_summary["win_rate"],
                    "baseline_avg_win": base_summary["avg_win"],
                    "enhanced_avg_win": enh_summary["avg_win"],
                    "baseline_avg_loss": base_summary["avg_loss"],
                    "enhanced_avg_loss": enh_summary["avg_loss"],
                    "baseline_total_fees": base_summary["total_fees"],
                    "enhanced_total_fees": enh_summary["total_fees"],
                    "baseline_total_slippage": base_summary["total_slippage"],
                    "enhanced_total_slippage": enh_summary["total_slippage"],
                    "baseline_turnover": base_summary["turnover"],
                    "enhanced_turnover": enh_summary["turnover"],
                    "baseline_worst_5pct_trade": base_tail["worst_5pct_trade"],
                    "baseline_expected_shortfall_95": base_tail["expected_shortfall_95"],
                    "enhanced_worst_5pct_trade": enh_tail["worst_5pct_trade"],
                    "enhanced_expected_shortfall_95": enh_tail["expected_shortfall_95"],
                    "baseline_worst_week_pnl": base_tail["worst_week_pnl"],
                    "enhanced_worst_week_pnl": enh_tail["worst_week_pnl"],
                    "baseline_avg_time_in_trade_min": base_time,
                    "enhanced_avg_time_in_trade_min": enh_time,
                    "delta_avg_time_in_trade_min": enh_time - base_time,
                    "hazard_exits": hazard_counts["hazard_exits"],
                    "fail_fast_exits": hazard_counts["fail_fast_exits"],
                    "sl_hit_reduction_rate": sl_reduction,
                    "hazard_enabled": diagnostics.get("hazard_enabled", 0),
                }
            )

        trades_baseline_all.append(base_trades.assign(fold_id=fold["fold_id"]))
        trades_enhanced_all.append(enh_trades.assign(fold_id=fold["fold_id"], policy_variant=variant))

    compare = pd.DataFrame(rows)
    trades_baseline = pd.concat(trades_baseline_all, ignore_index=True) if trades_baseline_all else pd.DataFrame()
    trades_enhanced = pd.concat(trades_enhanced_all, ignore_index=True) if trades_enhanced_all else pd.DataFrame()
    return compare, trades_baseline, trades_enhanced


def _date_range(start: str, end: str) -> List[str]:
    start_dt = pd.Timestamp(start, tz="UTC").normalize()
    end_dt = pd.Timestamp(end, tz="UTC").normalize()
    if end_dt <= start_dt:
        return [start_dt.strftime("%Y-%m-%d")]
    dates = pd.date_range(start_dt, end_dt - pd.Timedelta(days=1), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def _date_range_from_ms(start_ms: int, end_ms: int) -> List[str]:
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC").normalize()
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC").normalize()
    if end_dt <= start_dt:
        return [start_dt.strftime("%Y-%m-%d")]
    dates = pd.date_range(start_dt, end_dt - pd.Timedelta(days=1), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates]


def _config_hash(cfg: Dict) -> str:
    import hashlib

    payload = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_run_diagnosis(artifacts_dir: Path, message: str) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    content = [
        "# RUN_DIAGNOSIS",
        "",
        message,
        "",
        f"timestamp_utc: {pd.Timestamp.utcnow().isoformat()}",
    ]
    (artifacts_dir / "RUN_DIAGNOSIS.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def _data_gap_report(
    raw_dir: Path,
    exchange: str,
    market: str,
    symbol: str,
    datasets: Iterable[str],
    start: str,
    end: str,
    cfg_hash: str,
    artifacts_dir: Path,
    coverage: Optional[Dict[str, float]] = None,
    dataset_roots: Optional[Dict[str, Path]] = None,
) -> Dict:
    dates = _date_range(start, end)
    missing = []
    dataset_roots = dataset_roots or {}
    for dataset in datasets:
        base = dataset_roots.get(dataset)
        if base is None:
            base = raw_dir / exchange / market / symbol / dataset
        for date_str in dates:
            if dataset in dataset_roots:
                part_dir = base / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date_str}"
                if not list(part_dir.glob("part-*.parquet")):
                    missing.append({"dataset": dataset, "date": date_str})
            else:
                part = base / f"date={date_str}" / "part-000.parquet"
                if not part.exists():
                    missing.append({"dataset": dataset, "date": date_str})

    report = {
        "config_hash": cfg_hash,
        "exchange": exchange,
        "symbol": symbol,
        "start": start,
        "end": end,
        "missing": missing,
        "coverage": coverage or {},
    }
    path = artifacts_dir / "data_gap_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _missing_datasets(
    raw_dir: Path,
    exchange: str,
    market: str,
    symbol: str,
    datasets: Iterable[str],
    start: str,
    end: str,
    dataset_roots: Optional[Dict[str, Path]] = None,
) -> List[Dict[str, str]]:
    dates = _date_range(start, end)
    missing: List[Dict[str, str]] = []
    dataset_roots = dataset_roots or {}
    for dataset in datasets:
        base = dataset_roots.get(dataset)
        if base is None:
            base = raw_dir / exchange / market / symbol / dataset
        for date_str in dates:
            if dataset in dataset_roots:
                part_dir = base / f"exchange={exchange}" / f"symbol={symbol}" / f"date={date_str}"
                if not list(part_dir.glob("part-*.parquet")):
                    missing.append({"dataset": dataset, "date": date_str})
            else:
                part = base / f"date={date_str}" / "part-000.parquet"
                if not part.exists():
                    missing.append({"dataset": dataset, "date": date_str})
    return missing


def _feature_health_report(
    features: pd.DataFrame,
    cfg_hash: str,
    artifacts_dir: Path,
    metadata: Optional[Dict] = None,
) -> Dict:
    numeric = features.select_dtypes(include=["number"]).copy()
    if "t" in numeric.columns:
        numeric = numeric.drop(columns=["t"])
    report = {"config_hash": cfg_hash, "features": {}, "metadata": metadata or {}}
    if numeric.empty:
        path = artifacts_dir / "feature_health_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    mid = len(numeric) // 2 if len(numeric) > 1 else 1
    first = numeric.iloc[:mid]
    second = numeric.iloc[mid:]
    for col in numeric.columns:
        series = numeric[col].astype(float)
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        std = std if std > 0 else 1.0
        z = (series - mean) / std
        outlier_frac = float((z.abs() > 5.0).mean())
        drift = float(second[col].mean() - first[col].mean()) if len(series) > 1 else 0.0
        drift_z = drift / std
        report["features"][col] = {
            "missing_frac": float(series.isna().mean()),
            "mean": mean,
            "std": std,
            "min": float(series.min()),
            "max": float(series.max()),
            "outlier_frac": outlier_frac,
            "drift_z": drift_z,
        }

    path = artifacts_dir / "feature_health_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _minute_coverage(df: pd.DataFrame, start_ms: int, end_ms: int) -> float:
    if df.empty or "ts" not in df.columns:
        return 0.0
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    minutes = ts.dt.floor("min").dropna().unique()
    expected = max(1, int((end_ms - start_ms) / 60_000))
    return float(len(minutes) / expected)


def _flag_diagnostics(compare: pd.DataFrame, hazard_report_path: Path, artifacts_dir: Path, cfg_hash: str) -> None:
    if compare.empty:
        return

    required_cols = {
        "delta_pnl_net",
        "delta_max_drawdown",
        "enhanced_turnover",
        "baseline_turnover",
        "enhanced_total_fees",
        "baseline_total_fees",
    }
    if not required_cols.issubset(compare.columns):
        return

    def _write_flag(name: str, payload: Dict) -> None:
        payload["config_hash"] = cfg_hash
        path = artifacts_dir / f"flag_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    delta_pnl = compare["delta_pnl_net"]
    churn = compare["enhanced_turnover"] > (compare["baseline_turnover"] * 2.0)
    fee_spike = compare["enhanced_total_fees"] > (compare["baseline_total_fees"] * 2.0)
    if (delta_pnl > 0).any() and (churn & fee_spike).any():
        _write_flag("churn_risk", {"reason": "PnL improved with >2x turnover and fees."})

    over_exit = (compare["delta_pnl_net"] < 0) & (compare["delta_max_drawdown"] < 0)
    if over_exit.any():
        _write_flag("over_exiting", {"reason": "DD improved while PnL declined."})

    if hazard_report_path.exists():
        report = json.loads(hazard_report_path.read_text(encoding="utf-8"))
        auc = report.get("overall", {}).get("auc")
        if auc is not None and auc > 0.6 and (compare["delta_pnl_net"] < 0).any():
            _write_flag("model_not_decision_aligned", {"reason": "AUC good but trading PnL worsened."})


def _print_summary(compare: pd.DataFrame, artifacts_dir: Path) -> None:
    if compare.empty:
        print("No folds produced.")
        return

    metrics = [
        "pnl_net",
        "sharpe",
        "max_drawdown",
        "expected_shortfall_95",
        "turnover",
        "total_fees",
        "total_slippage",
    ]
    for label in ("baseline", "enhanced"):
        rows = {}
        for metric in metrics:
            col = f"{label}_{metric}"
            if col in compare.columns:
                rows[metric] = float(compare[col].mean())
        if rows:
            print(f"{label.upper()} mean metrics:", rows)

    flags = list(artifacts_dir.glob("flag_*.json"))
    if flags:
        print("Flags:", [f.stem.replace("flag_", "").upper() for f in flags])


def run_all(
    cfg: Dict,
    exchange: str,
    symbol: str,
    start: str,
    end: str,
    source: str = "vision",
    datasets: Optional[str] = None,
    vision_dir: Optional[str] = None,
    vision_auto_download: bool = True,
    okx_dir: Optional[str] = None,
    okx_cache_dir: Optional[str] = None,
    okx_auto_download: bool = False,
    okx_modules: Optional[str] = None,
    okx_level: Optional[int] = None,
    okx_agg: Optional[str] = None,
    okx_manual_candles_dir: Optional[str] = None,
    okx_manual_trades_dir: Optional[str] = None,
    okx_manual_book_dir: Optional[str] = None,
    okx_store_top_levels: Optional[int] = None,
    bybit_manual_root: Optional[str] = None,
    bybit_manual_book_root: Optional[str] = None,
    bybit_manual_trades_root: Optional[str] = None,
    bybit_store_top_levels: Optional[int] = None,
    bybit_book_sample_ms: Optional[int] = None,
    end_inclusive_date: bool = False,
    policy_variants: Optional[List[str]] = None,
    meta_enabled: Optional[bool] = None,
    meta_threshold: Optional[float] = None,
    label_horizon_minutes: Optional[int] = None,
    disable_trend_pullback_long: bool = False,
    disable_trend_pullback_short: bool = False,
    disable_range_vwap_band: bool = False,
    enable_trend_pullback_long_filter: Optional[bool] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipeline_health: Dict[str, object] = {}

    if meta_enabled is not None:
        cfg.setdefault("model", {}).setdefault("meta", {})["enabled"] = bool(meta_enabled)
    if meta_threshold is not None:
        cfg.setdefault("model", {}).setdefault("meta", {})["threshold"] = float(meta_threshold)
    if label_horizon_minutes is not None:
        tb_cfg = cfg.setdefault("labeling", {}).setdefault("triple_barrier", {})
        horizon_val = int(label_horizon_minutes)
        tb_cfg["horizon_minutes"] = horizon_val
        by_regime = tb_cfg.get("by_regime")
        if isinstance(by_regime, dict):
            for overrides in by_regime.values():
                if isinstance(overrides, dict):
                    overrides["horizon"] = horizon_val
    if disable_trend_pullback_long or disable_trend_pullback_short or disable_range_vwap_band:
        subtypes = (
            cfg.setdefault("strategy", {})
            .setdefault("entries_5m", {})
            .setdefault("subtypes", {})
        )
        if disable_trend_pullback_long:
            subtypes["trend_pullback_long"] = False
        if disable_trend_pullback_short:
            subtypes["trend_pullback_short"] = False
        if disable_range_vwap_band:
            subtypes["range_vwap_band"] = False
    if enable_trend_pullback_long_filter is not None:
        long_filter = (
            cfg.setdefault("strategy", {})
            .setdefault("entries_5m", {})
            .setdefault("trend", {})
            .setdefault("pullback", {})
            .setdefault("long_filter", {})
        )
        long_filter["enabled"] = bool(enable_trend_pullback_long_filter)

    cfg_hash = _config_hash(cfg)
    pipeline_health["config_hash"] = cfg_hash

    if datasets is None:
        if exchange == "okx":
            datasets = "trades,book_depth"
        elif exchange == "bybit":
            datasets = "book_depth,agg_trades"
        else:
            datasets = "klines_1m,agg_trades,book_ticker,book_depth,premium_kline,mark_kline"

    if exchange == "binance":
        market = "usdm"
    elif exchange == "deepcoin":
        market = "perp"
    elif exchange == "okx":
        market = "swap"
    elif exchange == "bybit":
        market = "perp"
    else:
        raise ValueError("exchange must be 'binance', 'deepcoin', 'okx', or 'bybit'")

    required_datasets = ["book_depth", "agg_trades", "klines_1m"]
    if exchange == "okx":
        required_datasets = ["book_depth", "agg_trades"]
    elif exchange == "bybit":
        required_datasets = ["book_depth", "agg_trades"]

    skip_build = False
    if exchange == "bybit" and source == "bybit_manual":
        roots = {"agg_trades": processed_dir / "agg_trades"}
        missing = _missing_datasets(raw_dir, exchange, market, symbol, required_datasets, start, end, roots)
        skip_build = not missing
    if force_rebuild:
        skip_build = False

    if not skip_build:
        try:
            build_dataset(
                cfg,
                exchange=exchange,
                symbol=symbol,
                start=start,
                end=end,
                source=source,
                datasets=datasets,
                build_bars=True,
                vision_dir=vision_dir,
                vision_auto_download=vision_auto_download,
                okx_dir=okx_dir,
                okx_cache_dir=okx_cache_dir,
                okx_auto_download=okx_auto_download,
                okx_modules=okx_modules,
                okx_level=okx_level,
                okx_agg=okx_agg,
                okx_manual_candles_dir=okx_manual_candles_dir,
                okx_manual_trades_dir=okx_manual_trades_dir,
                okx_manual_book_dir=okx_manual_book_dir,
                okx_store_top_levels=okx_store_top_levels,
                bybit_manual_root=bybit_manual_root,
                bybit_manual_book_root=bybit_manual_book_root,
                bybit_manual_trades_root=bybit_manual_trades_root,
                bybit_store_top_levels=bybit_store_top_levels,
                bybit_book_sample_ms=bybit_book_sample_ms,
                end_inclusive_date=end_inclusive_date,
                force_rebuild=force_rebuild,
            )
        except Exception as exc:
            _data_gap_report(raw_dir, exchange, market, symbol, required_datasets, start, end, cfg_hash, artifacts_dir)
            _write_run_diagnosis(artifacts_dir, f"build_dataset failed: {exc}")
            raise

    start_ms = _parse_date_to_ms(start)
    end_ms = _parse_end_to_ms(end, end_inclusive_date=end_inclusive_date)

    l2_columns = _required_l2_columns(cfg)
    l2 = _load_raw_dataset(
        raw_dir,
        exchange,
        market,
        symbol,
        "book_depth",
        start_ms,
        end_ms,
        columns=l2_columns,
    )
    trade_metrics: Optional[Dict[str, object]] = None
    trade_paths: Optional[List[Path]] = None
    if exchange == "bybit":
        trade_paths = _bybit_trade_partitions(processed_dir, exchange, symbol, start_ms, end_ms)
        if trade_paths:
            trade_metrics = _scan_bybit_trade_metrics(trade_paths, start_ms, end_ms)
            agg_trades = pd.DataFrame()
            agg_min = (
                pd.to_datetime(trade_metrics["min_ts"], unit="ms", utc=True).isoformat()
                if trade_metrics.get("min_ts") is not None
                else None
            )
            agg_max = (
                pd.to_datetime(trade_metrics["max_ts"], unit="ms", utc=True).isoformat()
                if trade_metrics.get("max_ts") is not None
                else None
            )
            pipeline_health["agg_trades"] = {
                "rows_total": int(trade_metrics["rows_total"]),
                "rows_per_day": {str(k): int(v) for k, v in trade_metrics["rows_per_day"].items()},
                "min_ts_utc": agg_min,
                "max_ts_utc": agg_max,
            }
        else:
            agg_trades = _load_bybit_trades_processed(processed_dir, exchange, symbol, start_ms, end_ms)
            agg_min, agg_max = _ts_range_utc(agg_trades, "ts")
            pipeline_health["agg_trades"] = {
                "rows_total": int(len(agg_trades)),
                "rows_per_day": _rows_per_day(agg_trades, "ts"),
                "min_ts_utc": agg_min,
                "max_ts_utc": agg_max,
            }
    else:
        agg_trades = _load_raw_dataset(raw_dir, exchange, market, symbol, "agg_trades", start_ms, end_ms)
        agg_min, agg_max = _ts_range_utc(agg_trades, "ts")
        pipeline_health["agg_trades"] = {
            "rows_total": int(len(agg_trades)),
            "rows_per_day": _rows_per_day(agg_trades, "ts"),
            "min_ts_utc": agg_min,
            "max_ts_utc": agg_max,
        }

    book_min, book_max = _ts_range_utc(l2, "ts")
    pipeline_health["book_depth"] = {
        "rows_total": int(len(l2)),
        "rows_per_day": _rows_per_day(l2, "ts"),
        "min_ts_utc": book_min,
        "max_ts_utc": book_max,
        "sampling_rate_sec": _sampling_rate_sec(l2, "ts"),
    }
    _write_pipeline_health(pipeline_health, artifacts_dir)

    agg_rows = trade_metrics["rows_total"] if trade_metrics is not None else len(agg_trades)
    if int(agg_rows) == 0:
        expected = _expected_dataset_paths(
            raw_dir, processed_dir, exchange, market, symbol, "agg_trades", start, end
        )
        _write_run_diagnosis(
            artifacts_dir,
            "agg_trades has 0 rows. Expected input files:\n- " + "\n- ".join(expected),
        )
        raise ValueError("Missing required datasets or insufficient coverage; see artifacts/RUN_DIAGNOSIS.md")

    bars_1m_path = processed_dir / "bars_1m.parquet"
    bars_5m_path = processed_dir / "bars_5m.parquet"
    bars_1m = _load_bars(processed_dir, "bars_1m.parquet") if bars_1m_path.exists() else None
    bars_5m = _load_bars(processed_dir, "bars_5m.parquet") if bars_5m_path.exists() else None
    if bars_1m is not None:
        price_col = "mid_close" if "mid_close" in bars_1m.columns else "close" if "close" in bars_1m.columns else None
        if price_col is not None:
            non_nan_ratio = float(bars_1m[price_col].notna().mean())
            if non_nan_ratio < 0.5 and not agg_trades.empty:
                bars_1m = None
                bars_5m = None
    if bars_1m is not None:
        start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
        end_ts = pd.to_datetime(end_ms, unit="ms", utc=True) - pd.Timedelta(minutes=1)
        if bars_1m.index.min() > start_ts or bars_1m.index.max() < end_ts:
            bars_1m = None
            bars_5m = None
    if bars_1m is None or bars_5m is None:
        if agg_trades.empty and exchange != "bybit":
            raise ValueError("Missing klines_1m and agg_trades; cannot build bars.")
        if exchange == "bybit" and trade_paths:
            bars_1m_parts = []
            bars_5m_parts = []
            for date_str in _date_range_from_ms(start_ms, end_ms):
                day_start = pd.Timestamp(date_str, tz="UTC")
                day_start_ms = int(day_start.value // 1_000_000)
                day_end_ms = int((day_start + pd.Timedelta(days=1)).value // 1_000_000)
                day_files = [p for p in trade_paths if p.parent.name == f"date={date_str}"]
                if not day_files:
                    continue
                trades = pd.concat(
                    [pd.read_parquet(p, columns=["ts", "price", "qty", "is_buyer_maker"]) for p in day_files],
                    ignore_index=True,
                )
                if trades.empty:
                    continue
                trades["timestamp"] = pd.to_datetime(trades["ts"], unit="ms", utc=True)
                l2_day = _load_raw_dataset(
                    raw_dir,
                    exchange,
                    market,
                    symbol,
                    "book_depth",
                    day_start_ms,
                    day_end_ms,
                    columns=l2_columns,
                )
                l2_view = None
                if not l2_day.empty:
                    l2_cov = _minute_coverage(l2_day, day_start_ms, day_end_ms)
                    if l2_cov >= 0.5:
                        l2_view = l2_day.copy()
                        l2_view["timestamp"] = pd.to_datetime(l2_view["ts"], unit="ms", utc=True)
                        l2_view = l2_view.rename(
                            columns={
                                "bid_price_1": "bid",
                                "ask_price_1": "ask",
                                "bid_size_1": "bid_size",
                                "ask_size_1": "ask_size",
                            }
                        )
                bars_1m_day = build_time_bars(
                    trades,
                    l2=l2_view,
                    time_col="timestamp",
                    price_col="price",
                    qty_col="qty",
                    l2_time_col="timestamp",
                    bid_col="bid",
                    ask_col="ask",
                    bid_size_col="bid_size",
                    ask_size_col="ask_size",
                )
                bars_1m_parts.append(bars_1m_day.reset_index().rename(columns={"index": "timestamp"}))
                bars_5m_parts.append(
                    resample_time_bars(bars_1m_day, freq="5min")
                    .reset_index()
                    .rename(columns={"index": "timestamp"})
                )
            if not bars_1m_parts or not bars_5m_parts:
                raise ValueError("Missing agg_trades; cannot build bars.")
            bars_1m_df = pd.concat(bars_1m_parts, ignore_index=True).sort_values("timestamp")
            bars_5m_df = pd.concat(bars_5m_parts, ignore_index=True).sort_values("timestamp")
            bars_1m_df.to_parquet(bars_1m_path, index=False, engine="pyarrow")
            bars_5m_df.to_parquet(bars_5m_path, index=False, engine="pyarrow")
            bars_1m = _ensure_dt_index(bars_1m_df)
            bars_5m = _ensure_dt_index(bars_5m_df)
        else:
            trades = agg_trades.copy()
            trades["timestamp"] = pd.to_datetime(trades["ts"], unit="ms", utc=True)
            l2_view = None
            if not l2.empty:
                l2_cov = _minute_coverage(l2, start_ms, end_ms)
                if l2_cov >= 0.5:
                    l2_view = l2.copy()
                    l2_view["timestamp"] = pd.to_datetime(l2_view["ts"], unit="ms", utc=True)
                    l2_view = l2_view.rename(
                        columns={
                            "bid_price_1": "bid",
                            "ask_price_1": "ask",
                            "bid_size_1": "bid_size",
                            "ask_size_1": "ask_size",
                        }
                    )
            bars_1m_tmp = build_time_bars(
                trades,
                l2=l2_view,
                time_col="timestamp",
                price_col="price",
                qty_col="qty",
                l2_time_col="timestamp",
                bid_col="bid",
                ask_col="ask",
                bid_size_col="bid_size",
                ask_size_col="ask_size",
            )
            bars_1m_df = bars_1m_tmp.reset_index().rename(columns={"index": "timestamp"})
            bars_1m_df.to_parquet(bars_1m_path, index=False, engine="pyarrow")
            bars_5m_tmp = resample_time_bars(bars_1m_df.set_index("timestamp"), freq="5min")
            bars_5m_tmp.to_parquet(bars_5m_path, engine="pyarrow")
            bars_1m = _ensure_dt_index(bars_1m_df)
            bars_5m = bars_5m_tmp

    bars_1m_count = 0 if bars_1m is None else int(len(bars_1m))
    bars_5m_count = 0 if bars_5m is None else int(len(bars_5m))
    pipeline_health["bars_1m"] = {
        "count": bars_1m_count,
        "missing_minutes_ratio": _missing_minutes_ratio(bars_1m, start_ms, end_ms) if bars_1m is not None else 1.0,
    }
    pipeline_health["bars_5m"] = {"count": bars_5m_count}
    _write_pipeline_health(pipeline_health, artifacts_dir)

    if bars_1m_count == 0:
        expected = _expected_dataset_paths(
            raw_dir, processed_dir, exchange, market, symbol, "agg_trades", start, end
        )
        _write_run_diagnosis(
            artifacts_dir,
            "bars_1m has 0 rows. Expected agg_trades inputs:\n- " + "\n- ".join(expected),
        )
        raise ValueError("Missing required datasets or insufficient coverage; see artifacts/RUN_DIAGNOSIS.md")
    min_coverage = float(
        cfg.get("data_health", {}).get("min_coverage", cfg.get("data", {}).get("quality", {}).get("min_coverage", 0.99))
    )
    coverage = {}
    for dataset in required_datasets:
        if dataset == "book_depth":
            df = l2
            coverage[dataset] = _minute_coverage(df, start_ms, end_ms)
            continue
        if dataset == "agg_trades":
            if trade_metrics is not None:
                coverage[dataset] = float(trade_metrics["minute_coverage"])
                continue
            if exchange == "bybit":
                coverage[dataset] = _minute_coverage(agg_trades, start_ms, end_ms)
                continue
        df = _load_raw_dataset(raw_dir, exchange, market, symbol, dataset, start_ms, end_ms)
        coverage[dataset] = _minute_coverage(df, start_ms, end_ms)

    gap_report = _data_gap_report(
        raw_dir,
        exchange,
        market,
        symbol,
        required_datasets,
        start,
        end,
        cfg_hash,
        artifacts_dir,
        coverage=coverage,
        dataset_roots=(
            {"agg_trades": processed_dir / "agg_trades"} if exchange == "bybit" else None
        ),
    )
    allow_baseline_only = bool(
        cfg.get("data", {}).get(exchange, {}).get("manual", {}).get("allow_baseline_only", False)
    )
    enhanced_enabled = not l2.empty

    if exchange == "bybit":
        bars_cov = max(coverage.get("klines_1m", 0.0), coverage.get("agg_trades", 0.0))
        book_cov = coverage.get("book_depth", 0.0)
        trade_cov = coverage.get("agg_trades", 0.0)
        if bars_cov < min_coverage:
            _write_run_diagnosis(artifacts_dir, "Missing agg_trades coverage for bars.")
            raise ValueError("Missing required datasets or insufficient coverage; see artifacts/data_gap_report.json")
        hazard_ok = book_cov >= min_coverage and trade_cov >= min_coverage
        enhanced_enabled = hazard_ok
    else:
        if gap_report["missing"] or any(val < min_coverage for val in coverage.values()):
            if not allow_baseline_only or enhanced_enabled:
                _write_run_diagnosis(artifacts_dir, "Missing required datasets or insufficient coverage.")
                raise ValueError("Missing required datasets or insufficient coverage; see artifacts/data_gap_report.json")
            enhanced_enabled = False

    regime = classify_regime(bars_5m, cfg["regime"])
    events = generate_entries_5m(bars_5m, regime, cfg["strategy"]["entries_5m"])
    event_source = "baseline_entries"
    fallback_trades = pd.DataFrame()
    if events.empty:
        fallback_trades = _cusum_fallback_trades(bars_1m, cfg, symbol)
        event_source = "cusum_fallback"
        if fallback_trades.empty:
            _write_run_diagnosis(
                artifacts_dir,
                "CUSUM fallback produced 0 events. Check labeling.cusum thresholds or bar data coverage.",
            )
            raise ValueError("CUSUM fallback produced 0 events; cannot train hazard model.")
    meta_cfg = cfg.get("model", {}).get("meta", {})
    meta_enabled = bool(meta_cfg.get("enabled", False))
    meta_threshold = float(meta_cfg.get("threshold", 0.55))
    meta_features = None
    intrabar_features = None
    events_meta_veto = None
    if meta_enabled and enhanced_enabled and bool(meta_cfg.get("include_microstructure", True)):
        try:
            meta_features = _build_hazard_features(bars_1m, l2, agg_trades, cfg)
        except Exception:
            meta_features = None

    events_pre_meta = events.copy()
    if meta_enabled and event_source == "baseline_entries" and not events.empty:
        intrabar_cfg = meta_cfg.get("intrabar", {})
        lookback_min = int(intrabar_cfg.get("lookback_min", 5))
        intrabar_features = build_intrabar_features(bars_1m, events_pre_meta["entry_ts"], lookback_min)
        oof, meta_model, meta_report, meta_frame = train_meta_model(
            events_pre_meta,
            bars_5m,
            cfg,
            hazard_features=meta_features,
            intrabar_features=intrabar_features,
            output_dir=artifacts_dir,
        )
        scores_oof = oof.reindex(meta_frame.index)
        scores_full = pd.Series(predict_meta(meta_model, meta_frame), index=meta_frame.index)
        scores_combined = scores_oof.copy()
        missing_mask = scores_combined.isna()
        scores_combined.loc[missing_mask] = scores_full.loc[missing_mask]
        score_map = pd.Series(scores_combined.values, index=meta_frame["event_id"])
        events_scored = events_pre_meta.copy()
        events_scored["meta_score"] = events_scored["event_id"].map(score_map)
        nan_ratio = float(events_scored["meta_score"].isna().mean())
        missing_features = meta_report.get("missing_features", [])
        missing_examples = meta_report.get("missing_examples", [])
        if nan_ratio > 0.05:
            debug = {
                "nan_ratio": nan_ratio,
                "missing_features": missing_features,
                "missing_examples": missing_examples,
            }
            (artifacts_dir / "meta_score_debug.json").write_text(
                json.dumps(debug, indent=2), encoding="utf-8"
            )
            raise ValueError(
                "Meta score NaN ratio exceeds 5%. "
                f"nan_ratio={nan_ratio:.3f} missing_features={missing_features} "
                f"missing_examples={missing_examples}"
            )
        score_values = pd.to_numeric(scores_combined, errors="coerce").dropna()
        quantiles = {}
        if not score_values.empty:
            quantiles = {
                "min": float(score_values.min()),
                "median": float(score_values.quantile(0.5)),
                "p90": float(score_values.quantile(0.9)),
                "p99": float(score_values.quantile(0.99)),
                "max": float(score_values.max()),
            }
        debug = {
            "nan_ratio": nan_ratio,
            "quantiles": quantiles,
            "missing_features": missing_features,
            "missing_examples": missing_examples,
            "total_scores": int(len(scores_combined)),
            "oof_coverage": float(meta_report.get("oof_coverage", 0.0)),
            "filled_from_full_ratio": float(missing_mask.mean()) if len(scores_combined) else 0.0,
        }
        (artifacts_dir / "meta_score_debug.json").write_text(
            json.dumps(debug, indent=2), encoding="utf-8"
        )

        score_source = pd.Series("oof", index=meta_frame.index)
        score_source.loc[missing_mask] = "full"
        meta_scores = pd.DataFrame(
            {
                "event_id": meta_frame["event_id"].values,
                "entry_ts": pd.to_datetime(meta_frame["entry_ts"]).values,
                "regime": meta_frame.get("regime"),
                "label": meta_frame["y"].values,
                "score": scores_combined.values,
                "score_oof": scores_oof.values,
                "score_full": scores_full.values,
                "score_source": score_source.values,
                "gross_pnl_est": meta_frame.get("gross_pnl_est"),
                "net_pnl_est": meta_frame.get("net_pnl_est"),
            }
        )
        meta_scores.to_parquet(artifacts_dir / "meta_scores.parquet", index=False)
        events_meta_veto = events_scored.copy()
        before = int(len(events_meta_veto))
        events_meta_veto = events_meta_veto[events_meta_veto["meta_score"] >= meta_threshold].reset_index(drop=True)
        pipeline_health["meta"] = {
            "enabled": True,
            "events_before": before,
            "events_after": int(len(events_meta_veto)),
            "threshold": meta_threshold,
            "used_features": meta_report.get("used_features", []),
            "oof_coverage": float(meta_report.get("oof_coverage", 0.0)),
        }
        if events_meta_veto.empty:
            pipeline_health["meta_veto"] = {
                "enabled": True,
                "events_after": 0,
                "reason": "All events filtered by meta threshold.",
                "score_quantiles": quantiles,
            }

    pipeline_health["regime"] = regime["regime"].value_counts().to_dict() if not regime.empty else {}
    pipeline_health["entries_5m"] = {"events_count": int(len(events))}
    pipeline_health["event_source"] = event_source
    _write_pipeline_health(pipeline_health, artifacts_dir)

    baseline_trades, _, _ = run_backtest(bars_5m, events, cfg)
    if meta_enabled and event_source == "baseline_entries" and not events_pre_meta.empty:
        baseline_all, _, _ = run_backtest(bars_5m, events_pre_meta, cfg)
        baseline_all.to_parquet(artifacts_dir / "trades_baseline_all.parquet", index=False)
    pipeline_health["baseline"] = {"trades_count": int(len(baseline_trades))}
    _write_pipeline_health(pipeline_health, artifacts_dir)

    if not enhanced_enabled:
        skip_reason = "missing_hazard_coverage"
        hazard_report = {
            "status": "skipped",
            "reason": skip_reason,
            "config_hash": cfg_hash,
            "coverage": coverage,
        }
        (artifacts_dir / "hazard_report.json").write_text(json.dumps(hazard_report, indent=2), encoding="utf-8")
        empty_signal = pd.DataFrame(
            columns=["timestamp", "P_end", "hazard_state", "recommended_action"]
        )
        empty_signal.to_parquet(artifacts_dir / "signal_1m.parquet", index=False)
        _feature_health_report(
            pd.DataFrame(),
            cfg_hash,
            artifacts_dir,
            metadata={"hazard_enabled": 0, "coverage": coverage},
        )
        pipeline_health["hazard"] = {"enabled": False, "reason_if_disabled": hazard_report.get("reason")}
        _write_pipeline_health(pipeline_health, artifacts_dir)
        trades_baseline = baseline_trades
        trades_enhanced = pd.DataFrame()
        base_equity = pd.DataFrame({"equity": [cfg["backtest"]["initial_capital"]]}, index=[bars_5m.index[-1]])
        base_summary = compute_summary(trades_baseline, base_equity, cfg["backtest"]["initial_capital"])
        base_tail = compute_tail_metrics(trades_baseline)
        base_time = compute_time_in_trade(trades_baseline)
        compare_rows = [
            {
                "fold_id": 0,
                "policy_variant": "baseline_only",
                "entry_variant": "baseline",
                "baseline_pnl_net": base_summary["pnl_net"],
                "baseline_sharpe": base_summary["sharpe"],
                "baseline_max_drawdown": base_summary["max_drawdown"],
                "baseline_trade_count": base_summary["trade_count"],
                "baseline_win_rate": base_summary["win_rate"],
                "baseline_avg_win": base_summary["avg_win"],
                "baseline_avg_loss": base_summary["avg_loss"],
                "baseline_total_fees": base_summary["total_fees"],
                "baseline_total_slippage": base_summary["total_slippage"],
                "baseline_turnover": base_summary["turnover"],
                "baseline_worst_5pct_trade": base_tail["worst_5pct_trade"],
                "baseline_expected_shortfall_95": base_tail.get("expected_shortfall_95"),
                "baseline_worst_week_pnl": base_tail["worst_week_pnl"],
                "baseline_avg_time_in_trade_min": base_time,
                "hazard_enabled": 0,
            }
        ]
        if meta_enabled and event_source == "baseline_entries" and events_meta_veto is not None:
            meta_trades, _, _ = run_backtest(bars_5m, events_meta_veto, cfg)
            meta_equity = pd.DataFrame(
                {"equity": [cfg["backtest"]["initial_capital"]]}, index=[bars_5m.index[-1]]
            )
            meta_summary = compute_summary(meta_trades, meta_equity, cfg["backtest"]["initial_capital"])
            meta_tail = compute_tail_metrics(meta_trades)
            meta_time = compute_time_in_trade(meta_trades)
            compare_rows.append(
                {
                    "fold_id": 0,
                    "policy_variant": "baseline_only",
                    "entry_variant": "meta_veto",
                    "baseline_pnl_net": meta_summary["pnl_net"],
                    "baseline_sharpe": meta_summary["sharpe"],
                    "baseline_max_drawdown": meta_summary["max_drawdown"],
                    "baseline_trade_count": meta_summary["trade_count"],
                    "baseline_win_rate": meta_summary["win_rate"],
                    "baseline_avg_win": meta_summary["avg_win"],
                    "baseline_avg_loss": meta_summary["avg_loss"],
                    "baseline_total_fees": meta_summary["total_fees"],
                    "baseline_total_slippage": meta_summary["total_slippage"],
                    "baseline_turnover": meta_summary["turnover"],
                    "baseline_worst_5pct_trade": meta_tail["worst_5pct_trade"],
                    "baseline_expected_shortfall_95": meta_tail.get("expected_shortfall_95"),
                    "baseline_worst_week_pnl": meta_tail["worst_week_pnl"],
                    "baseline_avg_time_in_trade_min": meta_time,
                    "hazard_enabled": 0,
                }
            )
            meta_trades.to_parquet(artifacts_dir / "trades_meta_veto_baseline.parquet", index=False)
        compare = pd.DataFrame(compare_rows)
    else:
        hazard_trades = fallback_trades if event_source == "cusum_fallback" else baseline_trades
        hazard_df = build_hazard_dataset(hazard_trades, bars_1m, cfg)
        if hazard_df.empty:
            raise ValueError("Hazard dataset is empty; check input trades and bars.")
        hazard_df.to_parquet(artifacts_dir / "hazard_dataset.parquet", index=False)

        hazard_features = meta_features if meta_features is not None else _build_hazard_features(
            bars_1m, l2, agg_trades, cfg
        )
        if hazard_features.empty:
            raise ValueError("Hazard features are empty; check L2 data availability.")
        hazard_features.to_parquet(artifacts_dir / "hazard_features.parquet", index=False)
        _feature_health_report(
            hazard_features,
            cfg_hash,
            artifacts_dir,
            metadata={"hazard_enabled": 1, "coverage": coverage},
        )
        pipeline_health["hazard"] = {"enabled": True, "reason_if_disabled": None}
        _write_pipeline_health(pipeline_health, artifacts_dir)

        train_hazard_model(hazard_df, hazard_features, cfg, output_dir=artifacts_dir)
        model_path = artifacts_dir / "models" / "hazard_model.pkl"
        if model_path.exists():
            _write_signal_1m(hazard_features, model_path, cfg, artifacts_dir / "signal_1m.parquet")

        policy_variants = policy_variants or ["full_policy"]
        compare_base, trades_baseline, trades_enhanced = _run_enhanced_walkforward(
            bars_5m,
            bars_1m,
            hazard_df,
            hazard_features,
            cfg,
            artifacts_dir,
            policy_variants,
            events_override=events,
            entry_variant="baseline",
        )
        compare = compare_base
        if meta_enabled and event_source == "baseline_entries" and events_meta_veto is not None:
            compare_meta, trades_meta_base, trades_meta_enh = _run_enhanced_walkforward(
                bars_5m,
                bars_1m,
                hazard_df,
                hazard_features,
                cfg,
                artifacts_dir,
                policy_variants,
                events_override=events_meta_veto,
                entry_variant="meta_veto",
            )
            compare = pd.concat([compare_base, compare_meta], ignore_index=True)
            trades_meta_base.to_parquet(artifacts_dir / "trades_meta_veto_baseline.parquet", index=False)
            trades_meta_enh.to_parquet(artifacts_dir / "trades_meta_veto_enhanced.parquet", index=False)

    compare_path = artifacts_dir / "compare_summary.parquet"
    compare["config_hash"] = cfg_hash
    compare.to_parquet(compare_path, index=False)
    compare.to_json(artifacts_dir / "compare_summary.json", orient="records", date_format="iso")
    trades_baseline.to_parquet(artifacts_dir / "trades_baseline.parquet", index=False)
    trades_enhanced.to_parquet(artifacts_dir / "trades_enhanced.parquet", index=False)
    _flag_diagnostics(compare, artifacts_dir / "hazard_report.json", artifacts_dir, cfg_hash)

    report = {
        "config_hash": cfg_hash,
        "exchange": exchange,
        "symbol": symbol,
        "start": start,
        "end": end,
        "folds": int(compare["fold_id"].nunique()) if not compare.empty and "fold_id" in compare.columns else 0,
    }
    if not compare.empty:
        numeric = compare.select_dtypes(include=["number"])
        report["summary_mean"] = numeric.mean().to_dict()
        report["summary_median"] = numeric.median().to_dict()

    (artifacts_dir / "backtest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    hazard_report_path = artifacts_dir / "hazard_report.json"
    hazard_report = {}
    if hazard_report_path.exists():
        hazard_report = json.loads(hazard_report_path.read_text(encoding="utf-8"))
    auc = hazard_report.get("overall", {}).get("auc")
    model_card = (
        "# Hazard Model Card\n\n"
        f"- Config hash: {cfg_hash}\n"
        f"- Exchange: {exchange}\n"
        f"- Symbol: {symbol}\n"
        f"- Date range: {start} to {end}\n"
        f"- AUC (overall): {auc}\n"
    )
    (artifacts_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    return compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end dataset build, training, and backtests.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--exchange", required=True, choices=["binance", "deepcoin", "okx", "bybit"])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--source",
        default="vision",
        choices=["vision", "rest", "okx_hist", "okx_api", "okx_manual", "bybit_manual"],
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated datasets")
    parser.add_argument("--vision-dir", default=None, help="Local Binance Vision directory")
    parser.add_argument("--vision-auto-download", action="store_true", help="Download missing Binance Vision files")
    parser.add_argument("--okx-dir", default=None, help="Local OKX historical directory")
    parser.add_argument("--okx-cache-dir", default=None, help="Local OKX cache directory")
    parser.add_argument("--okx-auto-download", action="store_true", help="Download missing OKX files")
    parser.add_argument("--okx-modules", default=None, help="Comma-separated OKX datasets (overrides --datasets)")
    parser.add_argument("--okx-level", type=int, default=None, help="Order book depth level for OKX (default 50)")
    parser.add_argument("--okx-agg", default=None, help="OKX date aggregation (daily|monthly)")
    parser.add_argument("--okx-manual-candles-dir", default=None, help="Local OKX manual candles directory")
    parser.add_argument("--okx-manual-trades-dir", default=None, help="Local OKX manual trades directory")
    parser.add_argument("--okx-manual-book-dir", default=None, help="Local OKX manual book directory")
    parser.add_argument(
        "--okx-store-top-levels",
        type=int,
        default=None,
        help="Store top N levels for OKX orderbook (default 50)",
    )
    parser.add_argument(
        "--bybit-manual-root",
        default=None,
        help="Local BYBIT manual root directory (trades + book defaults)",
    )
    parser.add_argument("--bybit-manual-book-root", default=None, help="Local BYBIT manual book root directory")
    parser.add_argument(
        "--bybit-manual-trades-root",
        default=None,
        help="Local BYBIT manual trades root directory",
    )
    parser.add_argument(
        "--bybit-store-top-levels",
        type=int,
        default=None,
        help="Store top N levels for BYBIT orderbook (default 50)",
    )
    parser.add_argument(
        "--bybit-book-sample-ms",
        type=int,
        default=None,
        help="Emit BYBIT book snapshots every N ms (default 1000)",
    )
    parser.add_argument(
        "--meta-enabled",
        action="store_true",
        default=None,
        help="Enable meta-label entry filter (overrides config).",
    )
    parser.add_argument(
        "--meta-threshold",
        type=float,
        default=None,
        help="Meta-label threshold override.",
    )
    parser.add_argument(
        "--label-horizon-minutes",
        type=int,
        default=None,
        help="Override triple-barrier horizon minutes (applies to all regimes).",
    )
    parser.add_argument(
        "--disable-trend-pullback-long",
        action="store_true",
        help="Disable trend_pullback_long entry subtype.",
    )
    parser.add_argument(
        "--disable-trend-pullback-short",
        action="store_true",
        help="Disable trend_pullback_short entry subtype.",
    )
    parser.add_argument(
        "--disable-range-vwap-band",
        action="store_true",
        help="Disable range_vwap_band entry subtype.",
    )
    parser.add_argument(
        "--enable-trend-pullback-long-filter",
        action="store_true",
        default=None,
        help="Enable extra filter for trend_pullback_long (EMA slope + confirmation).",
    )
    parser.add_argument(
        "--end-inclusive-date",
        action="store_true",
        help="Treat end date as inclusive (internally adds 1 day to end).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild datasets even if cached partitions exist.",
    )
    parser.add_argument(
        "--policy-variants",
        default="full_policy",
        help="Comma-separated hazard policy variants (e.g., full_policy,hazard_exit_only).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    variants = [v.strip() for v in args.policy_variants.split(",") if v.strip()]

    compare = run_all(
        cfg,
        exchange=args.exchange,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        source=args.source,
        datasets=args.datasets,
        vision_dir=args.vision_dir,
        vision_auto_download=args.vision_auto_download,
        okx_dir=args.okx_dir,
        okx_cache_dir=args.okx_cache_dir,
        okx_auto_download=args.okx_auto_download,
        okx_modules=args.okx_modules,
        okx_level=args.okx_level,
        okx_agg=args.okx_agg,
        okx_manual_candles_dir=args.okx_manual_candles_dir,
        okx_manual_trades_dir=args.okx_manual_trades_dir,
        okx_manual_book_dir=args.okx_manual_book_dir,
        okx_store_top_levels=args.okx_store_top_levels,
        bybit_manual_root=args.bybit_manual_root,
        bybit_manual_book_root=args.bybit_manual_book_root,
        bybit_manual_trades_root=args.bybit_manual_trades_root,
        bybit_store_top_levels=args.bybit_store_top_levels,
        bybit_book_sample_ms=args.bybit_book_sample_ms,
        end_inclusive_date=args.end_inclusive_date,
        meta_enabled=args.meta_enabled,
        meta_threshold=args.meta_threshold,
        label_horizon_minutes=args.label_horizon_minutes,
        disable_trend_pullback_long=args.disable_trend_pullback_long,
        disable_trend_pullback_short=args.disable_trend_pullback_short,
        disable_range_vwap_band=args.disable_range_vwap_band,
        enable_trend_pullback_long_filter=args.enable_trend_pullback_long_filter,
        force_rebuild=args.force_rebuild,
        policy_variants=variants,
    )

    print("Comparison summary:")
    if compare.empty:
        print("No folds produced.")
    else:
        print(compare.to_string(index=False))
    _print_summary(compare, Path(cfg["paths"]["artifacts_dir"]))


if __name__ == "__main__":
    main()

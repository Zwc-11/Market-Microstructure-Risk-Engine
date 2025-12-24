from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.backtest.metrics import compute_summary
from src.backtest.simulator import run_backtest
from src.regime.regime import classify_regime
from src.strategy.entries_5m import generate_entries_5m


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
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
            raise ValueError("bars must have a DatetimeIndex or timestamp/ts column")
    return df.sort_index()


def generate_walkforward_folds(
    index: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int,
    embargo_minutes: int,
) -> List[Dict[str, pd.Timestamp]]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, step_days must be positive")

    index = pd.DatetimeIndex(index).dropna().sort_values()
    if index.empty:
        raise ValueError("index must contain at least one valid timestamp")
    start = index.min()
    end = index.max()
    embargo = pd.Timedelta(minutes=embargo_minutes)

    folds = []
    fold_id = 0
    train_start = start

    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end + embargo
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_start > end or test_end > end:
            break

        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        fold_id += 1
        train_start = train_start + pd.Timedelta(days=step_days)

    return folds


def _apply_mode(entries_cfg: Dict, mode: str) -> Dict:
    cfg = copy.deepcopy(entries_cfg)
    if mode == "both":
        return cfg
    if mode == "trend_only":
        cfg["range"]["enabled"] = False
        cfg["trend"]["enabled"] = True
        return cfg
    if mode == "range_only":
        cfg["trend"]["enabled"] = False
        cfg["range"]["enabled"] = True
        return cfg
    raise ValueError("mode must be one of: both, trend_only, range_only")


def _multiplied_config(config: Dict, fee_mult: float, slippage_mult: float) -> Dict:
    cfg = copy.deepcopy(config)
    cfg["backtest"]["fees_bps"]["taker"] = float(cfg["backtest"]["fees_bps"]["taker"]) * fee_mult
    cfg["backtest"]["slippage_bps"] = float(cfg["backtest"]["slippage_bps"]) * slippage_mult
    return cfg


def run_walkforward(
    bars: pd.DataFrame,
    config: Dict,
    modes: Optional[Sequence[str]] = None,
    fee_mults: Optional[Sequence[float]] = None,
    slippage_mults: Optional[Sequence[float]] = None,
    save_trades_dir: Optional[Path] = None,
) -> pd.DataFrame:
    bars = _ensure_datetime_index(bars)
    val_cfg = config["model"]["training"]["validation"]
    splits = val_cfg["splits"]
    folds = generate_walkforward_folds(
        bars.index,
        train_days=int(splits["train_days"]),
        test_days=int(splits["test_days"]),
        step_days=int(splits["step_days"]),
        embargo_minutes=int(val_cfg.get("embargo_minutes", 0)),
    )

    modes = list(modes) if modes is not None else ["both"]
    fee_mults = list(fee_mults) if fee_mults is not None else [1.0]
    slippage_mults = list(slippage_mults) if slippage_mults is not None else [1.0]

    rows = []

    if save_trades_dir is not None:
        save_trades_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        test_start = fold["test_start"]
        test_end = fold["test_end"]

        bars_slice = bars.loc[bars.index <= test_end]
        regime = classify_regime(bars_slice, config["regime"])

        for mode in modes:
            entries_cfg = _apply_mode(config["strategy"]["entries_5m"], mode)
            events_all = generate_entries_5m(bars_slice, regime, entries_cfg)

            if not events_all.empty:
                mask = (events_all["entry_ts"] >= test_start) & (events_all["entry_ts"] <= test_end)
                events = events_all.loc[mask].reset_index(drop=True)
            else:
                events = events_all

            for fee_mult in fee_mults:
                for slippage_mult in slippage_mults:
                    cfg = _multiplied_config(config, fee_mult, slippage_mult)
                    trades, equity, _ = run_backtest(bars_slice, events, cfg)

                    equity_test = equity.loc[(equity.index >= test_start) & (equity.index <= test_end)]
                    summary = compute_summary(
                        trades,
                        equity_test,
                        initial_capital=float(cfg["backtest"]["initial_capital"]),
                    )

                    row = {
                        "fold_id": fold["fold_id"],
                        "test_start": test_start,
                        "test_end": test_end,
                        "mode": mode,
                        "fee_mult": float(fee_mult),
                        "slippage_mult": float(slippage_mult),
                        "pnl_net": summary["pnl_net"],
                        "sharpe": summary["sharpe"],
                        "max_drawdown": summary["max_drawdown"],
                        "win_rate": summary["win_rate"],
                        "trade_count": summary["trade_count"],
                        "total_fees": summary["total_fees"],
                        "total_slippage": summary["total_slippage"],
                    }
                    rows.append(row)

                    if save_trades_dir is not None:
                        fee_tag = str(fee_mult).replace(".", "p")
                        slip_tag = str(slippage_mult).replace(".", "p")
                        out_path = save_trades_dir / f"fold_{fold['fold_id']}_{mode}_fee{fee_tag}_slip{slip_tag}.parquet"
                        trades.to_parquet(out_path, index=False)

    return pd.DataFrame(rows)

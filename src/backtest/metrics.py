from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_sharpe(returns: pd.Series, annualization_factor: Optional[float] = None) -> float:
    if returns.empty:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    sharpe = float(mean / std)
    if annualization_factor is not None:
        sharpe *= float(np.sqrt(annualization_factor))
    return sharpe


def compute_summary(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
    annualization_factor: Optional[float] = None,
) -> Dict[str, float]:
    if trades.empty:
        return {
            "pnl_net": 0.0,
            "pnl_gross": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "trade_count": 0,
            "turnover": 0.0,
        }

    pnl_net = trades["net_pnl"].sum()
    pnl_gross = trades["gross_pnl"].sum()
    total_fees = trades["fees"].sum()
    total_slippage = trades["slippage"].sum()
    trade_count = int(len(trades))

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] < 0]
    win_rate = float(len(wins) / trade_count) if trade_count else 0.0
    avg_win = float(wins["net_pnl"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["net_pnl"].mean()) if not losses.empty else 0.0

    if "equity" in equity.columns:
        returns = equity["equity"].pct_change().fillna(0.0)
    else:
        returns = pd.Series(dtype="float64")

    sharpe = compute_sharpe(returns, annualization_factor=annualization_factor)
    max_drawdown = compute_max_drawdown(equity["equity"]) if "equity" in equity.columns else 0.0

    turnover = float(trades["notional"].sum() / initial_capital) if initial_capital > 0 else 0.0

    return {
        "pnl_net": float(pnl_net),
        "pnl_gross": float(pnl_gross),
        "total_fees": float(total_fees),
        "total_slippage": float(total_slippage),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "trade_count": trade_count,
        "turnover": float(turnover),
    }


def compute_tail_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"worst_5pct_trade": 0.0, "expected_shortfall_95": 0.0, "worst_week_pnl": 0.0}

    worst_5pct = float(trades["net_pnl"].quantile(0.05))
    cutoff = max(1, int(np.ceil(0.05 * len(trades))))
    tail = trades["net_pnl"].nsmallest(cutoff)
    expected_shortfall = float(tail.mean()) if not tail.empty else 0.0
    if "exit_ts" in trades.columns:
        exit_ts = pd.to_datetime(trades["exit_ts"])
        weekly = trades.copy()
        weekly["week"] = exit_ts.dt.to_period("W").astype(str)
        weekly_pnl = weekly.groupby("week")["net_pnl"].sum()
        worst_week = float(weekly_pnl.min()) if not weekly_pnl.empty else 0.0
    else:
        worst_week = 0.0

    return {
        "worst_5pct_trade": worst_5pct,
        "expected_shortfall_95": expected_shortfall,
        "worst_week_pnl": worst_week,
    }


def compute_time_in_trade(trades: pd.DataFrame) -> float:
    if trades.empty or "entry_ts" not in trades.columns or "exit_ts" not in trades.columns:
        return 0.0
    entry = pd.to_datetime(trades["entry_ts"])
    exit_ts = pd.to_datetime(trades["exit_ts"])
    durations = (exit_ts - entry).dt.total_seconds() / 60.0
    return float(durations.mean())


def compute_hazard_exit_counts(trades: pd.DataFrame) -> Dict[str, int]:
    if trades.empty or "exit_reason" not in trades.columns:
        return {"hazard_exits": 0, "fail_fast_exits": 0}
    hazard_exits = int((trades["exit_reason"] == "hazard_exit").sum())
    fail_fast = int((trades["exit_reason"] == "hazard_fail_fast").sum())
    return {"hazard_exits": hazard_exits, "fail_fast_exits": fail_fast}

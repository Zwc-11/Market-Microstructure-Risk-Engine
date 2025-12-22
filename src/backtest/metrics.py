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

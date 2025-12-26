from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.backtest.metrics import compute_summary, compute_tail_metrics


def _find_latest_artifacts(root: Path) -> Path:
    root = Path(root)
    direct = root / "compare_summary.parquet"
    if direct.exists():
        return root
    candidates = list(root.rglob("compare_summary.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No compare_summary.parquet found under {root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def _load_trade_log(artifacts_dir: Path) -> Tuple[pd.DataFrame, str]:
    candidates = [
        "trades_enhanced.parquet",
        "trades_baseline.parquet",
        "trade_log_hazard.parquet",
        "trade_log_baseline.parquet",
        "trades.parquet",
    ]
    for name in candidates:
        path = artifacts_dir / name
        if path.exists():
            return pd.read_parquet(path), name
    raise FileNotFoundError(
        "Missing required trade log. Checked:\n"
        + "\n".join(str(artifacts_dir / name) for name in candidates)
    )


def _build_equity(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"equity": [initial_capital]})
    if "exit_ts" not in trades.columns:
        return pd.DataFrame({"equity": [initial_capital + trades["net_pnl"].sum()]})
    trades = trades.sort_values("exit_ts")
    equity = initial_capital + trades["net_pnl"].cumsum()
    return pd.DataFrame({"equity": equity.values}, index=pd.to_datetime(trades["exit_ts"]))


def sweep_costs(
    artifacts: Optional[Path] = None,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    artifacts_dir = _find_latest_artifacts(Path(artifacts or "artifacts"))
    trades, trade_log_name = _load_trade_log(artifacts_dir)

    if trades.empty:
        for col in ("gross_pnl", "notional", "fees", "slippage", "net_pnl"):
            if col not in trades.columns:
                trades[col] = pd.Series(dtype="float64")
    elif "gross_pnl" not in trades.columns or "notional" not in trades.columns:
        raise ValueError("Trade log missing required columns: gross_pnl, notional.")

    grid = [(0, 0), (1, 1), (2, 2), (4, 2), (4, 4)]
    rows: List[Dict[str, float]] = []

    for fee_bps, slip_bps in grid:
        fee_rate = float(fee_bps) / 10000.0
        slip_rate = float(slip_bps) / 10000.0
        fees = 2.0 * trades["notional"].astype(float) * fee_rate
        slippage = 2.0 * trades["notional"].astype(float) * slip_rate
        net_pnl = trades["gross_pnl"].astype(float) - fees - slippage

        adjusted = trades.copy()
        adjusted["fees"] = fees
        adjusted["slippage"] = slippage
        adjusted["net_pnl"] = net_pnl

        equity = _build_equity(adjusted, initial_capital)
        summary = compute_summary(adjusted, equity, initial_capital)
        tail = compute_tail_metrics(adjusted)

        rows.append(
            {
                "fees_bps": float(fee_bps),
                "slippage_bps": float(slip_bps),
                "pnl_net": summary["pnl_net"],
                "pnl_gross": summary["pnl_gross"],
                "total_fees": summary["total_fees"],
                "total_slippage": summary["total_slippage"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "win_rate": summary["win_rate"],
                "trade_count": summary["trade_count"],
                "turnover": summary["turnover"],
                "worst_5pct_trade": tail["worst_5pct_trade"],
                "expected_shortfall_95": tail["expected_shortfall_95"],
                "trade_log_used": trade_log_name,
            }
        )

    out = pd.DataFrame(rows)
    out_path = artifacts_dir / "cost_sweep.parquet"
    out.to_parquet(out_path, index=False)

    lines = [
        "# Cost Sweep",
        "",
        f"- Trade log: {trade_log_name}",
        f"- Initial capital (assumed): {initial_capital}",
        "",
        "| fees_bps | slippage_bps | pnl_net | sharpe | max_drawdown | trade_count | turnover |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['fees_bps']:.0f} | {row['slippage_bps']:.0f} | {row['pnl_net']:.4f} | "
            f"{row['sharpe']:.4f} | {row['max_drawdown']:.4f} | {int(row['trade_count'])} | "
            f"{row['turnover']:.4f} |"
        )
    (artifacts_dir / "cost_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep backtest costs using existing trades.")
    parser.add_argument("--artifacts", default=None, help="Artifacts directory (default: latest under ./artifacts).")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital to compute turnover/drawdown (default: 10000).",
    )
    args = parser.parse_args()
    sweep_costs(Path(args.artifacts) if args.artifacts else None, initial_capital=args.initial_capital)


if __name__ == "__main__":
    main()

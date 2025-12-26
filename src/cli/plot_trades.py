from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


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


def _load_trade_log(artifacts_dir: Path, trade_log: Optional[str]) -> pd.DataFrame:
    if trade_log:
        path = artifacts_dir / trade_log
        if not path.exists():
            raise FileNotFoundError(f"Trade log not found: {path}")
        return pd.read_parquet(path)
    for name in ("trades_baseline.parquet", "trades_enhanced.parquet"):
        path = artifacts_dir / name
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError("No trades_baseline.parquet or trades_enhanced.parquet found.")


def _select_trades(trades: pd.DataFrame, n_best: int, n_worst: int, n_random: int) -> pd.DataFrame:
    trades = trades.copy()
    if "net_pnl" not in trades.columns:
        raise ValueError("Trade log missing net_pnl column.")
    best = trades.nlargest(n_best, "net_pnl") if n_best > 0 else trades.iloc[0:0]
    worst = trades.nsmallest(n_worst, "net_pnl") if n_worst > 0 else trades.iloc[0:0]
    remaining = trades.drop(best.index.union(worst.index))
    random = remaining.sample(n=min(n_random, len(remaining)), random_state=42) if n_random > 0 else remaining.iloc[0:0]
    return pd.concat([best, worst, random], ignore_index=True)


def _load_bars(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "bars_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing bars_1m.parquet: {path}")
    bars = pd.read_parquet(path)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        bars = bars.set_index("timestamp")
    elif "ts" in bars.columns:
        bars["ts"] = pd.to_datetime(bars["ts"])
        bars = bars.set_index("ts")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bars_1m must have a DatetimeIndex or timestamp/ts column")
    return bars.sort_index()


def _pick_ohlc_cols(bars: pd.DataFrame) -> Tuple[str, str, str, str]:
    if {"mid_open", "mid_high", "mid_low", "mid_close"}.issubset(bars.columns):
        return "mid_open", "mid_high", "mid_low", "mid_close"
    if {"open", "high", "low", "close"}.issubset(bars.columns):
        return "open", "high", "low", "close"
    raise ValueError("bars_1m missing required OHLC columns.")


def _plot_trade(
    bars: pd.DataFrame,
    trade: pd.Series,
    out_path: Path,
    symbol: Optional[str],
) -> None:
    import matplotlib.pyplot as plt

    open_col, high_col, low_col, close_col = _pick_ohlc_cols(bars)
    entry_ts = pd.to_datetime(trade["entry_ts"])
    exit_ts = pd.to_datetime(trade["exit_ts"])
    window_start = entry_ts - pd.Timedelta(minutes=30)
    window_end = exit_ts + pd.Timedelta(minutes=30)
    window = bars.loc[(bars.index >= window_start) & (bars.index <= window_end)]
    if window.empty:
        return

    x = range(len(window))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (_, row) in enumerate(window.iterrows()):
        o = float(row[open_col])
        h = float(row[high_col])
        l = float(row[low_col])
        c = float(row[close_col])
        color = "green" if c >= o else "red"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, min(o, c)),
                0.6,
                abs(c - o) if abs(c - o) > 0 else 0.0001,
                color=color,
                alpha=0.6,
            )
        )

    entry_idx = window.index.searchsorted(entry_ts, side="left")
    exit_idx = window.index.searchsorted(exit_ts, side="left")
    ax.axvline(entry_idx, color="blue", linestyle="--", linewidth=1)
    ax.axvline(exit_idx, color="black", linestyle="--", linewidth=1)

    side = "LONG" if int(trade.get("side", 1)) > 0 else "SHORT"
    pnl = float(trade.get("net_pnl", 0.0))
    reason = str(trade.get("exit_reason", ""))
    title = f"{symbol or ''} {side} pnl={pnl:.2f} reason={reason}"
    ax.set_title(title.strip())
    ax.set_xticks([])
    ax.set_ylabel("Price")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trades(
    artifacts: Optional[Path],
    processed_dir: Optional[Path],
    trade_log: Optional[str],
    symbol: Optional[str],
    n_best: int,
    n_worst: int,
    n_random: int,
) -> Path:
    artifacts_dir = _find_latest_artifacts(Path(artifacts or "artifacts"))
    trades = _load_trade_log(artifacts_dir, trade_log)
    if trades.empty:
        raise ValueError("Trade log is empty.")
    bars = _load_bars(Path(processed_dir or "data/processed"))

    selected = _select_trades(trades, n_best=n_best, n_worst=n_worst, n_random=n_random)
    if selected.empty:
        raise ValueError("No trades selected for plotting.")

    plot_dir = artifacts_dir / "plots" / "trades"
    lines = ["# Trade Plots", ""]

    for idx, trade in selected.iterrows():
        event_id = trade.get("event_id") or f"trade_{idx}"
        filename = f"{event_id}.png"
        out_path = plot_dir / filename
        _plot_trade(bars, trade, out_path, symbol or trade.get("symbol"))
        lines.append(f"- {filename} | net_pnl={float(trade.get('net_pnl', 0.0)):.4f}")

    index_path = plot_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sample trades with 1m candlesticks.")
    parser.add_argument("--artifacts", default=None, help="Artifacts directory (default: latest under ./artifacts).")
    parser.add_argument("--processed-dir", default=None, help="Processed data directory (default: data/processed).")
    parser.add_argument("--trade-log", default=None, help="Trade log file name inside artifacts.")
    parser.add_argument("--symbol", default=None, help="Symbol label for plot titles.")
    parser.add_argument("--n_best", type=int, default=5, help="Number of best trades to plot.")
    parser.add_argument("--n_worst", type=int, default=5, help="Number of worst trades to plot.")
    parser.add_argument("--n_random", type=int, default=5, help="Number of random trades to plot.")
    args = parser.parse_args()
    plot_trades(
        Path(args.artifacts) if args.artifacts else None,
        Path(args.processed_dir) if args.processed_dir else None,
        args.trade_log,
        args.symbol,
        args.n_best,
        args.n_worst,
        args.n_random,
    )


if __name__ == "__main__":
    main()

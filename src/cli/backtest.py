from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.simulator import run_backtest
from src.regime.regime import classify_regime
from src.strategy.entries_5m import generate_entries_5m


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_bars(processed_dir: Path) -> pd.DataFrame:
    bars_path = processed_dir / "bars_5m.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"Missing bars file: {bars_path}")
    return pd.read_parquet(bars_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline backtest.")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bars_5m = _load_bars(processed_dir)
    regime = classify_regime(bars_5m, cfg["regime"])
    events = generate_entries_5m(bars_5m, regime, cfg["strategy"]["entries_5m"])

    trades, equity, summary = run_backtest(bars_5m, events, cfg)

    trades_path = artifacts_dir / "trades.parquet"
    equity_path = artifacts_dir / "equity.parquet"
    trades.to_parquet(trades_path, index=False)
    equity.to_parquet(equity_path)

    print("Backtest summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

from pathlib import Path

import pandas as pd

from src.cli.sweep_costs import sweep_costs


def test_sweep_costs_smoke(tmp_path):
    artifacts = tmp_path
    trades = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "entry_ts": pd.Timestamp("2025-01-01 00:00:00"),
                "exit_ts": pd.Timestamp("2025-01-01 00:05:00"),
                "gross_pnl": 10.0,
                "fees": 0.0,
                "slippage": 0.0,
                "net_pnl": 10.0,
                "notional": 100.0,
            }
        ]
    )
    trades.to_parquet(artifacts / "trades_baseline.parquet", index=False)
    pd.DataFrame([{"fold_id": 0}]).to_parquet(artifacts / "compare_summary.parquet", index=False)

    out = sweep_costs(artifacts, initial_capital=10000.0)
    assert (artifacts / "cost_sweep.parquet").exists()
    assert (artifacts / "cost_sweep.md").exists()
    assert len(out) == 5

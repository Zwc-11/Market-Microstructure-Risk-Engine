from pathlib import Path

import pandas as pd

from src.cli.analyze_run import analyze_run


def test_analyze_run_smoke(tmp_path):
    artifacts = tmp_path
    compare = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "policy_variant": "baseline_only",
                "baseline_pnl_net": 1.0,
                "baseline_trade_count": 1,
            }
        ]
    )
    compare.to_parquet(artifacts / "compare_summary.parquet", index=False)
    trades = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "entry_ts": pd.Timestamp("2025-01-01 00:00:00"),
                "exit_ts": pd.Timestamp("2025-01-01 00:05:00"),
                "gross_pnl": 5.0,
                "fees": 1.0,
                "slippage": 0.5,
                "net_pnl": 3.5,
                "notional": 100.0,
                "regime": "RANGE",
                "reason": "range_upper_touch",
                "exit_reason": "barrier_pt",
            }
        ]
    )
    trades.to_parquet(artifacts / "trades_enhanced.parquet", index=False)
    (artifacts / "pipeline_health.json").write_text("{}", encoding="utf-8")
    (artifacts / "hazard_report.json").write_text("{}", encoding="utf-8")

    signal = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01 00:01:00"), pd.Timestamp("2025-01-01 00:02:00")],
            "P_end": [0.1, 0.9],
            "hazard_state": ["normal", "high"],
            "recommended_action": ["hold", "exit"],
        }
    )
    signal.to_parquet(artifacts / "signal_1m.parquet", index=False)

    report = analyze_run(artifacts)
    assert (artifacts / "attribution.json").exists()
    assert (artifacts / "attribution.md").exists()
    assert report["net_pnl"] == 3.5

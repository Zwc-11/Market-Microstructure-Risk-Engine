from pathlib import Path

import pandas as pd

from src.cli.plot_trades import plot_trades


def test_plot_trades_smoke(tmp_path):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": idx,
            "mid_open": [100.0] * len(idx),
            "mid_high": [101.0] * len(idx),
            "mid_low": [99.0] * len(idx),
            "mid_close": [100.5] * len(idx),
        }
    )
    bars.to_parquet(processed / "bars_1m.parquet", index=False)

    trades = pd.DataFrame(
        {
            "event_id": ["evt1"],
            "entry_ts": [idx[2]],
            "exit_ts": [idx[6]],
            "side": [1],
            "entry_price": [100.0],
            "exit_price": [100.5],
            "net_pnl": [1.0],
            "exit_reason": ["barrier_pt"],
        }
    )
    trades.to_parquet(artifacts / "trades_baseline.parquet", index=False)
    (artifacts / "compare_summary.parquet").write_bytes(b"PAR1")

    index_path = plot_trades(artifacts, processed, None, "BTCUSDT", 1, 0, 0)
    assert index_path.exists()
    pngs = list((artifacts / "plots" / "trades").glob("*.png"))
    assert pngs

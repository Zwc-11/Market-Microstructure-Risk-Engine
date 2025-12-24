from pathlib import Path

import pandas as pd

from src.data.bybit_manual import load_bybit_manual_book_depth


def test_bybit_orderbook_replay_emits_snapshots(tmp_path):
    root = tmp_path / "bybit_root"
    folder = root / "2025-12-01_BTCUSDT_ob200.data"
    folder.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        [
            '{"type":"snapshot","ts":1764547200000,"data":{"s":"BTCUSDT","seq":1,"u":1,"a":[["101","1"]],"b":[["100","1"]]}}',
            '{"type":"delta","ts":1764547205000,"data":{"s":"BTCUSDT","seq":2,"u":2,"a":[["102","1"]],"b":[["99","1"]]}}',
            '{"type":"delta","ts":1764547215000,"data":{"s":"BTCUSDT","seq":3,"u":3,"a":[["103","1"]],"b":[["98","1"]]}}',
        ]
    )
    (folder / "orderbook.data").write_text(payload + "\n", encoding="utf-8")

    start_ms = int(pd.Timestamp("2025-12-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-12-02", tz="UTC").timestamp() * 1000)

    df, _ = load_bybit_manual_book_depth(
        book_root=root,
        symbol="BTCUSDT",
        start_ms=start_ms,
        end_ms=end_ms,
        store_levels=2,
        diagnostics_dir=tmp_path,
        book_sample_ms=1000,
        start_str="2025-12-01",
        end_str="2025-12-02",
    )

    assert len(df) >= 2
    assert (tmp_path / "bybit_book_depth_sampling_debug.json").exists()

from pathlib import Path
import shutil

import pandas as pd

from src.data.bybit_manual import load_bybit_manual_book_depth


def _setup_root(tmp_path: Path) -> Path:
    fixture = Path(__file__).resolve().parent / "fixtures" / "bybit_small" / "orderbook.jsonl"
    root = tmp_path / "bybit_root"
    folder = root / "2025-12-01_BTCUSDT_ob200.data"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, folder / "orderbook.data")
    return root


def test_bybit_manual_parse_snapshot_delta(tmp_path):
    root = _setup_root(tmp_path)
    start_ms = int(pd.Timestamp("2025-12-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-12-02", tz="UTC").timestamp() * 1000)

    df, debug = load_bybit_manual_book_depth(
        book_root=root,
        symbol="BTCUSDT",
        start_ms=start_ms,
        end_ms=end_ms,
        store_levels=2,
        diagnostics_dir=tmp_path,
        start_str="2025-12-01",
        end_str="2025-12-02",
    )

    assert debug["rows_seen"] > 0
    assert len(df) >= 2

    first = df.iloc[0]
    assert first["bid_price_1"] == 50000.0
    assert first["bid_price_2"] == 49990.0
    assert first["ask_price_1"] == 50010.0
    assert first["ask_price_2"] == 50020.0

    last = df.iloc[-1]
    assert last["bid_price_1"] == 49980.0
    assert last["ask_price_1"] == 50030.0
    assert (tmp_path / "bybit_book_depth_debug.json").exists()
    assert (tmp_path / "bybit_book_depth_sampling_debug.json").exists()

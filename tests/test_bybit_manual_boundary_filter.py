from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.data.bybit_manual import load_bybit_manual_book_depth


def _setup_root(tmp_path: Path) -> Path:
    fixture = Path(__file__).resolve().parent / "fixtures" / "bybit_small" / "orderbook.jsonl"
    root = tmp_path / "bybit_root"
    folder = root / "2025-12-01_BTCUSDT_ob200.data"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, folder / "orderbook.data")
    return root


def test_bybit_manual_boundary_filter(tmp_path):
    root = _setup_root(tmp_path)
    start_ms = int(pd.Timestamp("2025-12-02", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-12-03", tz="UTC").timestamp() * 1000)

    with pytest.raises(ValueError, match="All book_depth rows excluded by date filter"):
        load_bybit_manual_book_depth(
            book_root=root,
            symbol="BTCUSDT",
            start_ms=start_ms,
            end_ms=end_ms,
            store_levels=2,
            diagnostics_dir=tmp_path,
            start_str="2025-12-02",
            end_str="2025-12-03",
        )

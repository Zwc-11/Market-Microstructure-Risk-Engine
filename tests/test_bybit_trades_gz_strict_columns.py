import gzip
from pathlib import Path

import pytest

from src.data.bybit_manual import iter_bybit_trades_gz_strict


def test_bybit_trades_gz_strict_columns(tmp_path):
    root = tmp_path / "trades"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "BTCUSDT_2025-12-01.csv.gz"
    payload = "timestamp,symbol,side,price\n1764547200000,BTCUSDT,Buy,100.0\n"

    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(payload)

    with pytest.raises(ValueError) as excinfo:
        list(
            iter_bybit_trades_gz_strict(
                root,
                "BTCUSDT",
                start_ms=0,
                end_ms=10**15,
                chunksize=2,
            )
        )

    msg = str(excinfo.value).lower()
    assert "missing" in msg
    assert "size" in msg

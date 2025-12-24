import gzip
from pathlib import Path

import pandas as pd

from src.data.bybit_manual import iter_bybit_trades_gz_strict


def test_bybit_trades_gz_seconds_timestamp(tmp_path):
    root = tmp_path / "trades"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "BTCUSDT_2025-12-01.csv.gz"
    payload = (
        "timestamp,symbol,side,price,size,trdMatchID\n"
        "1764547200,BTCUSDT,Buy,100.0,0.5,abc1\n"
    )

    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(payload)

    chunks = list(
        iter_bybit_trades_gz_strict(
            root,
            "BTCUSDT",
            start_ms=1764547200000,
            end_ms=1764547260000,
            chunksize=10,
        )
    )
    assert chunks
    df = pd.concat(chunks, ignore_index=True)
    assert df.loc[0, "ts_ms"] == 1764547200000

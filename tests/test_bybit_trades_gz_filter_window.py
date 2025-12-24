from pathlib import Path

import pandas as pd

from src.data.bybit_manual import iter_bybit_trades_gz_strict


def test_bybit_trades_gz_filter_window():
    root = Path(__file__).resolve().parent / "fixtures" / "bybit_trades_small"
    start_ms = 1764547200000
    end_ms = 1764547440000

    chunks = list(
        iter_bybit_trades_gz_strict(
            root,
            "BTCUSDT",
            start_ms=start_ms,
            end_ms=end_ms,
            chunksize=3,
        )
    )
    assert chunks
    df = pd.concat(chunks, ignore_index=True)

    assert list(df.columns) == [
        "ts_ms",
        "symbol",
        "side",
        "price",
        "size",
        "trade_id",
        "raw_source",
    ]
    assert len(df) == 4
    assert df["ts_ms"].min() >= start_ms
    assert df["ts_ms"].max() < end_ms
    assert df["ts_ms"].is_monotonic_increasing
    assert set(df["side"].unique()) <= {"buy", "sell"}
    assert set(df["symbol"].str.upper().unique()) == {"BTCUSDT"}
    assert set(df["raw_source"].unique()) == {"bybit_csv_gz"}
    assert df.duplicated(subset=["ts_ms", "side", "price", "size", "trade_id"]).sum() == 0

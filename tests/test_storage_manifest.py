from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schemas import SCHEMAS
from src.data.storage import validate_dataframe, write_partitioned_parquet


def _klines_df():
    return pd.DataFrame(
        {
            "ts": [0, 60_000, 120_000],
            "open": [1.0, 1.1, 1.2],
            "high": [1.5, 1.6, 1.7],
            "low": [0.9, 1.0, 1.1],
            "close": [1.2, 1.3, 1.4],
            "volume": [10.0, 11.0, 12.0],
            "close_ts": [60_000, 120_000, 180_000],
            "quote_volume": [0.0, 0.0, 0.0],
            "trade_count": [1, 1, 1],
            "taker_buy_base": [0.0, 0.0, 0.0],
            "taker_buy_quote": [0.0, 0.0, 0.0],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        }
    )


def test_manifest_written(tmp_path):
    df = _klines_df()
    manifest_path, manifest = write_partitioned_parquet(
        df,
        tmp_path,
        exchange="binance",
        market="usdm",
        symbol="BTCUSDT",
        dataset="klines_1m",
        expected_interval_ms=60_000,
        max_gap_ms=60_000,
    )

    assert manifest_path.exists()
    assert manifest["row_count"] == 3
    assert manifest["time_range_covered"][0] == 0
    assert manifest["time_range_covered"][1] == 120_000


def test_validate_duplicates_and_gaps():
    df = _klines_df()
    schema = SCHEMAS["klines_1m"]

    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_dataframe(df_dup, schema, expected_interval_ms=60_000, max_gap_ms=60_000)

    df_gap = df.copy()
    df_gap.loc[1, "ts"] = 300_000
    with pytest.raises(ValueError):
        validate_dataframe(df_gap, schema, expected_interval_ms=60_000, max_gap_ms=60_000)

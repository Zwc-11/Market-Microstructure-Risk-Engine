from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.okx import OKXHistoricalDownloader


def test_okx_parser_trades():
    root = Path(__file__).resolve().parent / "fixtures" / "okx_small"
    downloader = OKXHistoricalDownloader(local_dir=root, auto_download=False, date_aggr_type="daily")
    start_ms = int(pd.Timestamp("2025-09-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-09-02", tz="UTC").timestamp() * 1000)

    df = downloader.fetch_dataset("trades", "BTC-USDT-SWAP", start_ms, end_ms)

    assert df["agg_id"].tolist() == [1, 2]
    assert df["price"].tolist() == [26000.0, 26001.0]
    assert df["is_buyer_maker"].tolist() == [False, True]
    assert df["symbol"].unique().tolist() == ["BTC-USDT-SWAP"]

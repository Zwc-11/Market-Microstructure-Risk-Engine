from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.okx import OKXHistoricalDownloader


def test_okx_parser_book_depth():
    root = Path(__file__).resolve().parent / "fixtures" / "okx_small"
    downloader = OKXHistoricalDownloader(
        local_dir=root,
        auto_download=False,
        depth_levels=2,
        date_aggr_type="daily",
        orderbook_level=50,
    )
    start_ms = int(pd.Timestamp("2025-09-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-09-02", tz="UTC").timestamp() * 1000)

    df = downloader.fetch_dataset("book_depth", "BTC-USDT-SWAP", start_ms, end_ms)

    assert df["update_id"].tolist() == [10, 11]
    assert df["bid_price_1"].tolist() == [26000.0, 26000.0]
    assert df["ask_price_2"].tolist() == [26001.0, 26001.0]
    assert df["symbol"].unique().tolist() == ["BTC-USDT-SWAP"]

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.binance import BinanceVision


def _vision_root() -> Path:
    path = Path(__file__).resolve().parent / "fixtures" / "vision_small"
    if not path.exists():
        raise FileNotFoundError("Vision fixtures not found")
    return path


def test_vision_parser_bookticker():
    vision = BinanceVision(local_dir=_vision_root(), auto_download=False)
    start_ms = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-01-02", tz="UTC").timestamp() * 1000)

    df = vision.fetch_dataset("book_ticker", "BTCUSDT", start_ms, end_ms)

    assert df["update_id"].tolist() == [10, 11]
    assert df["bid_price"].tolist() == [100.0, 100.1]
    assert df["ask_price"].tolist() == [100.1, 100.2]
    assert df["symbol"].unique().tolist() == ["BTCUSDT"]

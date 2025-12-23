from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.binance import BinanceVision


def _vision_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "vision"


def test_vision_parser_aggtrades():
    vision = BinanceVision(local_dir=_vision_root(), auto_download=False)
    start_ms = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2025-01-02", tz="UTC").timestamp() * 1000)

    df = vision.fetch_dataset("agg_trades", "BTCUSDT", start_ms, end_ms)

    assert df["agg_id"].tolist() == [1, 2]
    assert df["is_buyer_maker"].tolist() == [True, False]
    assert df["symbol"].unique().tolist() == ["BTCUSDT"]

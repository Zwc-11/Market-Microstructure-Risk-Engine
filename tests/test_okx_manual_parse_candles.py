from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.okx_manual import sniff_file, _read_dataset_file, parse_candles


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "okx_manual_small" / "candles" / "candles.csv"


def test_okx_manual_parse_candles():
    path = _fixture_path()
    sniff = sniff_file(path)
    df = _read_dataset_file(path, sniff)
    parsed = parse_candles(df, "BTC-USDT-SWAP")

    assert list(parsed["ts"]) == [1764547200000, 1764547260000]
    assert parsed["open"].tolist() == [26000.0, 26050.0]
    assert parsed["close"].tolist() == [26050.0, 26100.0]
    assert parsed["symbol"].unique().tolist() == ["BTC-USDT-SWAP"]

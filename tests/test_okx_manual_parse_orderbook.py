from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.okx_manual import sniff_file, _read_dataset_file, parse_book_depth


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "okx_manual_small" / "book" / "book.jsonl"


def test_okx_manual_parse_orderbook():
    path = _fixture_path()
    sniff = sniff_file(path)
    df = _read_dataset_file(path, sniff)
    parsed, source_depth = parse_book_depth(df, "BTC-USDT-SWAP", store_levels=2)

    assert parsed["bid_price_1"].tolist() == [26000.0, 26000.1]
    assert parsed["ask_price_2"].tolist() == [26001.0, 26001.1]
    assert source_depth >= 2

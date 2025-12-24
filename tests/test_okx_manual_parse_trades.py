from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.okx_manual import sniff_file, _read_dataset_file, parse_trades


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "okx_manual_small" / "trades" / "trades.csv"


def test_okx_manual_parse_trades():
    path = _fixture_path()
    sniff = sniff_file(path)
    df = _read_dataset_file(path, sniff)
    parsed, unknown_side = parse_trades(df, "BTC-USDT-SWAP")

    assert parsed["agg_id"].tolist() == [1, 2]
    assert parsed["price"].tolist() == [26010.0, 26020.0]
    assert parsed["is_buyer_maker"].tolist() == [False, True]
    assert unknown_side == 0

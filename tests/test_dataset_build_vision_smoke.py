from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import build_dataset as build_mod


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "processed")},
        "data": {
            "quality": {"max_gap_seconds": 120},
            "exchanges": [{"name": "binance", "symbols": ["BTCUSDT"], "depth_levels": 10}],
        },
    }


def test_build_dataset_vision_smoke(tmp_path):
    cfg = _config(tmp_path)
    vision_root = Path(__file__).resolve().parent / "fixtures" / "vision"

    build_mod.build_dataset(
        cfg,
        exchange="binance",
        symbol="BTCUSDT",
        start="2025-01-01",
        end="2025-01-02",
        source="vision",
        datasets="klines_1m,agg_trades,book_ticker,book_depth",
        build_bars=False,
        vision_dir=str(vision_root),
    )

    raw_dir = Path(cfg["paths"]["raw_dir"])
    base = raw_dir / "binance" / "usdm" / "BTCUSDT"
    for dataset in ["klines_1m", "agg_trades", "book_ticker", "book_depth"]:
        manifest = base / dataset / "MANIFEST.json"
        assert manifest.exists()

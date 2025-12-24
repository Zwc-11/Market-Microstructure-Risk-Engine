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
            "exchanges": [{"name": "okx", "symbols": ["BTC-USDT-SWAP"], "depth_levels": 2}],
            "okx": {"depth_levels": 2, "date_aggr_type": "daily"},
        },
    }


def test_okx_build_dataset_smoke(tmp_path):
    cfg = _config(tmp_path)
    okx_root = Path(__file__).resolve().parent / "fixtures" / "okx_small"

    build_mod.build_dataset(
        cfg,
        exchange="okx",
        symbol="BTC-USDT-SWAP",
        start="2025-09-01",
        end="2025-09-02",
        source="okx_hist",
        datasets="trades,book_depth",
        build_bars=True,
        okx_cache_dir=str(okx_root),
        okx_auto_download=False,
    )

    raw_dir = Path(cfg["paths"]["raw_dir"])
    base = raw_dir / "okx" / "swap" / "BTC-USDT-SWAP"
    assert (base / "agg_trades" / "MANIFEST.json").exists()
    assert (base / "book_depth" / "MANIFEST.json").exists()
    assert (Path(cfg["paths"]["processed_dir"]) / "bars_1m.parquet").exists()
    assert (Path(cfg["paths"]["processed_dir"]) / "bars_5m.parquet").exists()

from pathlib import Path

from src.cli.build_dataset import build_dataset


def test_build_dataset_bybit_manual_smoke(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "bybit_trades_small"

    cfg = {
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "artifacts_dir": str(artifacts_dir),
        },
        "data": {
            "quality": {"min_coverage": 0.0},
            "bybit": {"manual": {"root": str(fixture_root), "allow_baseline_only": True}},
        },
    }

    build_dataset(
        cfg,
        exchange="bybit",
        symbol="BTCUSDT",
        start="2025-12-01",
        end="2025-12-02",
        source="bybit_manual",
        datasets="agg_trades",
        bybit_manual_root=str(fixture_root),
        bybit_manual_trades_root=str(fixture_root),
        bybit_manual_book_root=str(tmp_path / "missing_book"),
    )

    part_dir = processed_dir / "agg_trades" / "exchange=bybit" / "symbol=BTCUSDT" / "date=2025-12-01"
    parts = list(part_dir.glob("part-*.parquet"))
    assert parts
    assert (artifacts_dir / "diagnostics" / "bybit_trades_debug.json").exists()
    assert (artifacts_dir / "run_manifest.json").exists()

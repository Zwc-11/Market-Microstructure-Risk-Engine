from pathlib import Path

import pandas as pd

from src.cli.build_dataset import _partition_ready, _write_partition_manifest


def test_partition_ready_matches_manifest(tmp_path):
    part_dir = tmp_path / "date=2025-01-01"
    part_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ts": [1, 2, 3], "price": [1.0, 1.1, 1.2]})
    df.to_parquet(part_dir / "part-000.parquet", index=False)

    source = tmp_path / "source.csv"
    source.write_text("x", encoding="utf-8")

    stats = {"rows": 3, "min_ts": 1, "max_ts": 3}
    _write_partition_manifest(part_dir, "agg_trades", [source], stats)

    assert _partition_ready(part_dir, "agg_trades", [source], False, time_col="ts") is True


def test_partition_ready_mismatch_raises(tmp_path):
    part_dir = tmp_path / "date=2025-01-01"
    part_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ts": [1, 2, 3], "price": [1.0, 1.1, 1.2]})
    df.to_parquet(part_dir / "part-000.parquet", index=False)

    source = tmp_path / "source.csv"
    source.write_text("x", encoding="utf-8")

    stats = {"rows": 3, "min_ts": 1, "max_ts": 3}
    _write_partition_manifest(part_dir, "agg_trades", [source], stats)

    source.write_text("changed", encoding="utf-8")

    try:
        _partition_ready(part_dir, "agg_trades", [source], False, time_col="ts")
    except ValueError as exc:
        assert "Cached partition mismatch" in str(exc)
    else:
        raise AssertionError("Expected mismatch to raise ValueError")

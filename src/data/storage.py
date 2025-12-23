from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.data.schemas import SCHEMAS, Schema, enforce_schema


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _partition_path(
    root: Path, exchange: str, market: str, symbol: str, dataset: str, date_str: str
) -> Path:
    return root / exchange / market / symbol / dataset / f"date={date_str}"


def _manifest_path(root: Path, exchange: str, market: str, symbol: str, dataset: str) -> Path:
    return root / exchange / market / symbol / dataset / "MANIFEST.json"


def _load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(path: Path, manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def validate_dataframe(
    df: pd.DataFrame,
    schema: Schema,
    time_col: str = "ts",
    expected_interval_ms: Optional[int] = None,
    max_gap_ms: Optional[int] = None,
) -> None:
    if df.empty:
        return

    if time_col not in df.columns:
        raise ValueError("Missing timestamp column for validation")

    ordered = df.sort_values(time_col)
    if not ordered[time_col].is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic")

    if schema.primary_key:
        dup = ordered.duplicated(subset=schema.primary_key)
        if dup.any():
            raise ValueError("Duplicate primary keys detected")

    if expected_interval_ms is not None:
        diffs = ordered[time_col].diff().dropna()
        if max_gap_ms is None:
            max_gap_ms = expected_interval_ms
        if (diffs > max_gap_ms).any():
            raise ValueError("Time gaps exceed allowed threshold")


def write_partitioned_parquet(
    df: pd.DataFrame,
    root: Path,
    exchange: str,
    market: str,
    symbol: str,
    dataset: str,
    schema_version: str = "v1",
    time_col: str = "ts",
    expected_interval_ms: Optional[int] = None,
    max_gap_ms: Optional[int] = None,
) -> Tuple[Path, Dict]:
    if dataset not in SCHEMAS:
        raise ValueError(f"Unknown dataset schema: {dataset}")
    schema = SCHEMAS[dataset]

    df = enforce_schema(df, schema)
    df = df.sort_values(time_col)

    if df.empty:
        raise ValueError("No data to write")

    ts = pd.to_datetime(df[time_col], unit="ms", utc=True)
    df = df.assign(_date=ts.dt.strftime("%Y-%m-%d"))

    root = Path(root)
    for date_str, part in df.groupby("_date"):
        part = part.drop(columns=["_date"])
        path = _partition_path(root, exchange, market, symbol, dataset, date_str)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "part-000.parquet"

        if file_path.exists():
            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, part], ignore_index=True)
            combined = combined.sort_values(time_col)
            combined = combined.drop_duplicates(subset=schema.primary_key, keep="last")
            validate_dataframe(
                combined, schema, time_col=time_col, expected_interval_ms=expected_interval_ms, max_gap_ms=max_gap_ms
            )
            combined.to_parquet(file_path, index=False)
        else:
            validate_dataframe(
                part, schema, time_col=time_col, expected_interval_ms=expected_interval_ms, max_gap_ms=max_gap_ms
            )
            part.to_parquet(file_path, index=False)

    manifest_path = _manifest_path(root, exchange, market, symbol, dataset)
    files = sorted(manifest_path.parent.glob("date=*/part-000.parquet"))
    row_count = 0
    min_ts = None
    max_ts = None
    file_entries = []

    for fpath in files:
        part = pd.read_parquet(fpath, columns=[time_col])
        if not part.empty:
            row_count += len(part)
            part_min = int(part[time_col].min())
            part_max = int(part[time_col].max())
            min_ts = part_min if min_ts is None else min(min_ts, part_min)
            max_ts = part_max if max_ts is None else max(max_ts, part_max)

        file_entries.append(
            {
                "path": str(fpath.relative_to(root)),
                "sha256": _sha256_file(fpath),
                "size_bytes": fpath.stat().st_size,
                "mtime": int(fpath.stat().st_mtime),
            }
        )

    manifest = {
        "schema_version": schema_version,
        "dataset_name": dataset,
        "exchange": exchange,
        "market": market,
        "symbol": symbol,
        "time_range_covered": [min_ts, max_ts],
        "row_count": row_count,
        "files": file_entries,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    _write_manifest(manifest_path, manifest)
    return manifest_path, manifest

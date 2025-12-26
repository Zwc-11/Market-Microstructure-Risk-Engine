from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.bybit_manual import (
    _iter_l2_records,
    _parse_ts_ms,
    scan_orderbook_payloads,
    sniff_file,
)


def _to_utc(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return pd.to_datetime(value, unit="ms", utc=True).isoformat()


def _merge_counts(dest: Dict[str, int], src: Dict[str, int]) -> Dict[str, int]:
    for key, value in src.items():
        dest[key] = int(dest.get(key, 0)) + int(value)
    return dest


def inspect_bybit_l2(
    root: Path,
    symbol: str,
    date: str,
    level: int,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    root = Path(root)
    payloads = scan_orderbook_payloads(root, symbol)
    start = pd.Timestamp(date, tz="UTC")
    start_ms = int(start.value // 1_000_000)
    end_ms = int((start + pd.Timedelta(days=1)).value // 1_000_000)

    results: Dict[str, object] = {
        "root": str(root.resolve()),
        "symbol": symbol,
        "date": date,
        "level": int(level),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "files": [],
        "ts_scale_counts": {
            "seconds": 0,
            "milliseconds": 0,
            "microseconds": 0,
            "nanoseconds": 0,
            "invalid": 0,
        },
        "totals": {
            "parsed_records": 0,
            "events_in_window": 0,
            "min_ts_ms": None,
            "max_ts_ms": None,
            "min_ts_utc": None,
            "max_ts_utc": None,
            "type_counts": {},
            "decode_errors": 0,
        },
    }

    totals_type_counts: Counter = Counter()
    min_ts_total: Optional[int] = None
    max_ts_total: Optional[int] = None
    output_dir = Path(output_dir) if output_dir is not None else Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bybit_l2_inspect.json"

    for payload in payloads:
        path = Path(payload["selected"])
        sniff = sniff_file(path)
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None

        file_debug: Dict[str, object] = {
            "lines_seen": 0,
            "parsed_records": 0,
            "decode_errors": 0,
            "first_error": None,
            "first_error_sample": None,
            "ts_scale_counts": {
                "seconds": 0,
                "milliseconds": 0,
                "microseconds": 0,
                "nanoseconds": 0,
                "invalid": 0,
            },
        }
        first_events = []
        last_events: deque = deque(maxlen=3)
        type_counts: Counter = Counter()
        events_in_window = 0
        min_ts = None
        max_ts = None

        try:
            for record in _iter_l2_records(path, sniff, file_debug):
                if len(first_events) < 3:
                    first_events.append(record)
                last_events.append(record)
                if not isinstance(record, dict):
                    continue
                data = record.get("data")
                if not isinstance(data, dict):
                    continue
                ts_ms = _parse_ts_ms(record.get("ts") or data.get("ts"), debug=file_debug)
                if ts_ms is None:
                    continue
                min_ts = ts_ms if min_ts is None else min(min_ts, ts_ms)
                max_ts = ts_ms if max_ts is None else max(max_ts, ts_ms)
                if start_ms <= ts_ms < end_ms:
                    events_in_window += 1
                event_type = record.get("type") or record.get("event_type") or ""
                if event_type:
                    type_counts[str(event_type).lower()] += 1
        except Exception as exc:
            file_info = {
                "folder": str(payload["folder"]),
                "selected": str(path.resolve()),
                "candidates": [str(Path(p).resolve()) for p in payload["candidates"]],
                "size_bytes": size_bytes,
                "container": sniff.container,
                "format": sniff.data_format,
                "inner_name": sniff.inner_name,
                "magic_bytes": sniff.magic_bytes,
                "parse_error": str(exc),
            }
            results["files"].append(file_info)
            results["error"] = str(exc)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            raise

        lines_seen = int(file_debug.get("lines_seen") or 0)
        decode_errors = int(file_debug.get("decode_errors") or 0)
        error_rate = decode_errors / lines_seen if lines_seen else 0.0

        file_info = {
            "folder": str(payload["folder"]),
            "selected": str(path.resolve()),
            "candidates": [str(Path(p).resolve()) for p in payload["candidates"]],
            "size_bytes": size_bytes,
            "container": sniff.container,
            "format": sniff.data_format,
            "inner_name": sniff.inner_name,
            "magic_bytes": sniff.magic_bytes,
            "parsed_records": int(file_debug.get("parsed_records") or 0),
            "lines_seen": lines_seen,
            "decode_errors": decode_errors,
            "error_rate": error_rate,
            "first_error": file_debug.get("first_error"),
            "first_error_sample": file_debug.get("first_error_sample"),
            "first_events": first_events,
            "last_events": list(last_events),
            "min_ts_ms": min_ts,
            "max_ts_ms": max_ts,
            "min_ts_utc": _to_utc(min_ts),
            "max_ts_utc": _to_utc(max_ts),
            "events_in_window": events_in_window,
            "type_counts": {k: int(v) for k, v in type_counts.items()},
            "ts_scale_counts": {
                k: int(v) for k, v in (file_debug.get("ts_scale_counts") or {}).items()
            },
        }
        results["files"].append(file_info)

        parsed_records = int(file_debug.get("parsed_records") or 0)
        results["totals"]["parsed_records"] += parsed_records
        results["totals"]["events_in_window"] += events_in_window
        totals_type_counts.update(type_counts)
        results["totals"]["decode_errors"] += decode_errors
        if min_ts is not None:
            min_ts_total = min_ts if min_ts_total is None else min(min_ts_total, min_ts)
        if max_ts is not None:
            max_ts_total = max_ts if max_ts_total is None else max(max_ts_total, max_ts)
        results["ts_scale_counts"] = _merge_counts(
            results["ts_scale_counts"],
            {k: int(v) for k, v in (file_debug.get("ts_scale_counts") or {}).items()},
        )

    results["totals"]["min_ts_ms"] = min_ts_total
    results["totals"]["max_ts_ms"] = max_ts_total
    results["totals"]["min_ts_utc"] = _to_utc(min_ts_total)
    results["totals"]["max_ts_utc"] = _to_utc(max_ts_total)
    results["totals"]["type_counts"] = {k: int(v) for k, v in totals_type_counts.items()}

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Bybit manual L2 exports.")
    parser.add_argument("--root", required=True, help="Root directory containing Bybit manual exports.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT.")
    parser.add_argument("--date", required=True, help="UTC date YYYY-MM-DD.")
    parser.add_argument("--level", type=int, default=200, help="Orderbook level (default: 200).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for bybit_l2_inspect.json (default: ./artifacts).",
    )
    args = parser.parse_args()

    inspect_bybit_l2(
        root=Path(args.root),
        symbol=args.symbol,
        date=args.date,
        level=args.level,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()

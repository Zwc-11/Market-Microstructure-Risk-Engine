from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile

import pandas as pd

from src.data.schemas import L2_LEVELS, SCHEMAS, enforce_schema


@dataclass(frozen=True)
class SniffResult:
    container: str
    data_format: str
    inner_name: Optional[str]


def _init_book_debug(start_ms: int, end_ms: int, start_str: str, end_str: str) -> Dict:
    return {
        "rows_seen": 0,
        "rows_parsed_ok": 0,
        "rows_kept": 0,
        "min_ts_seen": None,
        "max_ts_seen": None,
        "min_ts_utc": None,
        "max_ts_utc": None,
        "instId_counts": Counter(),
        "action_counts": Counter(),
        "bid_levels_max": 0,
        "ask_levels_max": 0,
        "bid_level_columns": Counter(),
        "ask_level_columns": Counter(),
        "requested_start": start_str,
        "requested_end": end_str,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
    }


def _finalize_book_debug(debug: Dict) -> Dict:
    min_ts = debug.get("min_ts_seen")
    max_ts = debug.get("max_ts_seen")
    if min_ts is not None:
        debug["min_ts_utc"] = pd.to_datetime(min_ts, unit="ms", utc=True).isoformat()
    if max_ts is not None:
        debug["max_ts_utc"] = pd.to_datetime(max_ts, unit="ms", utc=True).isoformat()

    inst_counts = debug.get("instId_counts", Counter())
    if not isinstance(inst_counts, Counter):
        inst_counts = Counter(inst_counts)
    debug["instId_top5"] = [
        {"instId": inst, "count": int(count)} for inst, count in inst_counts.most_common(5)
    ]
    debug["instId_counts"] = {k: int(v) for k, v in inst_counts.items()}

    action_counts = debug.get("action_counts", Counter())
    debug["action_counts"] = {k: int(v) for k, v in action_counts.items()}
    debug["bid_level_columns"] = {k: int(v) for k, v in debug["bid_level_columns"].items()}
    debug["ask_level_columns"] = {k: int(v) for k, v in debug["ask_level_columns"].items()}
    return debug


def _write_book_debug(debug: Dict, diagnostics_dir: Optional[Path]) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _finalize_book_debug(debug)
    path = diagnostics_dir / "book_depth_debug.json"
    path.write_text(json.dumps(debug, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_skip(path: Path) -> bool:
    for part in path.parts:
        name = part.lower()
        if name in {"logs", "log", "__macosx"}:
            return True
    return False


def scan_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and not _should_skip(p)]
    return sorted(files, key=lambda p: str(p).lower())


def sniff_file(path: Path) -> SniffResult:
    with open(path, "rb") as f:
        head = f.read(4)

    container = "raw"
    if head.startswith(b"PK"):
        container = "zip"
    elif head.startswith(b"\x1f\x8b"):
        container = "gzip"

    sample = _read_text_sample(path, container)
    data_format = _detect_text_format(sample)
    inner_name = None
    if container == "zip":
        inner_name = _select_zip_member(path)
    return SniffResult(container=container, data_format=data_format, inner_name=inner_name)


def _read_text_sample(path: Path, container: str, max_bytes: int = 4096) -> str:
    if container == "zip":
        with ZipFile(path) as zf:
            name = _select_zip_member(path)
            with zf.open(name) as f:
                data = f.read(max_bytes)
    elif container == "gzip":
        with gzip.open(path, "rb") as f:
            data = f.read(max_bytes)
    else:
        with open(path, "rb") as f:
            data = f.read(max_bytes)
    return data.decode("utf-8", errors="ignore")


def _select_zip_member(path: Path) -> str:
    with ZipFile(path) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n]
    if not members:
        raise ValueError(f"No files in zip archive: {path}")
    return members[0]


def _detect_text_format(sample: str) -> str:
    sample = sample.strip()
    if not sample:
        return "unknown"
    first = sample[0]
    if first in {"{", "["}:
        lines = [line.strip() for line in sample.splitlines() if line.strip()]
        if len(lines) > 1 and all(line.startswith("{") for line in lines[:3]):
            return "jsonl"
        return "json"
    if "," in sample.splitlines()[0]:
        return "csv"
    return "unknown"


def _open_inner(path: Path, sniff: SniffResult):
    if sniff.container == "zip":
        zf = ZipFile(path)
        name = sniff.inner_name or _select_zip_member(path)
        return zf.open(name), zf
    if sniff.container == "gzip":
        return gzip.open(path, "rb"), None
    return open(path, "rb"), None


def _read_csv(path: Path, sniff: SniffResult) -> pd.DataFrame:
    handle, parent = _open_inner(path, sniff)
    try:
        return pd.read_csv(handle)
    finally:
        handle.close()
        if parent is not None:
            parent.close()


def _read_json(path: Path, sniff: SniffResult) -> pd.DataFrame:
    handle, parent = _open_inner(path, sniff)
    try:
        text = handle.read().decode("utf-8", errors="ignore")
    finally:
        handle.close()
        if parent is not None:
            parent.close()

    if sniff.data_format == "jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return pd.DataFrame(rows)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return pd.DataFrame(data["data"])
        return pd.DataFrame([data])
    return pd.DataFrame()


def _parse_ts_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype("int64")
        return values.where(values >= 10**12, values * 1000)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return (parsed.view("int64") // 1_000_000).astype("int64")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(name: str) -> str:
        name = name.strip().lower().replace(" ", "")
        name = name.replace("-", "_")
        return name

    return df.rename(columns={c: _norm(c) for c in df.columns})


def parse_candles(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    rename = {
        "instid": "symbol",
        "instrument_name": "symbol",
        "ts": "ts",
        "ts_ms": "ts",
        "timestamp": "ts",
        "open_time": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "volume",
        "vol_ccy": "volume_ccy",
        "vol_quote": "quote_volume",
        "volccyquote": "quote_volume",
        "volccy": "volume_ccy",
    }
    df = df.rename(columns=rename)

    required = ["ts", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OKX candles missing columns: {missing}")

    df["ts"] = _parse_ts_ms(df["ts"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    df["quote_volume"] = pd.to_numeric(df.get("quote_volume", 0.0), errors="coerce").fillna(0.0)
    df["close_ts"] = df["ts"] + 60_000
    df["trade_count"] = 0
    df["taker_buy_base"] = 0.0
    df["taker_buy_quote"] = 0.0
    df["symbol"] = symbol
    return enforce_schema(df, SCHEMAS["klines_1m"])


def parse_trades(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, int]:
    df = _normalize_columns(df)
    rename = {
        "instid": "symbol",
        "instrument_name": "symbol",
        "trade_id": "agg_id",
        "tradeid": "agg_id",
        "id": "agg_id",
        "ts": "ts",
        "ts_ms": "ts",
        "timestamp": "ts",
        "time": "ts",
        "created_time": "ts",
        "price": "price",
        "px": "price",
        "size": "qty",
        "sz": "qty",
        "qty": "qty",
        "side": "side",
    }
    df = df.rename(columns=rename)

    required = ["ts", "price", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OKX trades missing columns: {missing}")

    if "agg_id" not in df.columns:
        df["agg_id"] = range(1, len(df) + 1)

    df["ts"] = _parse_ts_ms(df["ts"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["symbol"] = symbol

    side = df.get("side")
    unknown_side = 0
    if side is None:
        df["is_buyer_maker"] = False
        unknown_side = len(df)
    else:
        side = side.astype(str).str.lower()
        unknown_side = int((~side.isin({"buy", "sell"})).sum())
        df["is_buyer_maker"] = side == "sell"

    return enforce_schema(df, SCHEMAS["agg_trades"]), unknown_side


def _parse_depth_arrays(raw, levels: int) -> List[float]:
    if isinstance(raw, list):
        arr = raw
    else:
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return [float("nan")] * (levels * 2)
    out = []
    for i in range(levels):
        if i < len(arr):
            price = float(arr[i][0])
            size = float(arr[i][1])
        else:
            price = float("nan")
            size = float("nan")
        out.extend([price, size])
    return out


def parse_book_depth(
    df: pd.DataFrame, symbol: str, store_levels: int
) -> Tuple[pd.DataFrame, int]:
    df = _normalize_columns(df)
    ts_col = _pick_column(df, ["ts", "ts_ms", "timestamp", "time", "timems", "time_ms"])
    if ts_col is None:
        raise ValueError("OKX orderbook missing timestamp column")

    update_col = _pick_column(df, ["update_id", "seq_id", "seqid", "exchtimems", "checksum"])
    if update_col is None:
        update_col = ts_col

    has_arrays = "bids" in df.columns and "asks" in df.columns
    has_levels = any(col.startswith("bid_") or col.startswith("ask_") for col in df.columns)

    out = pd.DataFrame({"ts": _parse_ts_ms(df[ts_col])})
    out["update_id"] = pd.to_numeric(df[update_col], errors="coerce").fillna(out["ts"]).astype("int64")

    source_depth = 0
    if has_arrays:
        bids = df["bids"].apply(lambda x: _parse_depth_arrays(x, store_levels))
        asks = df["asks"].apply(lambda x: _parse_depth_arrays(x, store_levels))
        bid_matrix = pd.DataFrame(bids.tolist(), columns=[f"bid_{i}" for i in range(store_levels * 2)])
        ask_matrix = pd.DataFrame(asks.tolist(), columns=[f"ask_{i}" for i in range(store_levels * 2)])

        for lvl in range(1, store_levels + 1):
            out[f"bid_price_{lvl}"] = bid_matrix[f"bid_{(lvl - 1) * 2}"].astype(float)
            out[f"bid_size_{lvl}"] = bid_matrix[f"bid_{(lvl - 1) * 2 + 1}"].astype(float)
            out[f"ask_price_{lvl}"] = ask_matrix[f"ask_{(lvl - 1) * 2}"].astype(float)
            out[f"ask_size_{lvl}"] = ask_matrix[f"ask_{(lvl - 1) * 2 + 1}"].astype(float)
        source_depth = _infer_depth_from_arrays(df["bids"])
    elif has_levels:
        for lvl in range(1, store_levels + 1):
            bid_px = _pick_column(df, _depth_candidates("bid", "price", lvl))
            bid_sz = _pick_column(df, _depth_candidates("bid", "size", lvl))
            ask_px = _pick_column(df, _depth_candidates("ask", "price", lvl))
            ask_sz = _pick_column(df, _depth_candidates("ask", "size", lvl))
            out[f"bid_price_{lvl}"] = pd.to_numeric(df[bid_px], errors="coerce") if bid_px else float("nan")
            out[f"bid_size_{lvl}"] = pd.to_numeric(df[bid_sz], errors="coerce") if bid_sz else float("nan")
            out[f"ask_price_{lvl}"] = pd.to_numeric(df[ask_px], errors="coerce") if ask_px else float("nan")
            out[f"ask_size_{lvl}"] = pd.to_numeric(df[ask_sz], errors="coerce") if ask_sz else float("nan")
        source_depth = _infer_depth_from_columns(df.columns)
    else:
        raise ValueError("OKX orderbook missing bids/asks arrays or per-level columns")

    if store_levels < L2_LEVELS:
        for lvl in range(store_levels + 1, L2_LEVELS + 1):
            out[f"bid_price_{lvl}"] = float("nan")
            out[f"bid_size_{lvl}"] = float("nan")
            out[f"ask_price_{lvl}"] = float("nan")
            out[f"ask_size_{lvl}"] = float("nan")

    out["symbol"] = symbol
    return enforce_schema(out, SCHEMAS["book_depth"]), source_depth


def _infer_depth_from_arrays(series: pd.Series) -> int:
    depth = 0
    for item in series.head(10):
        try:
            arr = item if isinstance(item, list) else json.loads(item)
            depth = max(depth, len(arr))
        except json.JSONDecodeError:
            continue
    return depth


def _infer_depth_from_columns(columns: Iterable[str]) -> int:
    depth = 0
    for col in columns:
        parts = col.split("_")
        if len(parts) < 2:
            continue
        if parts[0] in {"bid", "ask"} and parts[1].isdigit():
            depth = max(depth, int(parts[1]))
        if len(parts) >= 3 and parts[0] in {"bid", "ask"} and parts[1].isdigit():
            depth = max(depth, int(parts[1]))
    return depth


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _depth_candidates(side: str, field: str, level: int) -> List[str]:
    field_candidates = ["px", "price"] if field == "price" else ["qty", "size", "sz"]
    candidates = []
    for f in field_candidates:
        candidates.extend(
            [
                f"{side}_{level}_{f}",
                f"{side}_{f}_{level}",
                f"{side}{level}_{f}",
                f"{side}{f}{level}",
                f"{side}_{level}_{f}",
            ]
        )
    return candidates


def _read_dataset_file(path: Path, sniff: SniffResult) -> pd.DataFrame:
    if sniff.data_format == "csv":
        return _read_csv(path, sniff)
    if sniff.data_format in {"json", "jsonl"}:
        return _read_json(path, sniff)
    raise ValueError(f"Unsupported format: {sniff.data_format}")


def _filter_time(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    if "ts" not in df.columns:
        return df
    return df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]


def _date_coverage(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "ts" not in df.columns:
        return {}
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
    counts = ts.dt.strftime("%Y-%m-%d").value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def load_manual_datasets(
    candles_dir: Path,
    trades_dir: Path,
    book_dir: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    store_levels: int,
    allow_missing_book: bool = False,
    book_writer: Optional[Callable[[pd.DataFrame], None]] = None,
    diagnostics_dir: Optional[Path] = None,
    start_str: str = "",
    end_str: str = "",
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    store_levels = min(int(store_levels), L2_LEVELS)
    datasets: Dict[str, pd.DataFrame] = {}
    run_manifest = {
        "source": "okx_manual",
        "symbol": symbol,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "files": [],
        "datasets": {},
    }

    if candles_dir:
        datasets["klines_1m"] = _parse_folder(
            candles_dir,
            symbol,
            start_ms,
            end_ms,
            store_levels,
            "klines_1m",
            run_manifest,
        )
    if trades_dir:
        datasets["agg_trades"] = _parse_folder(
            trades_dir,
            symbol,
            start_ms,
            end_ms,
            store_levels,
            "agg_trades",
            run_manifest,
        )
    if book_dir:
        book_debug = _init_book_debug(start_ms, end_ms, start_str, end_str)
        try:
            if book_writer is None:
                datasets["book_depth"] = _parse_folder(
                    book_dir,
                    symbol,
                    start_ms,
                    end_ms,
                    store_levels,
                    "book_depth",
                    run_manifest,
                    book_debug=book_debug,
                )
            else:
                _stream_book_folder(
                    book_dir,
                    symbol,
                    start_ms,
                    end_ms,
                    store_levels,
                    run_manifest,
                    book_writer,
                    book_debug=book_debug,
                )
                datasets["book_depth"] = pd.DataFrame()

            if book_debug["rows_seen"] > 0 and book_debug["rows_kept"] == 0:
                _write_book_debug(book_debug, diagnostics_dir)
                min_utc = book_debug.get("min_ts_utc", "unknown")
                max_utc = book_debug.get("max_ts_utc", "unknown")
                raise ValueError(
                    f"All book_depth rows excluded by date filter. Observed ts range: {min_utc}.. {max_utc}. "
                    f"Requested: [{start_str},{end_str}). Fix: set end=next_day or add --end-inclusive-date."
                )
        except Exception as exc:
            _write_book_debug(book_debug, diagnostics_dir)
            if not allow_missing_book:
                raise
            run_manifest.setdefault("warnings", []).append(
                {"dataset": "book_depth", "error": str(exc)}
            )
            datasets["book_depth"] = pd.DataFrame()

    for name, df in datasets.items():
        existing = run_manifest.get("datasets", {}).get(name)
        if existing is None:
            run_manifest["datasets"][name] = {
                "rows": int(len(df)),
                "date_coverage": _date_coverage(df),
            }
        else:
            existing["rows"] = int(len(df))

    run_manifest["stored_depth"] = store_levels
    return datasets, run_manifest


def _stream_book_folder(
    folder: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    store_levels: int,
    run_manifest: Dict,
    writer: Callable[[pd.DataFrame], None],
    book_debug: Optional[Dict] = None,
) -> None:
    files = scan_files(folder)
    if not files:
        return

    for path in files:
        sniff = sniff_file(path)
        if sniff.data_format not in {"json", "jsonl"}:
            df = _read_dataset_file(path, sniff)
            parsed, source_depth = parse_book_depth(df, symbol, store_levels)
            filtered = _filter_time(parsed, start_ms, end_ms)
            if not filtered.empty:
                writer(filtered)
            run_manifest["files"].append(
                {
                    "dataset": "book_depth",
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                    "container": sniff.container,
                    "format": sniff.data_format,
                    "inner_name": sniff.inner_name,
                    "rows_parsed": len(parsed),
                    "rows_kept": len(filtered),
                }
            )
            run_manifest.setdefault("metadata", {})["source_depth"] = int(source_depth)
            continue

        rows_parsed, rows_kept, source_depth, coverage = _stream_book_file(
            path, sniff, symbol, store_levels, start_ms, end_ms, writer, book_debug=book_debug
        )
        run_manifest["files"].append(
            {
                "dataset": "book_depth",
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "container": sniff.container,
                "format": sniff.data_format,
                "inner_name": sniff.inner_name,
                "rows_parsed": rows_parsed,
                "rows_kept": rows_kept,
            }
        )
        run_manifest.setdefault("metadata", {})["source_depth"] = int(source_depth)
        dataset_meta = run_manifest.setdefault("datasets", {}).setdefault("book_depth", {})
        date_cov = dataset_meta.get("date_coverage", {})
        for date_str, count in coverage.items():
            date_cov[date_str] = date_cov.get(date_str, 0) + int(count)
        dataset_meta["date_coverage"] = date_cov


def _stream_book_file(
    path: Path,
    sniff: SniffResult,
    symbol: str,
    store_levels: int,
    start_ms: int,
    end_ms: int,
    writer: Callable[[pd.DataFrame], None],
    book_debug: Optional[Dict] = None,
    chunk_size: int = 5000,
) -> Tuple[int, int, int, Dict[str, int]]:
    handle, parent = _open_inner(path, sniff)
    rows = []
    rows_seen = 0
    rows_parsed = 0
    rows_kept = 0
    source_depth = 0
    coverage: Dict[str, int] = {}
    try:
        for raw_line in handle:
            line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
            line = line.strip()
            if not line:
                continue
            rows_seen += 1
            if book_debug is not None:
                book_debug["rows_seen"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_parsed += 1
            if book_debug is not None:
                book_debug["rows_parsed_ok"] += 1
            ts = _record_ts_ms(record)
            if ts is None or ts < start_ms or ts >= end_ms:
                if book_debug is not None and ts is not None:
                    _update_book_debug_range(book_debug, ts)
                _update_book_debug_from_record(book_debug, record)
                continue
            row, depth = _book_row_from_record(record, symbol, store_levels)
            source_depth = max(source_depth, depth)
            rows.append(row)
            rows_kept += 1
            if book_debug is not None:
                book_debug["rows_kept"] += 1
                if ts is not None:
                    _update_book_debug_range(book_debug, ts)
                _update_book_debug_from_record(book_debug, record)
            date_str = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")
            coverage[date_str] = coverage.get(date_str, 0) + 1
            if len(rows) >= chunk_size:
                df = pd.DataFrame(rows)
                df = enforce_schema(df, SCHEMAS["book_depth"])
                writer(df)
                rows = []
        if rows:
            df = pd.DataFrame(rows)
            df = enforce_schema(df, SCHEMAS["book_depth"])
            writer(df)
    finally:
        handle.close()
        if parent is not None:
            parent.close()
    return rows_parsed, rows_kept, source_depth, coverage


def _record_ts_ms(record: Dict) -> Optional[int]:
    for key in ("ts", "ts_ms", "timestamp", "time", "time_ms"):
        if key in record:
            value = record[key]
            try:
                value_int = int(value)
                return value_int if value_int >= 10**12 else value_int * 1000
            except (TypeError, ValueError):
                ts = pd.to_datetime(value, errors="coerce", utc=True)
                if pd.isna(ts):
                    return None
                return int(ts.value // 1_000_000)
    return None


def _book_row_from_record(record: Dict, symbol: str, store_levels: int) -> Tuple[Dict, int]:
    bids = record.get("bids", [])
    asks = record.get("asks", [])
    depth = max(len(bids), len(asks))
    ts = _record_ts_ms(record)
    update_id = record.get("seqId") or record.get("checksum") or ts

    row: Dict[str, object] = {"ts": ts, "update_id": update_id, "symbol": symbol}

    for lvl in range(1, store_levels + 1):
        if lvl - 1 < len(bids):
            row[f"bid_price_{lvl}"] = float(bids[lvl - 1][0])
            row[f"bid_size_{lvl}"] = float(bids[lvl - 1][1])
        else:
            row[f"bid_price_{lvl}"] = float("nan")
            row[f"bid_size_{lvl}"] = float("nan")
        if lvl - 1 < len(asks):
            row[f"ask_price_{lvl}"] = float(asks[lvl - 1][0])
            row[f"ask_size_{lvl}"] = float(asks[lvl - 1][1])
        else:
            row[f"ask_price_{lvl}"] = float("nan")
            row[f"ask_size_{lvl}"] = float("nan")

    if store_levels < L2_LEVELS:
        for lvl in range(store_levels + 1, L2_LEVELS + 1):
            row[f"bid_price_{lvl}"] = float("nan")
            row[f"bid_size_{lvl}"] = float("nan")
            row[f"ask_price_{lvl}"] = float("nan")
            row[f"ask_size_{lvl}"] = float("nan")

    return row, depth


def _update_book_debug_ts(debug: Optional[Dict], ts_values: pd.Series) -> None:
    if debug is None or ts_values.empty:
        return
    ts_min = int(ts_values.min())
    ts_max = int(ts_values.max())
    _update_book_debug_range(debug, ts_min)
    _update_book_debug_range(debug, ts_max)


def _update_book_debug_range(debug: Dict, ts_ms: int) -> None:
    min_ts = debug.get("min_ts_seen")
    max_ts = debug.get("max_ts_seen")
    debug["min_ts_seen"] = ts_ms if min_ts is None else min(min_ts, ts_ms)
    debug["max_ts_seen"] = ts_ms if max_ts is None else max(max_ts, ts_ms)


def _update_book_debug_from_df(debug: Optional[Dict], df: pd.DataFrame) -> None:
    if debug is None or df.empty:
        return
    if "instid" in df.columns:
        for inst in df["instid"].dropna().astype(str):
            debug["instId_counts"][inst] += 1
    if "instId" in df.columns:
        for inst in df["instId"].dropna().astype(str):
            debug["instId_counts"][inst] += 1
    if "action" in df.columns:
        for action in df["action"].dropna().astype(str):
            debug["action_counts"][action] += 1
    if "bids" in df.columns:
        for item in df["bids"].head(10):
            _update_level_counts(debug, item, side="bids")
    if "asks" in df.columns:
        for item in df["asks"].head(10):
            _update_level_counts(debug, item, side="asks")


def _update_book_debug_from_record(debug: Optional[Dict], record: Dict) -> None:
    if debug is None:
        return
    inst = record.get("instId") or record.get("instrument_name")
    if inst is not None:
        debug["instId_counts"][str(inst)] += 1
    action = record.get("action")
    if action is not None:
        debug["action_counts"][str(action)] += 1
    bids = record.get("bids")
    asks = record.get("asks")
    if bids is not None:
        _update_level_counts(debug, bids, side="bids")
    if asks is not None:
        _update_level_counts(debug, asks, side="asks")


def _update_level_counts(debug: Dict, item, side: str) -> None:
    try:
        arr = item if isinstance(item, list) else json.loads(item)
    except json.JSONDecodeError:
        return
    if not isinstance(arr, list):
        return
    level_count = len(arr)
    if side == "bids":
        debug["bid_levels_max"] = max(debug["bid_levels_max"], level_count)
    else:
        debug["ask_levels_max"] = max(debug["ask_levels_max"], level_count)
    if level_count > 0 and isinstance(arr[0], list):
        cols = len(arr[0])
        key = str(cols)
        if side == "bids":
            debug["bid_level_columns"][key] += 1
        else:
            debug["ask_level_columns"][key] += 1


def _parse_folder(
    folder: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    store_levels: int,
    dataset: str,
    run_manifest: Dict,
    book_debug: Optional[Dict] = None,
) -> pd.DataFrame:
    files = scan_files(folder)
    if not files:
        return pd.DataFrame()

    frames = []
    unknown_side_total = 0
    source_depth = 0

    for path in files:
        sniff = sniff_file(path)
        if sniff.data_format == "unknown":
            raise ValueError(f"Unknown format for {dataset}: {path}")

        df = _read_dataset_file(path, sniff)
        rows_parsed = len(df)

        if dataset == "klines_1m":
            parsed = parse_candles(df, symbol)
        elif dataset == "agg_trades":
            parsed, unknown_side = parse_trades(df, symbol)
            unknown_side_total += unknown_side
        elif dataset == "book_depth":
            _update_book_debug_from_df(book_debug, df)
            parsed, source_depth = parse_book_depth(df, symbol, store_levels)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        filtered = _filter_time(parsed, start_ms, end_ms)
        rows_kept = len(filtered)
        if dataset == "book_depth" and book_debug is not None:
            book_debug["rows_seen"] += rows_parsed
            book_debug["rows_parsed_ok"] += rows_parsed
            book_debug["rows_kept"] += rows_kept
            _update_book_debug_ts(book_debug, parsed["ts"])
        frames.append(filtered)

        run_manifest["files"].append(
            {
                "dataset": dataset,
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "container": sniff.container,
                "format": sniff.data_format,
                "inner_name": sniff.inner_name,
                "rows_parsed": rows_parsed,
                "rows_kept": rows_kept,
            }
        )

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).sort_values("ts").drop_duplicates()
    if dataset == "agg_trades" and unknown_side_total > 0:
        run_manifest.setdefault("warnings", []).append(
            {"dataset": "agg_trades", "unknown_side_rows": int(unknown_side_total)}
        )
    if dataset == "book_depth":
        run_manifest.setdefault("metadata", {})["source_depth"] = int(source_depth)

    return out

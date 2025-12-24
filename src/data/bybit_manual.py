from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from zipfile import ZipFile

import pandas as pd

from src.data.schemas import L2_LEVELS, SCHEMAS, enforce_schema

try:
    import orjson as _fast_json

    def _loads_json(payload):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return _fast_json.loads(payload)

except ImportError:  # pragma: no cover - optional speedup

    def _loads_json(payload):
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="ignore")
        return json.loads(payload)


@dataclass(frozen=True)
class SniffResult:
    container: str
    data_format: str
    inner_name: Optional[str]


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


def scan_orderbook_files(root: Path, symbol: str) -> List[Path]:
    if not root.exists():
        return []
    sym = symbol.lower()
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if sym in name and "ob" in name:
            for fpath in path.rglob("*"):
                if fpath.is_file() and not _should_skip(fpath):
                    files.append(fpath)
    return sorted(set(files), key=lambda p: str(p).lower())


def _extract_date_from_path(path: Path) -> Optional[str]:
    for part in (path.name, path.parent.name):
        match = re.search(r"\d{4}-\d{2}-\d{2}", part)
        if match:
            return match.group(0)
    return None


def _date_in_window(date_str: str, start_ms: int, end_ms: int) -> bool:
    try:
        ts = pd.Timestamp(date_str, tz="UTC")
    except (ValueError, TypeError):
        return True
    if end_ms <= start_ms:
        return False
    try:
        start = pd.Timestamp(start_ms, unit="ms", tz="UTC").normalize()
        end_exclusive = pd.Timestamp(end_ms - 1, unit="ms", tz="UTC").normalize() + pd.Timedelta(days=1)
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return True
    return start <= ts < end_exclusive


def detect_bybit_trades_gz(root: Path, symbol: str) -> List[Tuple[str, Path]]:
    if not root.exists():
        return []
    pattern = re.compile(
        rf"^{re.escape(symbol)}_?(\d{{4}}-\d{{2}}-\d{{2}})\.csv\.gz$",
        re.IGNORECASE,
    )
    matches: List[Tuple[str, Path]] = []
    for path in root.rglob("*.csv.gz"):
        if _should_skip(path):
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        matches.append((match.group(1), path))
    return sorted(matches, key=lambda item: (item[0], str(item[1]).lower()))


def iter_bybit_trades_gz_strict(
    root: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    chunksize: int = 2_000_000,
    debug: Optional[Dict[str, object]] = None,
    files: Optional[List[Tuple[str, Path]]] = None,
) -> Iterator[pd.DataFrame]:
    files = files or detect_bybit_trades_gz(root, symbol)
    if not files:
        return
    files = [item for item in files if _date_in_window(item[0], start_ms, end_ms)]
    if not files:
        return
    symbol_upper = str(symbol).upper()

    for date_str, path in files:
        reader = pd.read_csv(path, compression="gzip", chunksize=chunksize, dtype=str)
        col_map: Dict[str, str] = {}
        trade_id_col: Optional[str] = None
        required_cols = ["timestamp", "symbol", "side", "price", "size"]
        validated = False
        file_rows_kept = 0

        for chunk in reader:
            if chunk.empty:
                continue
            if not validated:
                col_map = {c.strip().lower(): c for c in chunk.columns if c is not None}
                missing = [name for name in required_cols if name not in col_map]
                if missing:
                    present = [str(c) for c in chunk.columns]
                    raise ValueError(
                        "Bybit trades missing columns in "
                        f"{path}: missing={missing} present={present}"
                    )
                trade_id_col = col_map.get("trdmatchid")
                validated = True

            ts_raw = pd.to_numeric(chunk[col_map["timestamp"]], errors="coerce")
            ts = ts_raw.where(ts_raw >= 10**12, ts_raw * 1000)
            price = pd.to_numeric(chunk[col_map["price"]], errors="coerce")
            size = pd.to_numeric(chunk[col_map["size"]], errors="coerce")
            sym = chunk[col_map["symbol"]].astype(str).str.upper()
            side_raw = chunk[col_map["side"]].astype(str).str.strip().str.lower()
            side = side_raw.replace({"b": "buy", "buy": "buy", "s": "sell", "sell": "sell"})

            if debug is not None:
                debug["rows_seen_total"] = int(debug.get("rows_seen_total", 0)) + int(len(chunk))
                valid_ts = ts.dropna()
                if not valid_ts.empty:
                    min_ts = int(valid_ts.min())
                    max_ts = int(valid_ts.max())
                    debug["min_ts_seen"] = (
                        min_ts
                        if debug.get("min_ts_seen") is None
                        else min(int(debug["min_ts_seen"]), min_ts)
                    )
                    debug["max_ts_seen"] = (
                        max_ts
                        if debug.get("max_ts_seen") is None
                        else max(int(debug["max_ts_seen"]), max_ts)
                    )

                invalid = debug.get("invalid_rows_dropped", {})
                invalid["bad_ts"] = int(invalid.get("bad_ts", 0)) + int(ts_raw.isna().sum())
                invalid["bad_price"] = int(invalid.get("bad_price", 0)) + int(
                    price.isna().sum() + (price <= 0).sum()
                )
                invalid["bad_size"] = int(invalid.get("bad_size", 0)) + int(
                    size.isna().sum() + (size <= 0).sum()
                )
                invalid["bad_side"] = int(invalid.get("bad_side", 0)) + int(
                    (~side.isin({"buy", "sell"})).sum()
                )
                invalid["bad_symbol"] = int(invalid.get("bad_symbol", 0)) + int(
                    (sym != symbol_upper).sum()
                )
                debug["invalid_rows_dropped"] = invalid

            mask = (
                ts.notna()
                & price.notna()
                & size.notna()
                & (sym == symbol_upper)
                & (ts >= start_ms)
                & (ts < end_ms)
                & (price > 0)
                & (size > 0)
                & side.isin({"buy", "sell"})
            )
            if not mask.any():
                continue

            index = mask[mask].index
            trade_id = None
            if trade_id_col is not None:
                raw_trade_id = chunk.loc[index, trade_id_col]
                trade_id = raw_trade_id.where(raw_trade_id.notna(), None).astype("string")
            else:
                trade_id = pd.Series([None] * len(index), index=index, dtype="string")

            out = pd.DataFrame(
                {
                    "ts_ms": ts.loc[index].astype("int64"),
                    "symbol": pd.Series(symbol_upper, index=index, dtype="string"),
                    "side": side.loc[index].astype("string"),
                    "price": price.loc[index].astype("float64"),
                    "size": size.loc[index].astype("float64"),
                    "trade_id": trade_id.astype("string"),
                    "raw_source": "bybit_csv_gz",
                }
            )
            out = out.sort_values("ts_ms", kind="mergesort")
            out = out.drop_duplicates(
                subset=["ts_ms", "side", "price", "size", "trade_id"], keep="first"
            )
            file_rows_kept += len(out)

            if debug is not None and not out.empty:
                debug["rows_kept_total"] = int(debug.get("rows_kept_total", 0)) + int(len(out))
                min_kept = int(out["ts_ms"].min())
                max_kept = int(out["ts_ms"].max())
                debug["min_ts_kept"] = (
                    min_kept
                    if debug.get("min_ts_kept") is None
                    else min(int(debug["min_ts_kept"]), min_kept)
                )
                debug["max_ts_kept"] = (
                    max_kept
                    if debug.get("max_ts_kept") is None
                    else max(int(debug["max_ts_kept"]), max_kept)
                )
                sample = debug.get("sample_rows", [])
                if isinstance(sample, list) and len(sample) < 5:
                    take = min(5 - len(sample), len(out))
                    sample.extend(out.head(take).to_dict("records"))
                    debug["sample_rows"] = sample
            yield out

        if debug is not None and file_rows_kept > 0:
            used = debug.get("files_used", [])
            used.append({"date": date_str, "path": str(path.resolve())})
            debug["files_used"] = used


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


def _iter_records(path: Path, sniff: SniffResult) -> Iterator[Dict]:
    handle, parent = _open_inner(path, sniff)
    try:
        if sniff.data_format == "jsonl":
            for raw in handle:
                line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw
                line = line.strip()
                if not line:
                    continue
                try:
                    record = _loads_json(line)
                except Exception:
                    continue
                if isinstance(record, dict):
                    yield record
            return

        if sniff.data_format == "json":
            wrapper = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
            parsed_any = False
            for raw in wrapper:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = _loads_json(raw)
                except Exception:
                    parsed_any = False
                    break
                if isinstance(record, dict):
                    parsed_any = True
                    yield record
            if parsed_any:
                return

            max_bytes = 50 * 1024 * 1024
            try:
                size = path.stat().st_size
            except OSError:
                size = max_bytes + 1
            if size > max_bytes:
                return

            handle.close()
            if parent is not None:
                parent.close()
            handle, parent = _open_inner(path, sniff)
            try:
                text = handle.read().decode("utf-8", errors="ignore")
            except OSError:
                return
            try:
                data = _loads_json(text)
            except Exception:
                return
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
                return
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        if isinstance(item, dict):
                            yield item
                    return
                yield data
            return

        if sniff.data_format == "csv":
            wrapper = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
            reader = csv.DictReader(wrapper)
            for row in reader:
                if not row:
                    continue
                if "data" in row and row["data"]:
                    try:
                        data = _loads_json(row["data"])
                    except Exception:
                        data = None
                    record = {k: v for k, v in row.items() if k != "data"}
                    if data is not None:
                        record["data"] = data
                    yield record
                else:
                    yield row
    finally:
        handle.close()
        if parent is not None:
            parent.close()


def _read_dataset_file(path: Path, sniff: SniffResult) -> pd.DataFrame:
    if sniff.data_format == "csv":
        handle, parent = _open_inner(path, sniff)
        try:
            return pd.read_csv(handle)
        finally:
            handle.close()
            if parent is not None:
                parent.close()
    rows = list(_iter_records(path, sniff))
    return pd.DataFrame(rows)


def _parse_ts_ms(value) -> Optional[int]:
    if value is None:
        return None
    try:
        ts = int(float(value))
    except (TypeError, ValueError):
        return None
    return ts if ts >= 10**12 else ts * 1000


def _parse_level_list(raw) -> List[Tuple[float, float]]:
    if not raw:
        return []
    out: List[Tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            px = float(item[0])
            sz = float(item[1])
        except (TypeError, ValueError):
            continue
        out.append((px, sz))
    return out


class _OrderBookState:
    def __init__(self) -> None:
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}

    def apply_snapshot(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        self.bids = {px: sz for px, sz in bids if sz > 0}
        self.asks = {px: sz for px, sz in asks if sz > 0}

    def apply_delta(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        for px, sz in bids:
            if sz <= 0:
                self.bids.pop(px, None)
            else:
                self.bids[px] = sz
        for px, sz in asks:
            if sz <= 0:
                self.asks.pop(px, None)
            else:
                self.asks[px] = sz


def _emit_snapshot_row(
    ts_ms: int,
    symbol: str,
    state: _OrderBookState,
    store_levels: int,
    update_id: int,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    bids = sorted(state.bids.items(), key=lambda x: x[0], reverse=True)[:store_levels]
    asks = sorted(state.asks.items(), key=lambda x: x[0])[:store_levels]
    row: Dict[str, object] = {"ts": ts_ms, "update_id": update_id, "symbol": symbol}
    for lvl in range(1, store_levels + 1):
        if lvl <= len(bids):
            px, sz = bids[lvl - 1]
            row[f"bid_price_{lvl}"] = float(px)
            row[f"bid_size_{lvl}"] = float(sz)
        else:
            row[f"bid_price_{lvl}"] = float("nan")
            row[f"bid_size_{lvl}"] = float("nan")
        if lvl <= len(asks):
            px, sz = asks[lvl - 1]
            row[f"ask_price_{lvl}"] = float(px)
            row[f"ask_size_{lvl}"] = float(sz)
        else:
            row[f"ask_price_{lvl}"] = float("nan")
            row[f"ask_size_{lvl}"] = float("nan")
    if store_levels < L2_LEVELS:
        for lvl in range(store_levels + 1, L2_LEVELS + 1):
            row[f"bid_price_{lvl}"] = float("nan")
            row[f"bid_size_{lvl}"] = float("nan")
            row[f"ask_price_{lvl}"] = float("nan")
            row[f"ask_size_{lvl}"] = float("nan")
    row.update(metadata)
    return row


def _init_debug(start_ms: int, end_ms: int, start_str: str, end_str: str) -> Dict:
    return {
        "files_scanned": [],
        "files_skipped_out_of_range": 0,
        "rows_seen": 0,
        "rows_emitted": 0,
        "rows_kept": 0,
        "min_ts_seen": None,
        "max_ts_seen": None,
        "min_ts_emitted": None,
        "max_ts_emitted": None,
        "symbols": Counter(),
        "type_counts": Counter(),
        "u_eq_1_count": 0,
        "requested_start": start_str,
        "requested_end": end_str,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
    }


def _init_sampling_debug(start_ms: int, end_ms: int, start_str: str, end_str: str, sample_ms: int) -> Dict:
    return {
        "events_seen": 0,
        "snapshots_seen": 0,
        "deltas_seen": 0,
        "snapshots_emitted": 0,
        "min_ts": None,
        "max_ts": None,
        "gap_count": 0,
        "resync_count": 0,
        "sample_ms": int(sample_ms),
        "requested_start": start_str,
        "requested_end": end_str,
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
    }


def _finalize_debug(debug: Dict) -> Dict:
    def _to_utc(value: Optional[int]) -> Optional[str]:
        if value is None:
            return None
        return pd.to_datetime(value, unit="ms", utc=True).isoformat()

    debug["min_ts_utc"] = _to_utc(debug.get("min_ts_seen"))
    debug["max_ts_utc"] = _to_utc(debug.get("max_ts_seen"))
    debug["min_ts_emitted_utc"] = _to_utc(debug.get("min_ts_emitted"))
    debug["max_ts_emitted_utc"] = _to_utc(debug.get("max_ts_emitted"))

    debug["symbols"] = {k: int(v) for k, v in debug.get("symbols", {}).items()}
    debug["type_counts"] = {k: int(v) for k, v in debug.get("type_counts", {}).items()}
    return debug


def _finalize_sampling_debug(debug: Dict) -> Dict:
    def _to_utc(value: Optional[int]) -> Optional[str]:
        if value is None:
            return None
        return pd.to_datetime(value, unit="ms", utc=True).isoformat()

    debug["min_ts_utc"] = _to_utc(debug.get("min_ts"))
    debug["max_ts_utc"] = _to_utc(debug.get("max_ts"))
    return debug


def _write_debug(debug: Dict, diagnostics_dir: Optional[Path]) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _finalize_debug(debug)
    path = diagnostics_dir / "bybit_book_depth_debug.json"
    path.write_text(json.dumps(debug, indent=2), encoding="utf-8")


def _write_sampling_debug(debug: Dict, diagnostics_dir: Optional[Path]) -> None:
    if diagnostics_dir is None:
        return
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _finalize_sampling_debug(debug)
    path = diagnostics_dir / "bybit_book_depth_sampling_debug.json"
    path.write_text(json.dumps(debug, indent=2), encoding="utf-8")


def _update_min_max(debug: Dict, key_min: str, key_max: str, ts: int) -> None:
    current_min = debug.get(key_min)
    current_max = debug.get(key_max)
    debug[key_min] = ts if current_min is None else min(current_min, ts)
    debug[key_max] = ts if current_max is None else max(current_max, ts)


def load_bybit_manual_book_depth(
    book_root: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    store_levels: int,
    diagnostics_dir: Optional[Path] = None,
    emit_every_delta: bool = False,
    book_sample_ms: int = 1000,
    start_str: str = "",
    end_str: str = "",
) -> Tuple[pd.DataFrame, Dict]:
    store_levels = min(int(store_levels), L2_LEVELS)
    files = scan_orderbook_files(book_root, symbol)
    debug = _init_debug(start_ms, end_ms, start_str, end_str)
    sample_ms = max(1, int(book_sample_ms))
    sampling_debug = _init_sampling_debug(start_ms, end_ms, start_str, end_str, sample_ms)
    filtered_files: List[Path] = []
    for path in files:
        date_str = _extract_date_from_path(path)
        if date_str and not _date_in_window(date_str, start_ms, end_ms):
            debug["files_skipped_out_of_range"] += 1
            continue
        filtered_files.append(path)
    files = filtered_files
    if not files and debug["files_skipped_out_of_range"] > 0:
        _write_debug(debug, diagnostics_dir)
        _write_sampling_debug(sampling_debug, diagnostics_dir)
        raise ValueError(
            "All book_depth rows excluded by date filter. "
            f"Requested: [{start_str},{end_str}). Fix: set end=next_day or add --end-inclusive-date."
        )
    rows: List[Dict[str, object]] = []
    state = _OrderBookState()
    row_counter = 0
    last_emit_bucket: Optional[int] = None
    last_seq: Optional[int] = None
    state_valid = False

    for path in files:
        sniff = sniff_file(path)
        debug["files_scanned"].append(
            {
                "dataset": "book_depth",
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "container": sniff.container,
                "format": sniff.data_format,
                "inner_name": sniff.inner_name,
            }
        )
        for record in _iter_records(path, sniff):
            debug["rows_seen"] += 1
            if not isinstance(record, dict):
                continue
            data = record.get("data")
            if not isinstance(data, dict):
                continue
            ts_ms = _parse_ts_ms(record.get("ts") or data.get("ts"))
            if ts_ms is None:
                continue
            sampling_debug["events_seen"] += 1
            _update_min_max(sampling_debug, "min_ts", "max_ts", ts_ms)

            symbol_raw = data.get("s") or record.get("symbol") or record.get("instrument")
            if symbol_raw:
                debug["symbols"][str(symbol_raw)] += 1
                if str(symbol_raw).upper() != str(symbol).upper():
                    continue
            event_type = record.get("type") or record.get("event_type") or ""
            event_type = str(event_type).lower()
            if event_type:
                debug["type_counts"][event_type] += 1

            update_id = data.get("u")
            seq = data.get("seq")
            if update_id == 1 or update_id == "1":
                debug["u_eq_1_count"] += 1

            bids = _parse_level_list(data.get("b"))
            asks = _parse_level_list(data.get("a"))

            _update_min_max(debug, "min_ts_seen", "max_ts_seen", ts_ms)

            treat_snapshot = event_type == "snapshot" or update_id == 1 or update_id == "1"
            if treat_snapshot:
                sampling_debug["snapshots_seen"] += 1
                state.apply_snapshot(bids, asks)
                if not state_valid:
                    sampling_debug["resync_count"] += 1
                state_valid = True
            else:
                sampling_debug["deltas_seen"] += 1
                if not state_valid:
                    continue
                seq_val = None
                for candidate in (seq, update_id):
                    try:
                        seq_val = int(candidate)
                        break
                    except (TypeError, ValueError):
                        continue
                if seq_val is not None and last_seq is not None:
                    if seq_val <= last_seq:
                        sampling_debug["gap_count"] += 1
                        state_valid = False
                        last_seq = seq_val
                        continue
                if seq_val is not None:
                    last_seq = seq_val
                state.apply_delta(bids, asks)

            if treat_snapshot:
                try:
                    last_seq = int(seq)
                except (TypeError, ValueError):
                    last_seq = last_seq

            should_emit = False
            if emit_every_delta:
                should_emit = True
            else:
                bucket = int(ts_ms // sample_ms)
                if last_emit_bucket is None or bucket != last_emit_bucket:
                    should_emit = True
                    last_emit_bucket = bucket
            if not should_emit:
                continue

            row_counter += 1
            update_id_final = int(ts_ms) * 1_000_000 + row_counter

            metadata = {
                "source_depth": int(max(len(bids), len(asks), 0)),
                "stored_depth": int(store_levels),
                "seq": seq,
                "u": update_id,
                "event_type": event_type or ("snapshot" if treat_snapshot else "delta"),
            }
            row = _emit_snapshot_row(
                ts_ms=ts_ms,
                symbol=str(symbol),
                state=state,
                store_levels=store_levels,
                update_id=update_id_final,
                metadata=metadata,
            )
            rows.append(row)
            debug["rows_emitted"] += 1
            sampling_debug["snapshots_emitted"] += 1
            _update_min_max(debug, "min_ts_emitted", "max_ts_emitted", ts_ms)

    df = pd.DataFrame(rows)
    rows_kept = 0
    if not df.empty:
        df = df.sort_values("ts")
        df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]
        rows_kept = len(df)
    debug["rows_kept"] = rows_kept
    _write_debug(debug, diagnostics_dir)
    _write_sampling_debug(sampling_debug, diagnostics_dir)

    if debug["rows_seen"] > 0 and debug["rows_emitted"] == 0:
        min_utc = debug.get("min_ts_utc", "unknown")
        max_utc = debug.get("max_ts_utc", "unknown")
        raise ValueError(
            f"No book_depth snapshots emitted. Observed ts range: {min_utc}.. {max_utc}. "
            f"Requested: [{start_str},{end_str}). Check emit policy or format."
        )
    if debug["rows_seen"] > 0 and rows_kept == 0:
        min_utc = debug.get("min_ts_utc", "unknown")
        max_utc = debug.get("max_ts_utc", "unknown")
        raise ValueError(
            f"All book_depth rows excluded by date filter. Observed ts range: {min_utc}.. {max_utc}. "
            f"Requested: [{start_str},{end_str}). Fix: set end=next_day or add --end-inclusive-date."
        )

    if df.empty:
        return df, debug

    return df, debug


def parse_candles(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    rename = {
        "ts": "ts",
        "timestamp": "ts",
        "open_time": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vol": "volume",
        "turnover": "quote_volume",
        "quote_volume": "quote_volume",
    }
    df = df.rename(columns=rename)
    required = ["ts", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bybit candles missing columns: {missing}")

    df["ts"] = df["ts"].apply(_parse_ts_ms)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df["ts"] = df["ts"].astype("int64")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    if "quote_volume" in df.columns:
        df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce").fillna(0.0)
    else:
        df["quote_volume"] = 0.0
    df["close_ts"] = df["ts"].astype("int64") + 60_000
    df["trade_count"] = 0
    df["taker_buy_base"] = 0.0
    df["taker_buy_quote"] = 0.0
    df["symbol"] = symbol
    return enforce_schema(df, SCHEMAS["klines_1m"])


def parse_trades(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, int]:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    rename = {
        "id": "agg_id",
        "trade_id": "agg_id",
        "tradeid": "agg_id",
        "ts": "ts",
        "timestamp": "ts",
        "time": "ts",
        "price": "price",
        "px": "price",
        "qty": "qty",
        "size": "qty",
        "sz": "qty",
        "volume": "qty",
        "side": "side",
    }
    df = df.rename(columns=rename)
    required = ["ts", "price", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bybit trades missing columns: {missing}")

    if "agg_id" not in df.columns:
        df["agg_id"] = range(1, len(df) + 1)

    df["ts"] = df["ts"].apply(_parse_ts_ms)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["ts", "price", "qty"])
    df["ts"] = df["ts"].astype("int64")
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


def _filter_time(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return df
    return df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]


def _date_coverage(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "ts" not in df.columns:
        return {}
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
    counts = ts.dt.strftime("%Y-%m-%d").value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def load_bybit_manual_datasets(
    book_root: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    store_levels: int,
    book_sample_ms: int = 1000,
    diagnostics_dir: Optional[Path] = None,
    emit_every_delta: bool = False,
    start_str: str = "",
    end_str: str = "",
    candles_dir: Optional[Path] = None,
    trades_dir: Optional[Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    store_levels = min(int(store_levels), L2_LEVELS)
    datasets: Dict[str, pd.DataFrame] = {}
    run_manifest = {
        "source": "bybit_manual",
        "symbol": symbol,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "files": [],
        "datasets": {},
    }

    if candles_dir:
        frames = []
        for path in scan_files(candles_dir):
            sniff = sniff_file(path)
            df = _read_dataset_file(path, sniff)
            try:
                parsed = parse_candles(df, symbol)
            except ValueError as exc:
                run_manifest.setdefault("warnings", []).append(
                    {"dataset": "klines_1m", "path": str(path.resolve()), "error": str(exc)}
                )
                continue
            filtered = _filter_time(parsed, start_ms, end_ms)
            frames.append(filtered)
            run_manifest["files"].append(
                {
                    "dataset": "klines_1m",
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                    "container": sniff.container,
                    "format": sniff.data_format,
                    "inner_name": sniff.inner_name,
                    "rows_parsed": len(parsed),
                    "rows_kept": len(filtered),
                }
            )
        datasets["klines_1m"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if trades_dir:
        frames = []
        unknown_side = 0
        for path in scan_files(trades_dir):
            sniff = sniff_file(path)
            df = _read_dataset_file(path, sniff)
            try:
                parsed, unknown = parse_trades(df, symbol)
            except ValueError as exc:
                run_manifest.setdefault("warnings", []).append(
                    {"dataset": "agg_trades", "path": str(path.resolve()), "error": str(exc)}
                )
                continue
            filtered = _filter_time(parsed, start_ms, end_ms)
            frames.append(filtered)
            unknown_side += unknown
            run_manifest["files"].append(
                {
                    "dataset": "agg_trades",
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                    "container": sniff.container,
                    "format": sniff.data_format,
                    "inner_name": sniff.inner_name,
                    "rows_parsed": len(parsed),
                    "rows_kept": len(filtered),
                }
            )
        datasets["agg_trades"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if unknown_side:
            run_manifest.setdefault("warnings", []).append(
                {"dataset": "agg_trades", "unknown_side": int(unknown_side)}
            )

    if book_root:
        book_df, debug = load_bybit_manual_book_depth(
            book_root=book_root,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            store_levels=store_levels,
            diagnostics_dir=diagnostics_dir,
            emit_every_delta=emit_every_delta,
            book_sample_ms=book_sample_ms,
            start_str=start_str,
            end_str=end_str,
        )
        datasets["book_depth"] = book_df
        run_manifest["files"].extend(debug.get("files_scanned", []))
        run_manifest.setdefault("metadata", {})["book_depth_debug"] = {
            "rows_seen": debug.get("rows_seen", 0),
            "rows_emitted": debug.get("rows_emitted", 0),
            "rows_kept": debug.get("rows_kept", 0),
            "min_ts_seen": debug.get("min_ts_seen"),
            "max_ts_seen": debug.get("max_ts_seen"),
        }

    for name, df in datasets.items():
        run_manifest["datasets"][name] = {"rows": int(len(df)), "date_coverage": _date_coverage(df)}

    run_manifest["stored_depth"] = store_levels
    return datasets, run_manifest
